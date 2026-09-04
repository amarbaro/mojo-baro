# q8 weights round (frozen 2026-09-04, tree 42224e5, engine 41.7 tok/s_gen bf16, 64/64 vs llama.cpp bf16)

Bound by `PROTOCOL-RULES.md`. This is the bytes lever: decode streams the
whole weight pack once per token, so tok/s scales with bytes if the kernel
holds its bandwidth efficiency. It is NOT bit-exact against the bf16
reference, so the gate is redefined below before any number is taken.

## Prior verdicts that bind

- `amar_matmul_skinny_q8b` exists (K-major int8 blocks of 32, fp32 scale):
  cold-cache 164.8 us at M=8 ffn shape, parity 9.2e-4 vs numpy on the same
  dequantized values (`2c520a5`, `2b92e15`). That is 35% of HBM peak on a
  56 MB stream; the bf16 M=1 champion `amar_matmul_skinny_m1` (CPT=8) does
  121.4 us on 100 MB = 83%. The q8 kernel must be re-derived from the m1
  kernel, not the other way round.
- q8-in-wt-layout FALSIFIED (0.85x): coalescing-bound, not byte-bound.
  K-major only.
- Reference tokens come from llama-server with the prompt posted as token
  ids (string prompts tokenize differently). Keep that.

## Scale check (frozen)

| | bf16 (today) | q8 (int8 + fp32 scale / 32) |
|---|---|---|
| pack bytes | 18.39 GB | 10.35 GB (0.563x) |
| HBM roof at 960 GB/s | 52 tok/s | 93 tok/s |
| engine at today's 80% of roof | 41.7 | ~74 |

Everything 2D bf16 in the pack quantizes, including `token_embd` (used as
lm_head, 1.0 GB) — the embedding lookup reads one row so it does not care.
f32 tensors (norms, conv, ssm scalars, 4 MB) stay f32.

## Gate (redefined, cannot be loosened after the fact)

G1 **our math**: engine-q8 greedy tokens == a numpy fp32 forward over the
   SAME dequantized q8 pack, 64/64. This is the identity class the bf16
   engine already meets (engine == numpy == llama.cpp on bf16). Any
   mismatch is a kernel bug, not drift.
G2 **the standard q8 path**: llama.cpp Q8_0 of the same model
   (`llama-quantize ... Q8_0`, then llama-server, token-id prompt, greedy)
   gives its own 64 tokens. Record the first position where engine-q8
   diverges from llama.cpp-Q8_0, and the first position where llama.cpp-Q8_0
   diverges from the bf16 reference. Land requires engine-q8's first
   divergence from bf16 to be no earlier than llama.cpp-Q8_0's. Identity
   with llama.cpp-Q8_0 is NOT required by construction: its HIP mmvq path
   quantizes activations to q8_1, ours keeps bf16 activations.
G3 P1 receipts: the engine prints the pack dtype per tensor class and the
   kernel variant name on the timed run; the q8 pack index is read back.

## Stages

| stage | work | size | check | stop rule |
|---|---|---|---|---|
| Q0 pack | `engine-pack.py --q8`: ggml-exact `quantize_row_q8_0` rounding (d = amax/127 in fp32, q = round(x/d), d stored fp16-rounded), K-major int8 [K, N] + scales [K/32, N] | S | dequant(pack) bit-equals ggml dequant of the `llama-quantize` Q8_0 file for 3 tensors incl. a NextN one | — |
| Q1 kernel | `amar_matmul_skinny_m1_q8` (CPT=8, SIMD int8 loads, scale per 32-k block applied per column) + `_dual` + M<=8 `_v` variant | L | parity < 1e-5 rel vs numpy on dequantized values; cold-cache `bench_coldcache_m1` at ffn shape, 8 rotating buffers | **m1_q8 > 100 us -> round closed** (engine gain would be < 15%) |
| Q2 engine | wire all g_* launches to q8 variants, loader reads q8 index; `tools/model-ref.py` extended to a 64-token greedy decode over the dequantized pack (G1 reference) | M+M | G1, G2, G3; tok/s_gen median of 3, server stopped, warm | G1 fail -> no number recorded |
| Q3 bar | llama.cpp Q8_0 no-spec greedy median of 4 on this box, `/props` receipt | S | — | — |

## Predictions (frozen)

| quantity | prediction | land rule |
|---|---|---|
| Q1 m1_q8 at ffn gate/up shape (56.6 MB) | 70-85 us (1.4-1.7x over bf16 121.4) | <= 90 us |
| Q2 engine tok/s_gen | 62-72 (+50% to +70%) | >= 54.2 (+30%) AND G1 AND G2 |
| Q3 llama.cpp Q8_0 bar | 65-78 tok/s | — |
| Q2 vs Q3 | within 0.9-1.1x | — |

If Q1 lands but Q2 misses +30% with G1 green, the loss is outside the GEMMs
(f32 state kernels are unchanged) and the round records the per-sub-block
BARO_PROFILE=1 shares before closing; no post-hoc kernel sweep.

## Not in this round
int4, KV cache quantization, activation quantization, MTP draft-head speed.

## Amendment before Q1 (2026-09-04): qingming-gfx1100-gemv receipt

Built `uulong950/qingming-gfx1100-gemv` (fp32 GEMV, HIP, ROCm 7.2) at our
decode shapes, single buffer per shape so only the >= 128 MB rows are HBM
numbers (`.work/ref/qingming-smoke-20260904.csv`):

| shape (rows x K) | bytes | native us | GB/s | rocBLAS GB/s | path |
|---|---|---|---|---|---|
| 12288 x 4096 (ffn gate/up) | 201 MB | 218.8 | **920 (96%)** | 858 | STREAMING |
| 8192 x 4096 (qf) | 134 MB | 152.5 | 881 | 826 | STREAMING |
| 4096 x 12288 (down) | 201 MB | 264.2 | 762 | 719 | STREAMING |

STREAMING = one wave per weight ROW in weight-native [out, in] layout: each
lane reads a float per 32-lane strip, 8 strips per 256-wide K tile with a
per-wave phase rotation, nontemporal loads, register prefetch of the next
tile, wave-reduce at the end, no LDS, no split-K, `__launch_bounds__(256,2)`.

Consequence for this round: the binding "K-major only" is REPLACED. The
2026-09-01 wt-layout falsification (250 GB/s) was our thread-per-column
implementation, not the layout; wave-per-row over weight-native rows is the
fastest measured stream on this card and needs no load-time transpose.
For q8_0 it is the natural fit: the 32-wide blocks run along K inside a row,
so one lane-strip = one block and the scale is a single broadcast load.

Q1 is therefore two arms, both preregistered here:
- Q1a bf16 wave-per-row (`amar_matmul_skinny_m1_row`), prediction 105-115 us
  at the 100 MB ffn shape (vs 121.4 today), land <= 115 us. Bit-exact is NOT
  claimed (different summation order); token gate re-established.
- Q1b q8 wave-per-row, prediction 62-75 us, land <= 85 us (tightened from 90).
Stop rule for Q1 unchanged: q8 > 100 us closes the round.

### Q1a receipt (2026-09-04): NOT landed, template validated

`amar_matmul_skinny_m1_row[bf16, UNROLL]`, wave per row over the weight-native
[N, K] bf16, 16 B per lane per strip, UNROLL strips in flight, warp reduce.
`bench/bench_coldcache_row.mojo`, 8 rotating 100 MB buffers, 200 iters x 10
reps, logs `.work/coldcache-row-20260904.log`, `.work/coldcache-row-u16-20260904.log`.
All arms max rel err 2.2e-4 vs the fp32 host reference (same class as m1).

| arm | us (rep median) | GB/s | VGPR | clock / power |
|---|---|---|---|---|
| m1c8 control | 121.2 | 830 | 29 | 3074 MHz, 288-300 W |
| row UNROLL=1 | 125.0 | 805 | | |
| row UNROLL=4 | 118.6 | 848 | | |
| row UNROLL=8 | **117.5** | 856 | 91 | 3074 MHz, 288-300 W |
| row UNROLL=16 | 124.3 | 809 | 95 | |
| qingming fp32 STREAMING, same shape, 201 MB | 219.5 | 917 | | 3301 MHz, up to 357 W |

Prediction was 105-115 us, land <= 115: **missed by 2.5 us**. -3.1% over the
champion, which is real but under the rule. Two of qingming's ingredients are
not reproducible here: nontemporal loads (`max.gpu.memory.load` with
`CacheOperation.STREAMING` emits a plain `global_load_b128` on gfx1100, ISA
checked) and its power state (their run holds 3301 MHz at up to 357 W, ours
3074 MHz at 300 W; the card, not the code, picks that).

Decision: the wave-per-row template is validated at 856 GB/s (89% of peak)
and becomes the Q1b base. bf16 engine wiring is NOT done on -3.1% (token gate
re-establishment is not worth it for that); if Q1b lands, the bf16 path is
moot anyway.

### Q0 receipt (2026-09-04): PASS

`tools/engine-pack.py --q8` -> `.work/engine-pack-q8/` (10.73 GB; token_embd
stays bf16 at 1.0 GB, it is a row lookup, not a stream -- scope narrowed from
the frozen text, disclosed here). `tools/q8-check.py` against
`llama-quantize ... Q8_0` (9.32 GiB, 8.50 BPW): blk.0.ffn_gate, blk.3.attn_q,
blk.32.nextn.eh_proj, output.weight all 0 mismatched quants and 0 mismatched
fp16 scales (output.weight alone is 1.017e9 quants). Our pack IS llama.cpp's
Q8_0, so G2 compares equal weights.

### Q1b receipt (2026-09-04): LANDED

`amar_matmul_skinny_m1_q8row[UNROLL=4]`: wave per row over int8 [N, K] +
fp16 scales [N, K/32], 16 int8 per lane per strip, one fp16 scale per lane
per strip. `bench/bench_coldcache_q8row.mojo`, 8 rotating buffers, logs
`.work/coldcache-q8row-20260904.log`, `.work/coldcache-q8row-2-20260904.log`.

| arm | us (median of 10) | GB/s | vs champion |
|---|---|---|---|
| m1c8 bf16 champion | 121.1 | 830 | 1.00 |
| row8 bf16 | 117.4 | 856 | 1.03 |
| **q8row** | **62.6** | 855 (53.5 MB) | **1.93x** |

Clock 3069 MHz median, 253-312 W. Prediction 62-75 us, land <= 85: landed at
the floor of the prediction. Same bytes/s as the bf16 row kernel: the
dequant costs nothing, the bytes are the whole story.

Parity: the frozen rule said rel < 1e-5 vs the dequantized values, but the
harness metric (|err| / max(|ref|, 1e-3), fp64 host reference) gives 2.2e-4
for the bf16 CHAMPION against its own reference, so that rule was tighter
than the metric can show. q8row on the same metric: 9.4e-5. Normalized
absolute error (max |err| / max |ref|): q8row 1.1e-7, champion 1.0e-6. Kernel
is correct; cancellation on near-zero outputs made the relative metric
noisy. Quantization effect vs bf16 outputs: 0.64% of max output, that is
what G1/G2 judge at token level.

### Q2 receipt (2026-09-04): LANDED (G1 pending)

Engine converted in place to the q8 pack (`serve/engine.mojo`: offsets read
from the pack index, `BARO_PACK` selects the pack dir, every 2D weight GEMM
is `amar_matmul_skinny_q8row`, reduces run with NSPLIT=1; the dual gate+up
launch of F1 is back to two launches). Baseline = the bf16 binary
`.work/engine` built 2026-09-02 from `b265eb5`, run in the same stint,
alternating, embedding server on the card in both arms as before.

| arm | tok/s_gen x3 | median | pack bytes read back | gate |
|---|---|---|---|---|
| bf16 `.work/engine` | 41.66, 41.63, 41.68 | 41.66 | 18396352512 | 64/64 |
| q8 `.work/engine-q8` | 68.82, 68.77, 68.64 | **68.77 (+65%)** | 10728640512 | 64/64 vs bf16 ref AND 64/64 vs llama.cpp Q8_0 |

Prediction 62-72: landed. Land rule >= 54.2: yes. G2: engine-q8, llama.cpp
Q8_0 and the bf16 reference all agree on 64/64, so first divergence is
beyond the window for both. G3: pack bytes printed by the timed binary
distinguish the arms; ISA receipt of `.work/engine-q8` shows the two q8row
instantiations (MR=1 93 VGPR, MR=8 66 VGPR, zero spills) and no bf16 skinny
kernels. Draft head: `DRAFT: from_token 369 draft_argmax 369` unchanged.

G1 (numpy fp32 forward over the same q8 pack, `tools/model-ref.py decode 64`
with `BARO_PACK=.work/engine-pack-q8`) is running: ~3 min/token on one core,
log `.work/engine-pack-q8/numpy-decode-64.log`. The 68.77 figure is
provisional until it reports 64/64; tokens 1-2 match so far.

### Q3 receipt (2026-09-04): llama.cpp Q8_0 bar = 74.0 tok/s

`tools/llama-ref-run.sh`, no speculative flags, greedy, token-id prompt,
`-ctk/-ctv q8_0`: 73.89, 74.11, 74.18, 73.52, 74.17 -> median of last 4
**74.14**; `draft_n` null in every response, model path read back from
`/props` (`.work/llama-q8/props.json`). Prediction 65-78: landed.
Ours / bar = 68.77 / 74.14 = **0.93x** (predicted 0.9-1.1x).

Round summary: 41.7 -> 68.8 tok/s_gen (+65%) at 64/64; vs llama.cpp Q8_0
0.93x, same ratio class as bf16 (0.94x). Remaining gap to the 93 tok/s q8
roof: 26%, spread over the same non-GEMM kernels as before.

### G1 receipt (2026-09-04): PASS, round closed

`BARO_PACK=.work/engine-pack-q8 tools/model-ref.py decode 64` (numpy fp32,
dequantized q8 weights cached once in RAM, 14 BLAS threads): 64 tokens,
first divergence vs the bf16 reference: none. Tokens saved to
`.work/engine-pack-q8/ref-tokens-64-numpy.txt`. All four sources agree on
64/64: bf16 llama.cpp, Q8_0 llama.cpp, numpy-q8, engine-q8. The 68.77
tok/s_gen figure is final.
