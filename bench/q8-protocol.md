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
