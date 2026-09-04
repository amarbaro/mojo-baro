# Multi-row window protocol — frozen before first run

> Binds `bench/PROTOCOL-RULES.md` P1-P6. Frozen 2026-09-04 on `41d0361`
> (+ the rules commit). Nothing below was measured before it was written
> except the two facts marked (measured).

## Problem on the record (measured)

- A 1-row decode pass costs ~15 ms; a 2-row speculative window ~1.28x that;
  a 4-row prefill window 43 ms = 2.9x (`.work/mr2-A.log`, `41d0361`).
  Bandwidth says 1.05-1.1x: the weights are read once either way.
- Real-prompt MTP at k=2 is 100.7 median (1.47x); llama.cpp 123.5 (1.66x).
  The gap is the window cost, not acceptance (ours 69%, theirs 58%).

## Diagnosis to be confirmed by stage M0, not assumed

`amar_matmul_skinny_q8row[UNROLL, MR]` streams weights once per wave-row but
loads the A slice per row per step from global (L2): A traffic per weight
byte = 2 x MR (bf16 A vs int8 W). At MR = 4 that is 8x the weight stream
through L2. Second suspect: the SSM sub-block runs its 5 kernels per row per
layer (24 layers x 5 x (m-1) extra launches per window).

## Stages

- **M0 receipt (lane B, first).** `bench/bench_coldcache_mrow.mojo`: the
  current q8row at MR = M for M in 1, 2, 4, 8 on the ffn shape (N=12288,
  K=4096), cold-cache 8-buffer rotation as in `bench_coldcache_q8row.mojo`,
  fp64 host check on every row. Prints us and the M-row/1-row ratio. This is
  the P5 receipt and the baseline for M1.
- **M1 kernel (lane A).** `amar_matmul_skinny_q8mrow[MR]`: block stages its
  A rows (MR x K bf16, <= 64 KB at MR = 8) into LDS once, waves stream their
  weight rows from global as q8row does, A comes from LDS. Same fp32 accumulate
  and reduction order per row as q8row so the m=1 path and the engine tokens
  stay bit-identical; gate = M0 bench parity (fp64 host) and engine 64/64.
- **M2 SSM fold (lane C).** The five SSM kernels take `m` and loop rows
  inside (recurrence stays sequential per row, same op order); engine dispatch
  drops the per-row loop for the SSM sub-block. Bit-exact by construction;
  gate = `test_ssm_block` and engine 64/64 on the 20-prompt set.
- **M3 engine (driver).** Registry dispatch to q8mrow for m > 1; MTP sweep
  and race per P4.

## Frozen predictions

| stage | prediction | land rule |
|---|---|---|
| M0 ratio today (m=4 / m=1) | 2.4-3.2x | recorded |
| M1 ratio (m=2, m=4, m=8) | <= 1.15x, <= 1.25x, <= 1.5x | m=4 <= 1.3x AND parity |
| M2 | ssm sub-block per extra row -0.5 to -0.8 ms | 64/64 all 20 prompts |
| M3 real prompts k=2 median | 100.7 -> **112-122** | >= 110 AND 100/100 identity |
| M3 race prompt k=4 | 145.6 -> **165-185** | recorded |
| M3 vs llama.cpp MTP real text | 0.78x -> 0.90-1.0x | recorded |
| falsifier | M1 m=4 ratio > 1.6x after LDS staging: the A-traffic diagnosis is wrong; stop, profile with rocprof, re-preregister | |

## Result

### M0 (2026-09-04, lane B `4af9e5c`, driver re-run under gpu.lock)

`flock .work/gpu.lock ./.work/bench_coldcache_mrow`, grid 1536 x 256,
ROW_WAVES=8, UNROLL=4, ffn_gate N=12288 K=4096, ITERS=200, fp64 parity on
every row: correct=true for all four arms.

| MR | us | GB/s (weight bytes) | ratio to MR=1 |
|---|---|---|---|
| 1 | 88.3 | 605 | 1.00 |
| 2 | 92.2 | 580 | 1.04 |
| 4 | 112.9 | 473 | 1.28 |
| 8 | 201.2 | 266 | 2.28 |

**Frozen m=4 prediction 2.4-3.2x was wrong: measured 1.28x.** The 2.9x
4-row prefill window and the 1.28x 2-row speculative window are therefore
NOT the GEMM at m=2 (1.04x here). The A-traffic diagnosis is falsified for
m <= 4; it holds only toward m=8. Lane A's LDS-staged q8mrow (`d7f765c`,
branch `lane-mrow-kernel`): m=4 1.29x, m=8 2.08x, bit-identical outputs;
no gain at the window sizes the engine uses, not wired into the engine.
Stage M1 is re-scoped to an ablation (traffic vs ALU vs occupancy at m=4/8)
before any second kernel attempt; M2 (SSM per-row fold) is now the lead
lever for the 2-row window.

### Per-window cost split (2026-09-04, driver, race prompt, BARO_PROFILE=1)

1-row decode window vs 2-row speculative window (k=1), ms per window,
profile syncs inflate absolutes, deltas are the signal:

| block | 1-row | 2-row | delta |
|---|---|---|---|
| attn | 1.55 | 1.76 | +0.21 |
| ssm | 4.76 | 5.82 | +1.06 |
| ffn | 8.41 | 9.34 | +0.93 |
| head | 1.54 | 1.61 | +0.07 |
| draft path (process + draft) | 0 | 2.22 | +2.22 |

Sum +4.4 ms on 16.3 = 1.27x, matching the measured window cost. Lever order
by size: draft path 2.2 (blk.32 layer + 1.06 GB q8 LM head + argmax + host
sync per window), SSM per-row 1.06 (M2, lane C), ffn 0.93 (GEMM m=2 is
1.04x = 0.34 of it; the rest is in the SPLITK reduce / swiglu at m rows,
unmeasured), attn 0.21. Draft-path split and the q4 draft head are the
next preregistered items; draft quantization cannot change output tokens,
only acceptance, so it is not gated by bit-exactness.

### M1 ablation (2026-09-04, lane A stint 2, `.work/briefs/status-mrowA.md`)

Original q8row, cold-cache ffn shape, ratio to FULL MR=1 (70.5 us):
FULL m=4 1.46x, m=8 2.71x. NOTRAFFIC (all rows read A row 0): 1.38x /
2.53x. NOFMA (all rows load A, only row 0 accumulates): **1.06x / 1.11x**.
OCC (ROW_WAVES=4): 1.42x / 2.69x. VGPR 89-133 across variants, no spills.
**The multi-row cost is FMA count, not traffic and not occupancy.** The
kernel is compute-bound above m=2: per weight element it does one dequant
and MR fp32 SIMD FMAs on the bf16 activations. LDS staging (stint 1) could
not help and did not. Lever, preregistered as M1b below: packed int8 dot
products (activations quantized to q8 per 32-block at window start, weights
already q8, `v_dot4_i32_i8`-class math, 4 MACs/lane/op) for m >= 3 only;
m <= 2 keeps the bf16 path and its bit-exactness.

### M2 (2026-09-04, lane C `871faca`, merged `9808a9d`)

Five SSM kernels take `m`, row loop inside; `amar_ssm_delta_step` is
VGPR-spill-bound (192 VGPR, 122 spills at MR=1; a runtime row loop made it
288) so it is instantiated per MR via `delta_dispatch`. `test_ssm_block`
m=1/m=4 bit-exact PASS. p09 k=2: 77.6 -> 78.4 (+1%), `profile: ssm` 0.273
-> 0.267 s. Small, as the split predicted (1.06 ms of which launches were a
fraction). The 122-spill delta step at m=1 is a standing defect of the
decode path itself: `bench/ssm-occupancy-protocol.md` was frozen for it
and never run; lane C stint 2 runs it.

### Draft-path split (2026-09-04, lane B stint 2, `BARO_PROFILE=3`, merged `03f6a13`)

Per 2-row window, race prompt k=1: layer 0.61, **head 1.51**, argmax 0.21,
accept (copies + sync + compare) 0.04, other 0.02 = 2.40 ms. p09 k=1 the
same within 0.1. k=2 doubles everything except accept: draft steps are
one row each, never batched. The host sync is 2% and not a lever. The
draft LM head (1.06 GB q8 read per drafted token) is 63% of the draft path.

## M1b, M4 frozen (2026-09-04, after the ablation, before any run)

| stage | prediction | land rule |
|---|---|---|
| M1b q8 x q8 dot kernel, cold-cache ffn shape | m=4 <= 1.15x, m=8 <= 1.4x of bf16 m=1 | parity vs fp64 on dequantized q8 activations <= 2e-3 max_rel; engine 64/64 race AND >= 18/20 identity vs arm A on the prompt set with m>=3 windows only |
| M1b engine effect | prefill 4-row 43 ms -> <= 25 ms; k=4 race 146 -> >= 160 | recorded |
| M4 q4 draft head (`bench/draft-q4-protocol.md`) | draft path 2.4 -> <= 1.7 ms/window; real-prompt k=2 median 100.7 -> **108-115**; acceptance within 3 points of q8 draft | >= 106 AND acceptance >= 66% |
| falsifier M1b | m=4 > 1.3x: int8 dot path does not beat fp32 FMA on this card; stop | |
| falsifier M4 | acceptance drops > 5 points: q4 draft too lossy, try q6/q5 pack before giving up | |
