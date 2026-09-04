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

Not yet run.
