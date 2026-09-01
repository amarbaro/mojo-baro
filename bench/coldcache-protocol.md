# Cold-cache GEMM protocol — frozen before first run

Question: q8 (int8+scales, 1.19 B/elem) vs bf16 (2 B/elem) weight-native
skinny GEMM throughput when weights are NOT Infinity-Cache resident — the
real decode pattern, where ~30 distinct weight matrices cycle per token.
Prior result retracted: single-buffer timing swung 2.70 ms ↔ 0.33 ms across
identical runs (IC state contamination at W ≥ IC size).

## Instrument

`bench/bench_coldcache.mojo`. M=8, K=4096, N=12288 (real blk.0 ffn gate
weight, replicated). NBUF=8 distinct device buffers per dtype (bf16 working
set 8x100.7 MB = 805 MB; q8 8x(48+6) MB = 430 MB; both >> 96 MB IC, so every
launch streams from HBM). Launches round-robin the 8 buffers; 1 s clock
warm (same rotation), then 200 timed launches, ms/launch reported.
The whole measurement repeats 10x in-process.

## Frozen predictions

1. **Stability**: with the rotation, per-measurement spread collapses to
   < 5% of mean (vs the >8x swing seen single-buffer). If spread stays
   > 5%, the instrument is still broken and no ratio may be claimed.
2. **bf16**: >= 105 us/launch (100.7 MB / 960 GB/s floor); predicted range
   105–200 us.
3. **q8/bf16 ratio**: byte ratio is 0.563, so predicted speedup 1.5–1.9x
   (sub-linear: dequant adds VALU work and the scale stream breaks
   locality slightly).

## Claim rule

Report mean ± min/max over the 10 repeats. Claim a speedup only if the
bf16 and q8 ranges do not overlap and prediction 1 held. Any deviation
from this protocol gets disclosed next to the result.

## Result 2026-09-01 (10 repeats, spread <1% — prediction 1 HELD)

| arm | us/launch | vs predicted |
|---|---|---|
| bf16 wt-layout | 399–403 | predicted 105–200: **FALSIFIED** (2x over) |
| q8 wt-layout | 468–474 | predicted 1.5–1.9x faster: **FALSIFIED** (0.85x — slower) |
| bf16 B-layout (post-hoc arm, transposed at load) | 194–195 | not preregistered; disclosed |

Reading: wt-layout is coalescing-bound (~250 GB/s effective, 26% of HBM
peak), not byte-bound — so halving bytes with q8 only added dequant ALU
work. B-layout reaches ~519 GB/s (54% peak). Decision: engine loader
transposes weights once at load; wt kernels remain for transpose-unaffordable
cases only. q8 must be re-laid-out K-major (B-layout blocks) before its
byte advantage can show; re-preregister before claiming.
