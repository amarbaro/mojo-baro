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

# Protocol v2 — K-major q8 + vendor arm (frozen before first run)

New arms in the same instrument (8 rotating buffers, 10 repeats):
- `q8b`: K-major q8 (quants [K,N] int8 + scales [K/32,N] fp32, 56.6 MB
  vs bf16's 100.7 MB), parity-gated at 9.2e-4 before this freeze.
- `hipblaslt_f16`: vendor tuned GEMM through the shim, fp16 copy of the
  same transposed weights, same rotation.

## Frozen predictions (v2)

1. Spread per arm < 1.5% of mean, else no claims.
2. q8b: 105–145 us (byte ratio 0.563 x 194 us = 109 us + dequant ALU).
3. q8b vs bf16 B-layout: 1.45–1.85x.
4. hipblaslt_f16: 110–200 us (vendor should exceed our 54%-of-peak).
5. "Beat vendor" claimed only if the q8b range sits strictly below the
   vendor range. Disclosed asymmetry: q8-vs-f16 is a product-level
   comparison (different dtypes); the like-for-like lane is
   bf16-B vs hipblaslt_f16.

## Result v2 2026-09-01 (10 repeats; rep-1 blay outlier 510us disclosed —
## llama-server held 22 GB VRAM throughout; wt arms dropped for VRAM, disclosed)

| arm | us/launch | prediction | verdict |
|---|---|---|---|
| bf16 B-layout | 195.0–195.9 | (v1 baseline) | reproduced |
| q8b K-major | 164.5–165.3 | 105–145 | **MISSED high** (faster than bf16, less than modeled) |
| q8b vs bf16-B | 1.18x | 1.45–1.85x | **MISSED low** |
| hipblaslt f16 | 123.0–126.2 | 110–200 | held — **vendor wins**, 1.33x over q8b |

Reading: vendor f16 achieves ~812 GB/s effective (85% of peak); our
B-layout 517 GB/s (54%); q8b only 344 GB/s effective on its byte stream —
dequant ALU + per-32 scale reload eats most of the byte win. Beating the
vendor cold requires raising bandwidth efficiency (multi-column per
thread, wider loads), not shrinking bytes further. No beat-vendor claim.

## v3 — multi-column/thread B-layout skinny (frozen before first run)

Question: does widening each thread from 1 column (2 B/load) to CPT
contiguous columns (2*CPT B vector load) close the bf16 B-layout gap
(517 GB/s, 54% of peak) toward the vendor's 812 GB/s (85%)?

Instrument: same bench_coldcache rotation (NBUF=8, 1 s warm, 200
timed, 10 repeats), arms = v1 (194–196 us baseline), v2 CPT=2, CPT=4,
CPT=8, plus hipblaslt f16 reference. Same A/W tensors, same reduce.

Frozen predictions:
1. Spread per arm < 1.5% of mean, else no claim.
2. Best CPT arm: 130–165 us (64–80% of peak). Mechanism: same
   coalesced footprint but 4x fewer load instructions and 4x fewer
   resident columns per wave -> better MALL/latency hiding.
3. CPT=8 regresses vs CPT=4 (32 accumulator lanes ok, 64 collapses
   occupancy — the "many resident waves" rule).
4. Falsifier: if best v2 arm >= 185 us, wide loads are not the
   bottleneck on this pattern and item-6 must move to
   split-K/grid-shape changes instead.
Claim rule: engine adoption only if the winning arm beats v1 by
>= 10% AND the engine token gate stays 16/16.

## Result v3 2026-09-01 (10 repeats, llama-server DOWN, GPU exclusive)

| arm | us/launch | prediction | verdict |
|---|---|---|---|
| v1 (1 col/thread) | 194.1–194.8 | baseline | reproduced |
| v2 CPT=2 | 150.7–151.4 | — | 1.29x |
| v2 CPT=4 | 140.0–140.7 | 130–165 | held |
| v2 CPT=8 | 138.8–139.8 | regress vs CPT=4 | **MISSED — best arm, 1.40x** |
| hipblaslt f16 | 125.1–127.8 | reference | vendor still 1.10x ahead |

Effective bandwidth: v1 517 -> v2c8 723 GB/s (75% of peak). The
wide-load mechanism held; the occupancy-collapse prediction for 64
accumulator lanes did not (32 more f32 regs is affordable here).
Adoption rule met (1.40x >= 1.10x floor) -> engine moves to CPT=8,
token gate must stay 16/16.

## v4 — M=1 specialization (frozen before first run)

Question: engine decode is M=1 but v2 carries SM=8 machinery (8-row
LDS staging, 64 accumulator lanes at CPT=8). Does an M=1-specialized
kernel (scalar-A staging, SIMD[CPT] accumulator) buy the remaining
gap to the vendor (139 -> 126 us at M=8; M=1 unmeasured cold)?

Instrument: bench_coldcache_m1.mojo — same 8-buffer rotation, 1 s
warm, 200 timed, 10 repeats, but M=1, K=4096, N=12288. Arms:
v2 CPT=8 at M=1 (engine's current kernel = baseline), m1 CPT=8,
m1 CPT=4, hipblaslt f16 at M=1.

Frozen predictions:
1. Spread per arm < 1.5% of mean, else no claim.
2. v2c8 at M=1 lands 135–145 us (B traffic dominates; wasted A rows
   cost little).
3. m1c8 beats v2c8 by 3–12% (117–135 us) via register/LDS pressure
   relief -> more resident waves; NOT via bytes (traffic identical).
4. Falsifier: m1c8 within 3% of v2c8 -> the kernel is latency/traffic
   -bound, not occupancy-bound; stop chasing GEMM at bf16 and move to
   MTP/q8 (steps 4–8 of the plan).
Claim rule: engine adoption only if m1 beats v2 by >= 5% at M=1 AND
the 64-token gate passes bit-identically.
