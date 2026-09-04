# Launch fusion round (frozen 2026-09-04, tree 3a995be, engine binary built from b265eb5, 41.7 tok/s_gen unprofiled)

Bound by `PROTOCOL-RULES.md`. Prior verdicts bind this round: micro-fusion was
falsified twice (M5 item 2, `616bc62`: ~28 -> ~20 launches/ssm-layer = +5%;
`ffn-fusion-protocol.md` F1/F2/F3 = +1.0% / -0.2% / +0.8%). The open question
from that series is whether the ~20% to the HBM roof is launch floor at all,
and if so whether anything short of a persistent megakernel reaches it.

## Launch census (read from `serve/engine.mojo` decode loop, m = 1)

| sub-block | launches | count/token |
|---|---|---|
| embed | 1 | 1 |
| attn layer (8): rmsc, 3 gemm + 3 reduce, split, hrms_q, hrms_kv, rope_q, rope_k, append x2, att, gmul, gemm + add | 18 | 144 |
| ssm layer (24): rmsc, 4 gemm + 2 reduce, rgates, conv, l2, delta, gated, gemm + add | 14 | 336 |
| ffn (32): rmsc, gemm_dual, swiglu, gemm, add | 5 | 160 |
| head: rmsc, rms_m, gemm, reduce, argmax | 5 | 5 |
| **total** | | **646** |

Decode = 24 ms/token at 41.7 tok/s -> 37 us mean per launch. "~90 launches"
on the board was the site count, not the per-token count.

## BARO_PROFILE=2 receipt (2026-09-04, `.work/profile-pf2-20260904.log`, 36.5 tok/s under profiling)

ssm per-kernel, 64 tokens x 24 layers = 1536 instances each, synchronize()
after every launch so these are launch + host round-trip, an UPPER bound:

| kernel | total s | share of ssm | us per instance |
|---|---|---|---|
| gemm4+reduce2 | 0.337 | 56% | 219 |
| rgates | 0.037 | 6.2% | 24 |
| conv | 0.029 | 4.9% | 19 |
| l2 | 0.029 | 4.8% | 19 |
| delta | 0.045 | 7.4% | 29 |
| gated | 0.027 | 4.5% | 18 |
| out_gemm+add | 0.098 | 16% | 64 |

The five small kernels are 24% of ssm time = 8% of decode under profiling.
Their own work is sub-us (state 2 MB/layer at 0.6% of traffic, per
`ssm-occupancy-protocol.md`); the 18-29 us each is floor.

## Stage 0: launch floor (S, `bench/bench_launch_floor.mojo`)

Measure on this box, no synchronize between launches, 1000 iterations:
(a) empty kernel grid 1 block 64; (b) empty kernel grid 96 block 256;
(c) the five ssm small kernels back-to-back at engine shapes;
(d) one kernel that does (c)'s work in a single launch (grid NH_V, block SSTATE,
    with a block-local barrier between the five phases) -- feasibility of the
    fused shape, not yet bit-exact.
Receipt: us per launch for a-d, GPU clock read back via `clock-probe.sh`.

Ceiling for the whole round = 646 x floor(b) / 24 ms. Predicted floor(b) 4-8 us
-> ceiling 11-22%. **Stop rule S0: floor(b) < 3 us (ceiling < 8%) -> round
closed, per-launch cost is not where the gap lives.**

## Stage 1: fused ssm step, bit-exact (M, `amar_ssm_step`)

rgates + conv + l2 + delta + gated in one launch, grid NH_V, block SSTATE.
rgates today is grid 1 block NH_V: each fused block recomputes its own head's
gate from Pab/Pab2 (identical arithmetic, no cross-head data). conv today is
grid ceildiv(CONV,256): fused block r covers columns [r*SSTATE, (r+1)*SSTATE)
of its head; same per-column expression. delta/gated already grid NH_V. Per-
element summation order unchanged -> 64/64 identity by construction.
Launches/token 646 -> 550.

| prediction (tok/s_gen) | land rule |
|---|---|
| 96 fewer launches x floor(b) from stage 0 = **+1.5% to +3%** (41.7 -> 42.3..43.0) | 64/64 AND median of 3 >= +1.5%, spread < 5%; baseline in same stint |

Stage 1 runs only if S0 did not fire. The attn small chain (split, hrms x2,
rope x2, append x2: 7 -> 1, 48 launches/token) is the same lever at half the
count; it is NOT a separate stage -- if stage 1 lands at its prediction,
fold attn in the same commit series under the same land rule, else skip.

## Stage 2: grid barrier spike (S, gates the megakernel)

If S0 did not fire and stage 1 lands, the remaining ~500 launches are GEMM +
reduce + rmsc + add pairs across 32 layers; the only lever left is one
persistent per-token kernel with grid-wide sync between stages. Mojo stdlib
grid sync availability on gfx1100 is unverified. Spike: one kernel, 96..384
resident blocks, atomic-counter barrier, 1000 barriers, measure us/barrier.

**Stop rule S2: us/barrier >= floor(b) -> megakernel cannot beat launches,
close the round at stage 1's result.** Otherwise the megakernel is its own
preregistered round (XL, not this file).

## Not in this round
q8 weights (bytes lever, not bit-exact). MTP verify width. Prefill.

## Stage 0 receipt (2026-09-04, `bench/bench_launch_floor.mojo`, `.work/launch-floor-{1,2,3}.log`)

2000 launches per arm after 200 warm, no sync between launches. Run 1 was at
2105 MHz (cold), runs 2-3 at 3313 MHz (clock-probe receipts in the logs); the
warm runs are the arms.

| arm | run 2 | run 3 |
|---|---|---|
| (a) empty g1 b64, us/launch | 2.83 | 2.76 |
| (b) empty g96 b256, us/launch | 2.59 | 2.57 |
| (c) five ssm kernels, us/chain (us/launch) | 23.8 (4.77) | 24.5 (4.91) |
| (d) fused shape, one launch, us | 16.25 | 16.26 |
| ceiling = 646 x (b) / 24 ms | 6.96% | 6.91% |

**S0 fired: floor(b) = 2.57 us < 3 us, ceiling 6.9% < 8%. Round closed at
stage 0; stages 1 and 2 do not run.**

What the receipt adds: the five-kernel chain is 24 us/layer of which ~13 us is
launch floor and ~11 us is the state pass itself; the one-launch fused shape
still costs 16 us, so stage 1 would have saved ~8 us x 24 layers = 0.2 ms
= 0.8% per token, under its own +1.5% land rule. The ~20% to the HBM roof is
not per-launch floor. Remaining candidates are per-kernel tail/ramp inside the
GEMM launches (SPLITK partial + reduce pairs), which is a GEMM-shape question,
and the bytes lever (q8).
