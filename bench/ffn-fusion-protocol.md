# ffn micro-fusion, bit-exact (frozen 2026-09-02 at commit a634fbf-era tree, engine 41.3 tok/s_gen)

Bound by `PROTOCOL-RULES.md`. ffn is 52% of decode GPU time and already
GEMM-bound at ~80% of HBM (skinny m1 121 us vs vendor 126 us per 100 MB
weight). Launch count is the only bit-exact lever left; each launch gap is
~2 us against ~390 us per layer.

| step | change | prediction (tok/s_gen) | land rule |
|---|---|---|---|
| F1 | gate+up in one kernel `amar_matmul_skinny_m1_dual` (A staged once, two B streams, two partial outputs; summation order per output unchanged) | +1% to +3% | 64/64 identity AND median of 3 >= baseline median +1%, spread < 5% |

Baseline and candidate measured in the same stint, server stopped, 3 runs each,
tok/s_gen read back from the engine's own output.

## F1 receipt (2026-09-02)

| arm | tok/s_gen x3 | median | gate |
|---|---|---|---|
| baseline `.work/engine` | 41.32, 41.34, 41.30 | 41.32 | 64/64 |
| F1 dual gate+up | 41.43, 41.73, 41.82 | 41.73 (+1.0%, spread 0.9%) | 64/64 |

Landed at the floor of the prediction. ffn is GEMM-bound; launch removal is
worth ~1% per pair. No further bit-exact micro-fusions are worth a stint.

## F2 (frozen 2026-09-02, after F1 at 41.73): narrow-N skinny GEMMs launch too few blocks

Bytes-per-region arithmetic from the pack index and BARO_PROFILE shares:
ffn 772 GB/s (80% of 960), head 794 (83%), **ssm 483 (50%), attn 487 (51%)**.
The ssm/attn projections have N = 4096 / 8192 / KV and launch with the
2048-column block: 64 / 128 / 32 blocks on 96 CUs, plus alpha/beta on ONE
column block. Fix: `amar_matmul_skinny_m1_t[..., THREADS]` with 64 threads
(512 columns/block) for those launches only; per-output k order and split-K
grouping unchanged, so bit-exact.

| step | prediction (tok/s_gen) | land rule |
|---|---|---|
| F2 | ssm+attn sub-block GEMM time -30..-40% => decode -12..-16% => **+13% to +19%** (41.7 -> 47..50) | 64/64 AND median of 3 >= +5% |

### F2 receipt: NOT landed. F1 41.85/41.81/41.83 vs F2 41.72/41.77/41.77 (0.998x), 64/64, ssm/attn shares unchanged (28.2% / 9.3%). 64 blocks of 256 threads already saturate HBM for a 33 MB stream; the ssm sub-block's missing half is in its non-GEMM kernels. Next: per-launch profile inside the ssm sub-block.

## F3 (frozen 2026-09-02 after F2): fuse the ssm qf and h projections into one launch

BARO_PROFILE=2: ssm projections+reduces = 56% of ssm time at ~500 GB/s; the
ffn's 100 MB streams reach 770. Same activation row feeds qf (N=8192) and h
(N=4096); `amar_matmul_skinny_m1_dual` gains a second N so the two streams
share one launch (blocks past N2 skip the second stream). Bit-exact.

| step | prediction (tok/s_gen) | land rule |
|---|---|---|
| F3 | +3% to +6% (41.8 -> 43..44) | 64/64 AND median of 3 >= +2% |

### F3 receipt: NOT landed. F1 41.79/41.68/41.90 vs F3 42.13/42.17/42.13 (+0.8%), 64/64. Below the +3% prediction floor and the +2% land rule.

## Conclusion of the bit-exact fusion series (2026-09-02)

Three preregistered fusions, one landed at +1.0%, two measured +0.8% and
-0.2%. Launch count and block count are not where the ~20% to the HBM roof
lives; it is spread over ~90 launches as per-launch ramp/tail and over the
five launch-floor kernels in each ssm layer. The remaining bit-exact lever is
structural: a persistent per-token megakernel (one launch, grid-wide sync
between stages). XL. The bytes lever (q8 weights) is the only other 2x and is
not bit-exact.
