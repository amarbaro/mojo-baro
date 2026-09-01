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
