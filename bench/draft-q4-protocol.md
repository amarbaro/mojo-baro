# q4 draft head protocol (M4) — frozen before first run, 2026-09-04

Binds `bench/PROTOCOL-RULES.md` P1-P6. Facts: the MTP draft path costs
2.4 ms per window, 63% of it the 1.06 GB q8 LM-head read per drafted token
(`bench/mrow-gemm-protocol.md`, draft-path split). The draft's tokens never
reach the output: a wrong draft costs acceptance, not correctness. So the
draft head may be quantized below the trunk without touching the 64/64 gate.

Plan: `tools/engine-pack.py --q4-draft` appends a Q4_0 copy of
`output.weight` (and optionally the blk.32 ffn/attn 2D weights) as extra
pack entries (18 B / 32 weights, byte-equal to `llama-quantize Q4_0` blocks,
checked by `tools/q4-check.py` as in the q8 round). Kernel
`amar_matmul_skinny_q4row[UNROLL, MR]`: wave-per-row like q8row, nibble
unpack in register, same fp32 accumulate. `blk32_forward` uses it for the
head when the pack has the q4 entry; the trunk never does.

| prediction | value | rule |
|---|---|---|
| q4row ffn-shape cold-cache stream | 100 MB-equiv in 36-44 us (q8row 62.6) | <= 48 us AND fp64 parity on dequantized values |
| draft path per window (k=1) | 2.4 -> 1.5-1.7 ms | recorded |
| real prompts k=2 median | 100.7 -> 108-115 | land >= 106 AND 100/100 identity AND acceptance >= 66% (q8 draft: 69%) |
| race k=4 | 146 -> 155-165 | recorded |
| falsifier | acceptance < 64%: q4 draft too lossy; q6_K/q5_0 pack next, not more q4 tuning | |

## Result

Not yet run.
