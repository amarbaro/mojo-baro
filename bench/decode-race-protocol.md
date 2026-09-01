# Decode race protocol — frozen before first run (M5 item 1)

Question: what is the llama.cpp GPU decode bar on this box for
Qwythos-9B (bf16 GGUF, RX 7900 XTX), with and without MTP speculative
decode, measured the same way our engine is measured — so that every
M5 optimization claim has a fixed denominator.

## Instrument

Running llama-server (port 8083, cmdline preserved in
`~/Brain/mojo-baro/llama-server-cmdline.txt`: `-ngl 99 -fa on`,
`-ctk/-ctv q8_0`, `--spec-type draft-mtp --spec-draft-n-max 6`).

Per arm: POST `/completion` with
`{"prompt": "The capital of France is", "n_predict": 64,
"temperature": 0, "top_k": 1, "cache_prompt": false}`.

- Arm A (MTP on): request as-is (server default spec settings).
- Arm B (MTP off): add `"speculative.n_max": 0`. If the per-request
  override is ignored (verified via timings/draft fields), arm B is
  measured after a controlled server restart without the two spec
  flags, then the original cmdline is restored.
- Ours: `./.work/engine` printed tok/s (BARO pack, 5-token prompt,
  16 generated, f32 KV, no spec) — 25.5 tok/s as of `f9d4ddb`.

5 repeats per arm, discard the first (cache/clock warm), report
median of the remaining 4 from the response's
`timings.predicted_per_second`. GPU otherwise idle (our engine not
running during server arms).

## Known asymmetries (disclosed, not corrected)

- llama.cpp uses q8_0 KV cache; ours f32. Negligible at n<=70 tokens.
- llama.cpp samples 64 tokens from a 5-token prompt; ours 16. Decode
  is stateless per token at these lengths; per-token cost flat.
- Both read bf16 weights (~17.9 GB working set per token).

## Frozen predictions

1. Arm B (no MTP): HBM roofline is ~960 GB/s / 17.9 GB = 53.6 tok/s
   ceiling. Predicted 32–48 tok/s (60–90% of roof, mature kernels).
2. Arm A (MTP): acceptance-dependent multiplier over B, predicted
   1.3–2.2x, i.e. 45–100 tok/s.
3. Ours (25.5) trails arm B by 1.3–1.9x; the gap is launch count
   (~20 launches/layer at M=1), not GEMM bandwidth.

## Verdict (2026-09-01, run after freeze at `adbb48c`)

- **Arm A (MTP on): 109.8 tok/s** (median of 4; 109.6–110.4, spread
  0.7%). Draft acceptance 52/62 = 84% every run. ABOVE the predicted
  45–100 band — MTP multiplier is 2.49x over arm B, past the frozen
  2.2x cap. Prediction 2 missed high.
- **Arm B (no MTP): 44.1 tok/s** (median of 4; spread 0.2%). Inside
  the predicted 32–48 band; 82% of the 53.6 tok/s HBM roof. First
  arm-B attempt (39.7, spread 11.6%) VOIDED by the spread rule —
  taken immediately after model load, clocks/cache cold; a clean
  repeat converged.
- Per-request `speculative.n_max: 0` was IGNORED by this server
  build (draft_n unchanged); arm B required the controlled restart.
  Original cmdline restored and health-checked after.
- Ours 25.5 → trails arm B 1.73x (inside predicted 1.3–1.9x) and
  arm A 4.3x. The bar for M5: **44.1 (kernel race) / 109.8 (with
  MTP, item 5's target)**.

## Claim rule

Only medians produced by this exact procedure may be recorded as
"the bar". Any deviation (different prompt, n_predict, server flags)
requires a new preregistration. Per-arm spread > 10% of median voids
the arm.
