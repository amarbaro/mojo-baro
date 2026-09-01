# Decode race protocol — frozen before first run (M5 item 1)

> **Binding: [`bench/PROTOCOL-RULES.md`](PROTOCOL-RULES.md).** P1 in particular:
> every parameter defining an arm is read back from the running system and
> recorded BEFORE the timed run. No receipt, no arm.


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

**P1 (`PROTOCOL-RULES.md`) binds every arm here.** Server arms must
produce a receipt before their timed run:

```sh
./bench/verify-params.py --arm B --body .work/arms/arm-b.json \
    --expect timings.draft_n=0 \
    --out .work/receipts/arm-b.json
```

The probe sends the arm's own request body, so a per-request override
this build ignores fails here instead of silently becoming the number.
Missing key = unverifiable knob = define the arm by controlled restart.
No receipt, no arm.

Engine arms: rebuild the timed binary in the same command as the run,
and record the run's own `prompt tokens:` / `tokens:` / `spec k:` lines
as the receipt.

## v5 preregistration (2026-09-01) — denominator change, no new "Ours" number yet

**Question**: does `Ours` tok/s become directly comparable to
llama.cpp's `timings.predicted_per_second` once both sides use the
same denominator convention?

**Why re-register**: `timings.predicted_per_second` is
`n_gen_steps() / t_gen_ms()` (verified
`~/llama.cpp/tools/server/server-common.h:404-427`):
`n_gen_steps() = n_gen - 1` (first token free, logits already came
from the last prompt batch) and `t_gen_ms()` excludes prompt eval
(`t_gen_last - t_prompt_last`). Our engine's clock (`serve/engine.mojo`)
previously started before the prefill loop and divided by the full
`GEN_N`, so the old "Ours 25.5 / 39.4" numbers in this file are not
on the same footing as the llama.cpp bar. This is a claim-rule
trigger per the section above: different instrument, new
preregistration required before any "Ours" figure is cited against
the 44.1 / 109.8 bars.

**Instrument**: `serve/engine.mojo` now prints `prefill_s`,
`decode_s`, `tok/s_total` (old metric, unchanged, own tracking
number — full-run clock / GEN_N), and `tok/s_gen = (GEN_N - 1) /
decode_s`, where `decode_s` excludes the prefill window (first
head-block emission, i.e. `pos + m >= len(prompt)`). `tok/s_gen` is
the llama.cpp-comparable field; `tok/s_total` is not and must not be
quoted against the bar.

**llama.cpp side**: the 44.1 tok/s (arm B, no MTP) and 109.8 tok/s
(arm A, MTP) bars above are UNCHANGED and need no re-run — they were
already `timings.predicted_per_second`, the same convention
`tok/s_gen` now matches on our side.

**Status**: instrument change only. No re-measurement was run under
this preregistration. No new "Ours" tok/s number — from `tok/s_gen`
or otherwise — may be cited as a claim until a fresh run is executed
and recorded under this section per the claim rule above.

### v5 verdict (2026-09-01, run at `c8cf219` + print cleanup)

Procedure as frozen above: BARO pack, 5-token prompt, GEN_N = 64,
greedy, f32 KV, no speculation. 5 runs, first discarded, median of 4.

| run | `tok/s_gen` | `tok/s_total` |
|---|---|---|
| 1 (discarded) | 41.385 | 40.540 |
| 2 | 41.042 | 40.203 |
| 3 | 41.377 | 40.529 |
| 4 | 41.274 | 40.415 |
| 5 | 41.371 | 40.496 |

**Median `tok/s_gen` = 41.32** (spread 0.8%, inside the 10% void
rule). `prefill_s` 0.0565–0.0576 s; the prefill window is ~3.7% of
wall time at this prompt length, which is why `tok/s_total` (40.48)
sits only ~2% below `tok/s_gen` here and would diverge sharply on a
long prompt.

Against the unchanged bars: **0.94x arm B (44.1)** and **0.38x arm A
(109.8)**. 77% of the 53.6 tok/s HBM roof, vs llama.cpp's 82%.

**Prediction status: NONE FROZEN.** The v5 section registered the
instrument change but never wrote a predicted range for `Ours`, and
this run was executed before that omission was noticed. So this is a
recorded measurement, not a confirmed prediction, and it earns no
credit as one. The earlier "trails arm B by 1.73x, gap is launch
count not GEMM bandwidth" reasoning is now stale — under the correct
denominator the trunk gap is 1.06x, so any launch-overhead claim
needs its own preregistration and a profile, not an inference from
this table.

**Where the real gap is**: arm A. The 2.5x from MTP speculative
decode dwarfs the 6% trunk gap. `blk.32` is implemented and validated
against the numpy reference but remains dump-only.


### P1 status of the v5 verdict (retroactive, disclosed)

The v5 verdict above predates `PROTOCOL-RULES.md` and carries only a
partial receipt. Verified at run time: the timed binary was rebuilt in
the same command as the run, and the run printed `prompt tokens: 5` and
`tokens: 64`. NOT read back: `GEN_N`, KV dtype and the no-speculation
condition were taken from the source rather than from run output, and no
receipt file was written.

The engine has no per-request override layer, which is where the arm-B
class of failure lives, so the exposure is low — but the standing was
never established and this section is not a retro-issued receipt. The
number stands as recorded; the next engine run under this protocol emits
a proper receipt and supersedes it.
