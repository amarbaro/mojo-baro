# MTP draft-head protocol — frozen before first run (M5 item 5)

Question: does the model's own `blk.32` draft head (NextN, qwen35
`nextn_predict_layers=1`) earn its launches in our engine — i.e. does
wiring speculative decode through the draft head pay back the extra
draft-forward-pass cost with a net decode speedup, without changing
what the model says?

## Correctness precondition (gates before any speed number is recorded)

No `tok/s_gen` figure from the MTP arm may be recorded until the
draft head passes its own numerical gate against llama.cpp's draft
logits, independent of and prior to any timing run:

1. Feed the same prompt/position to our draft head and to llama.cpp's
   `blk.32` forward pass (same GGUF, same eh_proj/concat wiring per
   `docs/mtp-notes.md` §3).
2. Compare draft logits: **top-1 argmax agreement**, **cosine
   similarity**, and **max relative error**, all three reported
   together — high cosine with a differing argmax is the known
   silent-corruption signature and must not be waved through on
   cosine alone.
3. Gate: top-1 argmax must agree, AND cosine >= 0.999, AND max
   relative error below a threshold tight enough that no top-k logit
   flips rank. If any of the three fails, the draft head is broken —
   fix it and re-run the gate. No timing run occurs until this
   section reads PASS.

## Instrument

Our engine's existing 64-token regression gate (`docs/M5-PLAN.md`
item 1 instrument; same gate used for `serve/engine.mojo`
token-identity checks), run twice:

- **Arm A (no spec)**: engine's current decode path, MTP head not
  invoked. Existing `tok/s_gen` metric.
- **Arm B (MTP on)**: engine wired to draft with `blk.32`, verify via
  `process()`/`draft()`/`accept()` pattern (`docs/mtp-notes.md` §3),
  greedy sampling, same prompt as the existing gate, same 64-token
  output length.

GPU discipline: llama-server (holds ~22 GB VRAM) MUST be down for
every timed run, GPU exclusive to our engine — cold-cache v3/v4
already document contention silently distorting results when the
server was up; that mistake is not repeated here.

Report per arm: `tok/s_gen` ((tokens_generated-1)/decode_seconds,
generation-time-EXCLUDING-prefill — the same definition used for our
own no-spec baseline, NOT llama.cpp's `server-common.h:404-427`
metric), draft acceptance count/rate, and drafted-vs-accepted per
window. Repeat 4x per arm (median reported, matching
`decode-race-protocol.md`'s repeat/median convention); report
min/max spread alongside the median.

## Metric-caveat guard (must not be violated)

The comparison in this protocol is **self-relative only**: our
with-MTP `tok/s_gen` vs. our own without-MTP `tok/s_gen`, both
measured by our engine's own metric definition. It is NOT a
comparison to llama.cpp's 44.1 / 109.8 tok/s bar
(`bench/decode-race-protocol.md`) — that bar used llama.cpp's own
`tok/s_gen` definition, was frozen and measured at commit `adbb48c`,
and has not been (and is not being) re-measured here. The llama.cpp
numbers below are cited only as an external reference point for the
plausibility of the acceptance-rate and speedup predictions, never
as the denominator of a claim this protocol makes.

## Reference facts (measured on this box, frozen, not re-derived here)

- llama.cpp bar, commit `adbb48c`, verdict `6b99693`: no speculation
  44.1 tok/s (median of 4, spread 0.2%); with its MTP draft head
  109.8 tok/s (median of 4, spread 0.7%); draft acceptance 52/62 =
  84%, every run; multiplier 2.49x.
- HBM roofline: ~960 GB/s / 17.9 GB per token = 53.6 tok/s
  non-speculative decode ceiling.
- Our engine currently (no spec): 38.97 tok/s by our own metric,
  64-token greedy output bit-identical to llama.cpp.

## Frozen predictions

1. **Acceptance rate**: 60–85%. llama.cpp achieves 84% on the exact
   model/prompt, but our draft head is a fresh implementation of the
   same wiring — some slack below llama.cpp's figure is expected
   from any small numerical divergence in the eh_proj/attention path
   that survives the correctness gate above (the gate bounds
   correctness, not bit-exactness of every intermediate).
2. **Speedup**: 1.6–2.4x over our own no-spec `tok/s_gen` (38.97).
   Lower bound reflects our engine's higher per-launch overhead
   (M5 item 2 not yet landed at time of freeze) eating into the
   draft-head's savings; upper bound tracks llama.cpp's 2.49x
   ceiling on the same model.
3. **Stability**: spread across the 4 repeats must be < 5% of the
   median for BOTH arms, else no speedup claim may be made — same
   discipline as `coldcache-protocol.md` v2/v3 (their < 1.5%
   threshold is tighter because that instrument is a tight GEMM
   loop; this protocol uses the looser 5% cap already used by
   `decode-race-protocol.md`'s 10%-of-median void rule, halved for a
   controlled single-machine, single-process comparison).

## Falsifier

Speedup < 1.3x over our own no-spec baseline falsifies the claim
that the draft head earns its launches in this engine — below that
line, the extra draft-forward-pass cost is not paying for itself and
MTP should not be adopted as-is (return to item 2's launch-count
work instead, per `docs/M5-PLAN.md`).

## Claim rule

A speedup number may be stated publicly only if:

1. The correctness precondition above reads PASS (top-1 argmax
   agreement + cosine >= 0.999 + bounded max relative error) BEFORE
   any timing run.
2. Both arms' 64-token greedy output remain bit-identical to
   llama.cpp's reference tokens (`.work/engine-pack/ref-tokens.txt`)
   — speculative decode is an optimization, not a change to what the
   model says. If Arm B's output moves off that reference, the run
   is void regardless of measured speed, full stop — no partial
   credit for "faster but different."
3. The stability requirement (prediction 3) holds for both arms.
4. Only the self-relative comparison (Arm B `tok/s_gen` vs Arm A
   `tok/s_gen`, both our engine's own metric) may be reported as a
   speedup claim, per the metric-caveat guard above.

Any deviation from this protocol (different prompt, different
n_predict, GPU shared with llama-server, gate skipped) voids the run
and requires a fresh preregistration before re-attempting.

## Result

**Not yet run.** To be filled in after the correctness precondition
passes and both timed arms complete, following the exact structure
of `bench/coldcache-protocol.md`'s Result sections: per-repeat
spread, verdict per prediction (HELD / MISSED HIGH / MISSED LOW /
FALSIFIED), and any disclosed asymmetry or contamination observed
during the run. No result may be written into this section before
the run happens — this document is frozen pre-outcome.
