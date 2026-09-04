# MTP draft-head protocol — frozen before first run (M5 item 5)

> **Binding: [`bench/PROTOCOL-RULES.md`](PROTOCOL-RULES.md).** P1 in particular:
> every parameter defining an arm is read back from the running system and
> recorded BEFORE the timed run. No receipt, no arm.


Question: does the model's own blk.32 draft head (NextN/MTP, `qwen35`
architecture, `nextn_predict_layers=1`) earn its launch cost in our
engine — i.e. does spending extra GPU launches on a draft-then-verify
loop beat plain greedy decode at our engine's own actual shape?

## Correctness precondition (gates ALL speed numbers)

No speedup number may be recorded until the draft head passes a
numerical parity gate against llama.cpp's draft logits (`ctx_dft`
`res->t_logits`, `common/speculative.cpp`'s `draft()` path, per
`docs/mtp-notes.md` §2-§3):

1. Top-1 argmax match on every position sampled during a fixed
   16-token draft trace.
2. Cosine similarity of the full logit vector >= threshold TBD-at-gate-
   time, held to the same bar `q8b` parity used (9.2e-4-class), recorded
   in the Result section — not loosened after the fact.
3. Max relative error bound recorded alongside cosine — cosine-alone is
   the known silent-corruption signature (high cosine, differing
   argmax = wrong token picked with a deceptively "close" logit vector).

Any one of the three failing voids the run: no tok/s, no acceptance
rate, no multiplier gets recorded that day. Fix the kernel, re-freeze
if the fix changes the instrument, re-run.

## Metric caveat (binds prediction 2 below)

llama.cpp's `tok/s` (`tools/server/server-common.h:404-427`,
`timings.predicted_per_second`) is generation-time-INCLUDING nothing
but decode: `(n_decoded - 1) / t_decode_s`. Our engine's own
`tok/s_gen` metric (see `bench/decode-race-protocol.md`) is computed
the same way, generation-time-EXCLUDING prefill — but the two are NOT
directly comparable across engines beyond what `decode-race-protocol.md`
already disclosed and froze. This protocol does NOT re-measure or
re-litigate llama.cpp's numbers. All predictions below are **self-
relative**: our-engine-with-MTP `tok/s_gen` vs our-engine-without-MTP
`tok/s_gen`, both measured by our own metric, same instrument, same run
session. llama.cpp's 44.1 / 109.8 tok/s figures (below) are cited only
as prior art informing the prediction ranges, never as the denominator
of any claim made here.

## Facts on the record (measured previously, not re-derived here)

- llama.cpp bar, frozen `adbb48c`, verdict `6b99693`
  (`bench/decode-race-protocol.md`): no-spec 44.1 tok/s (median of 4,
  spread 0.2%); with its MTP head 109.8 tok/s (median of 4, spread
  0.7%), draft acceptance 52/62 = 84% every run, multiplier 2.49x.
- HBM roofline ~960 GB/s / 17.9 GB-per-token weight traffic =
  53.6 tok/s non-speculative ceiling.
- Our engine, no MTP: 38.97 tok/s by our own metric, 64-token greedy
  output bit-identical to llama.cpp.
- GPU: llama-server holds ~22 GB VRAM at idle — MUST be down for this
  run, GPU exclusive (same discipline as `coldcache-protocol.md` v3/v4).

## Instrument

64-token gate run (`./.work/engine`, prompt "The capital France is",
greedy, `n_predict=64`), llama-server DOWN, GPU exclusive, existing
regression-gate prompt/tokens (`.work/engine-pack/ref-tokens.txt`).
Two arms, same process/binary build, same prompt:

- **Arm A (no MTP)**: current engine path, plain greedy decode.
- **Arm B (with MTP)**: blk.32 draft head wired per `docs/mtp-notes.md`
  §2-§4 (`process()`/`draft()`/`accept()` loop mirrored from
  `common/speculative.cpp`), same greedy sampling, same prompt.

Report per arm: `tok/s_gen` ((n_decoded-1)/decode_s, our own metric,
generation-time only), and for arm B additionally: draft acceptance
count/rate, drafted-vs-accepted token counts per verification window.
5 repeats per arm, discard first (cache/clock warm), report
median of remaining 4.

## Frozen predictions

1. **Stability**: per-arm spread on the 4 kept repeats < 5% of median.
   If spread exceeds 5% on either arm, no speedup claim may be made
   that day — re-run under the stability gate before recording
   anything (copy discipline: `coldcache-protocol.md` v2/v3).
2. **Acceptance rate**: 60-85%. Reasoning: llama.cpp gets 84% on this
   exact model/head; ours is a fresh implementation of the same weights
   through different kernels, so parity is expected but not assumed —
   range widened below llama.cpp's number to allow for real
   implementation slop while still requiring the head to be doing
   useful work.
3. **Speedup**: arm B `tok/s_gen` / arm A `tok/s_gen` in 1.6-2.4x.
   Reasoning: llama.cpp's own multiplier is 2.49x; our engine starts
   from a lower no-spec baseline (38.97 vs 44.1) and untuned draft-path
   kernels, so the range is shifted down and narrowed versus
   llama.cpp's observed ceiling, not copied from it.

## Falsifier

Arm B / Arm A speedup < 1.3x — the draft head is not earning its
extra launches in our engine; MTP work stops until the underlying
inefficiency (kernel launch overhead, draft-path fusion, acceptance
rate) is diagnosed and re-preregistered.

## Claim rule

A speedup number may be published ONLY if:

- the correctness precondition (top-1 AND cosine AND max-relative-error
  gate) passed and is recorded in the Result section, AND
- both arms' greedy 64-token output stayed bit-identical to the
  existing `ref-tokens.txt` gate — MTP is a launch-scheduling
  optimization, not a model-behavior change; if the output moves at
  all, the run is void regardless of how fast it was, AND
- per-arm spread stayed under the prediction-1 threshold.

Any deviation from this protocol (different prompt, different
`n_predict`, different sampling, GPU not exclusive) requires a new
preregistration before any number from that run may be cited as "the"
MTP result.

## Result

**Not yet run. This section is a placeholder — filled in only after
the correctness precondition passes and the instrument above executes
exactly as specified. No prose, no number, no verdict belongs here
until then.**

## Amendment 2026-09-04 (frozen before the draft loop is written, tree 87e0f81)

Base has moved: the engine is q8-only at **68.77 tok/s_gen** (`q8-protocol.md`),
64/64 identical to llama.cpp Q8_0 and bf16. Arm A = that binary/pack. Arm B
= same binary with `BARO_SPEC=1`, draft window k from `spec-k.txt` (4).
The correctness precondition of this file was met on 2026-09-01 (draft head
argmax and logits validated vs llama.cpp, `tools/draft-ref.py`); the DRAFT
line still prints `from_token 369 draft_argmax 369` on the q8 engine.

Design (mirrors `common/speculative.cpp` process/draft/accept, greedy):
- State per iteration: `pos` = last known token; trunk ring/KV valid through
  `pos-1`; `Hnm` rows hold the trunk's post-output-norm hidden for the last
  window; draft KV (`kc32/vc32`) valid through the previous iteration.
- **process**: one blk.32 pass, M = tokens not yet in the draft KV with
  truth h (token p pairs with h(p-1)): after prefill the P-1 prompt tokens,
  later the n_acc+1 accepted+bonus tokens. Its last row IS draft step 0:
  argmax -> d1, its own post-shared-head-norm hidden -> h for step 1.
- **draft**: k-1 further M=1 blk.32 passes at pos+j with (d_j, draft h_{j-1}).
- **verify**: trunk window M = min(k+1, remaining) rows at pos..pos+k over
  [Toks[pos], d1..dk]; argmax into `dtok_d`, host compares, n_acc = leading
  matches, bonus = argmax(row n_acc) written to Toks[pos+n_acc+1]; ring and
  pos advance by n_acc+1. Rejected rows' trunk KV / ring slots / draft KV are
  overwritten by the next window, never read.
- Greedy verification accepts iff equal, so the 64 tokens are identical to
  arm A by construction; any drift voids the run (claim rule unchanged).

Byte model per iteration at k=4, 84% acceptance (llama.cpp's rate):
verify 10.7 GB + process ~0.2 GB + LM head x4 (1.06 GB each) + 3 draft
bodies x 0.2 GB ~= 15.7 GB for ~4.4 tokens -> 3.6 GB/token vs 10.7 today.

| prediction | value | rule |
|---|---|---|
| acceptance (drafted->accepted) | 60-85% (unchanged) | recorded |
| arm B / arm A | **1.6x-2.4x** (110-165 tok/s_gen) | land >= 1.5x AND 64/64 AND spread < 5% |
| falsifier | < 1.3x: stop, diagnose, re-preregister | |

Both arms 5 repeats, discard first, median of 4; server down; `BARO_SPEC`
and `spec k` printed by the run are the P1 receipts, plus per-run
`drafted/accepted` counts.
