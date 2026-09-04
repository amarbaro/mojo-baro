# Protocol rules

Rules that bind every protocol in this directory. A protocol may add
constraints; it may not relax these. Cited by `coldcache-protocol.md`,
`decode-race-protocol.md`, `mtp-protocol.md`, `ssm-occupancy-protocol.md`.

## P1. Parameter verification is mandatory and precedes every run

**Passing a parameter is not evidence that the parameter took effect.**
Before any timed run, every parameter that defines an arm must be proven
active from the instrument's own reporting, and that proof recorded in the
run record next to the numbers.

An arm's parameters are the ones whose values distinguish it from another
arm, plus any value quoted in the result (dtype, shape, KV type, context
length, spec type and draft width, thread/block dims, n_predict, temperature,
buffer count, clock state).

The check is: **read the value back from the running system, not from the
command you typed.** Sources that count as read-back —

- llama-server: `GET /props` for effective load-time settings; the response's
  `timings` block (`draft_n`, `draft_n_accepted`, `speculative`) for per-request
  settings; server stderr at load for what the flags resolved to.
- our engine: values printed by the run itself (`prompt tokens:`, `spec k:`,
  `tokens:`), and for comptime constants the binary being timed rebuilt in the
  same command as the run.
- hipBLASLt / vendor calls: the algo actually selected and its tuned attributes
  (splitK, wgm, workspace), dumped from the shim, not assumed from the heuristic.
- kernels: grid/block dims and template parameters echoed by the bench harness.

Sources that do NOT count: the flag string, the request JSON, the protocol
text, a previous run's receipt, or a maintainer's recollection.

**No receipt, no arm.** An arm whose parameters were not verified before its
timed run is VOID and its numbers may not be recorded, cited, or compared —
identical standing to the >10% spread rule. A void arm is re-run, not
retro-justified.

If a parameter cannot be read back, it may not be set per-request. Define the
arm by a controlled restart or rebuild whose configuration IS observable,
record that this was necessary, and restore the original configuration with a
health check afterwards.

### Why this rule exists

Both of this repo's known poisonings were silently-inert parameters, not bad
math:

- **decode-race arm B**: `"speculative.n_max": 0` was accepted by the server
  and ignored by that build. `draft_n` in the response was unchanged, so the
  "no-MTP" arm was still speculating. Caught only because someone read the
  timings block. Had it gone unread, the published no-spec bar would have been
  the MTP number and every later comparison would have inherited it.
- **hipBLASLt shim**: splitK and wgm were never actually set, so our GEMM
  measured ~2x faster than "the vendor". The parameter absence *was* the
  result. Fixing it took hipBLASLt 2497 -> 5201 GFLOP/s and erased the lead.

Both cases produced clean-looking numbers with tight spreads. **Spread does not
detect an inert parameter** — a consistently mis-configured arm is consistently
mis-configured. P1 is the only check that catches this class.

## P2. Verification comes before prediction freeze

Verify parameters, then freeze predictions, then run. Verifying after the run
lets the observed number inform which knobs get scrutinised, which is the same
defect as an unfrozen prediction.

## P3. The receipt is part of the result

Record, in the protocol's Result/Verdict section: what was checked, where the
value was read from, and the observed value. A verdict without its receipt is
incomplete and may not be promoted to "the bar".

## P4. Decode and speculation claims come from the prompt set, never one prompt

Any tok/s, speedup or acceptance number quoted as "ours" or "the bar" is the
median over `bench/mtp-prompts/` (20 prompts, `bench/mtp-prompts.sh`), with
the min-max range beside it, and the competitor measured on the same set in
the same session (`tools/llama-mtp-prompts.sh`). A single-prompt run is an
instrument receipt or a bug-isolation probe; it may appear in a protocol as
such, never in README or a verdict line.

Why: 2026-09-04 the MTP loop was reported at 128 tok/s and 1.17x llama.cpp
from the 5-token race prompt (94% acceptance). On 20 prompts it was 82 and
0.66x. Both numbers were correct; only one described the engine.

## P5. Every kernel run at m > 1 carries a row-scaling receipt

Before a kernel is dispatched at m > 1 on a timed path (prefill, speculative
window), the cold-cache bench records its cost at m = 1, 2, 4, 8 on the ffn
shape, and the m-row / 1-row ratio is in the landing commit. A ratio above
1.25x at m = 4 on a bandwidth-bound shape is a defect to fix or a documented
falsifier, not a property to live with.

Why: the q8row kernel's m > 1 path shipped inside the q8 round on the m = 1
receipt alone. It cost 8x per window (runtime-indexed accumulators), then
2.9x (8-row template with half-width loads), and every MTP number sat on it
for a day.

## P6. Profile the harness before the kernel

When a sweep's wall time is more than 2x its summed GPU time, or a run prints
a load/setup time above 10% of its total, the harness (loader, host copies,
per-window synchronizes) is profiled and fixed before any kernel is touched.
Run receipts include `pack loaded in` next to `tok/s_gen`.

Why: the engine loader copied 10.7 GB with a per-byte loop, 12.6 s of every
17 s run; a 100-run sweep cost 30 min and a whole optimisation stint was
planned around waiting for it. memcpy: 1.8 s.
