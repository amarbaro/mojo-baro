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
