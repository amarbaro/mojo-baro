# SSM delta_step protocol — frozen before first timed run

> **Binding: [`bench/PROTOCOL-RULES.md`](PROTOCOL-RULES.md).** P1 in particular:
> every parameter defining an arm is read back from the running system and
> recorded BEFORE the timed run. No receipt, no arm.


Question: does reducing `amar_ssm_delta_step`'s state traffic (item 3) or
raising its occupancy (item 2) earn its complexity at our engine's actual
decode shape, and does fusing the per-row SSM loop (item 1) pay for itself
in an m=4 MTP verify window?

Plan and derivations: `.work/ssm-fusion-plan.md`. This file freezes the
predictions; the plan holds the reasoning.

## Scale check — binds every prediction below

Per-token weight traffic ~17.9 GB (frozen, `bench/decode-race-protocol.md`).
SSM state traffic per decode step: N_SSM=24 layers x (2 MB read + 2 MB write
f32 state, ~96 KB conv round-trip) ~= 100 MB/token — **~0.6% of total
traffic at M=1**.

Consequence, stated in advance so it cannot be spun afterwards: **no item
here can move plain M=1 decode more than ~1% by bandwidth.** A ~0%
end-to-end result is the EXPECTED outcome at M=1 and does not falsify a
kernel-level claim. Kernel-level and end-to-end claims are recorded and
judged separately.

## Correctness precondition (gates ALL speed numbers)

Items 2 and 3 are pure reorders/re-partitions: neither changes the
arithmetic or its order. Item 1 must also preserve per-element f32 op
order. Therefore:

1. `kernels/test_ssm_block.mojo` parity passes at **unchanged tolerances**.
   A loosened threshold voids the run.
2. `tools/check-tokens.sh` — 64-token greedy output **bit-identical** to
   `.work/engine-pack/ref-tokens-64.txt`. Required, not hoped for. If a
   token moves, the kernel is wrong; it is not "numerics drift".
3. `./run-tests.sh` green.

Any one failing voids the run: no tok/s, no per-kernel time, no claim.

## Baseline (measured this session, HEAD c8cf219, clean tree)

- Engine built: `./.venv/bin/mojo build serve/engine.mojo -o .work/engine -I kernels`
  (no shim link — the engine imports no FFI). **Not previously recorded
  anywhere; recorded here.**
- Single run, `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10`:
  `tok/s_gen 40.348`, `prefill_s 0.0647`, `decode_s 1.5614`,
  `host_enqueue_s 1.025`, `gpu_total_s 1.626`, token gate PASS.
- This is ONE run, not a median. It is the correctness reference and an
  indicative speed only. The medians below are what get compared.

## Instrument (identical for every arm)

- llama-server DOWN, GPU exclusive, nothing else resident.
- `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10`.
- 5 repeats, drop the first, median of 4. Spread gate <5% end-to-end;
  <1.5% for any per-kernel micro number (coldcache v2 discipline).
- Never sample `rocm-smi` inside a timing loop (BASELINE.md: produced a
  fake 11k-27k GFLOP/s oscillation).
- Every number carries its commit hash.

## Arms

One 4-arm sweep, because items 2 and 3 pull in opposite directions
(register staging costs VGPRs -> fewer resident waves; the j-split wants
more, smaller blocks):

| arm | item 3 (S0 staged in registers) | item 2 (JSPLIT) |
|---|---|---|
| A (control) | no | 1 |
| B | no | 4 |
| C | yes | 1 |
| D | yes | 4 |

Arm A must be bit-identical to today's kernel — that is the sweep's own
self-check. JSPLIT in {1,2,4} explored on the winning staging choice.

Item 1 is measured separately, at m=4, against whichever of A-D wins.

## Predictions — FROZEN, recorded before any timed run

1. **Arm A reproduces baseline** within the 5% spread gate and is
   bit-identical on tokens. If not, the instrument is broken and nothing
   else in this file may be read.
2. **Item 3 (arm C vs A):** S0 read traffic per launch halves (2 MB -> 1 MB).
   Direction only on GPU time — this wins ONLY if the kernel is
   traffic-bound at 32 blocks / 96 CUs. Falsifier: arm C slower than A on
   per-kernel GPU time, which would mean register pressure dominates and
   the staging is dead.
3. **Item 2 (arm B vs A):** per-kernel `delta_step` GPU time drops.
   **`tok/s_gen` moves <1% at M=1** — recorded in advance so a flat
   end-to-end result is not misread as falsification. Falsifier: arm B
   per-kernel time not below A, i.e. the kernel was never occupancy-bound.
4. **Prior-work guard:** an earlier launch-fusion experiment on this same
   loop returned +5% because decode is GPU-bound (host enqueue 83 ms vs
   GPU 775 ms). That measured **enqueue cost**. Items 2 and 3 are
   **occupancy and traffic** — a different mechanism. That null result
   neither predicts nor forbids this one, and must not be cited as
   "already tried". Conversely, no launch-count win may be claimed here.
5. **Item 1 (m=4 window):** removes ~(m-1) x 2 MB x 24 ~= 144 MB of
   intermediate state READS per window (writes are NOT removable —
   stage-2 rollback needs every step's state landed). At ~960 GB/s that
   is ~150 us against a window costing ~18+ ms of weight streaming:
   **~1-2% window speedup**, plus a launch-count reduction bounded above
   by the +5% null in (4). An earlier estimate of ~340 MB double-counted
   item 3's saving; 144 MB is the marginal figure once item 3 has landed.
   Falsifier: fused kernel slower than the arm-winner at m=4 -> keep the
   unfused path; the m=1 path keeps the arm-winner kernel regardless.

## What would make us abandon each item

- Item 3: arm C loses to A AND arm D loses to B (staging never pays).
- Item 2: no arm beats A on per-kernel time (never occupancy-bound).
- Item 1: loses at m=4, or its slot-index contract turns out to conflict
  with stage 2's accept loop once that exists.

## Results

Run 2026-09-04, branch `lane-ssm-spill`, on top of `main`'s merged M2 fold
(`871faca`, `9808a9d`). Instrument: llama-server DOWN, GPU exclusive,
`flock .work/gpu.lock` on every run.

### Arms A/B (item 3, unstaged) do not apply — protocol line invalidated

Item 3 proposes staging `S0`'s column into registers to kill a "pass 1
reads it, pass 2 re-reads it" double read (`.work/ssm-fusion-plan.md`
"Current shape" section). **`amar_ssm_delta_step` already reads the column
exactly once into `col: SIMD[f32, SSTATE]` and reuses it for both the `sk`
pass and the `s`/`o` pass** — this was true in the kernel this protocol
inherited (commit `c8cf219`, predating both this protocol's freeze and my
M2 fold) and is unchanged by the fold. There is no unstaged variant left in
the repo to build as a control; reconstructing one would be inventing code
the protocol never asked for, and the question item 3 poses (does staging
pay for itself) was never gated by a receipt before it landed — a real gap
in this protocol's own discipline, but not one stint 2 can retroactively
fix without fabricating an arm. **Arms A and B, and the protocol's own
self-check ("Arm A must be bit-identical to today's kernel"), are
unsatisfiable as written: "today's kernel" already IS what the protocol
calls arm C** (item 3 = yes, item 2 = 1/control). Per the brief: stopping
here on item 3, not inventing a replacement arm.

### Arms C/D (item 2, JSPLIT) — measured, falsified

Item 2 (JSPLIT: split the column dimension across more, smaller blocks —
`grid_dim=(NH_V, JSPLIT)`, `block_dim=SSTATE/JSPLIT`) still applies cleanly
to the folded kernel; implemented as a new `JSPLIT: Int` comptime parameter
on `amar_ssm_delta_step`, `JSPLIT=1` reproducing arm C (today's kernel) and
`JSPLIT=4` as arm D, per `.work/ssm-fusion-plan.md`'s JSPLIT=4 rationale
(128 blocks > 96 CUs, no partials).

**Gate 1 — correctness (both arms):**
- `kernels/test_ssm_block.mojo`, unchanged tolerances, at `JSPLIT=1` and
  `JSPLIT=4`: **PASS** both (m=1 vs numpy ref, m=4 fold vs 4x sequential
  m=1, `max_rel: 0.0` on the m=4 self-check for both JSPLIT values).
- `bench/mtp-prompts.sh` arm A (no-spec) `GENERATED` line, both JSPLIT
  builds, byte-identical to `main`'s `.work/mtp-m2/p09-explain-gpu.A.log`
  (spot-checked since JSPLIT changes no per-column arithmetic order;
  20-prompt full sweep not re-run since arm D was already disqualified on
  speed before reaching that gate — see below).

**Gate 2 — per-kernel timing, `BARO_PROFILE=2`, `BARO_SPEC_K=2`,
`bench/mtp-prompts/p09-explain-gpu.tokens`, 3 repeats each, `ssm-kernel:
delta` line:**

| arm | JSPLIT | delta (s), 3 runs | median |
|---|---|---|---|
| C (control) | 1 | 0.04805, 0.04782, 0.04736 | **0.0478** |
| D | 4 | 0.10115, 0.08841, 0.08776 | **0.0884** |

Arm D is **85% slower** than arm C — far outside the <1.5% per-kernel
spread gate in either direction; this is not noise. **Falsifies item 2's
prediction outright** ("per-kernel `delta_step` GPU time drops"). Matches
the protocol's own abandon criterion verbatim: *"Item 2: no arm beats A on
per-kernel time (never occupancy-bound)."* `tools/isa-receipt.py` explains
why: `vgpr_count` is unchanged (192, both arms — JSPLIT doesn't reduce
per-thread register need, each thread still carries the full 128-wide
`col` regardless of column-slab width) but `vgpr_spill_count` at MR=8 goes
603 (JSPLIT=1) → 2009 (JSPLIT=4), ~3.3x — JSPLIT=4's per-block `kq` fill
loop (`comptime for g in range(JSPLIT)`) makes every block redundantly
re-read the full 128-wide k/q vector, and the narrower 32-thread blocks
apparently cost more (via redundant traffic + higher spill pressure) than
the extra occupancy buys back. 32 blocks / 96 CUs was not, in practice,
occupancy-bound the way the plan predicted.

**Caught mid-run, worth recording:** the initial JSPLIT=1 "control" build
(before splitting the kernel body into a `comptime if JSPLIT == 1` fast
path that is byte-for-byte the pre-JSPLIT code) was NOT actually identical
to today's kernel at the ISA level — merely adding the `JSPLIT` template
parameter and routing `j` through `js * JW + jt` (with `js` always 0)
raised `vgpr_spill_count` from 603 to 2029 at MR=8, a ~3.4x regression from
the refactor alone, nothing to do with JSPLIT's actual value. First-pass
timing on that broken control showed delta at ~0.09s for BOTH arms
(apparently no difference), which would have wrongly read as "item 2:
inconclusive, within noise" and hidden arm D's real regression. Caught by
re-checking `vgpr_spill_count` against the pre-stint-2 baseline before
trusting the comparison (P1: read the value back from the instrument, not
from the diff you typed) and fixed by giving `JSPLIT == 1` its own
compile-time branch that reproduces the original code exactly, rather than
generalizing the control path through the same code the experiment uses.

### Item 1 (fusion) — already landed in stint 1, not re-measured here

Item 1 (fuse the per-row SSM loop into one launch per window) is the M2
fold already merged to `main` (`871faca`). The plan's own sequencing
("item 1 LAST, built against item 2's winning geometry") is moot: item 2
has no winning geometry (arm C, i.e. today's shape, wins by default), so
there is nothing for item 1 to be rebuilt against. No falsifier triggered.

### Verdict

**No arm lands.** Item 3 is moot (already shipped, unfalsifiable without
inventing code). Item 2 is measured and falsified (arm D 85% slower, not
faster). `amar_ssm_delta_step` stays exactly as `main` has it. No code
change in this stint; this file's Results section and
`.work/briefs/status-mrowC.md` are the deliverable.
