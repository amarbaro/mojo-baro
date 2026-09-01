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

(empty — nothing measured yet beyond the single baseline run above)
