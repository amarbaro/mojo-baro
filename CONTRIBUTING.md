# Contributing

The most useful contribution to this repo is a benchmark receipt from a card we
do not have.

Every performance number here was measured on one AMD RX 7900 XTX. The kernels
target RDNA3 (`gfx1100`) and are tuned against that card's 96 compute units,
96 MB Infinity Cache and 960 GB/s of bandwidth. Whether any of it holds on an
RX 7900 XT, a 7800 XT, a W7900, or an RDNA4 card is unknown — not "probably
fine", genuinely unmeasured.

## Send a hardware report

Needs an AMD GPU, ROCm, and about fifteen minutes. **No model weights are
required** — the fp16 GEMM benchmark is self-contained.

```sh
git clone https://github.com/amarbaro/mojo-baro && cd mojo-baro
uv sync                    # repo-local .venv, pinned Mojo/MAX; nothing machine-wide
./bench/report.sh --check  # preflight: GPU, ROCm, hipBLASLt, toolchain
./bench/report.sh          # the real run
```

It prints a markdown block and writes `results/report-<gfx>-<card>-<commit>.json`.
Open a [hardware report issue](../../issues/new?template=hardware-report.yml),
paste the block, attach the JSON.

`./bench/report.sh --quick` runs three sizes instead of ten if you are short on
time. Prefer the full run when you can: the interesting disagreements have so
far been at the small end, where filling the GPU is the constraint.

**File the issue even if it fails.** A build error on gfx1201, a card that
throttles, a hipBLASLt version that behaves differently — those are results. The
script writes a receipt marked `"valid": false` rather than nothing, and says
why.

## What the script protects

Two ways to produce a number that looks fine and is not, both of which have
bitten this repo:

- **Short warm-up.** The same binary reads 66k GFLOP/s at 4096³ after one second
  and 91k after ten, purely because clocks have not settled. The benches warm for
  10 s and record `warmup_s`; the receipt is marked invalid below that floor.
- **An untuned baseline.** An early version of `docs/BASELINE.md` claimed ~2x
  over hipBLASLt. It was measuring a badly-configured vendor call. The shim now
  sets per-call splitK/wgm, and the receipt records the algorithm hipBLASLt
  actually chose.

Numbers are not quoted anywhere in this repo unless they can be traced to a
commit, a warm-up and a correctness check. That is why the script refuses to
emit a clean-looking receipt it cannot stand behind.

## Other benches on unfamiliar hardware

`bench/wmma-peak.sh` measures the WMMA issue-rate roofline. The grid is sized to
fill the card, so the compute-unit count is arm-defining; the script reads it
from the running GPU rather than trusting the 96 hardcoded in the source for
this box's XTX. Pass a count explicitly to override.

The rest of the benches need model weights, which are not distributed here, so
`bench/report.sh` and `bench/wmma-peak.sh` are the two that will run on a fresh
clone.

## Code changes

Read `bench/PROTOCOL-RULES.md` first, and `docs/BASELINE.md` for current truth.
Performance claims follow the preregistration flow: the question, the
instrument, the predicted range and the condition that would falsify it are
committed **before** the run, and the result is recorded whether or not it
agreed. Missed predictions stay in the file — three of the four cold-cache
rounds falsified their own predictions, and that history is the point.

Kernel files (`kernels/matmul*.mojo`, `kernels/elementwise.mojo`) carry no
comments or docstrings by convention; rationale goes in commit messages and
`docs/`.

Correctness gates before anything else: `./run-tests.sh` builds the shim and
runs the parity tests, and `tools/kernel-census.py` fails if a kernel is
unreachable from the engine, a bench or a test.
