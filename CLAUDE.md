# mojo-baro

- **Kernel files (`kernels/matmul*.mojo`, `kernels/elementwise.mojo`) carry
  ZERO comments and ZERO docstrings.** Rationale, sweep numbers, and design
  notes go in commit messages, `docs/`, and `~/Brain/mojo-baro/` — never in
  kernel source. Test files may keep docstrings.
- `docs/BASELINE.md` = current truth; read before kernel work.
- **Before ANY timed run: read every arm-defining parameter back from the running
  system and record it.** Passing a flag is not evidence it took effect; a
  silently-inert parameter produces clean numbers with tight spread and spread
  will not catch it. No receipt, no arm — see `bench/PROTOCOL-RULES.md` P1. **Decode numbers = 20-prompt median (P4); any m>1 kernel needs a row-scaling receipt (P5); harness before kernel (P6).**
- Perf claims require the preregistration flow in `bench/coldcache-protocol.md`
  style: freeze predictions by commit BEFORE running; single-buffer GEMM
  timings at W >= 96 MB are invalid (Infinity Cache contamination).
- Verify: `./run-tests.sh`, `./bench/run.py [bench-src.mojo]`, parity tests
  in `kernels/test_*.mojo` (build AOT into `.work/`, `-I kernels`).
