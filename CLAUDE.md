# mojo-baro

- **Kernel files (`kernels/matmul*.mojo`, `kernels/elementwise.mojo`) carry
  ZERO comments and ZERO docstrings.** Rationale, sweep numbers, and design
  notes go in commit messages, `docs/`, and `~/Brain/mojo-baro/` — never in
  kernel source. Test files may keep docstrings.
- `docs/BASELINE.md` = current truth; read before kernel work.
- Perf claims require the preregistration flow in `bench/coldcache-protocol.md`
  style: freeze predictions by commit BEFORE running; single-buffer GEMM
  timings at W >= 96 MB are invalid (Infinity Cache contamination).
- Verify: `./run-tests.sh`, `./bench/run.py [bench-src.mojo]`, parity tests
  in `kernels/test_*.mojo` (build AOT into `.work/`, `-I kernels`).
