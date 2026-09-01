# Baseline — what is verified working

Current truth for `mojo-baro`. Every number here was measured on this machine and
is reproducible with the command given. If you are an agent picking up a kernel
task, this is your starting point: **do not re-derive it, and do not trust a
number that is not in this file or produced by `bench/run.py`.**

Last verified: 2026-08-31.

## Hardware and toolchain

| | |
|---|---|
| GPU | AMD RX 7900 XTX, **gfx1100 (RDNA3)** — 24 GB, ~22.3 GB free to MAX |
| Warp size | **32** (RDNA3, *not* 64 like CDNA) |
| LDS | 64 KB/block |
| ROCm | 7.2, hipBLASLt present at `/opt/rocm/lib/libhipblaslt.so` |
| Mojo | **1.0.0** — repo-local venv, `./.venv/bin/mojo` |
| MAX | 26.5.0 |
| Rust | 1.97.1 (service layer not started) |

Nothing is installed machine-wide. `uv` created `.venv` from the Modular nightly index.

## Layers

```
Python/app  →  Mojo (GPU kernels)  →  C++ shim (vendor SDK)  →  hipBLASLt
                      ↑
                    Rust (network/API shell) — NOT STARTED
```

Rust talks only to Mojo's C-ABI surface. It must **not** bind the C++ shim
directly: two independent owners of one hipBLASLt context is a lifetime bug.

## What runs

| Component | Path | Verify with |
|---|---|---|
| C++ hipBLASLt shim | `shim/` | `./run-tests.sh` |
| Mojo FFI binding | `kernels/amarbaro.mojo` | `./run-tests.sh` |
| GEMM kernels | `kernels/matmul.mojo` | `./bench/run.py` |
| Bench + correctness engine | `bench/run.py` | `./bench/run.py` |
| Parameter sweep | `bench/sweep.py` | `./bench/sweep.py` |

Related work lives in `~/AMDHQ` — this box's AMD/ROCm evaluation lab (`lab` CLI,
`tools/`, `runs/` ledger, rocprofv3 captures). Its 2026-08-27 shortlist already
recorded verdicts worth knowing before reaching for ROCm ecosystem pieces:
**aiter/ATOM have no RDNA3 build path** (gfx942/gfx950 only), **hip-ep's autotune
LUT is gfx1151-only**, and MIGraphX EP is already installed system-wide.

`./run-tests.sh` builds the shim and checks an fp16 GEMM through the C ABI
against a host reference. `./bench/run.py` gates on correctness *before*
reporting throughput and exits non-zero if any variant is wrong.

## Measured kernel performance

512×512×512, fp32, 200 iterations, 10 warmup. Run-to-run spread <0.5%.

| variant | GFLOP/s (512³) | notes |
|---|---|---|
| `hipblaslt` | ~5200 | **vendor baseline**, tuned (algo × splitK × wgm) |
| `matmul_naive` | ~1250 | one thread per output element |
| `matmul_tiled` | ~2270 | 16×16 shared-memory tiles |
| `matmul_regtile` | ~5130 | BM32 BN32 BK8 TM2 TN2, swept |

**regtile and tuned hipBLASLt are a tie**, trading places by size:

| size | hipBLASLt | regtile | winner |
|---|---|---|---|
| 512³ | 5370 | 5103 | vendor +5.2% |
| 1024³ | 6905 | 6790 | vendor +1.7% |
| 2048³ | 7089 | 7349 | ours +3.7% |
| 4096³ | 6758 | 7236 | ours +7.1% |
| 8192³ | 6263 | 6290 | tie +0.4% |

Both peak around 2048–4096³ and decline after; neither scales past 4096³, which
points at L2/HBM traffic rather than compute. `naive` degrades monotonically
(1129 → 658) as the working set outgrows cache.

Both sustain ~7.1 TFLOPS at large sizes, roughly **23% of the ~30.7 TFLOPS
plain-FMA peak** (~61 TFLOPS is the dual-issue figure).

**Do not repeat the retracted claim.** An earlier version of this file recorded
regtile as ~2× faster than hipBLASLt. That was measuring an *untuned vendor
call*, not a fast kernel. Removing three defects in our own shim — per-call
workspace allocation, trusting the heuristic's ordering, and never setting
splitK/wgm — took hipBLASLt from 2497 to 5201 GFLOP/s at 512³ and erased the
lead entirely. Matching a tuned hipBLASLt is still a good result; it is a much
smaller claim than the one first recorded.

A vendor baseline that looks easy to beat is a bug in your harness until
proven otherwise.

## Hard-won facts — do not relearn these

**Mojo 1.0 API.** `fn`, `alias`, `let`, `inout`, `owned` are all gone; use `def`,
`comptime`, `var`, `mut`. Imports take a `std.` prefix.

**`DLHandle` no longer exists.** FFI goes through `external_call`, and the binary
must be linked against the shared library at build time:
`-Xlinker -L<dir> -Xlinker -lamarbaro_shim -Xlinker -rpath -Xlinker <dir>`.
`mojo run`'s JIT will **not** resolve external symbols and `LD_PRELOAD` does not
reach it — build AOT.

**`Pointer` is non-null by design** and cannot hold a C handle that may come back
null. Carry opaque handles as `Int`, or use `Optional[Pointer[...]]`.

**Kernel scalar arguments must be fixed-width.** `Int`/`UInt` are not
`DevicePassable` — use `Int32` and convert inside the kernel.

**Accumulators must be SIMD values with comptime-unrolled indices.** Holding them
in `stack_allocation` gives you *scratch memory*, which on AMD lives in device
memory. The first register-tiled kernel did this and ran at 570 GFLOP/s —
**slower than naive**. Same algorithm, 6.4× apart.

**Benchmark at ≥200 iterations.** At 20, launch overhead dominated: both variants
reported roughly half their true throughput with ~10% run-to-run spread.

**CMake: link `hip::host`, not `hip::device`,** for a shim with no device code.
`hip::device` injects `--offload-arch` flags a non-clang host compiler rejects.

**t-strings reject format specifiers** — build JSON with String concatenation.

**hipBLASLt needs three things done right or it benchmarks as garbage.**
(1) Cache the workspace and selected algorithm on the context — a per-call
heuristic plus `hipMalloc`/`hipFree` costs more than the GEMM and read 1373
GFLOP/s, *below our naive kernel*. (2) The heuristic returns candidates in
predicted, not measured, order — time all of them. (3) **`splitK` and `wgm` are
only reachable through `hipblaslt_ext`**, not the C API, and are worth more than
algorithm choice: they took 3097 → 5201 GFLOP/s. Winning `splitK` decays with
size (4 at 512³, 1 at 2048³), consistent with small problems failing to fill the
GPU with workgroups.

**Never let the GPU idle inside a benchmark.** The card sits at ~28 MHz / 31 W
and ramps to ~3000 MHz in **~0.4 s of back-to-back work**, then holds. It drops
again the moment anything interrupts. Consequences, all measured:

- A fixed 10-iteration warmup (~10 ms at 512³) never leaves idle clocks.
- The bias tracks *measurement order*: whichever variant runs first is penalised
  most. Fixing it moved naive +33%, regtile +9%, hipblaslt +4%.
- **Each variant re-warms immediately before it is timed.** One warmup at the
  start is not enough — the host-side correctness check between variants is long
  enough to lose the clocks.
- Do not call `rocm-smi` inside a timing loop. Doing so idled the GPU between
  samples and produced a fake 11k↔27k GFLOP/s oscillation at half true speed;
  power read 31–54 W against a 290 W cap, which is the tell. Sample clocks from
  a separate process.

**MAX claims ~90% of VRAM on `DeviceContext()` creation.** It is a pool, not a
leak: it appears instantly, sits at ~22.7 GB whether the problem is 4096³
(192 MB of matrices) or 8192³ (768 MB), and stops at whatever is free. Cap it
with `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` — the bench harness
defaults it to 10.

| cap | VRAM held | regtile GFLOP/s (1024³) |
|---|---|---|
| 100% | 23.06 GB | **segfault in `hipblasLtCreate`** |
| 10% | 2.72 GB | 7703 |
| 2% | 0.88 GB | 7591 |

At 100% there is nothing left for hipBLASLt to allocate its handle and the
process dies. Capping costs no throughput — it measured slightly *faster*.
This is the constraint on co-tenancy: uncapped, one MAX process owns the card.

**fp32 WMMA does not exist on gfx1100 — it is the ISA, not Mojo.** Verified
three ways: `llvm-mc -mcpu=gfx1100` accepts `v_wmma_f32_16x16x16_f16` and
rejects `..._f32`; `BuiltinsAMDGPU.def` has zero f32-input WMMA entries; and
Mojo's own constraint reads *"RDNA WMMA does not support FP32 inputs (only
FP16/BF16 -> FP32)"*. **Any fp16 TFLOPS figure (aiter's ~82–89) is therefore not
comparable to this fp32 benchmark** — they use different hardware inside the
same chip.

**fp16 WMMA kernel — measured (`bench/bench_fp16.mojo`, `kernels/matmul_wmma_lds.mojo`).**
`WARPS 4x4 WTILE 2x1`, swept over 72 configs with `./bench/sweep.py wmma`:

| size | ours | aiter tuned Triton |
|---|---|---|
| 512³ | **9285** | 6175 |
| 1024³ | 21732 | 47485 |
| 2048³ | 28758 | 82202 |
| 4096³ | 27133 | 88897 |

We lead at 512³ and trail by ~3x at 4096³. **`WTILE_N=1` is on every top-nine
config** — a wave owning more than one 16-wide column strip halves throughput —
but the winners use 512 threads and a 128x64 block tile. The rule here is *many
resident waves each owning a thin column strip*, not small blocks. `BLK_K` must
stay at 16: widening it to 64 puts 16 KB of LDS in a block and halves throughput.

**fp16 WMMA in Mojo works, and the fragment shape is the trap.** RDNA3 wave32
wants **a/b = 16 wide, c/d = 8 wide** (matching
`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` typed `V8fV16hV16hV8f`). The
8-wide fragments `TensorCore.load_a` builds are the CDNA/NVIDIA shape and fail
with *"no valid implementation of mma"*. Measured lane mapping:

```
A: lane L, elem i(0..15) -> A[L % 16][i]
B: lane L, elem i(0..15) -> B[i][L % 16]
D: lane L, elem i(0..7)  -> D[2*i + L // 16][L % 16]
```

D is not the obvious layout — consecutive `i` steps two rows and the half-waves
interleave. Verified 16×16×16 tile: 0 mismatches, max_err 5.96e-07. Working
probes in `.work/wmma/`. Also: `get_mma_shape` has **no working RDNA entry** for
any dtype/shape_id — pass `Index(16,16,16)` to `TensorCore` explicitly. And the
whole `TensorCore` surface takes `LayoutTensor` (built via `Layout.row_major`),
which does not unify with the `TileTensor` our kernels use; calling `mma`
directly with hand-built fragments avoids both problems.

**gfx1100 is thinly tuned in hipBLASLt.** 8 fp32 (`SS_SS`) Tensile libraries vs
134 for gfx942; 95 total gfx1100 files vs 1111. The fp32 heuristic offers only
**4** candidate algorithms at any size. This is the contribution opportunity.

**MAX `DeviceBuffer.unsafe_ptr()`** yields a raw device address that the shim's
hipBLASLt calls accept directly — no copy needed to compare against vendor.

## Tuning findings

Small tiles win on RDNA3, contradicting NVIDIA-derived tiling guidance. Every
top-5 swept configuration uses BM=32 or 64; hand-picked BM128/BN128/TM8/TN8
**regressed to 1651 GFLOP/s** (0.45×), almost certainly register pressure
collapsing occupancy. Occupancy is buying more than data reuse here.

Do not trust large-tile intuition on this card. Measure it.

## Rules for kernel work

1. **Correctness first.** A fast wrong kernel is a failure. `bench/run.py`
   checks before it times.
2. **One number, one commit.** Results go to the **AMDHQ experiment ledger**
   (`~/AMDHQ/runs/runs.jsonl` + `runs.sqlite`) under `role_key="mojo-baro-gemm"`,
   tagged with the commit that produced them. That ledger is this box's existing
   record of ROCm experiments — do not start a second one here. `bench/sweep.py`
   keeps its own `sweep.jsonl` because a 198-point parameter search is search
   output, not an experiment record.
3. **Numeric parameter search belongs in `bench/sweep.py`,** not in a human or
   an agent. It is exact, free, and already beat a hand-tuned config by 1.38×.
4. **New strategies go in their own file** (`kernels/matmul_<strategy>.mojo`) and
   register a variant in `bench/bench.mojo`. Never edit another agent's kernel.
