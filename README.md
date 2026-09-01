# mojo-baro

An inference stack for AMD RDNA3 (gfx1100 / RX 7900 XTX), with the GPU kernels
written in [Mojo](https://www.modular.com/mojo).

The point of the project is the kernels. RDNA3 is not the architecture the
vendor libraries are tuned for — hipBLASLt's attention is on CDNA — and consumer
cards are where most people actually have 24 GB of VRAM. This repo is an attempt
to find out how much of that gap is real.

## Result: a cold-cache vendor beat at the decode shape

Single-token decode is bound by streaming weights out of HBM, not by compute.
The shape that matters is a skinny GEMM: **M=1, K=4096, N=12288**, weights not
resident in Infinity Cache.

| kernel | µs/launch (10 repeats) |
|---|---|
| `amar_matmul_skinny_v2` CPT=8 @ M=1 | 137.8 – 138.9 |
| **`amar_matmul_skinny_m1` CPT=8** | **121.2 – 121.8** |
| hipBLASLt f16 @ M=1 | 122.1 – 123.3 |

The `m1` range sits strictly below the vendor's — margin ~1%, ranges disjoint.
The mechanism is occupancy relief, not fewer bytes: specializing for M=1 drops
the 8-row LDS staging and shrinks the accumulator from 64 lanes to `SIMD[CPT]`,
so more waves stay resident. Byte traffic is identical between the two.

Getting there took four preregistered rounds, three of which falsified their own
predictions:

| round | change | cold-cache result |
|---|---|---|
| v1 | bf16 B-layout (transpose weights at load) | 194 µs, 517 GB/s (54% of peak) |
| v2 | K-major q8 | 165 µs — **slower than modeled**; dequant ALU ate the byte win |
| v3 | CPT=8 columns/thread, wide loads | 139 µs, 723 GB/s (75% of peak) |
| v4 | M=1 specialization | 121 µs — below vendor |

Reproduce: `./bench/run.py`, or `bench/bench_coldcache_m1.mojo` for the table
above. The protocol — 8 rotating device buffers to defeat Infinity-Cache
contamination, 1 s clock warm, 200 timed launches, 10 in-process repeats — is in
[`bench/coldcache-protocol.md`](bench/coldcache-protocol.md), frozen before each
run alongside its predictions and its falsifier.

## How numbers get into this repo

Every performance claim here is preregistered: the question, the instrument, the
predicted range and the condition that would falsify it are committed **before**
the run, and the result is recorded against them whether or not it agreed. Missed
predictions stay in the file. `bench/run.py` gates on correctness before it will
report throughput, and exits non-zero if any variant is numerically wrong.

This is not ceremony. An earlier version of `docs/BASELINE.md` recorded our
register-tiled GEMM as ~2× faster than hipBLASLt. It was measuring an *untuned*
vendor call. Fixing three defects in our own shim — per-call workspace
allocation, trusting the heuristic's ordering, never setting splitK/wgm — took
hipBLASLt from 2497 to 5201 GFLOP/s and erased the lead completely.

**A vendor baseline that looks easy to beat is a bug in your harness until proven
otherwise.**

## Engine status

There is a working end-to-end decode engine (`serve/engine.mojo`) for a bf16
Qwen-architecture GGUF: full-model greedy decode, 64 tokens **bit-identical** to
llama.cpp on the same file, verified by `tools/check-tokens.sh` against a
reference token-id array. It is a correctness vehicle for the kernels, not a
product — no server, no batching, no sampler beyond greedy.

### Throughput against llama.cpp

The engine reports two numbers, and only one of them is comparable to anything.
`tok/s_gen` divides `GEN_N - 1` by decode time alone, which is exactly what
llama.cpp's `timings.predicted_per_second` measures; `tok/s_total` includes
prefill and is reported for completeness only. Quoting `tok/s_total` against
llama.cpp would flatter or damage us depending on prompt length, not on kernel
quality.

Measured on the same box and the same bf16 GGUF, 5-token prompt, 64 tokens,
greedy, no speculative decode. Repeat rule from
[`bench/decode-race-protocol.md`](bench/decode-race-protocol.md): 5 runs,
discard the first, median of the remaining 4.

| | tok/s | of HBM roof (53.6) |
|---|---|---|
| llama.cpp, no MTP | 44.1 | 82% |
| **mojo-baro `tok/s_gen`** | **41.3** (41.04–41.38, 0.8% spread) | 77% |
| llama.cpp, MTP speculative | 109.8 | — |

So the trunk decode path runs at **0.94x llama.cpp** with no speculative
decode. Disclosed asymmetries, uncorrected: llama.cpp uses a q8_0 KV cache and
ours is f32 (negligible at these context lengths), and llama.cpp's number came
through an HTTP server while ours is measured in-process.

The 2.5x sitting in llama.cpp's MTP column is the real gap, not the 6%. The
MTP draft head (`blk.32`) is implemented and numerically validated against a
numpy reference, but it is still dump-only — nothing is drafted or verified in
the decode loop yet, so no speculative number belongs in this table.

## Layout

```
Python/app  →  Mojo (GPU kernels)  →  C++ shim (vendor SDK)  →  hipBLASLt
                      ↑
                    Rust (network/API shell) — NOT STARTED
```

All boundaries are C ABI. Rust talks only to Mojo's C surface; it must not bind
the C++ shim directly, since two independent owners of one hipBLASLt context is a
lifetime bug.

| | |
|---|---|
| `kernels/` | Mojo GPU kernels + their parity tests |
| `bench/` | benchmark harness and the frozen protocols |
| `serve/` | the decode engine |
| `shim/` | C++ hipBLASLt shim behind a C ABI |
| `tools/` | GGUF loader/packer, numpy reference implementations, gates |
| `docs/BASELINE.md` | **current truth** — every verified number and trap |

## Building

Requires ROCm 7.2, Mojo 1.0.0 / MAX 26.5.0, and CMake. Nothing is installed
machine-wide; `uv` creates a repo-local `.venv` from the Modular nightly index.

```sh
./run-tests.sh    # builds the shim, checks an fp16 GEMM through the C ABI
./bench/run.py    # correctness gate, then throughput
```

Model weights are not distributed with this repo.

## Portability

Everything here is measured on one card. RDNA3 specifics are load-bearing —
warp size **32**, not 64 as on CDNA; 64 KB LDS per block; tile and CPT
parameters swept for this memory system. None of the results should be assumed
to transfer to gfx942/gfx950 or to NVIDIA without re-sweeping.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright 2026 AmarBaro Labs.
