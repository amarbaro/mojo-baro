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

## Result: an fp16 WMMA GEMM ahead of hipBLASLt at every size tested

The second kernel line is a square fp16 GEMM on RDNA3's WMMA units. The pipelined
kernel runs 4x2 warps over a 128x128 block with two LDS buffers, one barrier per
K-step, a two-deep global prefetch, XOR-swizzled A and transposed B — 188 VGPR,
zero spills, 32 KB LDS.

One block shape does not win everywhere: at small sizes the 128x128 tile cannot
fill the GPU with workgroups. Tile geometry is therefore a kernel parameter and
the launcher dispatches on how many blocks the grid would have — `>= 96` blocks
takes 128x128 (8 waves), `>= 64` takes 64x128, below that 64x64 (4 waves).

GFLOP/s, ten square sizes, ours vs hipBLASLt fp16 through the same shim, 10 s
clock warm-up, every arm-defining parameter read back from the binary's own JSON:

| size | 256 | 512 | 768 | 1024 | 1536 | 2048 | 2560 | 3072 | 3584 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|
| **ours** | 6372 | 30642 | 64204 | 74824 | 93632 | 91300 | 97974 | 99307 | 105786 | 90705 |
| hipBLASLt | 6288 | 26324 | 54924 | 63623 | 69332 | 80203 | 87147 | 97224 | 85671 | 82437 |
| ratio | 1.01 | 1.16 | 1.17 | 1.18 | 1.35 | 1.14 | 1.12 | 1.02 | 1.24 | 1.10 |

Reproduce with `bench/fp16-templates.sh`. Before the dispatch landed the same
kernel was *behind* the vendor at every size at or below 1024 (0.78–0.98x); the
single 128x128 tile was the whole deficit.

Warm-up is load-bearing and was worth more than any kernel change at the top
end: at a 1 s warm-up the identical binary read 66k GFLOP/s at 4096³ instead of
91k, because the clocks had not settled. Benches now warm for 10 s and log
`warmup_s` in their receipts.

Note that fp32 WMMA **does not exist** on gfx1100 — it is an ISA limitation, not
a Mojo one, verified against `llvm-mc` and the LLVM builtin table. fp16 and fp32
GEMM numbers in this repo are therefore not comparable to each other; they use
different hardware inside the same chip.

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

2026-09-04, q8 weights (`bench/q8-protocol.md`, pack bit-equal to llama.cpp Q8_0):

| | tok/s | of q8 HBM roof (93) |
|---|---|---|
| llama.cpp Q8_0, no MTP | 74.1 | 80% |
| **mojo-baro q8 `tok/s_gen`** | **68.8** (68.64–68.82) | 74% |

So the trunk decode path runs at **0.94x llama.cpp** with no speculative
decode. Disclosed asymmetries, uncorrected: llama.cpp uses a q8_0 KV cache and
ours is f32 (negligible at these context lengths), and llama.cpp's number came
through an HTTP server while ours is measured in-process.

2026-09-04, MTP speculative decode (`bench/mtp-protocol.md`, `BARO_SPEC=1`,
k=4, draft = the model's own `blk.32` NextN head, verified in one k+1-row
trunk window, greedy accept, output still bit-identical to the no-spec run):

| | tok/s | vs no-spec |
|---|---|---|
| llama.cpp Q8_0, MTP speculative | 109.8 | 1.48x |
| **mojo-baro q8 MTP `tok/s_gen`** | **128.0** (127.6–128.0, 0.3% spread) | 1.89x |

So with speculation the engine decodes at **1.17x llama.cpp** on the same
card, same GGUF, same quant, same tokens. Caveats that stay attached to
that number: one 5-token prompt, 64 greedy tokens, acceptance 50/53 on a
repetitive tail; per-prompt acceptance on a real prompt set is in
`bench/mtp-protocol.md` as it lands.

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

Requires ROCm 7.2, CMake, and [`uv`](https://docs.astral.sh/uv/). Nothing is
installed machine-wide: `uv sync` creates a repo-local `.venv` from
`pyproject.toml`, which pins `max[all]==26.5.0` (Mojo 1.0.0). Every script in
this repo invokes `./.venv/bin/mojo` directly and never a system Mojo.

```sh
uv sync           # creates .venv with the pinned Mojo/MAX toolchain
./run-tests.sh    # builds the shim, checks an fp16 GEMM through the C ABI
./bench/run.py    # correctness gate, then throughput
```

Verified from a clean clone: `uv sync` then `./run-tests.sh` prints
`GEMM OK — 4 x 3 @ 3 x 2 matches host reference` and exits 0.

Model weights are not distributed with this repo.

## Portability

Everything here is measured on one card. RDNA3 specifics are load-bearing —
warp size **32**, not 64 as on CDNA; 64 KB LDS per block; tile and CPT
parameters swept for this memory system. None of the results should be assumed
to transfer to gfx942/gfx950 or to NVIDIA without re-sweeping.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright 2026 AmarBaro Labs.
