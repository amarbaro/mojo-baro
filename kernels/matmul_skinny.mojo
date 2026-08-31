"""Tall-skinny split-K GEMM for decode shapes (M <= 8, large N/K).

The square-tuned kernels lose 2.3x to hipBLASLt at M=1/8 for two structural
reasons the 2026-09-01 roofline note pins down: BM32 pads 75% of every A tile
when M=8, and the grid collapses to ~128 blocks (~1.3/CU) -- far too few
resident waves to hide HBM latency on what is a pure bandwidth problem
(streaming B once is ~all the mandatory traffic; ceiling ~3.8 TFLOP/s fp32).

Design: BM = SM = 8 (no padding), one column per thread so B reads coalesce
across the block, and split-K to multiply the grid: grid = (N/SBN, SPLITK) =
32 x 16 = 512 blocks. Each block stages its A slice (8 x KCHUNK) in LDS --
reads of it broadcast, so no pad needed -- streams its K-slice of B straight
from global, and writes an fp32 partial tile. A second trivial kernel sums
the SPLITK partials. Two-pass instead of float atomics: deterministic, and
the partial traffic (2 x SPLITK x M x N x 4B = 4 MB) is ~6% of the 67 MB
B stream at 8x4096x4096.
"""

from std.gpu import block_idx, global_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from matmul import dtype

# Swept 2026-09-01 at 8x4096x4096: SPLITK 8/16/32 -> 2744/4731/4826,
# SBN 64/128/256 -> 3167/4826/4902, KCHUNK 128/512 -> 5028/4857.
comptime SM = 8          # rows a block carries = max supported M
comptime SBN = 256       # columns per block; one per thread
comptime SPLITK = 32     # K-dimension grid split; grid = (N/SBN) * SPLITK
comptime SK_THREADS = SBN
comptime KCHUNK = 128    # LDS A-slice depth: 8 * 128 * 4B = 4 KB


def matmul_skinny[
    ALayout: TensorLayout, BLayout: TensorLayout, PLayout: TensorLayout
](
    A: TileTensor[dtype, ALayout, MutAnyOrigin],
    B: TileTensor[dtype, BLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)

    var tid = thread_idx.x
    var col = block_idx.x * SBN + tid

    # This block's K slice.
    var kslice = (K + SPLITK - 1) // SPLITK
    var k0 = block_idx.y * kslice
    var k1 = min(k0 + kslice, K)

    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[SM, KCHUNK]()
    )

    var acc = SIMD[dtype, SM](0)

    var kk = k0
    while kk < k1:
        var clen = min(KCHUNK, k1 - kk)

        # Stage the 8 x clen A slab; consecutive tids hit consecutive k.
        comptime for i in range(SM * KCHUNK // SK_THREADS):
            var e = tid + i * SK_THREADS
            var r = e // KCHUNK
            var kc = e % KCHUNK
            var g = kk + kc
            sa[r, kc] = rebind[sa.ElementType](
                rebind[Scalar[dtype]](A[r, g]) if r < M and kc < clen else 0
            )
        barrier()

        if col < N:
            for k in range(clen):
                var b_val = rebind[Scalar[dtype]](B[kk + k, col])
                comptime for i in range(SM):
                    acc[i] += rebind[Scalar[dtype]](sa[i, k]) * b_val
        barrier()
        kk += KCHUNK

    if col < N:
        comptime for i in range(SM):
            Cp[block_idx.y, i, col] = rebind[Cp.ElementType](acc[i])


def skinny_reduce[
    PLayout: TensorLayout, CLayout: TensorLayout
](
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    C: TileTensor[dtype, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
):
    """Sum the SPLITK partial tiles into C. One thread per output element."""
    comptime assert Cp.flat_rank == 3 and C.flat_rank == 2

    var M = Int(m)
    var N = Int(n)
    var gid = global_idx.x
    if gid >= M * N:
        return
    var r = gid // N
    var c = gid % N

    var acc: Scalar[dtype] = 0
    comptime for s in range(SPLITK):
        acc += rebind[Scalar[dtype]](Cp[s, r, c])
    C[r, c] = rebind[C.ElementType](acc)
