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
from std.math import exp
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
    in_dtype: DType,
    ALayout: TensorLayout, BLayout: TensorLayout, PLayout: TensorLayout
](
    A: TileTensor[in_dtype, ALayout, MutAnyOrigin],
    B: TileTensor[in_dtype, BLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    """A/B in `in_dtype` (fp32 or bf16 weights); accumulate and emit fp32.

    bf16 halves the B stream, doubling the bandwidth roof; LDS keeps the A
    slab in fp32 so the inner loop is a single cast on the B load.
    """
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

    # k-major so the compute loop reads one SM-wide vector per k -- one LDS
    # read instead of SM scalar reads; the loop was issue-bound on LDS ops.
    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[KCHUNK, SM]()
    )
    var sav = sa.vectorize[1, SM]()
    comptime assert sav.flat_rank == 2

    var acc = SIMD[dtype, SM](0)

    var kk = k0
    while kk < k1:
        var clen = min(KCHUNK, k1 - kk)

        # Stage the clen x 8 A slab transposed; consecutive tids -> consecutive k.
        comptime for i in range(SM * KCHUNK // SK_THREADS):
            var e = tid + i * SK_THREADS
            var r = e // KCHUNK
            var kc = e % KCHUNK
            var g = kk + kc
            sa[kc, r] = rebind[sa.ElementType](
                rebind[Scalar[in_dtype]](A[r, g]).cast[
                    dtype
                ]() if r < M and kc < clen else 0
            )
        barrier()

        if col < N:
            for k in range(clen):
                var b_val = rebind[Scalar[in_dtype]](B[kk + k, col]).cast[
                    dtype
                ]()
                var a_vec = rebind[SIMD[dtype, SM]](sav[k, 0])
                acc += a_vec * b_val
        barrier()
        kk += KCHUNK

    if col < N:
        comptime for i in range(SM):
            Cp[block_idx.y, i, col] = rebind[Cp.ElementType](acc[i])


def matmul_skinny_wt[
    in_dtype: DType,
    ALayout: TensorLayout, WLayout: TensorLayout, PLayout: TensorLayout
](
    A: TileTensor[in_dtype, ALayout, MutAnyOrigin],
    W: TileTensor[in_dtype, WLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    """skinny with W in GGUF-native [out, in] row-major layout (= B transposed).

    C[m, col] = sum_k A[m, k] * W[col, k]. Each thread owns one output column
    and walks its W row contiguously in k, so per-thread reads are sequential
    rather than block-coalesced.
    """
    comptime assert A.flat_rank == 2 and W.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)

    var tid = thread_idx.x
    var col = block_idx.x * SBN + tid

    var kslice = (K + SPLITK - 1) // SPLITK
    var k0 = block_idx.y * kslice
    var k1 = min(k0 + kslice, K)

    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[KCHUNK, SM]()
    )
    var sav = sa.vectorize[1, SM]()
    comptime assert sav.flat_rank == 2

    var acc = SIMD[dtype, SM](0)

    var kk = k0
    while kk < k1:
        var clen = min(KCHUNK, k1 - kk)

        comptime for i in range(SM * KCHUNK // SK_THREADS):
            var e = tid + i * SK_THREADS
            var r = e // KCHUNK
            var kc = e % KCHUNK
            var g = kk + kc
            sa[kc, r] = rebind[sa.ElementType](
                rebind[Scalar[in_dtype]](A[r, g]).cast[
                    dtype
                ]() if r < M and kc < clen else 0
            )
        barrier()

        if col < N:
            # Per-thread W row is contiguous: 8-wide vector loads (16 B for
            # bf16), the scalar version was 40x off the bf16 roof.
            var Wv = W.vectorize[1, 8]()
            comptime assert Wv.flat_rank == 2
            for kv in range(clen // 8):
                var w8 = rebind[SIMD[in_dtype, 8]](
                    Wv[col, kk // 8 + kv]
                ).cast[dtype]()
                comptime for j in range(8):
                    var a_vec = rebind[SIMD[dtype, SM]](sav[kv * 8 + j, 0])
                    acc += a_vec * w8[j]
        barrier()
        kk += KCHUNK

    if col < N:
        comptime for i in range(SM):
            Cp[block_idx.y, i, col] = rebind[Cp.ElementType](acc[i])


comptime Q8_BLOCK = 32


def matmul_skinny_q8[
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Wq: TileTensor[DType.int8, QLayout, MutAnyOrigin],
    Ws: TileTensor[dtype, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    """skinny_wt with int8 weights dequantized in-register.

    Split layout (ours, not ggml's Q8_0): Wq [N, K] int8 quants,
    Ws [N, K/32] fp32 per-block scales. 1.125 B/elem vs bf16's 2 --
    the weight stream shrinks 1.8x, which is the whole point of decode
    quantization. K-slices stay multiples of 32 (SPLITK guarantees it
    for K % (SPLITK*32) == 0).
    """
    comptime assert A.flat_rank == 2 and Wq.flat_rank == 2
    comptime assert Ws.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)

    var tid = thread_idx.x
    var col = block_idx.x * SBN + tid

    var kslice = (K + SPLITK - 1) // SPLITK
    var k0 = block_idx.y * kslice
    var k1 = min(k0 + kslice, K)

    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[KCHUNK, SM]()
    )
    var sav = sa.vectorize[1, SM]()
    comptime assert sav.flat_rank == 2

    var acc = SIMD[dtype, SM](0)

    var kk = k0
    while kk < k1:
        var clen = min(KCHUNK, k1 - kk)

        comptime for i in range(SM * KCHUNK // SK_THREADS):
            var e = tid + i * SK_THREADS
            var r = e // KCHUNK
            var kc = e % KCHUNK
            var g = kk + kc
            sa[kc, r] = rebind[sa.ElementType](
                rebind[Scalar[DType.bfloat16]](A[r, g]).cast[
                    dtype
                ]() if r < M and kc < clen else 0
            )
        barrier()

        if col < N:
            # One 32 B vector load per block, dequantized in registers.
            var Qv = Wq.vectorize[1, Q8_BLOCK]()
            comptime assert Qv.flat_rank == 2
            for blk in range(clen // Q8_BLOCK):
                var kb = kk + blk * Q8_BLOCK
                var scale = rebind[Scalar[dtype]](Ws[col, kb // Q8_BLOCK])
                var wq = rebind[SIMD[DType.int8, Q8_BLOCK]](
                    Qv[col, kb // Q8_BLOCK]
                )
                var wf = wq.cast[dtype]() * scale
                comptime for j in range(Q8_BLOCK):
                    var a_vec = rebind[SIMD[dtype, SM]](
                        sav[blk * Q8_BLOCK + j, 0]
                    )
                    acc += a_vec * wf[j]
        barrier()
        kk += KCHUNK

    if col < N:
        comptime for i in range(SM):
            Cp[block_idx.y, i, col] = rebind[Cp.ElementType](acc[i])


def skinny_reduce_swiglu[
    PLayout: TensorLayout, CLayout: TensorLayout
](
    Gp: TileTensor[dtype, PLayout, MutAnyOrigin],
    Up: TileTensor[dtype, PLayout, MutAnyOrigin],
    C: TileTensor[dtype, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
):
    """Fused FFN epilogue: C = silu(sum Gp) * (sum Up).

    Replaces two skinny_reduce launches plus a swiglu pass -- the reduce
    already touches every output element, so the activation is free.
    """
    comptime assert Gp.flat_rank == 3 and Up.flat_rank == 3 and C.flat_rank == 2

    var M = Int(m)
    var N = Int(n)
    var gid = global_idx.x
    if gid >= M * N:
        return
    var r = gid // N
    var c = gid % N

    var g: Scalar[dtype] = 0
    var u: Scalar[dtype] = 0
    comptime for s in range(SPLITK):
        g += rebind[Scalar[dtype]](Gp[s, r, c])
        u += rebind[Scalar[dtype]](Up[s, r, c])
    var silu = g / (1 + exp(-g))
    C[r, c] = rebind[C.ElementType](silu * u)


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
