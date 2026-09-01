
from std.gpu import block_idx, global_idx, thread_idx
from std.math import exp
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from matmul import dtype

comptime SM = 8
comptime SBN = 256
comptime SPLITK = 32
comptime SK_THREADS = SBN
comptime KCHUNK = 128


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
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and Cp.flat_rank == 3

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


comptime Q8_BLOCK = 32


def matmul_skinny_q8b[
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Bq: TileTensor[DType.int8, QLayout, MutAnyOrigin],
    Bs: TileTensor[dtype, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Bq.flat_rank == 2
    comptime assert Bs.flat_rank == 2 and Cp.flat_rank == 3

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
            for blk in range(clen // Q8_BLOCK):
                var kb = kk + blk * Q8_BLOCK
                var scale = rebind[Scalar[dtype]](Bs[kb // Q8_BLOCK, col])
                comptime for j in range(Q8_BLOCK):
                    var q = rebind[Scalar[DType.int8]](Bq[kb + j, col])
                    var a_vec = rebind[SIMD[dtype, SM]](
                        sav[blk * Q8_BLOCK + j, 0]
                    )
                    acc += a_vec * (q.cast[dtype]() * scale)
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
