
from std.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from std.gpu.primitives import warp
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
comptime ROW_WAVES = 8
comptime ROW_THREADS = ROW_WAVES * WARP_SIZE
comptime ROW_VEC = 8


def amar_matmul_skinny[
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


def amar_matmul_skinny_v2[
    in_dtype: DType, CPT: Int,
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
    var c0 = (block_idx.x * SK_THREADS + tid) * CPT

    var kslice = (K + SPLITK - 1) // SPLITK
    var k0 = block_idx.y * kslice
    var k1 = min(k0 + kslice, K)

    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[KCHUNK, SM]()
    )
    var sav = sa.vectorize[1, SM]()
    var Bv = B.vectorize[1, CPT]()

    var acc = SIMD[dtype, SM * CPT](0)

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

        if c0 < N:
            for k in range(clen):
                var b_vec = rebind[SIMD[in_dtype, CPT]](
                    Bv[kk + k, c0 // CPT]
                ).cast[dtype]()
                var a_vec = rebind[SIMD[dtype, SM]](sav[k, 0])
                comptime for i in range(SM):
                    comptime for j in range(CPT):
                        acc[i * CPT + j] += a_vec[i] * b_vec[j]
        barrier()
        kk += KCHUNK

    if c0 < N:
        comptime for i in range(SM):
            comptime for j in range(CPT):
                Cp[block_idx.y, i, c0 + j] = rebind[Cp.ElementType](
                    acc[i * CPT + j]
                )


def amar_matmul_skinny_m1[
    in_dtype: DType, CPT: Int,
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

    var N = Int(n)
    var K = Int(k_dim)

    var tid = thread_idx.x
    var c0 = (block_idx.x * SK_THREADS + tid) * CPT

    var kslice = (K + SPLITK - 1) // SPLITK
    var k0 = block_idx.y * kslice
    var k1 = min(k0 + kslice, K)

    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[KCHUNK]()
    )
    var Bv = B.vectorize[1, CPT]()

    var acc = SIMD[dtype, CPT](0)

    var kk = k0
    while kk < k1:
        var clen = min(KCHUNK, k1 - kk)

        if tid < KCHUNK:
            sa[tid] = rebind[sa.ElementType](
                rebind[Scalar[in_dtype]](A[0, kk + tid]).cast[
                    dtype
                ]() if tid < clen else 0
            )
        barrier()

        if c0 < N:
            for k in range(clen):
                var b_vec = rebind[SIMD[in_dtype, CPT]](
                    Bv[kk + k, c0 // CPT]
                ).cast[dtype]()
                var a_val = rebind[Scalar[dtype]](sa[k])
                acc += b_vec * a_val
        barrier()
        kk += KCHUNK

    if c0 < N:
        comptime for j in range(CPT):
            Cp[block_idx.y, 0, c0 + j] = rebind[Cp.ElementType](acc[j])


def amar_matmul_skinny_m1_row[
    in_dtype: DType, UNROLL: Int,
    ALayout: TensorLayout, WLayout: TensorLayout, OLayout: TensorLayout
](
    A: TileTensor[in_dtype, ALayout, MutAnyOrigin],
    W: TileTensor[in_dtype, WLayout, MutAnyOrigin],
    O: TileTensor[dtype, OLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and W.flat_rank == 2 and O.flat_rank == 1

    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return

    var Wv = W.vectorize[1, ROW_VEC]()
    var Av = A.vectorize[1, ROW_VEC]()
    comptime STEP = WARP_SIZE * ROW_VEC
    var acc = SIMD[dtype, ROW_VEC](0)

    var kk = 0
    while kk + UNROLL * STEP <= K:
        var ws = InlineArray[SIMD[in_dtype, ROW_VEC], UNROLL](uninitialized=True)

        comptime for u in range(UNROLL):
            ws[u] = rebind[SIMD[in_dtype, ROW_VEC]](Wv[row, (kk + u * STEP) // ROW_VEC + lane])

        comptime for u in range(UNROLL):
            var a = rebind[SIMD[in_dtype, ROW_VEC]](Av[0, (kk + u * STEP) // ROW_VEC + lane]).cast[dtype]()
            acc += ws[u].cast[dtype]() * a
        kk += UNROLL * STEP
    while kk < K:
        var w = rebind[SIMD[in_dtype, ROW_VEC]](Wv[row, kk // ROW_VEC + lane]).cast[dtype]()
        var a = rebind[SIMD[in_dtype, ROW_VEC]](Av[0, kk // ROW_VEC + lane]).cast[dtype]()
        acc += w * a
        kk += STEP

    var total = warp.sum(acc.reduce_add())
    if lane == 0:
        O[row] = rebind[O.ElementType](total)


def amar_matmul_skinny_q8row[
    UNROLL: Int, MR: Int,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.int8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert S.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return

    comptime QV = 16 if MR <= 5 else 8
    comptime STEP = WARP_SIZE * QV
    var Qv = Q.vectorize[1, QV]()
    var Av = A.vectorize[1, QV]()
    var acc = InlineArray[SIMD[dtype, QV], MR](fill=SIMD[dtype, QV](0))

    var kk = 0
    while kk + UNROLL * STEP <= K:
        var qs = InlineArray[SIMD[DType.int8, QV], UNROLL](uninitialized=True)
        var ds = InlineArray[Scalar[DType.float16], UNROLL](uninitialized=True)

        comptime for u in range(UNROLL):
            var kb = kk + u * STEP
            qs[u] = rebind[SIMD[DType.int8, QV]](Qv[row, kb // QV + lane])
            ds[u] = rebind[Scalar[DType.float16]](S[row, (kb + lane * QV) // 32])

        comptime for u in range(UNROLL):
            var kb = kk + u * STEP
            var w = qs[u].cast[dtype]() * ds[u].cast[dtype]()
            comptime for r in range(MR):
                if r < M:
                    var a = rebind[SIMD[DType.bfloat16, QV]](Av[r, kb // QV + lane]).cast[dtype]()
                    acc[r] += w * a
        kk += UNROLL * STEP
    while kk < K:
        var q = rebind[SIMD[DType.int8, QV]](Qv[row, kk // QV + lane]).cast[dtype]()
        var d = rebind[Scalar[DType.float16]](S[row, (kk + lane * QV) // 32]).cast[dtype]()
        var w = q * d
        comptime for r in range(MR):
            if r < M:
                var a = rebind[SIMD[DType.bfloat16, QV]](Av[r, kk // QV + lane]).cast[dtype]()
                acc[r] += w * a
        kk += STEP

    comptime for r in range(MR):
        if r < M:
            var total = warp.sum(acc[r].reduce_add())
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](total)


comptime Q8_BLOCK = 32


def amar_matmul_skinny_q8b[
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


def amar_skinny_reduce_swiglu[
    PLayout: TensorLayout, CLayout: TensorLayout, NSPLIT: Int = SPLITK
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
    comptime for s in range(NSPLIT):
        g += rebind[Scalar[dtype]](Gp[s, r, c])
        u += rebind[Scalar[dtype]](Up[s, r, c])
    var silu = g / (1 + exp(-g))
    C[r, c] = rebind[C.ElementType](silu * u)


def amar_skinny_reduce_swiglu_bf16[
    PLayout: TensorLayout, CLayout: TensorLayout, NSPLIT: Int = SPLITK
](
    Gp: TileTensor[dtype, PLayout, MutAnyOrigin],
    Up: TileTensor[dtype, PLayout, MutAnyOrigin],
    C: TileTensor[DType.bfloat16, CLayout, MutAnyOrigin],
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
    comptime for s in range(NSPLIT):
        g += rebind[Scalar[dtype]](Gp[s, r, c])
        u += rebind[Scalar[dtype]](Up[s, r, c])
    var silu = g / (1 + exp(-g))
    C[r, c] = rebind[C.ElementType]((silu * u).cast[DType.bfloat16]())


def amar_skinny_reduce_add[
    PLayout: TensorLayout, YLayout: TensorLayout, NSPLIT: Int = SPLITK
](
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    Y: TileTensor[dtype, YLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
):
    comptime assert Cp.flat_rank == 3 and Y.flat_rank == 2

    var M = Int(m)
    var N = Int(n)
    var gid = global_idx.x
    if gid >= M * N:
        return
    var r = gid // N
    var c = gid % N

    var acc: Scalar[dtype] = 0
    comptime for s in range(NSPLIT):
        acc += rebind[Scalar[dtype]](Cp[s, r, c])
    Y[r, c] = rebind[Y.ElementType](rebind[Scalar[dtype]](Y[r, c]) + acc)


def amar_skinny_reduce[
    PLayout: TensorLayout, CLayout: TensorLayout, NSPLIT: Int = SPLITK
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
    comptime for s in range(NSPLIT):
        acc += rebind[Scalar[dtype]](Cp[s, r, c])
    C[r, c] = rebind[C.ElementType](acc)
