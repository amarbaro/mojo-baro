"""Mojo GEMM kernels for gfx1100 (RDNA3, warp=32).

Variants are registered here and benchmarked by bench/bench.mojo against the
hipBLASLt baseline reached through the C-ABI shim.
"""

from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime dtype = DType.float32
comptime TILE = 16


def matmul_naive[
    ALayout: TensorLayout, BLayout: TensorLayout, CLayout: TensorLayout
](
    A: TileTensor[dtype, ALayout, MutAnyOrigin],
    B: TileTensor[dtype, BLayout, MutAnyOrigin],
    C: TileTensor[dtype, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    """One thread per output element; every read goes to global memory."""
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)

    var row = block_idx.y * TILE + thread_idx.y
    var col = block_idx.x * TILE + thread_idx.x
    if row >= M or col >= N:
        return

    var acc: Scalar[dtype] = 0
    for k in range(K):
        acc += rebind[Scalar[dtype]](A[row, k]) * rebind[Scalar[dtype]](B[k, col])
    C[row, col] = rebind[C.ElementType](acc)


def matmul_tiled[
    ALayout: TensorLayout, BLayout: TensorLayout, CLayout: TensorLayout
](
    A: TileTensor[dtype, ALayout, MutAnyOrigin],
    B: TileTensor[dtype, BLayout, MutAnyOrigin],
    C: TileTensor[dtype, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    """Shared-memory tiling: each A/B tile is read once per block, not per thread."""
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)

    var tx = thread_idx.x
    var ty = thread_idx.y
    var row = block_idx.y * TILE + ty
    var col = block_idx.x * TILE + tx

    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[TILE, TILE]()
    )
    var sb = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[TILE, TILE]()
    )

    var acc: Scalar[dtype] = 0
    var k_tile = 0
    while k_tile < K:
        # Cooperative load; out-of-range lanes pad with zero so the inner
        # product stays correct for non-multiple-of-TILE shapes.
        if row < M and k_tile + tx < K:
            sa[ty, tx] = rebind[sa.ElementType](A[row, k_tile + tx])
        else:
            sa[ty, tx] = 0
        if col < N and k_tile + ty < K:
            sb[ty, tx] = rebind[sb.ElementType](B[k_tile + ty, col])
        else:
            sb[ty, tx] = 0
        barrier()

        for k in range(TILE):
            acc += rebind[Scalar[dtype]](sa[ty, k]) * rebind[Scalar[dtype]](sb[k, tx])
        barrier()
        k_tile += TILE

    if row < M and col < N:
        C[row, col] = rebind[C.ElementType](acc)


# --- Register-tiled variant -------------------------------------------------
# Each thread computes a TM x TN patch of C instead of a single element, so the
# values loaded from shared memory are reused TM/TN times in registers. This is
# where the arithmetic intensity comes from: the tiled kernel above is still
# bound by shared-memory traffic, one load per multiply-add.

comptime BM = 64
comptime BN = 64
comptime BK = 16
comptime TM = 4
comptime TN = 4
comptime NTHREADS = (BM // TM) * (BN // TN)  # 256


def matmul_regtile[
    ALayout: TensorLayout, BLayout: TensorLayout, CLayout: TensorLayout
](
    A: TileTensor[dtype, ALayout, MutAnyOrigin],
    B: TileTensor[dtype, BLayout, MutAnyOrigin],
    C: TileTensor[dtype, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)

    var tx = thread_idx.x
    var ty = thread_idx.y
    var tid = ty * (BN // TN) + tx

    var block_row = block_idx.y * BM
    var block_col = block_idx.x * BN

    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[BM, BK]()
    )
    var sb = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[BK, BN]()
    )
    # SIMD values, not stack_allocation: a stack allocation here is scratch
    # memory, which on AMD lives in device memory and made this kernel slower
    # than the naive one. Comptime-unrolled indices keep these in registers.
    var acc = SIMD[dtype, TM * TN](0)

    var k_tile = 0
    while k_tile < K:
        # 256 threads cooperatively stage BM*BK and BK*BN elements: 4 each.
        comptime for i in range(BM * BK // NTHREADS):
            var e = tid + i * NTHREADS
            var r = e // BK
            var c = e % BK
            var gr = block_row + r
            var gc = k_tile + c
            if gr < M and gc < K:
                sa[r, c] = rebind[sa.ElementType](A[gr, gc])
            else:
                sa[r, c] = 0

        comptime for i in range(BK * BN // NTHREADS):
            var e = tid + i * NTHREADS
            var r = e // BN
            var c = e % BN
            var gr = k_tile + r
            var gc = block_col + c
            if gr < K and gc < N:
                sb[r, c] = rebind[sb.ElementType](B[gr, gc])
            else:
                sb[r, c] = 0
        barrier()

        # Outer product: one shared read feeds TN (or TM) fused multiply-adds.
        for k in range(BK):
            var a_reg = SIMD[dtype, TM](0)
            comptime for i in range(TM):
                a_reg[i] = rebind[Scalar[dtype]](sa[ty * TM + i, k])
            var b_reg = SIMD[dtype, TN](0)
            comptime for j in range(TN):
                b_reg[j] = rebind[Scalar[dtype]](sb[k, tx * TN + j])
            comptime for i in range(TM):
                comptime for j in range(TN):
                    acc[i * TN + j] += a_reg[i] * b_reg[j]
        barrier()
        k_tile += BK

    comptime for i in range(TM):
        comptime for j in range(TN):
            var r = block_row + ty * TM + i
            var c = block_col + tx * TN + j
            if r < M and c < N:
                C[r, c] = rebind[C.ElementType](acc[i * TN + j])
