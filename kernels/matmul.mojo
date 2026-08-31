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
