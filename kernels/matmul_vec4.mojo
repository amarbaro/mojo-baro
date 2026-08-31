"""Register-tiled GEMM staging global memory through 128-bit (float4) loads.

Builds on `matmul_ldst` (transposed A tile, 64-bit LDS reads) and changes only
how the tiles are fetched from global memory.

At BM32/BN32/BK8 with 256 threads each tile is exactly 256 floats, so
`matmul_regtile` has every thread issue one scalar 32-bit load per tile -- eight
wavefront-wide load instructions per k-step.  Reading four contiguous floats per
thread instead needs 64 threads per tile, so A and B are fetched by disjoint
thread ranges (0..63 and 64..127) and the whole staging step costs four
wavefront-wide instructions.

The catch is that the transposed A destination cannot absorb a vector store: a
float4 read spans four k values, which the transpose scatters BM + LDS_PAD
floats apart, so it lands as four scalar LDS writes.  B keeps its vector store.
That asymmetry is the price of the 64-bit LDS reads in the inner loop, and it is
the thing this variant exists to price.

Requires BK and BN to be multiples of 4, which holds for the swept config.
"""

from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from matmul import dtype
from matmul_ldst import BM, BN, BK, TM, TN, NTHREADS, LDS_PAD

comptime VEC = 4
comptime A_LOADERS = BM * BK // VEC  # 64
comptime B_LOADERS = BK * BN // VEC  # 64


def matmul_vec4[
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
    comptime assert BK % VEC == 0 and BN % VEC == 0
    comptime assert A_LOADERS + B_LOADERS <= NTHREADS

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)

    var tx = thread_idx.x
    var ty = thread_idx.y
    var tid = ty * (BN // TN) + tx

    var block_row = block_idx.y * BM
    var block_col = block_idx.x * BN

    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[BK, BM + LDS_PAD]()
    )
    var sb = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[BK, BN]()
    )
    var sav = sa.vectorize[1, TM]()
    var sbv = sb.vectorize[1, TN]()
    var sbw = sb.vectorize[1, VEC]()
    comptime assert sav.flat_rank == 2 and sbv.flat_rank == 2 and sbw.flat_rank == 2

    # 128-bit views of the global operands; the last axis is in units of VEC.
    var Av = A.vectorize[1, VEC]()
    var Bv = B.vectorize[1, VEC]()
    comptime assert Av.flat_rank == 2 and Bv.flat_rank == 2

    var acc = SIMD[dtype, TM * TN](0)

    var k_tile = 0
    while k_tile < K:
        if tid < A_LOADERS:
            # Each loader owns VEC consecutive k values of one row of A.
            var r = tid * VEC // BK
            var c = tid * VEC % BK
            var gr = block_row + r
            var gc = k_tile + c
            var v = SIMD[dtype, VEC](0)
            if gr < M and gc + VEC <= K:
                v = rebind[SIMD[dtype, VEC]](Av[gr, gc // VEC])
            # Transposed destination: VEC scalar stores, BM + LDS_PAD apart.
            comptime for j in range(VEC):
                sa[c + j, r] = rebind[sa.ElementType](v[j])
        elif tid < A_LOADERS + B_LOADERS:
            var e = (tid - A_LOADERS) * VEC
            var r = e // BN
            var c = e % BN
            var gr = k_tile + r
            var gc = block_col + c
            var v = SIMD[dtype, VEC](0)
            if gr < K and gc + VEC <= N:
                v = rebind[SIMD[dtype, VEC]](Bv[gr, gc // VEC])
            sbw[r, c // VEC] = rebind[sbw.ElementType](v)
        barrier()

        for k in range(BK):
            var a_reg = rebind[SIMD[dtype, TM]](sav[k, ty])
            var b_reg = rebind[SIMD[dtype, TN]](sbv[k, tx])
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
