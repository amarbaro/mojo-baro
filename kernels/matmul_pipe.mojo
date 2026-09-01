"""Double buffering and 128-bit global staging combined.

`amar_matmul_dbuf` and `amar_matmul_vec4` each land ~+16% over `amar_matmul_regtile` and they
attack different things -- one hides global-load latency behind compute, the
other cuts the number of load instructions issued.  Neither subsumes the other,
so this variant runs both: float4 staging registers filled by disjoint thread
ranges, held across the compute of the current tile, then committed to the
other LDS buffer.

Transposed A (from `amar_matmul_ldst`) is kept throughout, so the inner loop reads
both operands 64 bits at a time.
"""

from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from matmul import dtype
from matmul_ldst import BM, BN, BK, TM, TN, NTHREADS, LDS_PAD
from matmul_vec4 import VEC, A_LOADERS, B_LOADERS


def amar_matmul_pipe[
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
        row_major[2 * BK, BM + LDS_PAD]()
    )
    var sb = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[2 * BK, BN]()
    )
    var sav = sa.vectorize[1, TM]()
    var sbv = sb.vectorize[1, TN]()
    var sbw = sb.vectorize[1, VEC]()
    comptime assert sav.flat_rank == 2 and sbv.flat_rank == 2 and sbw.flat_rank == 2

    var Av = A.vectorize[1, VEC]()
    var Bv = B.vectorize[1, VEC]()
    comptime assert Av.flat_rank == 2 and Bv.flat_rank == 2

    # Which slice of a tile this thread stages, fixed for the whole kernel.
    var ar = tid * VEC // BK
    var ac = tid * VEC % BK
    var be = (tid - A_LOADERS) * VEC
    var br = be // BN
    var bc = be % BN

    var acc = SIMD[dtype, TM * TN](0)
    var stage = SIMD[dtype, VEC](0)

    # --- prologue: stage tile 0 and commit it to buffer 0 -------------------
    if tid < A_LOADERS:
        if block_row + ar < M and ac + VEC <= K:
            stage = rebind[SIMD[dtype, VEC]](Av[block_row + ar, ac // VEC])
        comptime for j in range(VEC):
            sa[ac + j, ar] = rebind[sa.ElementType](stage[j])
    elif tid < A_LOADERS + B_LOADERS:
        if br < K and block_col + bc + VEC <= N:
            stage = rebind[SIMD[dtype, VEC]](Bv[br, (block_col + bc) // VEC])
        sbw[br, bc // VEC] = rebind[sbw.ElementType](stage)

    var cur = 0
    var k_tile = 0
    while k_tile < K:
        barrier()

        var next_k = k_tile + BK
        var have_next = next_k < K
        if have_next:
            stage = SIMD[dtype, VEC](0)
            if tid < A_LOADERS:
                if block_row + ar < M and next_k + ac + VEC <= K:
                    stage = rebind[SIMD[dtype, VEC]](
                        Av[block_row + ar, (next_k + ac) // VEC]
                    )
            elif tid < A_LOADERS + B_LOADERS:
                if next_k + br < K and block_col + bc + VEC <= N:
                    stage = rebind[SIMD[dtype, VEC]](
                        Bv[next_k + br, (block_col + bc) // VEC]
                    )

        var base = cur * BK
        for k in range(BK):
            var a_reg = rebind[SIMD[dtype, TM]](sav[base + k, ty])
            var b_reg = rebind[SIMD[dtype, TN]](sbv[base + k, tx])
            comptime for i in range(TM):
                comptime for j in range(TN):
                    acc[i * TN + j] += a_reg[i] * b_reg[j]

        if have_next:
            var nbase = (1 - cur) * BK
            if tid < A_LOADERS:
                comptime for j in range(VEC):
                    sa[nbase + ac + j, ar] = rebind[sa.ElementType](stage[j])
            elif tid < A_LOADERS + B_LOADERS:
                sbw[nbase + br, bc // VEC] = rebind[sbw.ElementType](stage)

        cur = 1 - cur
        k_tile = next_k

    comptime for i in range(TM):
        comptime for j in range(TN):
            var r = block_row + ty * TM + i
            var c = block_col + tx * TN + j
            if r < M and c < N:
                C[r, c] = rebind[C.ElementType](acc[i * TN + j])
