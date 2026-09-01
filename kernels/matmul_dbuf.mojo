"""Double-buffered register-tiled GEMM: overlap the next tile's global load
with the current tile's compute.

Builds on `amar_matmul_ldst` (transposed A tile, 64-bit LDS reads) and changes only
the pipeline.  `amar_matmul_regtile` runs strictly serially per k-tile:

    global load -> barrier -> compute -> barrier -> global load -> ...

so every thread stalls on global-memory latency once per BK steps with nothing
to hide it.  The swept result that small tiles beat large ones -- occupancy
buying more than reuse -- is what a latency-bound kernel looks like, which is
the case double buffering exists for.

With two LDS buffers the loop becomes:

    barrier -> issue next tile's global loads into registers
            -> compute from buf[cur]
            -> store those registers into buf[1 - cur]

The global loads are issued before the compute that hides them, and the store
into buf[1 - cur] cannot race: buf[1 - cur] was last *read* in the previous
iteration, and the barrier at the top of this iteration already separates the
two.  That also halves the barrier count -- one per k-tile instead of two.
"""

from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from matmul import dtype
from matmul_ldst import BM, BN, BK, TM, TN, NTHREADS, LDS_PAD

comptime A_PER_THREAD = BM * BK // NTHREADS
comptime B_PER_THREAD = BK * BN // NTHREADS


def amar_matmul_dbuf[
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

    # Two buffers each; the leading BK dimension is doubled rather than adding a
    # rank so the SIMD views below stay 2-D.
    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[2 * BK, BM + LDS_PAD]()
    )
    var sb = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[2 * BK, BN]()
    )
    var sav = sa.vectorize[1, TM]()
    var sbv = sb.vectorize[1, TN]()
    comptime assert sav.flat_rank == 2 and sbv.flat_rank == 2

    var acc = SIMD[dtype, TM * TN](0)
    var a_stage = SIMD[dtype, A_PER_THREAD](0)
    var b_stage = SIMD[dtype, B_PER_THREAD](0)

    # --- prologue: stage tile 0 and commit it to buffer 0 -------------------
    comptime for i in range(A_PER_THREAD):
        var e = tid + i * NTHREADS
        var gr = block_row + e // BK
        var gc = e % BK
        a_stage[i] = (
            rebind[Scalar[dtype]](A[gr, gc]) if gr < M and gc < K else 0
        )
    comptime for i in range(B_PER_THREAD):
        var e = tid + i * NTHREADS
        var gr = e // BN
        var gc = block_col + e % BN
        b_stage[i] = (
            rebind[Scalar[dtype]](B[gr, gc]) if gr < K and gc < N else 0
        )
    comptime for i in range(A_PER_THREAD):
        var e = tid + i * NTHREADS
        sa[e % BK, e // BK] = rebind[sa.ElementType](a_stage[i])
    comptime for i in range(B_PER_THREAD):
        var e = tid + i * NTHREADS
        sb[e // BN, e % BN] = rebind[sb.ElementType](b_stage[i])

    var cur = 0
    var k_tile = 0
    while k_tile < K:
        barrier()

        # Issue the next tile's global reads first so their latency is covered
        # by the compute below, not by a stall in front of it.
        var next_k = k_tile + BK
        var have_next = next_k < K
        if have_next:
            comptime for i in range(A_PER_THREAD):
                var e = tid + i * NTHREADS
                var gr = block_row + e // BK
                var gc = next_k + e % BK
                a_stage[i] = (
                    rebind[Scalar[dtype]](A[gr, gc]) if gr < M and gc < K else 0
                )
            comptime for i in range(B_PER_THREAD):
                var e = tid + i * NTHREADS
                var gr = next_k + e // BN
                var gc = block_col + e % BN
                b_stage[i] = (
                    rebind[Scalar[dtype]](B[gr, gc]) if gr < K and gc < N else 0
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
            comptime for i in range(A_PER_THREAD):
                var e = tid + i * NTHREADS
                sa[nbase + e % BK, e // BK] = rebind[sa.ElementType](a_stage[i])
            comptime for i in range(B_PER_THREAD):
                var e = tid + i * NTHREADS
                sb[nbase + e // BN, e % BN] = rebind[sb.ElementType](b_stage[i])

        cur = 1 - cur
        k_tile = next_k

    comptime for i in range(TM):
        comptime for j in range(TN):
            var r = block_row + ty * TM + i
            var c = block_col + tx * TN + j
            if r < M and c < N:
                C[r, c] = rebind[C.ElementType](acc[i * TN + j])
