"""Register-tiled GEMM with a transposed A tile in shared memory.

Same blocking as `matmul_regtile` (BM32 BN32 BK8 TM2 TN2) so the only variable
is the shared-memory layout of A.

In `matmul_regtile`, A's tile is `sa[BM][BK]` and the inner loop reads
`sa[ty * TM + i, k]` for i in 0..TM: those TM values are BK apart, so each one
costs its own scalar LDS read.  B's tile is already read contiguously.  With
TM = TN = 2 that is 4 LDS reads feeding 4 FMAs -- a 1:1 ratio against a VALU
that wants to be fed faster than the LDS pipe can issue.

Storing A transposed as `sa[BK][BM]` makes the TM values adjacent, so the pair
is one 64-bit LDS read.  Per k-step the inner loop drops from 4 LDS reads to 2.

The `+ LDS_PAD` on the row stride is load-bearing twice over:
  * without it the transposed store `sa[c, r]` has stride BM = 32 floats, so all
    BK threads sharing a row `r` hit the same bank -- an 8-way write conflict.
    A stride of BM + 2 spreads them across 8 distinct banks.
  * the pad must stay even so `2 * ty` keeps 8-byte alignment and the SIMD[2]
    read stays a single `ds_read_b64`.  An odd pad (the usual +1 trick) removes
    the conflict but breaks the alignment and costs more than it saves.
"""

from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from matmul import dtype

comptime BM = 32
comptime BN = 32
comptime BK = 8
comptime TM = 2
comptime TN = 2
comptime NTHREADS = (BM // TM) * (BN // TN)  # 256
comptime LDS_PAD = 2


def matmul_ldst[
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

    # A is held transposed: [BK][BM + LDS_PAD] instead of [BM][BK].
    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[BK, BM + LDS_PAD]()
    )
    var sb = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[BK, BN]()
    )
    # SIMD views over the same storage: one element is the TM (or TN) values a
    # thread consumes together, so the read lowers to a single 64-bit LDS op.
    var sav = sa.vectorize[1, TM]()
    var sbv = sb.vectorize[1, TN]()
    comptime assert sav.flat_rank == 2 and sbv.flat_rank == 2

    var acc = SIMD[dtype, TM * TN](0)

    var k_tile = 0
    while k_tile < K:
        # Staging indices are unchanged from regtile so the global read stays
        # coalesced (BK consecutive threads read BK consecutive floats of one
        # row); only the destination is transposed.
        comptime for i in range(BM * BK // NTHREADS):
            var e = tid + i * NTHREADS
            var r = e // BK
            var c = e % BK
            var gr = block_row + r
            var gc = k_tile + c
            if gr < M and gc < K:
                sa[c, r] = rebind[sa.ElementType](A[gr, gc])
            else:
                sa[c, r] = 0

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
