from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from layout.tensor_core import mma
from std.utils import StaticTuple

comptime WMMA_M = 16
comptime WMMA_N = 16
comptime WMMA_K = 16

comptime WARPS_M = 2
comptime WARPS_N = 2
comptime WTILE_M = 4
comptime WTILE_N = 4
comptime BLK_K = 32
comptime PAD_A = 0
comptime PAD_B = 0
comptime TRANS_B = 2
comptime ALIGNED = 1
comptime C_F16 = 1
comptime C_DTYPE = DType.float16 if C_F16 == 1 else DType.float32

comptime BLK_M = WARPS_M * WTILE_M * WMMA_M
comptime BLK_N = WARPS_N * WTILE_N * WMMA_N
comptime NTHREADS = WARPS_M * WARPS_N * 32
comptime VEC = 8
comptime KCH = BLK_K // VEC
comptime A_VECS = BLK_M * BLK_K // VEC
comptime B_VECS = BLK_K * BLK_N // VEC
comptime A_PER = A_VECS // NTHREADS
comptime B_PER = B_VECS // NTHREADS
comptime SA_W = BLK_K + PAD_A
comptime SWZ_A = 1 if PAD_A == 0 else 0
comptime SB_W = BLK_N if TRANS_B == 0 else (BLK_K + PAD_B if TRANS_B == 1 else BLK_K)
comptime SB_H = BLK_K if TRANS_B == 0 else BLK_N
comptime KSTEPS = BLK_K // WMMA_K


@__llvm_metadata(`rocdl.flat_work_group_size`=StaticTuple[Int32, 1](NTHREADS))
def amar_matmul_wmma_pipe[
    ALayout: TensorLayout, BLayout: TensorLayout, CLayout: TensorLayout
](
    A: TileTensor[DType.float16, ALayout, MutAnyOrigin],
    B: TileTensor[DType.float16, BLayout, MutAnyOrigin],
    C: TileTensor[C_DTYPE, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    comptime assert A_VECS % NTHREADS == 0 and B_VECS % NTHREADS == 0
    comptime assert SA_W % VEC == 0 and SB_W % VEC == 0

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)

    var tid = thread_idx.x
    var lane = tid % 32
    var warp = tid // 32
    var warp_m = warp // WARPS_N
    var warp_n = warp % WARPS_N
    var h = lane % 16
    var half = lane // 16

    var block_row = block_idx.y * BLK_M
    var block_col = block_idx.x * BLK_N

    var sa = stack_allocation[DType.float16, address_space=AddressSpace.SHARED](
        row_major[2 * BLK_M, SA_W]()
    )
    var sb = stack_allocation[DType.float16, address_space=AddressSpace.SHARED](
        row_major[2 * SB_H, SB_W]()
    )
    var sav = sa.vectorize[1, VEC]()
    var sbv = sb.vectorize[1, VEC]()
    var Av = A.vectorize[1, VEC]()
    var Bv = B.vectorize[1, VEC]()
    comptime assert sav.flat_rank == 2 and sbv.flat_rank == 2
    comptime assert Av.flat_rank == 2 and Bv.flat_rank == 2

    var acc = SIMD[DType.float32, WTILE_M * WTILE_N * 8](0)
    var sta = SIMD[DType.float16, A_PER * VEC](0)
    var stb = SIMD[DType.float16, B_PER * VEC](0)

    # prologue: tile 0 -> registers -> buffer 0
    comptime for s in range(A_PER):
        var v = tid + s * NTHREADS
        var r = v // KCH
        var c = (v % KCH) * VEC
        var gr = block_row + r
        var x = SIMD[DType.float16, VEC](0)
        if ALIGNED == 1 or (gr < M and c + VEC <= K):
            x = rebind[SIMD[DType.float16, VEC]](Av[gr, c // VEC])
        elif gr < M:
            comptime for i in range(VEC):
                if c + i < K:
                    x[i] = rebind[Scalar[DType.float16]](A[gr, c + i])
        comptime for i in range(VEC):
            sta[s * VEC + i] = x[i]
    comptime for s in range(B_PER):
        var v = tid + s * NTHREADS
        var r = (v % BLK_K) if TRANS_B == 2 else (v // (BLK_N // VEC))
        var c = ((v // BLK_K) * VEC) if TRANS_B == 2 else ((v % (BLK_N // VEC)) * VEC)
        var gc = block_col + c
        var x = SIMD[DType.float16, VEC](0)
        if ALIGNED == 1 or (r < K and gc + VEC <= N):
            x = rebind[SIMD[DType.float16, VEC]](Bv[r, gc // VEC])
        elif r < K:
            comptime for i in range(VEC):
                if gc + i < N:
                    x[i] = rebind[Scalar[DType.float16]](B[r, gc + i])
        comptime for i in range(VEC):
            stb[s * VEC + i] = x[i]
    comptime for s in range(A_PER):
        var v = tid + s * NTHREADS
        var r = v // KCH
        var c = (v % KCH) * VEC
        var x = SIMD[DType.float16, VEC](0)
        comptime for i in range(VEC):
            x[i] = sta[s * VEC + i]
        var ca = c // VEC
        comptime if SWZ_A == 1:
            ca = ca ^ ((r >> 1) & (KCH - 1))
        sav[r, ca] = rebind[sav.ElementType](x)
    comptime for s in range(B_PER):
        var v = tid + s * NTHREADS
        comptime if TRANS_B == 0:
            var r = v // (BLK_N // VEC)
            var c = (v % (BLK_N // VEC)) * VEC
            var x = SIMD[DType.float16, VEC](0)
            comptime for i in range(VEC):
                x[i] = stb[s * VEC + i]
            sbv[r, c // VEC] = rebind[sbv.ElementType](x)
        elif TRANS_B == 1:
            var r = v // (BLK_N // VEC)
            var c = (v % (BLK_N // VEC)) * VEC
            comptime for i in range(VEC):
                sb[c + i, r] = rebind[sb.ElementType](stb[s * VEC + i])
        else:
            var r = v % BLK_K
            var c = (v // BLK_K) * VEC
            comptime for i in range(VEC):
                var nn = c + i
                var g = ((nn >> 1) ^ (nn >> 3)) & (KCH - 1)
                sb[nn, ((r // VEC) ^ g) * VEC + r % VEC] = rebind[sb.ElementType](
                    stb[s * VEC + i]
                )

    var cur = 0
    var kt = 0
    while kt < K:
        barrier()
        var next_k = kt + BLK_K
        var have_next = next_k < K
        if have_next:
            comptime for s in range(A_PER):
                var v = tid + s * NTHREADS
                var r = v // KCH
                var c = (v % KCH) * VEC
                var gr = block_row + r
                var gc = next_k + c
                var x = SIMD[DType.float16, VEC](0)
                if ALIGNED == 1 or (gr < M and gc + VEC <= K):
                    x = rebind[SIMD[DType.float16, VEC]](Av[gr, gc // VEC])
                elif gr < M:
                    comptime for i in range(VEC):
                        if gc + i < K:
                            x[i] = rebind[Scalar[DType.float16]](A[gr, gc + i])
                comptime for i in range(VEC):
                    sta[s * VEC + i] = x[i]
            comptime for s in range(B_PER):
                var v = tid + s * NTHREADS
                var r = (v % BLK_K) if TRANS_B == 2 else (v // (BLK_N // VEC))
                var c = ((v // BLK_K) * VEC) if TRANS_B == 2 else ((v % (BLK_N // VEC)) * VEC)
                var gr = next_k + r
                var gc = block_col + c
                var x = SIMD[DType.float16, VEC](0)
                if ALIGNED == 1 or (gr < K and gc + VEC <= N):
                    x = rebind[SIMD[DType.float16, VEC]](Bv[gr, gc // VEC])
                elif gr < K:
                    comptime for i in range(VEC):
                        if gc + i < N:
                            x[i] = rebind[Scalar[DType.float16]](B[gr, gc + i])
                comptime for i in range(VEC):
                    stb[s * VEC + i] = x[i]

        comptime for ks in range(KSTEPS):
            var bfr = SIMD[DType.float16, WTILE_N * 16](0)
            comptime for tn in range(WTILE_N):
                var bn = (warp_n * WTILE_N + tn) * WMMA_N + h
                comptime if TRANS_B == 0:
                    comptime for i in range(16):
                        bfr[tn * 16 + i] = rebind[Scalar[DType.float16]](
                            sb[cur * SB_H + ks * WMMA_K + i, bn]
                        )
                else:
                    var brow = cur * SB_H + bn
                    var c0 = ks * 2
                    var c1 = ks * 2 + 1
                    comptime if TRANS_B == 2:
                        var g = ((bn >> 1) ^ (bn >> 3)) & (KCH - 1)
                        c0 = c0 ^ g
                        c1 = c1 ^ g
                    var lo = rebind[SIMD[DType.float16, VEC]](sbv[brow, c0])
                    var hi = rebind[SIMD[DType.float16, VEC]](sbv[brow, c1])
                    comptime for i in range(VEC):
                        bfr[tn * 16 + i] = lo[i]
                        bfr[tn * 16 + VEC + i] = hi[i]
            comptime for tm in range(WTILE_M):
                var arow = cur * BLK_M + (warp_m * WTILE_M + tm) * WMMA_M + h
                var a0 = ks * 2
                var a1 = ks * 2 + 1
                comptime if SWZ_A == 1:
                    var ga = (h >> 1) & (KCH - 1)
                    a0 = a0 ^ ga
                    a1 = a1 ^ ga
                var lo = rebind[SIMD[DType.float16, VEC]](sav[arow, a0])
                var hi = rebind[SIMD[DType.float16, VEC]](sav[arow, a1])
                var a = lo.join(hi)
                comptime for tn in range(WTILE_N):
                    var b = SIMD[DType.float16, 16](0)
                    comptime for i in range(16):
                        b[i] = bfr[tn * 16 + i]
                    var c = SIMD[DType.float32, 8](0)
                    comptime for i in range(8):
                        c[i] = acc[(tm * WTILE_N + tn) * 8 + i]
                    var d = SIMD[DType.float32, 8](0)
                    mma(d, a, b, c)
                    comptime for i in range(8):
                        acc[(tm * WTILE_N + tn) * 8 + i] = d[i]

        if have_next:
            var nb = 1 - cur
            comptime for s in range(A_PER):
                var v = tid + s * NTHREADS
                var r = v // KCH
                var c = (v % KCH) * VEC
                var x = SIMD[DType.float16, VEC](0)
                comptime for i in range(VEC):
                    x[i] = sta[s * VEC + i]
                var ca = c // VEC
                comptime if SWZ_A == 1:
                    ca = ca ^ ((r >> 1) & (KCH - 1))
                sav[nb * BLK_M + r, ca] = rebind[sav.ElementType](x)
            comptime for s in range(B_PER):
                var v = tid + s * NTHREADS
                comptime if TRANS_B == 0:
                    var r = v // (BLK_N // VEC)
                    var c = (v % (BLK_N // VEC)) * VEC
                    var x = SIMD[DType.float16, VEC](0)
                    comptime for i in range(VEC):
                        x[i] = stb[s * VEC + i]
                    sbv[nb * SB_H + r, c // VEC] = rebind[sbv.ElementType](x)
                elif TRANS_B == 1:
                    var r = v // (BLK_N // VEC)
                    var c = (v % (BLK_N // VEC)) * VEC
                    comptime for i in range(VEC):
                        sb[nb * SB_H + c + i, r] = rebind[sb.ElementType](stb[s * VEC + i])
                else:
                    var r = v % BLK_K
                    var c = (v // BLK_K) * VEC
                    comptime for i in range(VEC):
                        var nn = c + i
                        var g = ((nn >> 1) ^ (nn >> 3)) & (KCH - 1)
                        sb[nb * SB_H + nn, ((r // VEC) ^ g) * VEC + r % VEC] = rebind[
                            sb.ElementType
                        ](stb[s * VEC + i])
        cur = 1 - cur
        kt = next_k

    comptime for tm in range(WTILE_M):
        comptime for tn in range(WTILE_N):
            comptime for i in range(8):
                var row = block_row + (warp_m * WTILE_M + tm) * WMMA_M + 2 * i + half
                var col = block_col + (warp_n * WTILE_N + tn) * WMMA_N + h
                if ALIGNED == 1 or (row < M and col < N):
                    C[row, col] = rebind[C.ElementType](
                        acc[(tm * WTILE_N + tn) * 8 + i].cast[C_DTYPE]()
                    )
