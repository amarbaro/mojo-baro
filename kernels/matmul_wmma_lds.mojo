"""fp16 WMMA GEMM with LDS staging, gfx1100 (wave32).

The direct-from-global version re-reads A and B per warp with a strided access
per lane: no reuse, poor coalescing, and throughput that swings non-monotonically
with size (14.4 / 7.1 / 16.4 TFLOPS at 1024/2048/4096). Here each block stages
one A and one B tile into LDS cooperatively, then every warp builds its
fragments from LDS.

Fragment shape and lane mapping are as documented in matmul_wmma.mojo:
RDNA3 wave32 wants a/b 16 wide, c/d 8 wide.

Two things were tried here and are NOT worth retrying:

  Vectorizing the LDS fragment reads. A fragments are already contiguous and the
  compiler handles them; forcing 8-wide reads changed nothing (9208 vs 9285 at
  512^3, 28763 vs 28758 at 2048^3). B fragments are strided, and staging B
  transposed to make them contiguous cost ~3% everywhere -- the strided LDS
  store is dearer than the wide read is worth. Same result the fp32 `ldst`
  variant produced.

  BLK_K = 64, so one LDS stage feeds four matrix-core steps. Worse everywhere
  (2048^3: 28758 -> 8602 scale). 16 KB of LDS per block destroys occupancy.
"""

from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from layout.tensor_core import mma

comptime WMMA_M = 16
comptime WMMA_N = 16
comptime WMMA_K = 16

comptime WARPS_M = 4
comptime WARPS_N = 4
comptime WTILE_M = 2  # 16x16 tiles per warp, M direction
comptime WTILE_N = 1

comptime BLK_M = WARPS_M * WTILE_M * WMMA_M  # 64
comptime BLK_N = WARPS_N * WTILE_N * WMMA_N  # 64
# BLK_K=64 was tried and is worse everywhere (2048^3: 21901 -> 8602). It puts
# 16 KB of LDS in a block and craters occupancy; at 16 the footprint is 4 KB.
# Occupancy beats reuse on this card -- the same result the fp32 sweep and the
# aiter tuning both produced.
comptime BLK_K = WMMA_K
comptime NTHREADS = WARPS_M * WARPS_N * 32   # 128


def matmul_wmma_lds[
    ALayout: TensorLayout, BLayout: TensorLayout, CLayout: TensorLayout
](
    A: TileTensor[DType.float16, ALayout, MutAnyOrigin],
    B: TileTensor[DType.float16, BLayout, MutAnyOrigin],
    C: TileTensor[DType.float32, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2

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
        row_major[BLK_M, BLK_K]()
    )
    var sb = stack_allocation[DType.float16, address_space=AddressSpace.SHARED](
        row_major[BLK_K, BLK_N]()
    )

    var acc = SIMD[DType.float32, WTILE_M * WTILE_N * 8](0)

    var kt = 0
    while kt < K:
        # 128 threads stage 64x16 and 16x64 elements: 8 each, contiguous in the
        # fast axis so the global reads coalesce.
        comptime for s in range(BLK_M * BLK_K // NTHREADS):
            var e = tid + s * NTHREADS
            var r = e // BLK_K
            var c = e % BLK_K
            var gr = block_row + r
            var gc = kt + c
            sa[r, c] = rebind[sa.ElementType](
                A[gr, gc] if gr < M and gc < K else Scalar[DType.float16](0)
            )
        comptime for s in range(BLK_K * BLK_N // NTHREADS):
            var e = tid + s * NTHREADS
            var r = e // BLK_N
            var c = e % BLK_N
            var gr = kt + r
            var gc = block_col + c
            sb[r, c] = rebind[sb.ElementType](
                B[gr, gc] if gr < K and gc < N else Scalar[DType.float16](0)
            )
        barrier()

        # One LDS stage now feeds BLK_K/WMMA_K matrix-core steps, so the two
        # barriers are amortised instead of paid every 16 elements of K.
        comptime for ks in range(BLK_K // WMMA_K):
            comptime for tm in range(WTILE_M):
                var a = SIMD[DType.float16, 16](0)
                var arow = (warp_m * WTILE_M + tm) * WMMA_M + h
                comptime for i in range(16):
                    a[i] = rebind[Scalar[DType.float16]](sa[arow, ks * WMMA_K + i])

                comptime for tn in range(WTILE_N):
                    var b = SIMD[DType.float16, 16](0)
                    var bcol = (warp_n * WTILE_N + tn) * WMMA_N + h
                    comptime for i in range(16):
                        b[i] = rebind[Scalar[DType.float16]](
                            sb[ks * WMMA_K + i, bcol]
                        )

                    var c = SIMD[DType.float32, 8](0)
                    comptime for i in range(8):
                        c[i] = acc[(tm * WTILE_N + tn) * 8 + i]
                    var d = SIMD[DType.float32, 8](0)
                    mma(d, a, b, c)
                    comptime for i in range(8):
                        acc[(tm * WTILE_N + tn) * 8 + i] = d[i]
        barrier()
        kt += BLK_K

    comptime for tm in range(WTILE_M):
        comptime for tn in range(WTILE_N):
            comptime for i in range(8):
                var row = block_row + (warp_m * WTILE_M + tm) * WMMA_M + 2 * i + half
                var col = block_col + (warp_n * WTILE_N + tn) * WMMA_N + h
                if row < M and col < N:
                    C[row, col] = rebind[C.ElementType](
                        acc[(tm * WTILE_N + tn) * 8 + i]
                    )
