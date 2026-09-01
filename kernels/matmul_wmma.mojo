"""fp16 GEMM on RDNA3 matrix cores (gfx1100, wave32).

fp32 WMMA does not exist on this ISA -- `v_wmma_f32_16x16x16_f32` is rejected by
the assembler and there is no f32-input clang builtin. fp16 in / fp32 accumulate
is the only matrix-core path, so this kernel is a separate dtype from the fp32
variants and is NOT comparable to them.

Fragment shape is the trap: RDNA3 wave32 wants a/b 16 wide and c/d 8 wide,
matching __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 (V8fV16hV16hV8f).
`TensorCore.load_a` builds 8-wide CDNA/NVIDIA fragments and fails to select an
implementation, so fragments are built by hand and passed to the low-level `mma`.

Lane mapping, measured on this card:
    A: lane L, elem i (0..15) -> A[L % 16][i]
    B: lane L, elem i (0..15) -> B[i][L % 16]
    D: lane L, elem i (0..7)  -> D[2*i + L // 16][L % 16]
Lanes 0-15 and 16-31 hold identical A/B fragments and differ only in the rows of
D they own.
"""

from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout
from layout.tensor_core import mma

comptime WMMA_M = 16
comptime WMMA_N = 16
comptime WMMA_K = 16

# Each block is one wave computing a WARP_TILES_M x WARP_TILES_N grid of 16x16
# output tiles, so one A/B fragment load feeds several accumulators.
comptime WARP_TILES_M = 2
comptime WARP_TILES_N = 2


def amar_matmul_wmma[
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

    var lane = thread_idx.x
    var h = lane % 16
    var half = lane // 16

    var tile_row = block_idx.y * (WARP_TILES_M * WMMA_M)
    var tile_col = block_idx.x * (WARP_TILES_N * WMMA_N)

    var acc = SIMD[DType.float32, WARP_TILES_M * WARP_TILES_N * 8](0)

    var kt = 0
    while kt < K:
        # One A fragment per row-tile, one B fragment per column-tile.
        var a_frag = SIMD[DType.float16, WARP_TILES_M * 16](0)
        comptime for tm in range(WARP_TILES_M):
            var row = tile_row + tm * WMMA_M + h
            comptime for i in range(16):
                if row < M and kt + i < K:
                    a_frag[tm * 16 + i] = rebind[Scalar[DType.float16]](
                        A[row, kt + i]
                    )

        var b_frag = SIMD[DType.float16, WARP_TILES_N * 16](0)
        comptime for tn in range(WARP_TILES_N):
            var col = tile_col + tn * WMMA_N + h
            comptime for i in range(16):
                if col < N and kt + i < K:
                    b_frag[tn * 16 + i] = rebind[Scalar[DType.float16]](
                        B[kt + i, col]
                    )

        comptime for tm in range(WARP_TILES_M):
            var a = SIMD[DType.float16, 16](0)
            comptime for i in range(16):
                a[i] = a_frag[tm * 16 + i]
            comptime for tn in range(WARP_TILES_N):
                var b = SIMD[DType.float16, 16](0)
                comptime for i in range(16):
                    b[i] = b_frag[tn * 16 + i]

                var c = SIMD[DType.float32, 8](0)
                comptime for i in range(8):
                    c[i] = acc[(tm * WARP_TILES_N + tn) * 8 + i]
                var d = SIMD[DType.float32, 8](0)
                mma(d, a, b, c)
                comptime for i in range(8):
                    acc[(tm * WARP_TILES_N + tn) * 8 + i] = d[i]

        kt += WMMA_K

    comptime for tm in range(WARP_TILES_M):
        comptime for tn in range(WARP_TILES_N):
            comptime for i in range(8):
                var row = tile_row + tm * WMMA_M + 2 * i + half
                var col = tile_col + tn * WMMA_N + h
                if row < M and col < N:
                    C[row, col] = rebind[C.ElementType](
                        acc[(tm * WARP_TILES_N + tn) * 8 + i]
                    )
