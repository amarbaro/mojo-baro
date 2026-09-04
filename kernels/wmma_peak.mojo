from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout
from layout.tensor_core import mma

comptime NACC = 8
comptime ITERS = 4096
comptime WAVES_PER_BLOCK = 4
comptime BLOCKS_PER_CU = 4

def wmma_peak_kernel[CLayout: TensorLayout](
    C: TileTensor[DType.float32, CLayout, MutAnyOrigin],
):
    comptime assert C.flat_rank == 2
    var tid = thread_idx.x
    var lane = Int(tid) % 32
    var warp = Int(tid) // 32
    var global_wave = Int(block_idx.x) * WAVES_PER_BLOCK + warp

    var a = SIMD[DType.float16, 16](0)
    var b = SIMD[DType.float16, 16](0)
    for i in range(16):
        a[i] = Scalar[DType.float16]((lane + i + global_wave) % 5 - 2)
        b[i] = Scalar[DType.float16]((lane * 2 + i - global_wave) % 3 - 1)

    var acc = SIMD[DType.float32, NACC * 8](0)
    for j in range(NACC):
        for i in range(8):
            acc[j * 8 + i] = Float32((j + i + lane) % 4)

    for _ in range(ITERS):
        for j in range(NACC):
            var c = SIMD[DType.float32, 8](0)
            for i in range(8):
                c[i] = acc[j * 8 + i]
            var d = SIMD[DType.float32, 8](0)
            mma(d, a, b, c)
            for i in range(8):
                acc[j * 8 + i] = d[i]

    var s = Float32(0)
    for j in range(NACC):
        for i in range(8):
            s += acc[j * 8 + i]
    C[global_wave * 32 + lane, 0] = rebind[C.ElementType](s)
