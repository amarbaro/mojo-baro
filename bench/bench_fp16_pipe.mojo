"""fp16 WMMA pipelined GEMM benchmark, gfx1100 (kernels/matmul_wmma_pipe.mojo).

Separate from bench.mojo on purpose: this is fp16 in / fp32 accumulate on the
matrix cores, a different unit of the chip from the fp32 SIMD path. The numbers
are NOT comparable to the fp32 variants.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, HostBuffer
from layout import TileTensor, row_major

from matmul_wmma_pipe import amar_matmul_wmma_pipe, BLK_M, BLK_N, BLK_K, NTHREADS, WARPS_M, WARPS_N, WTILE_M, WTILE_N, PAD_A, PAD_B, TRANS_B, ALIGNED, C_DTYPE, PGR, LB

comptime M = 512
comptime N = 512
comptime K = 512
comptime ITERS = 200
comptime WARMUP_SECONDS = 10.0

comptime a_layout = row_major[M, K]()
comptime b_layout = row_major[K, N]()
comptime c_layout = row_major[M, N]()


def check(
    a: HostBuffer[DType.float16],
    b: HostBuffer[DType.float16],
    c: HostBuffer[C_DTYPE],
) -> Float64:
    comptime RSTEP = 37 if M <= 2048 else (M // 32)
    comptime CSTEP = 41 if N <= 2048 else (N // 32)
    var worst = Float64(0)
    for r in range(0, M, RSTEP):
        for col in range(0, N, CSTEP):
            var want = Float64(0)
            for p in range(K):
                want += Float64(a[r * K + p]) * Float64(b[p * N + col])
            var err = abs(Float64(c[r * N + col]) - want)
            if err > worst:
                worst = err
    return worst


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    var ah = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    var bh = ctx.enqueue_create_host_buffer[DType.float16](K * N)
    var ch = ctx.enqueue_create_host_buffer[C_DTYPE](M * N)
    ctx.synchronize()

    # Small integers so the fp16 product is exact and the gate catches real bugs
    # rather than rounding.
    for i in range(M * K):
        ah[i] = Scalar[DType.float16]((i % 7) - 3)
    for i in range(K * N):
        bh[i] = Scalar[DType.float16]((i % 5) - 2)

    var ad = ctx.enqueue_create_buffer[DType.float16](M * K)
    var bd = ctx.enqueue_create_buffer[DType.float16](K * N)
    var cd = ctx.enqueue_create_buffer[C_DTYPE](M * N)
    ctx.enqueue_copy(dst_buf=ad, src_buf=ah)
    ctx.enqueue_copy(dst_buf=bd, src_buf=bh)
    ctx.synchronize()

    var A = TileTensor(ad, a_layout)
    var B = TileTensor(bd, b_layout)
    var C = TileTensor(cd, c_layout)

    comptime GRID = (ceildiv(N, BLK_N), ceildiv(M, BLK_M))
    comptime kernel = amar_matmul_wmma_pipe[
        type_of(a_layout), type_of(b_layout), type_of(c_layout)
    ]
    comptime FLOPS = 2.0 * Float64(M) * Float64(N) * Float64(K)

    # Clocks idle at ~28 MHz and ramp only under back-to-back work.
    var w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < WARMUP_SECONDS:
        for _ in range(20):
            ctx.enqueue_function[kernel](
                A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=GRID, block_dim=NTHREADS
            )
        ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[kernel](
            A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=GRID, block_dim=NTHREADS
        )
    ctx.synchronize()
    var ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)

    ctx.enqueue_copy(dst_buf=ch, src_buf=cd)
    ctx.synchronize()
    var err = check(ah, bh, ch)

    var out = String("{")
    out += '"variant": "wmma_pipe_fp16", '
    out += '"m": ' + String(M) + ', "n": ' + String(N) + ', "k": ' + String(K) + ", "
    out += '"ms": ' + String(ms) + ", "
    out += '"gflops": ' + String(FLOPS / (ms * 1.0e6)) + ", "
    out += '"correct": ' + ("true" if err < 0.01 else "false") + ", "
    out += '"max_err": ' + String(err) + ", "
    out += '"iters": ' + String(ITERS) + ', "warmup_s": ' + String(WARMUP_SECONDS) + ', "pgr": ' + String(PGR) + ', "lb": ' + String(LB) + ', "tile": 16, "dtype": "float16", '
    out += '"blk": [' + String(BLK_M) + ", " + String(BLK_N) + ", " + String(BLK_K) + "], "
    out += '"warps": [' + String(WARPS_M) + ", " + String(WARPS_N) + "], "
    out += '"wtile": [' + String(WTILE_M) + ", " + String(WTILE_N) + "], "
    out += '"grid": [' + String(GRID[0]) + ", " + String(GRID[1]) + '], "block": ' + String(NTHREADS) + "}"
    print(out)
