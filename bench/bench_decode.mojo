"""Decode-shape benchmark: M=8, N=K=4096 (Qwythos-9B decode GEMM).

Same engine as bench.mojo (per-variant clock re-warm, correctness gate) but at
the shape that matters for single-request decode, with the skinny split-K
kernel included. Only the variants competitive at this shape are run: dbuf
(prior best), pipe, skinny, and the hipBLASLt baseline.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

comptime WARMUP_SECONDS = 1.0

from max.gpu.host import DeviceContext, HostBuffer
from layout import TileTensor, row_major

from baro import Baro
from matmul import dtype, BM, BN, TM, TN
from matmul_dbuf import matmul_dbuf
from matmul_pipe import matmul_pipe
from matmul_skinny import matmul_skinny, skinny_reduce, SM, SBN, SPLITK, SK_THREADS

comptime M = 8
comptime N = 4096
comptime K = 4096
comptime ITERS = 200

comptime a_layout = row_major[M, K]()
comptime b_layout = row_major[K, N]()
comptime c_layout = row_major[M, N]()
comptime p_layout = row_major[SPLITK, SM, N]()


def emit(name: String, ms: Float64, gflops: Float64, ok: Bool, max_err: Float64):
    var out = String("{")
    out += '"variant": "' + name + '", '
    out += '"m": ' + String(M) + ', "n": ' + String(N) + ', "k": ' + String(K) + ', '
    out += '"ms": ' + String(ms) + ', "gflops": ' + String(gflops) + ', '
    out += '"correct": ' + ("true" if ok else "false") + ', '
    out += '"max_err": ' + String(max_err) + ', '
    out += '"iters": ' + String(ITERS) + ', "tile": ' + String(SBN) + ', '
    out += '"dtype": "float32"}'
    print(out)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"

    var ctx = DeviceContext()

    var a_host = ctx.enqueue_create_host_buffer[dtype](M * K)
    var b_host = ctx.enqueue_create_host_buffer[dtype](K * N)
    var c_host = ctx.enqueue_create_host_buffer[dtype](M * N)
    ctx.synchronize()

    for i in range(M * K):
        a_host[i] = Scalar[dtype]((i % 13) - 6) * 0.25
    for i in range(K * N):
        b_host[i] = Scalar[dtype]((i % 7) - 3) * 0.5

    var a_dev = ctx.enqueue_create_buffer[dtype](M * K)
    var b_dev = ctx.enqueue_create_buffer[dtype](K * N)
    var c_dev = ctx.enqueue_create_buffer[dtype](M * N)
    var p_dev = ctx.enqueue_create_buffer[dtype](SPLITK * SM * N)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)
    ctx.synchronize()

    var A = TileTensor(a_dev, a_layout)
    var B = TileTensor(b_dev, b_layout)
    var C = TileTensor(c_dev, c_layout)
    var Cp = TileTensor(p_dev, p_layout)

    comptime FLOPS = 2.0 * Float64(M) * Float64(N) * Float64(K)

    comptime dbuf = matmul_dbuf[
        type_of(a_layout), type_of(b_layout), type_of(c_layout)
    ]
    comptime pipe = matmul_pipe[
        type_of(a_layout), type_of(b_layout), type_of(c_layout)
    ]
    comptime skinny = matmul_skinny[
        dtype, type_of(a_layout), type_of(b_layout), type_of(p_layout)
    ]
    comptime skinny16 = matmul_skinny[
        DType.bfloat16, type_of(a_layout), type_of(b_layout), type_of(p_layout)
    ]
    comptime reduce = skinny_reduce[type_of(p_layout), type_of(c_layout)]

    comptime RGRID = (ceildiv(N, BN), ceildiv(M, BM))
    comptime RBLOCK = (BN // TN, BM // TM)
    comptime SGRID = (ceildiv(N, SBN), SPLITK)
    comptime RED_THREADS = 256
    comptime RED_GRID = ceildiv(M * N, RED_THREADS)

    # --- dbuf ---
    var w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < WARMUP_SECONDS:
        for _ in range(20):
            ctx.enqueue_function[dbuf](
                A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=RGRID, block_dim=RBLOCK
            )
        ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[dbuf](
            A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=RGRID, block_dim=RBLOCK
        )
    ctx.synchronize()
    var dbuf_ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.synchronize()
    var dbuf_err = check(a_host, b_host, c_host)

    # --- pipe ---
    w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < WARMUP_SECONDS:
        for _ in range(20):
            ctx.enqueue_function[pipe](
                A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=RGRID, block_dim=RBLOCK
            )
        ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[pipe](
            A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=RGRID, block_dim=RBLOCK
        )
    ctx.synchronize()
    var pipe_ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.synchronize()
    var pipe_err = check(a_host, b_host, c_host)

    # --- skinny (split-K + reduce, both in the timed loop) ---
    w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < WARMUP_SECONDS:
        for _ in range(20):
            ctx.enqueue_function[skinny](
                A, B, Cp, Int32(M), Int32(N), Int32(K),
                grid_dim=SGRID, block_dim=SK_THREADS,
            )
            ctx.enqueue_function[reduce](
                Cp, C, Int32(M), Int32(N),
                grid_dim=RED_GRID, block_dim=RED_THREADS,
            )
        ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[skinny](
            A, B, Cp, Int32(M), Int32(N), Int32(K),
            grid_dim=SGRID, block_dim=SK_THREADS,
        )
        ctx.enqueue_function[reduce](
            Cp, C, Int32(M), Int32(N),
            grid_dim=RED_GRID, block_dim=RED_THREADS,
        )
    ctx.synchronize()
    var sk_ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.synchronize()
    var sk_err = check(a_host, b_host, c_host)

    # --- skinny bf16 (same values, exactly representable; result bit-equal) ---
    var a16_host = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
    var b16_host = ctx.enqueue_create_host_buffer[DType.bfloat16](K * N)
    ctx.synchronize()
    for i in range(M * K):
        a16_host[i] = a_host[i].cast[DType.bfloat16]()
    for i in range(K * N):
        b16_host[i] = b_host[i].cast[DType.bfloat16]()
    var a16_dev = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var b16_dev = ctx.enqueue_create_buffer[DType.bfloat16](K * N)
    ctx.enqueue_copy(dst_buf=a16_dev, src_buf=a16_host)
    ctx.enqueue_copy(dst_buf=b16_dev, src_buf=b16_host)
    ctx.synchronize()
    var A16 = TileTensor(a16_dev, a_layout)
    var B16 = TileTensor(b16_dev, b_layout)

    w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < WARMUP_SECONDS:
        for _ in range(20):
            ctx.enqueue_function[skinny16](
                A16, B16, Cp, Int32(M), Int32(N), Int32(K),
                grid_dim=SGRID, block_dim=SK_THREADS,
            )
            ctx.enqueue_function[reduce](
                Cp, C, Int32(M), Int32(N),
                grid_dim=RED_GRID, block_dim=RED_THREADS,
            )
        ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[skinny16](
            A16, B16, Cp, Int32(M), Int32(N), Int32(K),
            grid_dim=SGRID, block_dim=SK_THREADS,
        )
        ctx.enqueue_function[reduce](
            Cp, C, Int32(M), Int32(N),
            grid_dim=RED_GRID, block_dim=RED_THREADS,
        )
    ctx.synchronize()
    var sk16_ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.synchronize()
    var sk16_err = check(a_host, b_host, c_host)

    # --- hipBLASLt vendor baseline ---
    var baro = Baro()
    var pa = Int(a_dev.unsafe_ptr())
    var pb = Int(b_dev.unsafe_ptr())
    var pc = Int(c_dev.unsafe_ptr())

    w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < WARMUP_SECONDS:
        for _ in range(20):
            baro.gemm_f32(M, N, K, pa, pb, pc)
        baro.sync()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        baro.gemm_f32(M, N, K, pa, pb, pc)
    baro.sync()
    var lt_ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.synchronize()
    var lt_err = check(a_host, b_host, c_host)

    emit("hipblaslt", lt_ms, FLOPS / (lt_ms * 1.0e6), lt_err < 0.01, lt_err)
    emit("dbuf", dbuf_ms, FLOPS / (dbuf_ms * 1.0e6), dbuf_err < 0.01, dbuf_err)
    emit("pipe", pipe_ms, FLOPS / (pipe_ms * 1.0e6), pipe_err < 0.01, pipe_err)
    emit("skinny", sk_ms, FLOPS / (sk_ms * 1.0e6), sk_err < 0.01, sk_err)
    emit("skinny_bf16", sk16_ms, FLOPS / (sk16_ms * 1.0e6), sk16_err < 0.01, sk16_err)


def check(a: HostBuffer[dtype], b: HostBuffer[dtype], c: HostBuffer[dtype]) -> Float64:
    """Max abs error over a sampled set of output elements vs a host dot product."""
    var worst = Float64(0)
    for r in range(M):
        for col in range(0, N, 41):
            var want = Float64(0)
            for p in range(K):
                want += Float64(a[r * K + p]) * Float64(b[p * N + col])
            var err = abs(Float64(c[r * N + col]) - want)
            if err > worst:
                worst = err
    return worst
