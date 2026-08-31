"""Benchmark + correctness engine for the baro GEMM kernels.

Emits one JSON object per variant on stdout so runs accumulate into a log and
stay comparable across sessions. Correctness is checked every run: a fast
kernel that is wrong is a failure, not a result.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, HostBuffer
from layout import TileTensor, row_major

from matmul import (
    matmul_naive, matmul_tiled, matmul_regtile, dtype, TILE, BM, BN, TM, TN
)

comptime M = 512
comptime N = 512
comptime K = 512
comptime WARMUP = 10
comptime ITERS = 200

comptime a_layout = row_major[M, K]()
comptime b_layout = row_major[K, N]()
comptime c_layout = row_major[M, N]()


def emit(name: String, ms: Float64, gflops: Float64, ok: Bool, max_err: Float64):
    var out = String("{")
    out += '"variant": "' + name + '", '
    out += '"m": ' + String(M) + ', "n": ' + String(N) + ', "k": ' + String(K) + ', '
    out += '"ms": ' + String(ms) + ', "gflops": ' + String(gflops) + ', '
    out += '"correct": ' + ("true" if ok else "false") + ', '
    out += '"max_err": ' + String(max_err) + ', '
    out += '"iters": ' + String(ITERS) + ', "tile": ' + String(TILE) + ', '
    out += '"dtype": "float32"}'
    print(out)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"

    var ctx = DeviceContext()

    var a_host = ctx.enqueue_create_host_buffer[dtype](M * K)
    var b_host = ctx.enqueue_create_host_buffer[dtype](K * N)
    var c_host = ctx.enqueue_create_host_buffer[dtype](M * N)
    ctx.synchronize()

    # Deterministic, non-trivial data: a constant matrix would hide index bugs.
    for i in range(M * K):
        a_host[i] = Scalar[dtype]((i % 13) - 6) * 0.25
    for i in range(K * N):
        b_host[i] = Scalar[dtype]((i % 7) - 3) * 0.5

    var a_dev = ctx.enqueue_create_buffer[dtype](M * K)
    var b_dev = ctx.enqueue_create_buffer[dtype](K * N)
    var c_dev = ctx.enqueue_create_buffer[dtype](M * N)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)
    ctx.synchronize()

    var A = TileTensor(a_dev, a_layout)
    var B = TileTensor(b_dev, b_layout)
    var C = TileTensor(c_dev, c_layout)

    comptime GRID = (ceildiv(N, TILE), ceildiv(M, TILE))
    comptime BLOCK = (TILE, TILE)
    comptime FLOPS = 2.0 * Float64(M) * Float64(N) * Float64(K)

    comptime naive = matmul_naive[
        type_of(a_layout), type_of(b_layout), type_of(c_layout)
    ]
    comptime tiled = matmul_tiled[
        type_of(a_layout), type_of(b_layout), type_of(c_layout)
    ]

    # --- naive ---
    for _ in range(WARMUP):
        ctx.enqueue_function[naive](A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=GRID, block_dim=BLOCK)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[naive](A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=GRID, block_dim=BLOCK)
    ctx.synchronize()
    var naive_ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.synchronize()
    var naive_err = check(a_host, b_host, c_host)

    # --- tiled ---
    for _ in range(WARMUP):
        ctx.enqueue_function[tiled](A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=GRID, block_dim=BLOCK)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[tiled](A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=GRID, block_dim=BLOCK)
    ctx.synchronize()
    var tiled_ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.synchronize()
    var tiled_err = check(a_host, b_host, c_host)

    # --- register-tiled ---
    comptime regtile = matmul_regtile[
        type_of(a_layout), type_of(b_layout), type_of(c_layout)
    ]
    comptime RGRID = (ceildiv(N, BN), ceildiv(M, BM))
    comptime RBLOCK = (BN // TN, BM // TM)

    for _ in range(WARMUP):
        ctx.enqueue_function[regtile](
            A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=RGRID, block_dim=RBLOCK
        )
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[regtile](
            A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=RGRID, block_dim=RBLOCK
        )
    ctx.synchronize()
    var reg_ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.synchronize()
    var reg_err = check(a_host, b_host, c_host)

    emit("naive", naive_ms, FLOPS / (naive_ms * 1.0e6), naive_err < 0.01, naive_err)
    emit("tiled", tiled_ms, FLOPS / (tiled_ms * 1.0e6), tiled_err < 0.01, tiled_err)
    emit("regtile", reg_ms, FLOPS / (reg_ms * 1.0e6), reg_err < 0.01, reg_err)


def check(a: HostBuffer[dtype], b: HostBuffer[dtype], c: HostBuffer[dtype]) -> Float64:
    """Max abs error over a sampled set of output elements vs a host dot product."""
    var worst = Float64(0)
    for r in range(0, M, 37):
        for col in range(0, N, 41):
            var want = Float64(0)
            for p in range(K):
                want += Float64(a[r * K + p]) * Float64(b[p * N + col])
            var err = abs(Float64(c[r * N + col]) - want)
            if err > worst:
                worst = err
    return worst
