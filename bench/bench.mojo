"""Benchmark + correctness engine for the baro GEMM kernels.

Emits one JSON object per variant on stdout so runs accumulate into a log and
stay comparable across sessions. Correctness is checked every run: a fast
kernel that is wrong is a failure, not a result.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

# The GPU ramps to full clocks in ~0.4 s of BACK-TO-BACK work and drops as soon
# as it idles. The host-side correctness check between variants is long enough
# to lose the clocks, so each variant re-warms immediately before it is timed;
# a single warmup at the start biased whichever variant ran first.
comptime WARMUP_SECONDS = 1.0
from max.gpu.host import DeviceContext, HostBuffer
from layout import TileTensor, row_major

from baro import Baro
from matmul import (
    matmul_naive, matmul_tiled, matmul_regtile, dtype, TILE, BM, BN, TM, TN
)
from matmul_ldst import matmul_ldst
from matmul_dbuf import matmul_dbuf
from matmul_vec4 import matmul_vec4
from matmul_pipe import matmul_pipe

comptime M = 512
comptime N = 512
comptime K = 512
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

    comptime regtile = matmul_regtile[
        type_of(a_layout), type_of(b_layout), type_of(c_layout)
    ]
    comptime RGRID = (ceildiv(N, BN), ceildiv(M, BM))
    comptime RBLOCK = (BN // TN, BM // TM)

    comptime ldst = matmul_ldst[
        type_of(a_layout), type_of(b_layout), type_of(c_layout)
    ]
    comptime dbuf = matmul_dbuf[
        type_of(a_layout), type_of(b_layout), type_of(c_layout)
    ]
    comptime vec4 = matmul_vec4[
        type_of(a_layout), type_of(b_layout), type_of(c_layout)
    ]

    comptime pipe = matmul_pipe[
        type_of(a_layout), type_of(b_layout), type_of(c_layout)
    ]

    comptime naive = matmul_naive[
        type_of(a_layout), type_of(b_layout), type_of(c_layout)
    ]
    comptime tiled = matmul_tiled[
        type_of(a_layout), type_of(b_layout), type_of(c_layout)
    ]

    # --- naive ---
    var w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < WARMUP_SECONDS:
        for _ in range(20):
            ctx.enqueue_function[naive](
                A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=GRID, block_dim=BLOCK
            )
        ctx.synchronize()
    for _ in range(1):
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
    w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < WARMUP_SECONDS:
        for _ in range(20):
            ctx.enqueue_function[tiled](
                A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=GRID, block_dim=BLOCK
            )
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
    w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < WARMUP_SECONDS:
        for _ in range(20):
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

    # --- ldst ---
    w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < WARMUP_SECONDS:
        for _ in range(20):
            ctx.enqueue_function[ldst](
                A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=RGRID, block_dim=RBLOCK
            )
        ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[ldst](
            A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=RGRID, block_dim=RBLOCK
        )
    ctx.synchronize()
    var ldst_ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.synchronize()
    var ldst_err = check(a_host, b_host, c_host)

    # --- dbuf ---
    w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < WARMUP_SECONDS:
        for _ in range(20):
            ctx.enqueue_function[dbuf](
                A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=RGRID, block_dim=RBLOCK
            )
        ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[dbuf](
            A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=RGRID, block_dim=RBLOCK
        )
    ctx.synchronize()
    var dbuf_ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.synchronize()
    var dbuf_err = check(a_host, b_host, c_host)

    # --- vec4 ---
    w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < WARMUP_SECONDS:
        for _ in range(20):
            ctx.enqueue_function[vec4](
                A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=RGRID, block_dim=RBLOCK
            )
        ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[vec4](
            A, B, C, Int32(M), Int32(N), Int32(K), grid_dim=RGRID, block_dim=RBLOCK
        )
    ctx.synchronize()
    var vec4_ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.synchronize()
    var vec4_err = check(a_host, b_host, c_host)

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

    # --- hipBLASLt vendor baseline, same device buffers ---
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
    emit("naive", naive_ms, FLOPS / (naive_ms * 1.0e6), naive_err < 0.01, naive_err)
    emit("tiled", tiled_ms, FLOPS / (tiled_ms * 1.0e6), tiled_err < 0.01, tiled_err)
    emit("regtile", reg_ms, FLOPS / (reg_ms * 1.0e6), reg_err < 0.01, reg_err)
    emit("ldst", ldst_ms, FLOPS / (ldst_ms * 1.0e6), ldst_err < 0.01, ldst_err)
    emit("dbuf", dbuf_ms, FLOPS / (dbuf_ms * 1.0e6), dbuf_err < 0.01, dbuf_err)
    emit("vec4", vec4_ms, FLOPS / (vec4_ms * 1.0e6), vec4_err < 0.01, vec4_err)
    emit("pipe", pipe_ms, FLOPS / (pipe_ms * 1.0e6), pipe_err < 0.01, pipe_err)


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
