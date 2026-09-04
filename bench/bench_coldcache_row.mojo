"""Cold-cache M=1 ffn-shape GEMM: m1 champion (transposed B) vs wave-per-row
over the weight-native [N, K] layout (bench/q8-protocol.md, Q1a).

Same NBUF rotation as bench_coldcache_m1. Correctness: every arm vs the fp32
host reference row 0 of blk.0.ffn_gate (.c.bin), max rel err printed.

Needs .work/gguf/ from:
  tools/gguf-extract.py <model> blk.0.ffn_gate.weight --ref 8
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, row_major

from matmul_skinny import (
    amar_matmul_skinny_m1, amar_matmul_skinny_m1_row,
    SM, SPLITK, SK_THREADS, ROW_WAVES, ROW_THREADS,
)

comptime M = 1
comptime K = 4096
comptime N = 12288
comptime NBUF = 8
comptime ITERS = 200
comptime REPEATS = 10

comptime a_layout = row_major[M, K]()
comptime b_layout = row_major[K, N]()
comptime w_layout = row_major[N, K]()
comptime o_layout = row_major[N]()
comptime p_layout = row_major[SPLITK, SM, N]()

comptime bf16 = DType.bfloat16
comptime f32 = DType.float32


def load_into(
    path: String, dst: MutPointer[UInt8, MutUntrackedOrigin], size: Int
) raises:
    with open(path, "r") as f:
        var data = f.read_bytes()
        if len(data) < size:
            raise Error("size mismatch for " + path)
        for i in range(size):
            dst[unsafe_offset=i] = data[i]


def max_rel(r_buf: HostBuffer[f32], got: HostBuffer[f32]) -> Float64:
    var worst: Float64 = 0
    for i in range(N):
        var r = Float64(r_buf[i])
        var d = abs(Float64(got[i]) - r) / max(abs(r), 1e-3)
        if d > worst:
            worst = d
    return worst


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    comptime base = ".work/gguf/blk_0_ffn_gate_weight"

    var a_host = ctx.enqueue_create_host_buffer[bf16](M * K)
    var t_host = ctx.enqueue_create_host_buffer[bf16](N * K)
    var w_host = ctx.enqueue_create_host_buffer[bf16](N * K)
    var c_host = ctx.enqueue_create_host_buffer[f32](N)
    var o_host = ctx.enqueue_create_host_buffer[f32](N)
    var p_host = ctx.enqueue_create_host_buffer[f32](SPLITK * SM * N)
    ctx.synchronize()
    load_into(base + ".a.bin", a_host.unsafe_ptr().unsafe_bitcast[UInt8](), M * K * 2)
    load_into(base + ".t.bin", t_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K * 2)
    load_into(base + ".bin", w_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K * 2)
    load_into(base + ".c.bin", c_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * 4)

    var a_dev = ctx.enqueue_create_buffer[bf16](M * K)
    var t_dev = ctx.enqueue_create_buffer[bf16](NBUF * N * K)
    var w_dev = ctx.enqueue_create_buffer[bf16](NBUF * N * K)
    var p_dev = ctx.enqueue_create_buffer[f32](SPLITK * SM * N)
    var o_dev = ctx.enqueue_create_buffer[f32](N)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    var tp = t_dev.unsafe_ptr()
    var wp = w_dev.unsafe_ptr()
    for b in range(NBUF):
        var tb = DeviceBuffer[bf16](ctx, tp + b * N * K, N * K, owning=False)
        var wb = DeviceBuffer[bf16](ctx, wp + b * N * K, N * K, owning=False)
        ctx.enqueue_copy(dst_buf=tb, src_buf=t_host)
        ctx.enqueue_copy(dst_buf=wb, src_buf=w_host)
    ctx.synchronize()

    var A = TileTensor(a_dev, a_layout)
    var Cp = TileTensor(p_dev, p_layout)
    var O = TileTensor(o_dev, o_layout)

    comptime m1c8 = amar_matmul_skinny_m1[bf16, 8, type_of(a_layout), type_of(b_layout), type_of(p_layout)]
    comptime row1 = amar_matmul_skinny_m1_row[bf16, 8, type_of(a_layout), type_of(w_layout), type_of(o_layout)]
    comptime row4 = amar_matmul_skinny_m1_row[bf16, 16, type_of(a_layout), type_of(w_layout), type_of(o_layout)]
    comptime G8 = (ceildiv(N, SK_THREADS * 8), SPLITK)
    comptime GR = ceildiv(N, ROW_WAVES)

    # correctness
    var tb0 = DeviceBuffer[bf16](ctx, tp, N * K, owning=False)
    var wb0 = DeviceBuffer[bf16](ctx, wp, N * K, owning=False)
    ctx.enqueue_function[m1c8](A, TileTensor(tb0, b_layout), Cp, Int32(M), Int32(N), Int32(K), grid_dim=G8, block_dim=SK_THREADS)
    ctx.enqueue_copy(dst_buf=p_host, src_buf=p_dev)
    ctx.synchronize()
    for j in range(N):
        var s: Float32 = 0
        for sp in range(SPLITK):
            s += p_host[sp * SM * N + j]
        o_host[j] = s
    print("m1c8 max_rel:", max_rel(c_host, o_host))
    ctx.enqueue_function[row1](A, TileTensor(wb0, w_layout), O, Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
    ctx.enqueue_copy(dst_buf=o_host, src_buf=o_dev)
    ctx.synchronize()
    print("row8 max_rel:", max_rel(c_host, o_host))
    ctx.enqueue_function[row4](A, TileTensor(wb0, w_layout), O, Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
    ctx.enqueue_copy(dst_buf=o_host, src_buf=o_dev)
    ctx.synchronize()
    print("row16 max_rel:", max_rel(c_host, o_host))

    print("rep  m1c8_us  row8_us  row16_us")
    for rep in range(REPEATS):
        var w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var tb = DeviceBuffer[bf16](ctx, tp + b * N * K, N * K, owning=False)
                ctx.enqueue_function[m1c8](A, TileTensor(tb, b_layout), Cp, Int32(M), Int32(N), Int32(K), grid_dim=G8, block_dim=SK_THREADS)
            ctx.synchronize()
        var t0 = perf_counter_ns()
        for it in range(ITERS):
            var tb = DeviceBuffer[bf16](ctx, tp + (it % NBUF) * N * K, N * K, owning=False)
            ctx.enqueue_function[m1c8](A, TileTensor(tb, b_layout), Cp, Int32(M), Int32(N), Int32(K), grid_dim=G8, block_dim=SK_THREADS)
        ctx.synchronize()
        var m1_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var wb = DeviceBuffer[bf16](ctx, wp + b * N * K, N * K, owning=False)
                ctx.enqueue_function[row1](A, TileTensor(wb, w_layout), O, Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
            ctx.synchronize()
        t0 = perf_counter_ns()
        for it in range(ITERS):
            var wb = DeviceBuffer[bf16](ctx, wp + (it % NBUF) * N * K, N * K, owning=False)
            ctx.enqueue_function[row1](A, TileTensor(wb, w_layout), O, Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
        ctx.synchronize()
        var r1_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var wb = DeviceBuffer[bf16](ctx, wp + b * N * K, N * K, owning=False)
                ctx.enqueue_function[row4](A, TileTensor(wb, w_layout), O, Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
            ctx.synchronize()
        t0 = perf_counter_ns()
        for it in range(ITERS):
            var wb = DeviceBuffer[bf16](ctx, wp + (it % NBUF) * N * K, N * K, owning=False)
            ctx.enqueue_function[row4](A, TileTensor(wb, w_layout), O, Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
        ctx.synchronize()
        var r4_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        print(rep, " ", m1_us, " ", r1_us, " ", r4_us)
