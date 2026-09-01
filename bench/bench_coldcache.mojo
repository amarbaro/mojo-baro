"""Cold-cache q8 vs bf16 weight-native skinny GEMM.

Protocol and frozen predictions: bench/coldcache-protocol.md. NBUF distinct
weight buffers rotate per launch so no buffer is Infinity-Cache resident;
10 in-process repeats expose the run-to-run spread that invalidated the
single-buffer measurement.

Needs .work/gguf/ from:
  tools/gguf-extract.py <model> blk.0.ffn_gate.weight --ref 8 --q8
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from matmul_skinny import (
    matmul_skinny, matmul_skinny_wt, matmul_skinny_q8,
    SM, SBN, SPLITK, SK_THREADS, Q8_BLOCK,
)

comptime M = 8
comptime K = 4096
comptime N = 12288
comptime NBUF = 8
comptime ITERS = 200
comptime REPEATS = 10

comptime a_layout = row_major[M, K]()
comptime w_layout = row_major[N, K]()
comptime b_layout = row_major[K, N]()
comptime s_layout = row_major[N, K // Q8_BLOCK]()
comptime p_layout = row_major[SPLITK, SM, N]()

comptime bf16 = DType.bfloat16
comptime f32 = DType.float32


def load_into(
    path: String, dst: MutPointer[UInt8, MutUntrackedOrigin], size: Int
) raises:
    with open(path, "r") as f:
        var data = f.read_bytes()
        if len(data) != size:
            raise Error("size mismatch for " + path)
        for i in range(size):
            dst[unsafe_offset=i] = data[i]


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    comptime base = ".work/gguf/blk_0_ffn_gate_weight"

    var a_host = ctx.enqueue_create_host_buffer[bf16](M * K)
    var w_host = ctx.enqueue_create_host_buffer[bf16](N * K)
    var q_host = ctx.enqueue_create_host_buffer[DType.int8](N * K)
    var s_host = ctx.enqueue_create_host_buffer[f32](N * K // Q8_BLOCK)
    ctx.synchronize()
    load_into(base + ".a.bin", a_host.unsafe_ptr().unsafe_bitcast[UInt8](), M * K * 2)
    load_into(base + ".bin", w_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K * 2)
    load_into(base + ".q.bin", q_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K)
    load_into(base + ".scales.bin", s_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K // Q8_BLOCK * 4)

    var t_host = ctx.enqueue_create_host_buffer[bf16](N * K)
    ctx.synchronize()
    load_into(base + ".t.bin", t_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K * 2)

    var a_dev = ctx.enqueue_create_buffer[bf16](M * K)
    var t_dev = ctx.enqueue_create_buffer[bf16](NBUF * N * K)
    var w_dev = ctx.enqueue_create_buffer[bf16](NBUF * N * K)
    var q_dev = ctx.enqueue_create_buffer[DType.int8](NBUF * N * K)
    var s_dev = ctx.enqueue_create_buffer[f32](NBUF * N * K // Q8_BLOCK)
    var p_dev = ctx.enqueue_create_buffer[f32](SPLITK * SM * N)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    ctx.synchronize()

    var A = TileTensor(a_dev, a_layout)
    var Cp = TileTensor(p_dev, p_layout)

    comptime skinny_wt = matmul_skinny_wt[
        bf16, type_of(a_layout), type_of(w_layout), type_of(p_layout)
    ]
    comptime skinny_q8 = matmul_skinny_q8[
        type_of(a_layout), type_of(w_layout), type_of(s_layout), type_of(p_layout)
    ]
    comptime SGRID = (ceildiv(N, SBN), SPLITK)
    comptime FLOPS = 2.0 * Float64(M) * Float64(N) * Float64(K)

    comptime skinny_b = matmul_skinny[
        bf16, type_of(a_layout), type_of(b_layout), type_of(p_layout)
    ]
    var tp = t_dev.unsafe_ptr()
    var wp = w_dev.unsafe_ptr()
    var qp = q_dev.unsafe_ptr()
    var sp = s_dev.unsafe_ptr()
    for b in range(NBUF):
        var wb = DeviceBuffer[bf16](ctx, wp + b * N * K, N * K, owning=False)
        var qb = DeviceBuffer[DType.int8](ctx, qp + b * N * K, N * K, owning=False)
        var sb = DeviceBuffer[f32](
            ctx, sp + b * N * K // Q8_BLOCK, N * K // Q8_BLOCK, owning=False
        )
        var tb = DeviceBuffer[bf16](ctx, tp + b * N * K, N * K, owning=False)
        ctx.enqueue_copy(dst_buf=tb, src_buf=t_host)
        ctx.enqueue_copy(dst_buf=wb, src_buf=w_host)
        ctx.enqueue_copy(dst_buf=qb, src_buf=q_host)
        ctx.enqueue_copy(dst_buf=sb, src_buf=s_host)
    ctx.synchronize()

    print("repeat  bf16_us  q8_us  ratio")
    for rep in range(REPEATS):
        var w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var wb = DeviceBuffer[bf16](ctx, wp + b * N * K, N * K, owning=False)
                var W = TileTensor(wb, w_layout)
                ctx.enqueue_function[skinny_wt](
                    A, W, Cp, Int32(M), Int32(N), Int32(K),
                    grid_dim=SGRID, block_dim=SK_THREADS,
                )
            ctx.synchronize()
        var t0 = perf_counter_ns()
        for it in range(ITERS):
            var b = it % NBUF
            var wb = DeviceBuffer[bf16](ctx, wp + b * N * K, N * K, owning=False)
            var W = TileTensor(wb, w_layout)
            ctx.enqueue_function[skinny_wt](
                A, W, Cp, Int32(M), Int32(N), Int32(K),
                grid_dim=SGRID, block_dim=SK_THREADS,
            )
        ctx.synchronize()
        var bf16_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var qb = DeviceBuffer[DType.int8](ctx, qp + b * N * K, N * K, owning=False)
                var sb = DeviceBuffer[f32](
                    ctx, sp + b * N * K // Q8_BLOCK, N * K // Q8_BLOCK, owning=False
                )
                var Wq = TileTensor(qb, w_layout)
                var Ws = TileTensor(sb, s_layout)
                ctx.enqueue_function[skinny_q8](
                    A, Wq, Ws, Cp, Int32(M), Int32(N), Int32(K),
                    grid_dim=SGRID, block_dim=SK_THREADS,
                )
            ctx.synchronize()
        t0 = perf_counter_ns()
        for it in range(ITERS):
            var b = it % NBUF
            var qb = DeviceBuffer[DType.int8](ctx, qp + b * N * K, N * K, owning=False)
            var sb = DeviceBuffer[f32](
                ctx, sp + b * N * K // Q8_BLOCK, N * K // Q8_BLOCK, owning=False
            )
            var Wq = TileTensor(qb, w_layout)
            var Ws = TileTensor(sb, s_layout)
            ctx.enqueue_function[skinny_q8](
                A, Wq, Ws, Cp, Int32(M), Int32(N), Int32(K),
                grid_dim=SGRID, block_dim=SK_THREADS,
            )
        ctx.synchronize()
        var q8_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var tb = DeviceBuffer[bf16](ctx, tp + b * N * K, N * K, owning=False)
                var Bt = TileTensor(tb, b_layout)
                ctx.enqueue_function[skinny_b](
                    A, Bt, Cp, Int32(M), Int32(N), Int32(K),
                    grid_dim=SGRID, block_dim=SK_THREADS,
                )
            ctx.synchronize()
        t0 = perf_counter_ns()
        for it in range(ITERS):
            var b = it % NBUF
            var tb = DeviceBuffer[bf16](ctx, tp + b * N * K, N * K, owning=False)
            var Bt = TileTensor(tb, b_layout)
            ctx.enqueue_function[skinny_b](
                A, Bt, Cp, Int32(M), Int32(N), Int32(K),
                grid_dim=SGRID, block_dim=SK_THREADS,
            )
        ctx.synchronize()
        var blay_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        print(
            rep, " ", bf16_us, " ", q8_us, " ", blay_us, " ",
            bf16_us / q8_us, " ", bf16_us / blay_us,
        )
