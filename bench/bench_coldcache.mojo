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

from amarbaro import AmarBaro
from matmul_skinny import (
    matmul_skinny, matmul_skinny_q8b, matmul_skinny_v2,
    SM, SBN, SPLITK, SK_THREADS, Q8_BLOCK,
)

comptime M = 8
comptime K = 4096
comptime N = 12288
comptime NBUF = 8
comptime ITERS = 200
comptime REPEATS = 10

comptime a_layout = row_major[M, K]()
comptime b_layout = row_major[K, N]()
comptime qt_layout = row_major[K, N]()
comptime st_layout = row_major[K // Q8_BLOCK, N]()
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
    ctx.synchronize()
    load_into(base + ".a.bin", a_host.unsafe_ptr().unsafe_bitcast[UInt8](), M * K * 2)

    var t_host = ctx.enqueue_create_host_buffer[bf16](N * K)
    var qt_host = ctx.enqueue_create_host_buffer[DType.int8](N * K)
    var st_host = ctx.enqueue_create_host_buffer[f32](N * K // Q8_BLOCK)
    var h16_host = ctx.enqueue_create_host_buffer[DType.float16](N * K)
    var a16_host = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    ctx.synchronize()
    load_into(base + ".t.bin", t_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K * 2)
    load_into(base + ".qt.bin", qt_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K)
    load_into(base + ".scales_t.bin", st_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K // Q8_BLOCK * 4)
    for i in range(N * K):
        h16_host[i] = t_host[i].cast[DType.float16]()
    for i in range(M * K):
        a16_host[i] = a_host[i].cast[DType.float16]()

    var a_dev = ctx.enqueue_create_buffer[bf16](M * K)
    var a16_dev = ctx.enqueue_create_buffer[DType.float16](M * K)
    var c16_dev = ctx.enqueue_create_buffer[DType.float16](M * N)
    var t_dev = ctx.enqueue_create_buffer[bf16](NBUF * N * K)
    var qt_dev = ctx.enqueue_create_buffer[DType.int8](NBUF * N * K)
    var st_dev = ctx.enqueue_create_buffer[f32](NBUF * N * K // Q8_BLOCK)
    var h16_dev = ctx.enqueue_create_buffer[DType.float16](NBUF * N * K)
    var p_dev = ctx.enqueue_create_buffer[f32](SPLITK * SM * N)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    ctx.enqueue_copy(dst_buf=a16_dev, src_buf=a16_host)
    ctx.synchronize()

    var A = TileTensor(a_dev, a_layout)
    var Cp = TileTensor(p_dev, p_layout)

    comptime SGRID = (ceildiv(N, SBN), SPLITK)
    comptime FLOPS = 2.0 * Float64(M) * Float64(N) * Float64(K)

    comptime skinny_b = matmul_skinny[
        bf16, type_of(a_layout), type_of(b_layout), type_of(p_layout)
    ]
    comptime skinny_q8b = matmul_skinny_q8b[
        type_of(a_layout), type_of(qt_layout), type_of(st_layout), type_of(p_layout)
    ]
    comptime skinny_v2_c2 = matmul_skinny_v2[
        bf16, 2, type_of(a_layout), type_of(b_layout), type_of(p_layout)
    ]
    comptime skinny_v2_c4 = matmul_skinny_v2[
        bf16, 4, type_of(a_layout), type_of(b_layout), type_of(p_layout)
    ]
    comptime skinny_v2_c8 = matmul_skinny_v2[
        bf16, 8, type_of(a_layout), type_of(b_layout), type_of(p_layout)
    ]
    var tp = t_dev.unsafe_ptr()
    var qtp = qt_dev.unsafe_ptr()
    var stp = st_dev.unsafe_ptr()
    var hp = h16_dev.unsafe_ptr()
    for b in range(NBUF):
        var tb = DeviceBuffer[bf16](ctx, tp + b * N * K, N * K, owning=False)
        var qtb = DeviceBuffer[DType.int8](ctx, qtp + b * N * K, N * K, owning=False)
        var stb = DeviceBuffer[f32](
            ctx, stp + b * N * K // Q8_BLOCK, N * K // Q8_BLOCK, owning=False
        )
        var hb = DeviceBuffer[DType.float16](ctx, hp + b * N * K, N * K, owning=False)
        ctx.enqueue_copy(dst_buf=tb, src_buf=t_host)
        ctx.enqueue_copy(dst_buf=qtb, src_buf=qt_host)
        ctx.enqueue_copy(dst_buf=stb, src_buf=st_host)
        ctx.enqueue_copy(dst_buf=hb, src_buf=h16_host)
    ctx.synchronize()

    print("rep blay_us v2c2_us v2c4_us v2c8_us vendor_us")
    for rep in range(REPEATS):
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

        comptime G2 = (ceildiv(N, SK_THREADS * 2), SPLITK)
        w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var tb = DeviceBuffer[bf16](ctx, tp + b * N * K, N * K, owning=False)
                ctx.enqueue_function[skinny_v2_c2](
                    A, TileTensor(tb, b_layout), Cp, Int32(M), Int32(N), Int32(K),
                    grid_dim=G2, block_dim=SK_THREADS,
                )
            ctx.synchronize()
        t0 = perf_counter_ns()
        for it in range(ITERS):
            var tb = DeviceBuffer[bf16](ctx, tp + (it % NBUF) * N * K, N * K, owning=False)
            ctx.enqueue_function[skinny_v2_c2](
                A, TileTensor(tb, b_layout), Cp, Int32(M), Int32(N), Int32(K),
                grid_dim=G2, block_dim=SK_THREADS,
            )
        ctx.synchronize()
        var v2c2_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        comptime G4 = (ceildiv(N, SK_THREADS * 4), SPLITK)
        w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var tb = DeviceBuffer[bf16](ctx, tp + b * N * K, N * K, owning=False)
                ctx.enqueue_function[skinny_v2_c4](
                    A, TileTensor(tb, b_layout), Cp, Int32(M), Int32(N), Int32(K),
                    grid_dim=G4, block_dim=SK_THREADS,
                )
            ctx.synchronize()
        t0 = perf_counter_ns()
        for it in range(ITERS):
            var tb = DeviceBuffer[bf16](ctx, tp + (it % NBUF) * N * K, N * K, owning=False)
            ctx.enqueue_function[skinny_v2_c4](
                A, TileTensor(tb, b_layout), Cp, Int32(M), Int32(N), Int32(K),
                grid_dim=G4, block_dim=SK_THREADS,
            )
        ctx.synchronize()
        var v2c4_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        comptime G8 = (ceildiv(N, SK_THREADS * 8), SPLITK)
        w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var tb = DeviceBuffer[bf16](ctx, tp + b * N * K, N * K, owning=False)
                ctx.enqueue_function[skinny_v2_c8](
                    A, TileTensor(tb, b_layout), Cp, Int32(M), Int32(N), Int32(K),
                    grid_dim=G8, block_dim=SK_THREADS,
                )
            ctx.synchronize()
        t0 = perf_counter_ns()
        for it in range(ITERS):
            var tb = DeviceBuffer[bf16](ctx, tp + (it % NBUF) * N * K, N * K, owning=False)
            ctx.enqueue_function[skinny_v2_c8](
                A, TileTensor(tb, b_layout), Cp, Int32(M), Int32(N), Int32(K),
                grid_dim=G8, block_dim=SK_THREADS,
            )
        ctx.synchronize()
        var v2c8_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        var baro = AmarBaro()
        var pa = Int(a16_dev.unsafe_ptr())
        var pc = Int(c16_dev.unsafe_ptr())
        w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                baro.gemm_f16(M, N, K, pa, Int(hp + b * N * K), pc)
            baro.sync()
        t0 = perf_counter_ns()
        for it in range(ITERS):
            baro.gemm_f16(M, N, K, pa, Int(hp + (it % NBUF) * N * K), pc)
        baro.sync()
        var lt_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        print(
            rep, " ", blay_us, " ", v2c2_us, " ", v2c4_us, " ",
            v2c8_us, " ", lt_us,
        )
