"""Cold-cache M=1 ffn-shape GEMM, Q1b of bench/q8-protocol.md: bf16 m1
champion vs wave-per-row bf16 (Q1a) vs wave-per-row q8 over the q8 engine pack.

Reads blk.0.ffn_gate.weight from .work/engine-pack-q8 (int8 [N, K] + fp16
scales) and the bf16 layouts from .work/gguf. Correctness: fp64 host dot over
the SAME dequantized values (q8 arm) and the bf16 host reference (bf16 arms).
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, row_major

from matmul_skinny import (
    amar_matmul_skinny_m1, amar_matmul_skinny_m1_row, amar_matmul_skinny_q8row,
    SM, SPLITK, SK_THREADS, ROW_WAVES, ROW_THREADS,
)

comptime M = 1
comptime K = 4096
comptime N = 12288
comptime NBUF = 8
comptime ITERS = 200
comptime REPEATS = 10
comptime QBYTES = N * K + N * (K // 32) * 2

comptime a_layout = row_major[M, K]()
comptime b_layout = row_major[K, N]()
comptime w_layout = row_major[N, K]()
comptime q_layout = row_major[N, K]()
comptime s_layout = row_major[N, K // 32]()
comptime o_layout = row_major[N]()
comptime p_layout = row_major[SPLITK, SM, N]()

comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime i8 = DType.int8
comptime u8 = DType.uint8


def load_into(
    path: String, dst: MutPointer[UInt8, MutUntrackedOrigin], size: Int, skip: Int = 0
) raises:
    with open(path, "r") as f:
        _ = f.seek(skip)
        var data = f.read_bytes(size)
        if len(data) < size:
            raise Error("size mismatch for " + path)
        for i in range(size):
            dst[unsafe_offset=i] = data[i]


def pack_offset(name: String) raises -> Int:
    with open(".work/engine-pack-q8/index.txt", "r") as f:
        for line in f.read().splitlines():
            var parts = line.split(" ")
            if parts[0] == name:
                if String(parts[1]) != "q8":
                    raise Error("not q8: " + name)
                return Int(parts[2])
    raise Error("missing " + name)


def q_tensors(
    ctx: DeviceContext, q_dev: DeviceBuffer[u8], b: Int
) raises -> Tuple[TileTensor[i8, type_of(q_layout), MutAnyOrigin], TileTensor[f16, type_of(s_layout), MutAnyOrigin]]:
    var qdp = q_dev.unsafe_ptr()
    var qb = DeviceBuffer[i8](ctx, (qdp + b * QBYTES).unsafe_bitcast[Int8](), N * K, owning=False)
    var sb = DeviceBuffer[f16](ctx, (qdp + b * QBYTES + N * K).unsafe_bitcast[Float16](), N * (K // 32), owning=False)
    return (TileTensor(qb, q_layout), TileTensor(sb, s_layout))

def max_abs_norm(r_buf: HostBuffer[f32], got: HostBuffer[f32]) -> Float64:
    var worst: Float64 = 0
    var scale: Float64 = 0
    for i in range(N):
        var r = Float64(r_buf[i])
        if abs(r) > scale:
            scale = abs(r)
        var d = abs(Float64(got[i]) - r)
        if d > worst:
            worst = d
    return worst / scale


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
    var cq_host = ctx.enqueue_create_host_buffer[f32](N)
    var o_host = ctx.enqueue_create_host_buffer[f32](N)
    var p_host = ctx.enqueue_create_host_buffer[f32](SPLITK * SM * N)
    var q_host = ctx.enqueue_create_host_buffer[u8](QBYTES)
    ctx.synchronize()
    load_into(base + ".a.bin", a_host.unsafe_ptr().unsafe_bitcast[UInt8](), M * K * 2)
    load_into(base + ".t.bin", t_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K * 2)
    load_into(base + ".bin", w_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K * 2)
    load_into(base + ".c.bin", c_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * 4)
    var qoff = pack_offset("blk.0.ffn_gate.weight")
    load_into(".work/engine-pack-q8/pack.bin", q_host.unsafe_ptr().unsafe_bitcast[UInt8](), QBYTES, qoff)

    var qp = q_host.unsafe_ptr().unsafe_bitcast[Int8]()
    var sp = (q_host.unsafe_ptr() + N * K).unsafe_bitcast[Float16]()
    for j in range(N):
        var s: Float64 = 0
        for k in range(K):
            var d = Float64(sp[j * (K // 32) + k // 32])
            s += Float64(qp[j * K + k]) * d * Float64(a_host[k])
        cq_host[j] = Float32(s)

    var a_dev = ctx.enqueue_create_buffer[bf16](M * K)
    var t_dev = ctx.enqueue_create_buffer[bf16](NBUF * N * K)
    var w_dev = ctx.enqueue_create_buffer[bf16](NBUF * N * K)
    var q_dev = ctx.enqueue_create_buffer[u8](NBUF * QBYTES)
    var p_dev = ctx.enqueue_create_buffer[f32](SPLITK * SM * N)
    var o_dev = ctx.enqueue_create_buffer[f32](N)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    var tp = t_dev.unsafe_ptr()
    var wp = w_dev.unsafe_ptr()
    var qdp = q_dev.unsafe_ptr().unsafe_bitcast[UInt8]()
    for b in range(NBUF):
        var tb = DeviceBuffer[bf16](ctx, tp + b * N * K, N * K, owning=False)
        var wb = DeviceBuffer[bf16](ctx, wp + b * N * K, N * K, owning=False)
        var qb = DeviceBuffer[u8](ctx, qdp + b * QBYTES, QBYTES, owning=False)
        ctx.enqueue_copy(dst_buf=tb, src_buf=t_host)
        ctx.enqueue_copy(dst_buf=wb, src_buf=w_host)
        ctx.enqueue_copy(dst_buf=qb, src_buf=q_host)
    ctx.synchronize()

    var A = TileTensor(a_dev, a_layout)
    var Cp = TileTensor(p_dev, p_layout)
    var O = TileTensor(o_dev, o_layout)

    comptime m1c8 = amar_matmul_skinny_m1[bf16, 8, type_of(a_layout), type_of(b_layout), type_of(p_layout)]
    comptime row8 = amar_matmul_skinny_m1_row[bf16, 8, type_of(a_layout), type_of(w_layout), type_of(o_layout)]
    comptime q8row = amar_matmul_skinny_q8row[4, 1, type_of(a_layout), type_of(q_layout), type_of(s_layout), type_of(p_layout)]
    comptime G8 = (ceildiv(N, SK_THREADS * 8), SPLITK)
    comptime GR = ceildiv(N, ROW_WAVES)

    var tb0 = DeviceBuffer[bf16](ctx, tp, N * K, owning=False)
    var wb0 = DeviceBuffer[bf16](ctx, wp, N * K, owning=False)
    ctx.enqueue_function[m1c8](A, TileTensor(tb0, b_layout), Cp, Int32(M), Int32(N), Int32(K), grid_dim=G8, block_dim=SK_THREADS)
    ctx.enqueue_copy(dst_buf=p_host, src_buf=p_dev)
    ctx.synchronize()
    for j in range(N):
        var s: Float32 = 0
        for sp_ in range(SPLITK):
            s += p_host[sp_ * SM * N + j]
        o_host[j] = s
    print("m1c8 max_rel vs bf16 ref:", max_rel(c_host, o_host), " abs/maxref:", max_abs_norm(c_host, o_host))
    ctx.enqueue_function[row8](A, TileTensor(wb0, w_layout), O, Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
    ctx.enqueue_copy(dst_buf=o_host, src_buf=o_dev)
    ctx.synchronize()
    print("row8 max_rel vs bf16 ref:", max_rel(c_host, o_host), " abs/maxref:", max_abs_norm(c_host, o_host))
    var qt0 = q_tensors(ctx, q_dev, 0)
    ctx.enqueue_function[q8row](A, qt0[0], qt0[1], Cp, Int32(M), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
    ctx.enqueue_copy(dst_buf=p_host, src_buf=p_dev)
    ctx.synchronize()
    for j in range(N):
        o_host[j] = p_host[j]
    print("q8row max_rel vs dequant ref:", max_rel(cq_host, o_host), " abs/maxref:", max_abs_norm(cq_host, o_host), " vs bf16 ref abs/maxref:", max_abs_norm(c_host, o_host))

    print("rep  m1c8_us  row8_us  q8row_us")
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
                ctx.enqueue_function[row8](A, TileTensor(wb, w_layout), O, Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
            ctx.synchronize()
        t0 = perf_counter_ns()
        for it in range(ITERS):
            var wb = DeviceBuffer[bf16](ctx, wp + (it % NBUF) * N * K, N * K, owning=False)
            ctx.enqueue_function[row8](A, TileTensor(wb, w_layout), O, Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
        ctx.synchronize()
        var r8_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var qt = q_tensors(ctx, q_dev, b)
                ctx.enqueue_function[q8row](A, qt[0], qt[1], Cp, Int32(M), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
            ctx.synchronize()
        t0 = perf_counter_ns()
        for it in range(ITERS):
            var qt = q_tensors(ctx, q_dev, it % NBUF)
            ctx.enqueue_function[q8row](A, qt[0], qt[1], Cp, Int32(M), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
        ctx.synchronize()
        var q_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        print(rep, " ", m1_us, " ", r8_us, " ", q_us)
