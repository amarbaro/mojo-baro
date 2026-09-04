"""Cold-cache row-scaling receipt, stage M0 of bench/mrow-gemm-protocol.md:
the CURRENT amar_matmul_skinny_q8row[4, MR] at MR in (1, 2, 4, 8) rows on the
ffn shape (N=12288, K=4096), modeled on bench_coldcache_q8row.mojo (same
NBUF=8 cold-cache rotation over the q8 engine pack, same ROW_WAVES/ROW_THREADS
grid as serve/registry.mojo's gemm_q8 dispatch).

Correctness: fp64 host dot over the SAME dequantized int8*scale values, every
row, max_rel <= 1e-3. A failing row fails the whole arm; no timing is printed
for a failed arm.

Stage M4 (bench/draft-q4-protocol.md) adds a q4row MR=1 arm on the same ffn
shape: amar_matmul_skinny_q4row over a standalone Q4_0 quantization of
blk.0.ffn_gate.weight (tools/q4-ffn-fixture.py; not the engine pack -- this
shape is the fixed cold-cache stream comparison, not the draft head), same
fp64-per-row host check, timed against this run's own q8row MR=1 number.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, row_major

from matmul_skinny import (
    amar_matmul_skinny_q8row, amar_matmul_skinny_q8dot, amar_matmul_skinny_q4row,
    SM, SPLITK, ROW_WAVES, ROW_THREADS,
)
from elementwise import amar_quantize_q8_rows

comptime MAXM = 8
comptime K = 4096
comptime N = 12288
comptime NBUF = 8
comptime ITERS = 200
comptime QBYTES = N * K + N * (K // 32) * 2
comptime QBYTES4 = N * (K // 2) + N * (K // 32) * 2

comptime q_layout = row_major[N, K]()
comptime q4_layout = row_major[N, K // 2]()
comptime s_layout = row_major[N, K // 32]()
comptime p_layout = row_major[SPLITK, SM, N]()

comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime i8 = DType.int8
comptime u8 = DType.uint8

comptime GR = ceildiv(N, ROW_WAVES)


def load_into(
    path: String, dst: MutPointer[UInt8, MutUntrackedOrigin], size: Int, skip: Int = 0
) raises:
    with open(path, "r") as f:
        _ = f.seek(skip)
        var data = f.read_bytes(size)
        if len(data) < size:
            raise Error("size mismatch " + path)
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


def q4_tensors(
    ctx: DeviceContext, q4_dev: DeviceBuffer[u8], b: Int
) raises -> Tuple[TileTensor[u8, type_of(q4_layout), MutAnyOrigin], TileTensor[f16, type_of(s_layout), MutAnyOrigin]]:
    var qdp = q4_dev.unsafe_ptr()
    var qb = DeviceBuffer[u8](ctx, qdp + b * QBYTES4, N * (K // 2), owning=False)
    var sb = DeviceBuffer[f16](ctx, (qdp + b * QBYTES4 + N * (K // 2)).unsafe_bitcast[Float16](), N * (K // 32), owning=False)
    return (TileTensor(qb, q4_layout), TileTensor(sb, s_layout))


def median_us(durs: InlineArray[Float64, ITERS]) -> Float64:
    var arr = durs.copy()
    for i in range(ITERS):
        var j = i
        while j > 0 and arr[j - 1] > arr[j]:
            var tmp = arr[j - 1]
            arr[j - 1] = arr[j]
            arr[j] = tmp
            j -= 1
    return arr[ITERS // 2]


def run_arm[MR: Int](
    ctx: DeviceContext,
    a_host: HostBuffer[bf16],
    q_host: HostBuffer[u8],
    q_dev: DeviceBuffer[u8],
    base_us: Float64,
) raises -> Float64:
    comptime a_layout = row_major[MR, K]()
    comptime q8row = amar_matmul_skinny_q8row[
        4, MR, type_of(a_layout), type_of(q_layout), type_of(s_layout), type_of(p_layout)
    ]

    var a_host_mr = ctx.enqueue_create_host_buffer[bf16](MR * K)
    ctx.synchronize()
    for i in range(MR * K):
        a_host_mr[i] = a_host[i]
    var a_dev = ctx.enqueue_create_buffer[bf16](MR * K)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host_mr)
    var p_dev = ctx.enqueue_create_buffer[f32](SPLITK * SM * N)
    var p_host = ctx.enqueue_create_host_buffer[f32](SPLITK * SM * N)
    ctx.synchronize()

    var A = TileTensor(a_dev, a_layout)
    var Cp = TileTensor(p_dev, p_layout)

    print(
        "MR=" + String(MR), "UNROLL=4", "grid_dim=" + String(GR),
        "block_dim=" + String(ROW_THREADS), "ROW_WAVES=" + String(ROW_WAVES),
    )

    var qt0 = q_tensors(ctx, q_dev, 0)
    ctx.enqueue_function[q8row](
        A, qt0[0], qt0[1], Cp, Int32(MR), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS
    )
    ctx.enqueue_copy(dst_buf=p_host, src_buf=p_dev)
    ctx.synchronize()

    var qp = q_host.unsafe_ptr().unsafe_bitcast[Int8]()
    var sp = (q_host.unsafe_ptr() + N * K).unsafe_bitcast[Float16]()
    var worst: Float64 = 0
    for r in range(MR):
        for j in range(N):
            var s: Float64 = 0
            for k in range(K):
                var d = Float64(sp[j * (K // 32) + k // 32])
                s += Float64(qp[j * K + k]) * d * Float64(a_host_mr[r * K + k])
            var rel = abs(Float64(p_host[r * N + j]) - s) / max(abs(s), 1e-3)
            if rel > worst:
                worst = rel

    if worst > 1e-3:
        print("MR=" + String(MR), "m=" + String(MR), "FAIL max_rel=" + String(worst))
        return -1.0

    var w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
        for b in range(NBUF):
            var qtw = q_tensors(ctx, q_dev, b)
            ctx.enqueue_function[q8row](
                A, qtw[0], qtw[1], Cp, Int32(MR), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS
            )
            ctx.synchronize()

    var durs = InlineArray[Float64, ITERS](uninitialized=True)
    for it in range(ITERS):
        var qtw = q_tensors(ctx, q_dev, it % NBUF)
        var t0 = perf_counter_ns()
        ctx.enqueue_function[q8row](
            A, qtw[0], qtw[1], Cp, Int32(MR), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS
        )
        ctx.synchronize()
        durs[it] = Float64(perf_counter_ns() - t0) / 1.0e3

    var us = median_us(durs)
    var gbps = Float64(QBYTES) / (us * 1.0e-6) / 1.0e9
    var ratio = us / base_us if base_us > 0 else 1.0
    print(
        "MR=" + String(MR), "m=" + String(MR), "us=" + String(us), "GBps=" + String(gbps),
        "ratio=" + String(ratio), "correct=true",
    )
    return us


def run_dot_arm[MR: Int, UNROLL: Int = 4](
    ctx: DeviceContext,
    a_host: HostBuffer[bf16],
    q_host: HostBuffer[u8],
    q_dev: DeviceBuffer[u8],
    base_us: Float64,
) raises -> Float64:
    comptime a_layout = row_major[MR, K]()
    comptime aq_layout = row_major[MR, K]()
    comptime as_layout = row_major[MR, K // 32]()
    comptime quant = amar_quantize_q8_rows[
        type_of(a_layout), type_of(aq_layout), type_of(as_layout)
    ]
    comptime q8dot = amar_matmul_skinny_q8dot[
        UNROLL, MR, type_of(aq_layout), type_of(as_layout),
        type_of(q_layout), type_of(s_layout), type_of(p_layout)
    ]
    comptime GRQ = (MR, K // 32)

    var a_host_mr = ctx.enqueue_create_host_buffer[bf16](MR * K)
    ctx.synchronize()
    for i in range(MR * K):
        a_host_mr[i] = a_host[i]
    var a_dev = ctx.enqueue_create_buffer[bf16](MR * K)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host_mr)
    var aq_dev = ctx.enqueue_create_buffer[i8](MR * K)
    var as_dev = ctx.enqueue_create_buffer[f16](MR * (K // 32))
    var aq_host = ctx.enqueue_create_host_buffer[i8](MR * K)
    var as_host = ctx.enqueue_create_host_buffer[f16](MR * (K // 32))
    var p_dev = ctx.enqueue_create_buffer[f32](SPLITK * SM * N)
    var p_host = ctx.enqueue_create_host_buffer[f32](SPLITK * SM * N)
    ctx.synchronize()

    var A = TileTensor(a_dev, a_layout)
    var Aq = TileTensor(aq_dev, aq_layout)
    var As = TileTensor(as_dev, as_layout)
    var Cp = TileTensor(p_dev, p_layout)

    print(
        "DOT MR=" + String(MR), "UNROLL=" + String(UNROLL), "grid_dim=" + String(GR),
        "block_dim=" + String(ROW_THREADS), "ROW_WAVES=" + String(ROW_WAVES),
    )

    ctx.enqueue_function[quant](A, Aq, As, Int32(MR), Int32(K), grid_dim=GRQ, block_dim=32)
    ctx.enqueue_copy(dst_buf=aq_host, src_buf=aq_dev)
    ctx.enqueue_copy(dst_buf=as_host, src_buf=as_dev)
    ctx.synchronize()

    var qt0 = q_tensors(ctx, q_dev, 0)
    ctx.enqueue_function[q8dot](
        Aq, As, qt0[0], qt0[1], Cp, Int32(MR), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS
    )
    ctx.enqueue_copy(dst_buf=p_host, src_buf=p_dev)
    ctx.synchronize()

    var qp = q_host.unsafe_ptr().unsafe_bitcast[Int8]()
    var sp = (q_host.unsafe_ptr() + N * K).unsafe_bitcast[Float16]()
    var worst: Float64 = 0
    for r in range(MR):
        for j in range(N):
            var s: Float64 = 0
            for k in range(K):
                var wd = Float64(sp[j * (K // 32) + k // 32])
                var wq = Float64(qp[j * K + k])
                var ad = Float64(as_host[r * (K // 32) + k // 32])
                var aq = Float64(aq_host[r * K + k])
                s += wq * wd * aq * ad
            var rel = abs(Float64(p_host[r * N + j]) - s) / max(abs(s), 1e-3)
            if rel > worst:
                worst = rel

    if worst > 2e-3:
        print("DOT MR=" + String(MR), "m=" + String(MR), "FAIL max_rel=" + String(worst))
        return -1.0

    var w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
        for b in range(NBUF):
            var qtw = q_tensors(ctx, q_dev, b)
            ctx.enqueue_function[q8dot](
                Aq, As, qtw[0], qtw[1], Cp, Int32(MR), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS
            )
            ctx.synchronize()

    var durs = InlineArray[Float64, ITERS](uninitialized=True)
    for it in range(ITERS):
        var qtw = q_tensors(ctx, q_dev, it % NBUF)
        var t0 = perf_counter_ns()
        ctx.enqueue_function[q8dot](
            Aq, As, qtw[0], qtw[1], Cp, Int32(MR), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS
        )
        ctx.synchronize()
        durs[it] = Float64(perf_counter_ns() - t0) / 1.0e3

    var us = median_us(durs)
    var gbps = Float64(QBYTES) / (us * 1.0e-6) / 1.0e9
    var ratio = us / base_us if base_us > 0 else 1.0
    print(
        "DOT MR=" + String(MR), "m=" + String(MR), "us=" + String(us), "GBps=" + String(gbps),
        "ratio_vs_bf16_m1=" + String(ratio), "max_rel=" + String(worst), "correct=true",
    )
    return us


def run_q4_arm(
    ctx: DeviceContext,
    a_host: HostBuffer[bf16],
    q4_host: HostBuffer[u8],
    q4_dev: DeviceBuffer[u8],
    q8row_us1: Float64,
) raises -> Float64:
    comptime MR = 1
    comptime UNROLL = 2
    comptime a_layout = row_major[MR, K]()
    comptime q4row = amar_matmul_skinny_q4row[
        UNROLL, MR, type_of(a_layout), type_of(q4_layout), type_of(s_layout), type_of(p_layout)
    ]

    var a_host_mr = ctx.enqueue_create_host_buffer[bf16](MR * K)
    ctx.synchronize()
    for i in range(MR * K):
        a_host_mr[i] = a_host[i]
    var a_dev = ctx.enqueue_create_buffer[bf16](MR * K)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host_mr)
    var p_dev = ctx.enqueue_create_buffer[f32](SPLITK * SM * N)
    var p_host = ctx.enqueue_create_host_buffer[f32](SPLITK * SM * N)
    ctx.synchronize()

    var A = TileTensor(a_dev, a_layout)
    var Cp = TileTensor(p_dev, p_layout)

    print(
        "MR=" + String(MR) + "(q4)", "UNROLL=" + String(UNROLL), "grid_dim=" + String(GR),
        "block_dim=" + String(ROW_THREADS), "ROW_WAVES=" + String(ROW_WAVES),
    )

    var qt0 = q4_tensors(ctx, q4_dev, 0)
    ctx.enqueue_function[q4row](
        A, qt0[0], qt0[1], Cp, Int32(MR), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS
    )
    ctx.enqueue_copy(dst_buf=p_host, src_buf=p_dev)
    ctx.synchronize()

    var qp = q4_host.unsafe_ptr()
    var sp = (q4_host.unsafe_ptr() + N * (K // 2)).unsafe_bitcast[Float16]()
    var worst: Float64 = 0
    for j in range(N):
        var s: Float64 = 0
        for k in range(K):
            var blk = k // 32
            var byte = qp[j * (K // 2) + blk * 16 + (k % 32) % 16]
            var nib = (byte & UInt8(0xF)) if (k % 32) < 16 else (byte >> UInt8(4))
            var d = Float64(sp[j * (K // 32) + blk])
            s += (Float64(Int(nib)) - 8.0) * d * Float64(a_host_mr[k])
        var rel = abs(Float64(p_host[j]) - s) / max(abs(s), 1e-3)
        if rel > worst:
            worst = rel

    if worst > 1e-3:
        print("MR=1(q4)", "m=1", "FAIL max_rel=" + String(worst))
        return -1.0

    var w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
        for b in range(NBUF):
            var qtw = q4_tensors(ctx, q4_dev, b)
            ctx.enqueue_function[q4row](
                A, qtw[0], qtw[1], Cp, Int32(MR), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS
            )
            ctx.synchronize()

    var durs = InlineArray[Float64, ITERS](uninitialized=True)
    for it in range(ITERS):
        var qtw = q4_tensors(ctx, q4_dev, it % NBUF)
        var t0 = perf_counter_ns()
        ctx.enqueue_function[q4row](
            A, qtw[0], qtw[1], Cp, Int32(MR), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS
        )
        ctx.synchronize()
        durs[it] = Float64(perf_counter_ns() - t0) / 1.0e3

    var us = median_us(durs)
    var gbps = Float64(QBYTES4) / (us * 1.0e-6) / 1.0e9
    var ratio = us / q8row_us1 if q8row_us1 > 0 else 0.0
    print(
        "MR=1(q4)", "m=1", "us=" + String(us), "GBps=" + String(gbps),
        "ratio_vs_q8row_MR1=" + String(ratio), "correct=true",
    )
    return us


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    comptime base = ".work/gguf/blk_0_ffn_gate_weight"

    var a_host = ctx.enqueue_create_host_buffer[bf16](MAXM * K)
    var q_host = ctx.enqueue_create_host_buffer[u8](QBYTES)
    ctx.synchronize()
    load_into(base + ".a.bin", a_host.unsafe_ptr().unsafe_bitcast[UInt8](), MAXM * K * 2)
    var qoff = pack_offset("blk.0.ffn_gate.weight")
    load_into(".work/engine-pack-q8/pack.bin", q_host.unsafe_ptr().unsafe_bitcast[UInt8](), QBYTES, qoff)

    var q_dev = ctx.enqueue_create_buffer[u8](NBUF * QBYTES)
    var qdp = q_dev.unsafe_ptr().unsafe_bitcast[UInt8]()
    for b in range(NBUF):
        var qb = DeviceBuffer[u8](ctx, qdp + b * QBYTES, QBYTES, owning=False)
        ctx.enqueue_copy(dst_buf=qb, src_buf=q_host)
    ctx.synchronize()

    var us1 = run_arm[1](ctx, a_host, q_host, q_dev, 0.0)
    _ = run_arm[2](ctx, a_host, q_host, q_dev, us1)
    _ = run_arm[4](ctx, a_host, q_host, q_dev, us1)
    _ = run_arm[8](ctx, a_host, q_host, q_dev, us1)

    _ = run_dot_arm[1](ctx, a_host, q_host, q_dev, us1)
    _ = run_dot_arm[2](ctx, a_host, q_host, q_dev, us1)
    _ = run_dot_arm[4](ctx, a_host, q_host, q_dev, us1)
    _ = run_dot_arm[8](ctx, a_host, q_host, q_dev, us1)

    var q4_host = ctx.enqueue_create_host_buffer[u8](QBYTES4)
    ctx.synchronize()
    load_into(base + ".q4.bin", q4_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * (K // 2))
    load_into(base + ".q4scales.bin", (q4_host.unsafe_ptr() + N * (K // 2)).unsafe_bitcast[UInt8](), N * (K // 32) * 2)

    var q4_dev = ctx.enqueue_create_buffer[u8](NBUF * QBYTES4)
    var q4dp = q4_dev.unsafe_ptr()
    for b in range(NBUF):
        var qb = DeviceBuffer[u8](ctx, q4dp + b * QBYTES4, QBYTES4, owning=False)
        ctx.enqueue_copy(dst_buf=qb, src_buf=q4_host)
    ctx.synchronize()

    _ = run_q4_arm(ctx, a_host, q4_host, q4_dev, us1)
