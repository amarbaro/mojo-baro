"""Milestone-6 checks: int8 dequant-in-kernel GEMM + fused SwiGLU epilogue.

Consumes .work/gguf/ from:
  tools/gguf-extract.py <model> blk.0.ffn_gate.weight blk.0.ffn_up.weight --ref 8 --q8

Checks, in order:
1. skinny_q8 parity on real quantized gate/up weights vs the numpy fp32
   reference computed from the SAME dequantized values (gate: rel < 1e-2).
2. Fused skinny_reduce_swiglu vs the unfused three-launch path
   (reduce, reduce, swiglu) from identical partials: must agree to fp32
   round-off (both sum in the same order; tolerance 1e-6 rel).
3. Timing at the real FFN shape (M=8, K=4096, N=12288): bf16 vs q8 weight
   stream. bf16 W is 100 MB (> 96 MB Infinity Cache), q8 is 56 MB (fits),
   so this is the first shape where quantization must visibly pay.
"""
from std.math import ceildiv, exp
from std.memory import alloc
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from elementwise import swiglu
from matmul_skinny import (
    matmul_skinny_wt, matmul_skinny_q8, matmul_skinny_q8b,
    skinny_reduce, skinny_reduce_swiglu,
    SM, SBN, SPLITK, SK_THREADS, Q8_BLOCK,
)

comptime M = 8
comptime K = 4096
comptime N = 12288
comptime ITERS = 200

comptime a_layout = row_major[M, K]()
comptime w_layout = row_major[N, K]()
comptime s_layout = row_major[N, K // Q8_BLOCK]()
comptime qt_layout = row_major[K, N]()
comptime st_layout = row_major[K // Q8_BLOCK, N]()
comptime c_layout = row_major[M, N]()
comptime cf_layout = row_major[M * N]()
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

    comptime base_g = ".work/gguf/blk_0_ffn_gate_weight"
    comptime base_u = ".work/gguf/blk_0_ffn_up_weight"

    var a_host = ctx.enqueue_create_host_buffer[bf16](M * K)
    var wg_host = ctx.enqueue_create_host_buffer[bf16](N * K)
    var wu_host = ctx.enqueue_create_host_buffer[bf16](N * K)
    var qg_host = ctx.enqueue_create_host_buffer[DType.int8](N * K)
    var qu_host = ctx.enqueue_create_host_buffer[DType.int8](N * K)
    var sg_host = ctx.enqueue_create_host_buffer[f32](N * K // Q8_BLOCK)
    var su_host = ctx.enqueue_create_host_buffer[f32](N * K // Q8_BLOCK)
    var c_host = ctx.enqueue_create_host_buffer[f32](M * N)
    var c2_host = ctx.enqueue_create_host_buffer[f32](M * N)
    var ref_g = alloc[Float32](M * N)
    var ref_u = alloc[Float32](M * N)
    ctx.synchronize()

    load_into(base_g + ".a.bin", a_host.unsafe_ptr().unsafe_bitcast[UInt8](), M * K * 2)
    load_into(base_g + ".bin", wg_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K * 2)
    load_into(base_u + ".bin", wu_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K * 2)
    load_into(base_g + ".q.bin", qg_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K)
    load_into(base_u + ".q.bin", qu_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K)
    load_into(base_g + ".scales.bin", sg_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K // Q8_BLOCK * 4)
    load_into(base_u + ".scales.bin", su_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K // Q8_BLOCK * 4)
    load_into(base_g + ".cq.bin", ref_g.unsafe_bitcast[UInt8](), M * N * 4)
    load_into(base_u + ".cq.bin", ref_u.unsafe_bitcast[UInt8](), M * N * 4)

    var a_dev = ctx.enqueue_create_buffer[bf16](M * K)
    var wg_dev = ctx.enqueue_create_buffer[bf16](N * K)
    var qg_dev = ctx.enqueue_create_buffer[DType.int8](N * K)
    var qu_dev = ctx.enqueue_create_buffer[DType.int8](N * K)
    var sg_dev = ctx.enqueue_create_buffer[f32](N * K // Q8_BLOCK)
    var su_dev = ctx.enqueue_create_buffer[f32](N * K // Q8_BLOCK)
    var c_dev = ctx.enqueue_create_buffer[f32](M * N)
    var c2_dev = ctx.enqueue_create_buffer[f32](M * N)
    var pg_dev = ctx.enqueue_create_buffer[f32](SPLITK * SM * N)
    var pu_dev = ctx.enqueue_create_buffer[f32](SPLITK * SM * N)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    ctx.enqueue_copy(dst_buf=wg_dev, src_buf=wg_host)
    ctx.enqueue_copy(dst_buf=qg_dev, src_buf=qg_host)
    ctx.enqueue_copy(dst_buf=qu_dev, src_buf=qu_host)
    ctx.enqueue_copy(dst_buf=sg_dev, src_buf=sg_host)
    ctx.enqueue_copy(dst_buf=su_dev, src_buf=su_host)
    ctx.synchronize()

    var A = TileTensor(a_dev, a_layout)
    var Wg = TileTensor(wg_dev, w_layout)
    var Qg = TileTensor(qg_dev, w_layout)
    var Qu = TileTensor(qu_dev, w_layout)
    var Sg = TileTensor(sg_dev, s_layout)
    var Su = TileTensor(su_dev, s_layout)
    var C = TileTensor(c_dev, c_layout)
    var C2 = TileTensor(c2_dev, c_layout)
    var Pg = TileTensor(pg_dev, p_layout)
    var Pu = TileTensor(pu_dev, p_layout)

    comptime skinny_wt = matmul_skinny_wt[
        bf16, type_of(a_layout), type_of(w_layout), type_of(p_layout)
    ]
    comptime skinny_q8 = matmul_skinny_q8[
        type_of(a_layout), type_of(w_layout), type_of(s_layout), type_of(p_layout)
    ]
    comptime reduce = skinny_reduce[type_of(p_layout), type_of(c_layout)]
    comptime reduce_swiglu = skinny_reduce_swiglu[
        type_of(p_layout), type_of(c_layout)
    ]
    comptime swiglu_k = swiglu[
        type_of(cf_layout), type_of(cf_layout), type_of(cf_layout)
    ]

    comptime SGRID = (ceildiv(N, SBN), SPLITK)
    comptime RED_GRID = ceildiv(M * N, 256)

    # --- 1. q8 parity, gate and up ---
    ctx.enqueue_function[skinny_q8](
        A, Qg, Sg, Pg, Int32(M), Int32(N), Int32(K),
        grid_dim=SGRID, block_dim=SK_THREADS,
    )
    ctx.enqueue_function[reduce](
        Pg, C, Int32(M), Int32(N), grid_dim=RED_GRID, block_dim=256
    )
    ctx.enqueue_function[skinny_q8](
        A, Qu, Su, Pu, Int32(M), Int32(N), Int32(K),
        grid_dim=SGRID, block_dim=SK_THREADS,
    )
    ctx.enqueue_function[reduce](
        Pu, C2, Int32(M), Int32(N), grid_dim=RED_GRID, block_dim=256
    )
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.enqueue_copy(dst_buf=c2_host, src_buf=c2_dev)
    ctx.synchronize()
    var worst = Float64(0)
    for i in range(M * N):
        var eg = abs(Float64(c_host[i]) - Float64(ref_g[unsafe_offset=i])) / (
            abs(Float64(ref_g[unsafe_offset=i])) + 1e-3
        )
        var eu = abs(Float64(c2_host[i]) - Float64(ref_u[unsafe_offset=i])) / (
            abs(Float64(ref_u[unsafe_offset=i])) + 1e-3
        )
        worst = max(worst, max(eg, eu))
    print("q8 parity max_rel:", worst)
    if worst > 1e-2:
        raise Error("q8 parity failure")

    # --- 1b. q8b (K-major) parity on gate ---
    var qtg_host = ctx.enqueue_create_host_buffer[DType.int8](N * K)
    var stg_host = ctx.enqueue_create_host_buffer[f32](N * K // Q8_BLOCK)
    ctx.synchronize()
    load_into(base_g + ".qt.bin", qtg_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K)
    load_into(base_g + ".scales_t.bin", stg_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K // Q8_BLOCK * 4)
    var qtg_dev = ctx.enqueue_create_buffer[DType.int8](N * K)
    var stg_dev = ctx.enqueue_create_buffer[f32](N * K // Q8_BLOCK)
    ctx.enqueue_copy(dst_buf=qtg_dev, src_buf=qtg_host)
    ctx.enqueue_copy(dst_buf=stg_dev, src_buf=stg_host)
    ctx.synchronize()
    var Qtg = TileTensor(qtg_dev, qt_layout)
    var Stg = TileTensor(stg_dev, st_layout)
    comptime skinny_q8b = matmul_skinny_q8b[
        type_of(a_layout), type_of(qt_layout), type_of(st_layout), type_of(p_layout)
    ]
    ctx.enqueue_function[skinny_q8b](
        A, Qtg, Stg, Pg, Int32(M), Int32(N), Int32(K),
        grid_dim=SGRID, block_dim=SK_THREADS,
    )
    ctx.enqueue_function[reduce](
        Pg, C, Int32(M), Int32(N), grid_dim=RED_GRID, block_dim=256
    )
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.synchronize()
    worst = 0
    for i in range(M * N):
        var e = abs(Float64(c_host[i]) - Float64(ref_g[unsafe_offset=i])) / (
            abs(Float64(ref_g[unsafe_offset=i])) + 1e-3
        )
        worst = max(worst, e)
    print("q8b (K-major) parity max_rel:", worst)
    if worst > 1e-2:
        raise Error("q8b parity failure")
    ctx.enqueue_function[skinny_q8](
        A, Qg, Sg, Pg, Int32(M), Int32(N), Int32(K),
        grid_dim=SGRID, block_dim=SK_THREADS,
    )
    ctx.synchronize()

    # --- 2. fused swiglu epilogue vs unfused three-launch path ---
    ctx.enqueue_function[reduce_swiglu](
        Pg, Pu, C, Int32(M), Int32(N), grid_dim=RED_GRID, block_dim=256
    )
    # Unfused: reduce into two flat tensors, then elementwise swiglu.
    ctx.enqueue_function[reduce](
        Pg, C2, Int32(M), Int32(N), grid_dim=RED_GRID, block_dim=256
    )
    var c3_dev = ctx.enqueue_create_buffer[f32](M * N)
    var C3 = TileTensor(c3_dev, c_layout)
    ctx.enqueue_function[reduce](
        Pu, C3, Int32(M), Int32(N), grid_dim=RED_GRID, block_dim=256
    )
    var c4_dev = ctx.enqueue_create_buffer[f32](M * N)
    var C2f = TileTensor(c2_dev, cf_layout)
    var C3f = TileTensor(c3_dev, cf_layout)
    var C4f = TileTensor(c4_dev, cf_layout)
    ctx.enqueue_function[swiglu_k](
        C2f, C3f, C4f, Int32(M * N), grid_dim=RED_GRID, block_dim=256
    )
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.enqueue_copy(dst_buf=c2_host, src_buf=c4_dev)
    ctx.synchronize()
    worst = 0
    for i in range(M * N):
        var err = abs(Float64(c_host[i]) - Float64(c2_host[i])) / (
            abs(Float64(c2_host[i])) + 1e-6
        )
        worst = max(worst, err)
    print("fused swiglu vs unfused max_rel:", worst)
    if worst > 1e-6:
        raise Error("fused swiglu mismatch")

    # --- 3. timing: bf16 vs q8 weight stream at the real FFN shape ---
    comptime FLOPS = 2.0 * Float64(M) * Float64(N) * Float64(K)
    var w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
        for _ in range(20):
            ctx.enqueue_function[skinny_wt](
                A, Wg, Pg, Int32(M), Int32(N), Int32(K),
                grid_dim=SGRID, block_dim=SK_THREADS,
            )
        ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[skinny_wt](
            A, Wg, Pg, Int32(M), Int32(N), Int32(K),
            grid_dim=SGRID, block_dim=SK_THREADS,
        )
    ctx.synchronize()
    var bf16_ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)

    w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
        for _ in range(20):
            ctx.enqueue_function[skinny_q8](
                A, Qg, Sg, Pg, Int32(M), Int32(N), Int32(K),
                grid_dim=SGRID, block_dim=SK_THREADS,
            )
        ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[skinny_q8](
            A, Qg, Sg, Pg, Int32(M), Int32(N), Int32(K),
            grid_dim=SGRID, block_dim=SK_THREADS,
        )
    ctx.synchronize()
    var q8_ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)

    print(
        "bf16:", bf16_ms, "ms", FLOPS / (bf16_ms * 1.0e6) / 1e3, "TFLOP/s |",
        "q8:", q8_ms, "ms", FLOPS / (q8_ms * 1.0e6) / 1e3, "TFLOP/s |",
        "speedup:", bf16_ms / q8_ms,
    )
    print("PASS: q8 dequant-in-kernel + fused swiglu verified")
