"""Parity check: real Qwythos bf16 weight through skinny_wt vs numpy reference.

Consumes .work/gguf/ produced by:
  tools/gguf-extract.py <model.gguf> blk.0.ffn_up.weight --ref 8

W is blk.0.ffn_up.weight [12288, 4096] bf16 in GGUF-native [out, in] layout;
A is a fixed random bf16 [8, 4096]; the reference C = A @ W^T was computed in
fp32 by numpy. Gate: max relative error < 1e-2 over all 8 x 12288 outputs
(both sides accumulate in fp32 from identical bf16 inputs, so the real
disagreement is only summation order).
"""
from std.math import ceildiv
from std.memory import alloc
from std.sys import has_accelerator

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from matmul_skinny import matmul_skinny, skinny_reduce, SM, SBN, SPLITK, SK_THREADS

comptime M = 8
comptime K = 4096
comptime N = 12288

comptime a_layout = row_major[M, K]()
comptime w_layout = row_major[K, N]()
comptime c_layout = row_major[M, N]()
comptime p_layout = row_major[SPLITK, SM, N]()

comptime bf16 = DType.bfloat16


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

    var a_host = ctx.enqueue_create_host_buffer[bf16](M * K)
    var w_host = ctx.enqueue_create_host_buffer[bf16](N * K)
    var c_host = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    var cref = alloc[Float32](M * N)
    ctx.synchronize()

    load_into(
        ".work/gguf/blk_0_ffn_up_weight.a.bin",
        a_host.unsafe_ptr().unsafe_bitcast[UInt8](), M * K * 2,
    )
    load_into(
        ".work/gguf/blk_0_ffn_up_weight.t.bin",
        w_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * K * 2,
    )
    load_into(
        ".work/gguf/blk_0_ffn_up_weight.c.bin",
        cref.unsafe_bitcast[UInt8](), M * N * 4,
    )

    var a_dev = ctx.enqueue_create_buffer[bf16](M * K)
    var w_dev = ctx.enqueue_create_buffer[bf16](N * K)
    var c_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    var p_dev = ctx.enqueue_create_buffer[DType.float32](SPLITK * SM * N)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    ctx.enqueue_copy(dst_buf=w_dev, src_buf=w_host)
    ctx.synchronize()

    var A = TileTensor(a_dev, a_layout)
    var W = TileTensor(w_dev, w_layout)
    var C = TileTensor(c_dev, c_layout)
    var Cp = TileTensor(p_dev, p_layout)

    comptime skinny_wt = matmul_skinny[
        bf16, type_of(a_layout), type_of(w_layout), type_of(p_layout)
    ]
    comptime reduce = skinny_reduce[type_of(p_layout), type_of(c_layout)]

    ctx.enqueue_function[skinny_wt](
        A, W, Cp, Int32(M), Int32(N), Int32(K),
        grid_dim=(ceildiv(N, SBN), SPLITK), block_dim=SK_THREADS,
    )
    ctx.enqueue_function[reduce](
        Cp, C, Int32(M), Int32(N),
        grid_dim=ceildiv(M * N, 256), block_dim=256,
    )
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.synchronize()

    var worst_rel = Float64(0)
    var worst_abs = Float64(0)
    for i in range(M * N):
        var got = Float64(c_host[i])
        var want = Float64(cref[unsafe_offset=i])
        var err = abs(got - want)
        var rel = err / (abs(want) + 1e-3)
        if err > worst_abs:
            worst_abs = err
        if rel > worst_rel:
            worst_rel = rel
    print("max_abs_err:", worst_abs, " max_rel_err:", worst_rel)
    if worst_rel < 1e-2:
        print("PASS: real bf16 GGUF weight matches numpy fp32 reference")
    else:
        print("FAIL")
        raise Error("parity failure")
