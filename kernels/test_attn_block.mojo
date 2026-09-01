"""Parity: one decode token through the qwen35 gated full-attention block
(docs/qwen35-ssm-notes.md §7b) vs tools/attn-ref.py. Position 7 with 7 cached
tokens. Gates mirror test_ssm_block: exact-path intermediates at 1e-3, values
crossing a bf16 cast at wider gates (boundary flips, documented there).
"""
from std.math import ceildiv
from std.memory import alloc
from std.sys import has_accelerator

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from elementwise import rmsnorm
from matmul_skinny import matmul_skinny, skinny_reduce, SM, SBN, SPLITK, SK_THREADS
from ssm import cast_bf16, residual_add
from attn import (
    head_rmsnorm, attn_decode, gate_mul, qgate_split, rope_yarn, kv_append,
    HD, NQH, NKVH,
)

comptime H = 4096
comptime QF = 2 * H
comptime KV = NKVH * HD
comptime T_PRE = 7
comptime T = T_PRE + 1
comptime POS = 7
comptime bf16 = DType.bfloat16
comptime f32 = DType.float32

comptime x2_layout = row_major[1, H]()
comptime h_layout = row_major[H]()
comptime qf_layout = row_major[QF]()
comptime q_layout = row_major[NQH, HD]()
comptime kv_layout = row_major[NKVH, HD]()
comptime kvflat_layout = row_major[KV]()
comptime cache_layout = row_major[NKVH, T, HD]()
comptime hd_layout = row_major[HD]()

comptime wq_layout = row_major[H, QF]()
comptime wkv_layout = row_major[H, KV]()
comptime wo_layout = row_major[H, H]()
comptime p_qf = row_major[SPLITK, SM, QF]()
comptime p_kv = row_major[SPLITK, SM, KV]()
comptime p_h = row_major[SPLITK, SM, H]()
comptime c_qf = row_major[1, QF]()
comptime c_kv = row_major[1, KV]()
comptime c_h = row_major[1, H]()


def load_into(
    path: String, dst: MutPointer[UInt8, MutUntrackedOrigin], size: Int
) raises:
    with open(path, "r") as f:
        var data = f.read_bytes()
        if len(data) != size:
            raise Error("size mismatch for " + path)
        for i in range(size):
            dst[unsafe_offset=i] = data[i]


def check(name: String, got: MutPointer[Float32, MutUntrackedOrigin],
          want: MutPointer[Float32, MutUntrackedOrigin], n: Int,
          gate: Float64 = 1e-3) raises:
    var worst = Float64(0)
    for i in range(n):
        var e = abs(Float64(got[unsafe_offset=i]) - Float64(want[unsafe_offset=i]))
        var rel = e / (abs(Float64(want[unsafe_offset=i])) + 1e-2)
        if rel > worst:
            worst = rel
    print(name, "max_rel:", worst)
    if worst > gate:
        raise Error("parity failure: " + name)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    comptime D = ".work/gguf/"

    var x_h = ctx.enqueue_create_host_buffer[f32](H)
    var kc_h = ctx.enqueue_create_host_buffer[f32](NKVH * T_PRE * HD)
    var vc_h = ctx.enqueue_create_host_buffer[f32](NKVH * T_PRE * HD)
    var qn_h = ctx.enqueue_create_host_buffer[f32](HD)
    var kn_h = ctx.enqueue_create_host_buffer[f32](HD)
    var an_h = ctx.enqueue_create_host_buffer[f32](H)
    var wq_h = ctx.enqueue_create_host_buffer[bf16](H * QF)
    var wk_h = ctx.enqueue_create_host_buffer[bf16](H * KV)
    var wv_h = ctx.enqueue_create_host_buffer[bf16](H * KV)
    var wo_h = ctx.enqueue_create_host_buffer[bf16](H * H)
    ctx.synchronize()

    load_into(D + "attn_x.bin", x_h.unsafe_ptr().unsafe_bitcast[UInt8](), H * 4)
    load_into(D + "attn_kcache.bin", kc_h.unsafe_ptr().unsafe_bitcast[UInt8](), NKVH * T_PRE * HD * 4)
    load_into(D + "attn_vcache.bin", vc_h.unsafe_ptr().unsafe_bitcast[UInt8](), NKVH * T_PRE * HD * 4)
    load_into(D + "blk_3_attn_q_norm_weight.bin", qn_h.unsafe_ptr().unsafe_bitcast[UInt8](), HD * 4)
    load_into(D + "blk_3_attn_k_norm_weight.bin", kn_h.unsafe_ptr().unsafe_bitcast[UInt8](), HD * 4)
    load_into(D + "blk_3_attn_norm_weight.bin", an_h.unsafe_ptr().unsafe_bitcast[UInt8](), H * 4)
    load_into(D + "blk_3_attn_q_weight.t.bin", wq_h.unsafe_ptr().unsafe_bitcast[UInt8](), H * QF * 2)
    load_into(D + "blk_3_attn_k_weight.t.bin", wk_h.unsafe_ptr().unsafe_bitcast[UInt8](), H * KV * 2)
    load_into(D + "blk_3_attn_v_weight.t.bin", wv_h.unsafe_ptr().unsafe_bitcast[UInt8](), H * KV * 2)
    load_into(D + "blk_3_attn_output_weight.t.bin", wo_h.unsafe_ptr().unsafe_bitcast[UInt8](), H * H * 2)

    var y_ref = alloc[Float32](H)
    var knew_ref = alloc[Float32](NKVH * HD)
    var vnew_ref = alloc[Float32](NKVH * HD)
    load_into(D + "attn_y_ref.bin", y_ref.unsafe_bitcast[UInt8](), H * 4)
    load_into(D + "attn_knew_ref.bin", knew_ref.unsafe_bitcast[UInt8](), NKVH * HD * 4)
    load_into(D + "attn_vnew_ref.bin", vnew_ref.unsafe_bitcast[UInt8](), NKVH * HD * 4)

    var x_d = ctx.enqueue_create_buffer[f32](H)
    var cur_d = ctx.enqueue_create_buffer[f32](H)
    var curb_d = ctx.enqueue_create_buffer[bf16](H)
    var qn_d = ctx.enqueue_create_buffer[f32](HD)
    var kn_d = ctx.enqueue_create_buffer[f32](HD)
    var an_d = ctx.enqueue_create_buffer[f32](H)
    var wq_d = ctx.enqueue_create_buffer[bf16](H * QF)
    var wk_d = ctx.enqueue_create_buffer[bf16](H * KV)
    var wv_d = ctx.enqueue_create_buffer[bf16](H * KV)
    var wo_d = ctx.enqueue_create_buffer[bf16](H * H)
    var qf_d = ctx.enqueue_create_buffer[f32](QF)
    var q_d = ctx.enqueue_create_buffer[f32](NQH * HD)
    var gate_d = ctx.enqueue_create_buffer[f32](H)
    var k_d = ctx.enqueue_create_buffer[f32](KV)
    var v_d = ctx.enqueue_create_buffer[f32](KV)
    var kc_d = ctx.enqueue_create_buffer[f32](NKVH * T * HD)
    var vc_d = ctx.enqueue_create_buffer[f32](NKVH * T * HD)
    var o_d = ctx.enqueue_create_buffer[f32](NQH * HD)
    var ob_d = ctx.enqueue_create_buffer[bf16](H)
    var out_d = ctx.enqueue_create_buffer[f32](H)
    var p_qf_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * QF)
    var p_kv_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * KV)
    var p_h_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * H)

    ctx.enqueue_copy(dst_buf=x_d, src_buf=x_h)
    ctx.enqueue_copy(dst_buf=qn_d, src_buf=qn_h)
    ctx.enqueue_copy(dst_buf=kn_d, src_buf=kn_h)
    ctx.enqueue_copy(dst_buf=an_d, src_buf=an_h)
    ctx.enqueue_copy(dst_buf=wq_d, src_buf=wq_h)
    ctx.enqueue_copy(dst_buf=wk_d, src_buf=wk_h)
    ctx.enqueue_copy(dst_buf=wv_d, src_buf=wv_h)
    ctx.enqueue_copy(dst_buf=wo_d, src_buf=wo_h)
    ctx.synchronize()
    var kcp = kc_d.unsafe_ptr()
    var vcp = vc_d.unsafe_ptr()
    for h in range(NKVH):
        for t in range(T_PRE):
            for d in range(HD):
                kcp[unsafe_offset = h * T * HD + t * HD + d] = kc_h[h * T_PRE * HD + t * HD + d]
                vcp[unsafe_offset = h * T * HD + t * HD + d] = vc_h[h * T_PRE * HD + t * HD + d]

    var X2 = TileTensor(x_d, x2_layout)
    var X1 = TileTensor(x_d, h_layout)
    var Cur2 = TileTensor(cur_d, x2_layout)
    var Cur1 = TileTensor(cur_d, h_layout)
    var CurB1 = TileTensor(curb_d, h_layout)
    var CurB2 = TileTensor(curb_d, row_major[1, H]())
    var An = TileTensor(an_d, h_layout)
    var Qn = TileTensor(qn_d, hd_layout)
    var Kn = TileTensor(kn_d, hd_layout)
    var Wq = TileTensor(wq_d, wq_layout)
    var Wk = TileTensor(wk_d, wkv_layout)
    var Wv = TileTensor(wv_d, wkv_layout)
    var Wo = TileTensor(wo_d, wo_layout)
    var Pqf = TileTensor(p_qf_d, p_qf)
    var Pkv = TileTensor(p_kv_d, p_kv)
    var Ph = TileTensor(p_h_d, p_h)
    var Qf1 = TileTensor(qf_d, qf_layout)
    var Qf2 = TileTensor(qf_d, c_qf)
    var Q = TileTensor(q_d, q_layout)
    var Gate1 = TileTensor(gate_d, h_layout)
    var K1 = TileTensor(k_d, kvflat_layout)
    var K2c = TileTensor(k_d, c_kv)
    var Khd = TileTensor(k_d, kv_layout)
    var V2c = TileTensor(v_d, c_kv)
    var Vhd = TileTensor(v_d, kv_layout)
    var Kc = TileTensor(kc_d, cache_layout)
    var Vc = TileTensor(vc_d, cache_layout)
    var O = TileTensor(o_d, q_layout)
    var O1 = TileTensor(o_d, h_layout)
    var ObB1 = TileTensor(ob_d, h_layout)
    var ObB2 = TileTensor(ob_d, row_major[1, H]())
    var Out1 = TileTensor(out_d, h_layout)
    var Out2 = TileTensor(out_d, c_h)

    comptime rms_k = rmsnorm[type_of(x2_layout), type_of(h_layout), type_of(x2_layout)]
    comptime cast_k = cast_bf16[type_of(h_layout), type_of(h_layout)]
    comptime g_qf = matmul_skinny[bf16, type_of(row_major[1, H]()), type_of(wq_layout), type_of(p_qf)]
    comptime g_kv = matmul_skinny[bf16, type_of(row_major[1, H]()), type_of(wkv_layout), type_of(p_kv)]
    comptime g_h = matmul_skinny[bf16, type_of(row_major[1, H]()), type_of(wo_layout), type_of(p_h)]
    comptime r_qf = skinny_reduce[type_of(p_qf), type_of(c_qf)]
    comptime r_kv = skinny_reduce[type_of(p_kv), type_of(c_kv)]
    comptime r_h = skinny_reduce[type_of(p_h), type_of(c_h)]
    comptime split_k = qgate_split[type_of(c_qf), type_of(q_layout), type_of(h_layout)]
    comptime hrms_q = head_rmsnorm[type_of(q_layout), type_of(hd_layout)]
    comptime hrms_kv = head_rmsnorm[type_of(kv_layout), type_of(hd_layout)]
    comptime rope_q = rope_yarn[type_of(q_layout)]
    comptime rope_k = rope_yarn[type_of(kv_layout)]
    comptime append_k = kv_append[type_of(cache_layout), type_of(kv_layout)]
    comptime att_k = attn_decode[type_of(q_layout), type_of(cache_layout), type_of(q_layout)]
    comptime gmul_k = gate_mul[type_of(h_layout), type_of(h_layout)]
    comptime add_k = residual_add[type_of(h_layout), type_of(h_layout)]

    ctx.enqueue_function[rms_k](X2, An, Cur2, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
    ctx.enqueue_function[cast_k](Cur1, CurB1, Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.enqueue_function[g_qf](CurB2, Wq, Pqf, Int32(1), Int32(QF), Int32(H), grid_dim=(ceildiv(QF, SBN), SPLITK), block_dim=SK_THREADS)
    ctx.enqueue_function[r_qf](Pqf, Qf2, Int32(1), Int32(QF), grid_dim=ceildiv(QF, 256), block_dim=256)
    ctx.enqueue_function[g_kv](CurB2, Wk, Pkv, Int32(1), Int32(KV), Int32(H), grid_dim=(ceildiv(KV, SBN), SPLITK), block_dim=SK_THREADS)
    ctx.enqueue_function[r_kv](Pkv, K2c, Int32(1), Int32(KV), grid_dim=ceildiv(KV, 256), block_dim=256)
    ctx.enqueue_function[g_kv](CurB2, Wv, Pkv, Int32(1), Int32(KV), Int32(H), grid_dim=(ceildiv(KV, SBN), SPLITK), block_dim=SK_THREADS)
    ctx.enqueue_function[r_kv](Pkv, V2c, Int32(1), Int32(KV), grid_dim=ceildiv(KV, 256), block_dim=256)
    ctx.enqueue_function[split_k](Qf2, Q, Gate1, grid_dim=NQH, block_dim=HD)
    ctx.enqueue_function[hrms_q](Q, Qn, Float32(1e-6), grid_dim=NQH, block_dim=HD)
    ctx.enqueue_function[hrms_kv](Khd, Kn, Float32(1e-6), grid_dim=NKVH, block_dim=HD)
    ctx.enqueue_function[rope_q](Q, Int32(POS), Int32(NQH), grid_dim=NQH, block_dim=32)
    ctx.enqueue_function[rope_k](Khd, Int32(POS), Int32(NKVH), grid_dim=NKVH, block_dim=32)
    ctx.enqueue_function[append_k](Kc, Khd, Int32(T_PRE), grid_dim=NKVH, block_dim=HD)
    ctx.enqueue_function[append_k](Vc, Vhd, Int32(T_PRE), grid_dim=NKVH, block_dim=HD)
    ctx.enqueue_function[att_k](Q, Kc, Vc, O, Int32(T), Float32(0.0625), grid_dim=NQH, block_dim=HD)
    ctx.enqueue_function[gmul_k](O1, Gate1, Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.enqueue_function[cast_k](O1, ObB1, Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.enqueue_function[g_h](ObB2, Wo, Ph, Int32(1), Int32(H), Int32(H), grid_dim=(ceildiv(H, SBN), SPLITK), block_dim=SK_THREADS)
    ctx.enqueue_function[r_h](Ph, Out2, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.enqueue_function[add_k](X1, Out1, Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.synchronize()

    var y_got = ctx.enqueue_create_host_buffer[f32](H)
    var k_got = ctx.enqueue_create_host_buffer[f32](KV)
    var v_got = ctx.enqueue_create_host_buffer[f32](KV)
    ctx.enqueue_copy(dst_buf=y_got, src_buf=out_d)
    ctx.enqueue_copy(dst_buf=k_got, src_buf=k_d)
    ctx.enqueue_copy(dst_buf=v_got, src_buf=v_d)
    ctx.synchronize()

    check("k_new", k_got.unsafe_ptr(), knew_ref, KV)
    check("v_new", v_got.unsafe_ptr(), vnew_ref, KV)
    check("y", y_got.unsafe_ptr(), y_ref, H, 3e-2)
    print("PASS: qwen35 gated full-attention block matches numpy reference")
