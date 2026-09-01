"""Parity: one decode token through the qwen35 gated-delta-net block on GPU
vs the numpy reference (tools/ssm-ref.py implementing docs/qwen35-ssm-notes.md).

Gates: block output y, new recurrent state S1, and new conv window, all
rel < 1e-3 against the reference (shared bf16 weights, fp32 math both sides).
"""
from std.math import ceildiv
from std.memory import alloc
from std.sys import has_accelerator

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from elementwise import rmsnorm
from matmul_skinny import matmul_skinny, skinny_reduce, SM, SBN, SPLITK, SK_THREADS
from ssm import (
    cast_bf16, residual_add, ssm_gates, ssm_conv, ssm_qk_l2norm,
    ssm_delta_step, ssm_gated_out, CONV, NH_V, SSTATE,
)

comptime H = 4096
comptime bf16 = DType.bfloat16
comptime f32 = DType.float32

comptime x2_layout = row_major[1, H]()
comptime h_layout = row_major[H]()
comptime hb_layout = row_major[H]()
comptime conv_layout = row_major[CONV]()
comptime cs_layout = row_major[3, CONV]()
comptime cw_layout = row_major[CONV, 4]()
comptime s_layout = row_major[NH_V, SSTATE, SSTATE]()
comptime o_layout = row_major[NH_V, SSTATE]()
comptime g_layout = row_major[NH_V]()
comptime n_layout = row_major[SSTATE]()

comptime wqkv_layout = row_major[H, CONV]()
comptime wsq_layout = row_major[H, H]()
comptime wab_layout = row_major[H, NH_V]()

comptime p_qkv = row_major[SPLITK, SM, CONV]()
comptime p_h = row_major[SPLITK, SM, H]()
comptime p_ab = row_major[SPLITK, SM, NH_V]()
comptime c_qkv = row_major[1, CONV]()
comptime c_h = row_major[1, H]()
comptime c_ab = row_major[1, NH_V]()


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
    var worst_abs = Float64(0)
    var wi = 0
    for i in range(n):
        var e = abs(Float64(got[unsafe_offset=i]) - Float64(want[unsafe_offset=i]))
        if e > worst_abs:
            worst_abs = e
            wi = i
    print(
        name, "max_rel:", worst, "max_abs:", worst_abs, "at", wi,
        "got", got[unsafe_offset=wi], "want", want[unsafe_offset=wi],
    )
    if worst > gate:
        raise Error("parity failure: " + name)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    comptime D = ".work/gguf/"

    var x_h = ctx.enqueue_create_host_buffer[f32](H)
    var cs_h = ctx.enqueue_create_host_buffer[f32](3 * CONV)
    var s0_h = ctx.enqueue_create_host_buffer[f32](NH_V * SSTATE * SSTATE)
    var cw_h = ctx.enqueue_create_host_buffer[f32](CONV * 4)
    var ssma_h = ctx.enqueue_create_host_buffer[f32](NH_V)
    var dtb_h = ctx.enqueue_create_host_buffer[f32](NH_V)
    var nw_h = ctx.enqueue_create_host_buffer[f32](SSTATE)
    var an_h = ctx.enqueue_create_host_buffer[f32](H)
    var wqkv_h = ctx.enqueue_create_host_buffer[bf16](H * CONV)
    var wz_h = ctx.enqueue_create_host_buffer[bf16](H * H)
    var wa_h = ctx.enqueue_create_host_buffer[bf16](H * NH_V)
    var wb_h = ctx.enqueue_create_host_buffer[bf16](H * NH_V)
    var wo_h = ctx.enqueue_create_host_buffer[bf16](H * H)
    ctx.synchronize()

    load_into(D + "ssm_x.bin", x_h.unsafe_ptr().unsafe_bitcast[UInt8](), H * 4)
    load_into(D + "ssm_conv_state.bin", cs_h.unsafe_ptr().unsafe_bitcast[UInt8](), 3 * CONV * 4)
    load_into(D + "ssm_s0.bin", s0_h.unsafe_ptr().unsafe_bitcast[UInt8](), NH_V * SSTATE * SSTATE * 4)
    load_into(D + "blk_0_ssm_conv1d_weight.bin", cw_h.unsafe_ptr().unsafe_bitcast[UInt8](), CONV * 4 * 4)
    load_into(D + "blk_0_ssm_a.bin", ssma_h.unsafe_ptr().unsafe_bitcast[UInt8](), NH_V * 4)
    load_into(D + "blk_0_ssm_dt_bias.bin", dtb_h.unsafe_ptr().unsafe_bitcast[UInt8](), NH_V * 4)
    load_into(D + "blk_0_ssm_norm_weight.bin", nw_h.unsafe_ptr().unsafe_bitcast[UInt8](), SSTATE * 4)
    load_into(D + "blk_0_attn_norm_weight.bin", an_h.unsafe_ptr().unsafe_bitcast[UInt8](), H * 4)
    load_into(D + "blk_0_attn_qkv_weight.t.bin", wqkv_h.unsafe_ptr().unsafe_bitcast[UInt8](), H * CONV * 2)
    load_into(D + "blk_0_attn_gate_weight.t.bin", wz_h.unsafe_ptr().unsafe_bitcast[UInt8](), H * H * 2)
    load_into(D + "blk_0_ssm_alpha_weight.t.bin", wa_h.unsafe_ptr().unsafe_bitcast[UInt8](), H * NH_V * 2)
    load_into(D + "blk_0_ssm_beta_weight.t.bin", wb_h.unsafe_ptr().unsafe_bitcast[UInt8](), H * NH_V * 2)
    load_into(D + "blk_0_ssm_out_weight.t.bin", wo_h.unsafe_ptr().unsafe_bitcast[UInt8](), H * H * 2)

    var y_ref = alloc[Float32](H)
    var s1_ref = alloc[Float32](NH_V * SSTATE * SSTATE)
    var cs1_ref = alloc[Float32](3 * CONV)
    load_into(D + "ssm_y_ref.bin", y_ref.unsafe_bitcast[UInt8](), H * 4)
    load_into(D + "ssm_s1_ref.bin", s1_ref.unsafe_bitcast[UInt8](), NH_V * SSTATE * SSTATE * 4)
    load_into(D + "ssm_conv_state1_ref.bin", cs1_ref.unsafe_bitcast[UInt8](), 3 * CONV * 4)

    var x_d = ctx.enqueue_create_buffer[f32](H)
    var cur_d = ctx.enqueue_create_buffer[f32](H)
    var curb_d = ctx.enqueue_create_buffer[bf16](H)
    var cs_d = ctx.enqueue_create_buffer[f32](3 * CONV)
    var cs1_d = ctx.enqueue_create_buffer[f32](3 * CONV)
    var s0_d = ctx.enqueue_create_buffer[f32](NH_V * SSTATE * SSTATE)
    var s1_d = ctx.enqueue_create_buffer[f32](NH_V * SSTATE * SSTATE)
    var cw_d = ctx.enqueue_create_buffer[f32](CONV * 4)
    var ssma_d = ctx.enqueue_create_buffer[f32](NH_V)
    var dtb_d = ctx.enqueue_create_buffer[f32](NH_V)
    var nw_d = ctx.enqueue_create_buffer[f32](SSTATE)
    var an_d = ctx.enqueue_create_buffer[f32](H)
    var wqkv_d = ctx.enqueue_create_buffer[bf16](H * CONV)
    var wz_d = ctx.enqueue_create_buffer[bf16](H * H)
    var wa_d = ctx.enqueue_create_buffer[bf16](H * NH_V)
    var wb_d = ctx.enqueue_create_buffer[bf16](H * NH_V)
    var wo_d = ctx.enqueue_create_buffer[bf16](H * H)
    var qkv_d = ctx.enqueue_create_buffer[f32](CONV)
    var z_d = ctx.enqueue_create_buffer[f32](H)
    var araw_d = ctx.enqueue_create_buffer[f32](NH_V)
    var braw_d = ctx.enqueue_create_buffer[f32](NH_V)
    var eg_d = ctx.enqueue_create_buffer[f32](NH_V)
    var beta_d = ctx.enqueue_create_buffer[f32](NH_V)
    var conv_d = ctx.enqueue_create_buffer[f32](CONV)
    var o_d = ctx.enqueue_create_buffer[f32](NH_V * SSTATE)
    var res_d = ctx.enqueue_create_buffer[f32](H)
    var resb_d = ctx.enqueue_create_buffer[bf16](H)
    var out_d = ctx.enqueue_create_buffer[f32](H)
    var p_qkv_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * CONV)
    var p_h_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * H)
    var p_ab_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)

    ctx.enqueue_copy(dst_buf=x_d, src_buf=x_h)
    ctx.enqueue_copy(dst_buf=cs_d, src_buf=cs_h)
    ctx.enqueue_copy(dst_buf=s0_d, src_buf=s0_h)
    ctx.enqueue_copy(dst_buf=cw_d, src_buf=cw_h)
    ctx.enqueue_copy(dst_buf=ssma_d, src_buf=ssma_h)
    ctx.enqueue_copy(dst_buf=dtb_d, src_buf=dtb_h)
    ctx.enqueue_copy(dst_buf=nw_d, src_buf=nw_h)
    ctx.enqueue_copy(dst_buf=an_d, src_buf=an_h)
    ctx.enqueue_copy(dst_buf=wqkv_d, src_buf=wqkv_h)
    ctx.enqueue_copy(dst_buf=wz_d, src_buf=wz_h)
    ctx.enqueue_copy(dst_buf=wa_d, src_buf=wa_h)
    ctx.enqueue_copy(dst_buf=wb_d, src_buf=wb_h)
    ctx.enqueue_copy(dst_buf=wo_d, src_buf=wo_h)
    ctx.synchronize()

    var X2 = TileTensor(x_d, x2_layout)
    var Cur2 = TileTensor(cur_d, x2_layout)
    var An = TileTensor(an_d, h_layout)
    var X1 = TileTensor(x_d, h_layout)
    var Cur1 = TileTensor(cur_d, h_layout)
    var CurB1 = TileTensor(curb_d, hb_layout)
    var CurB2 = TileTensor(curb_d, row_major[1, H]())
    var Wqkv = TileTensor(wqkv_d, wqkv_layout)
    var Wz = TileTensor(wz_d, wsq_layout)
    var Wa = TileTensor(wa_d, wab_layout)
    var Wb = TileTensor(wb_d, wab_layout)
    var Wo = TileTensor(wo_d, wsq_layout)
    var Pq = TileTensor(p_qkv_d, p_qkv)
    var Ph = TileTensor(p_h_d, p_h)
    var Pab = TileTensor(p_ab_d, p_ab)
    var Qkv1 = TileTensor(qkv_d, conv_layout)
    var Qkv2 = TileTensor(qkv_d, c_qkv)
    var Z1 = TileTensor(z_d, h_layout)
    var Z2 = TileTensor(z_d, c_h)
    var Araw1 = TileTensor(araw_d, g_layout)
    var Araw2 = TileTensor(araw_d, c_ab)
    var Braw1 = TileTensor(braw_d, g_layout)
    var Braw2 = TileTensor(braw_d, c_ab)
    var Eg = TileTensor(eg_d, g_layout)
    var Beta = TileTensor(beta_d, g_layout)
    var SsmA = TileTensor(ssma_d, g_layout)
    var DtB = TileTensor(dtb_d, g_layout)
    var Cs = TileTensor(cs_d, cs_layout)
    var Cs1 = TileTensor(cs1_d, cs_layout)
    var Cw = TileTensor(cw_d, cw_layout)
    var Conv = TileTensor(conv_d, conv_layout)
    var S0 = TileTensor(s0_d, s_layout)
    var S1t = TileTensor(s1_d, s_layout)
    var O = TileTensor(o_d, o_layout)
    var Nw = TileTensor(nw_d, n_layout)
    var Res1 = TileTensor(res_d, h_layout)
    var ResB1 = TileTensor(resb_d, hb_layout)
    var ResB2 = TileTensor(resb_d, row_major[1, H]())
    var Out1 = TileTensor(out_d, h_layout)
    var Out2 = TileTensor(out_d, c_h)

    comptime rms_k = rmsnorm[type_of(x2_layout), type_of(h_layout), type_of(x2_layout)]
    comptime cast_k = cast_bf16[type_of(h_layout), type_of(hb_layout)]
    comptime g_qkv = matmul_skinny[bf16, type_of(row_major[1, H]()), type_of(wqkv_layout), type_of(p_qkv)]
    comptime g_h = matmul_skinny[bf16, type_of(row_major[1, H]()), type_of(wsq_layout), type_of(p_h)]
    comptime g_ab = matmul_skinny[bf16, type_of(row_major[1, H]()), type_of(wab_layout), type_of(p_ab)]
    comptime r_qkv = skinny_reduce[type_of(p_qkv), type_of(c_qkv)]
    comptime r_h = skinny_reduce[type_of(p_h), type_of(c_h)]
    comptime r_ab = skinny_reduce[type_of(p_ab), type_of(c_ab)]
    comptime gates_k = ssm_gates[
        type_of(g_layout), type_of(g_layout), type_of(g_layout),
        type_of(g_layout), type_of(g_layout)
    ]
    comptime conv_k = ssm_conv[
        type_of(conv_layout), type_of(cs_layout), type_of(cw_layout), type_of(conv_layout)
    ]
    comptime l2_k = ssm_qk_l2norm[type_of(conv_layout)]
    comptime delta_k = ssm_delta_step[
        type_of(s_layout), type_of(conv_layout), type_of(g_layout), type_of(o_layout)
    ]
    comptime gated_k = ssm_gated_out[
        type_of(o_layout), type_of(h_layout), type_of(n_layout), type_of(h_layout)
    ]
    comptime add_k = residual_add[type_of(h_layout), type_of(h_layout)]

    ctx.enqueue_function[rms_k](X2, An, Cur2, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
    ctx.enqueue_function[cast_k](Cur1, CurB1, Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.enqueue_function[g_qkv](CurB2, Wqkv, Pq, Int32(1), Int32(CONV), Int32(H), grid_dim=(ceildiv(CONV, SBN), SPLITK), block_dim=SK_THREADS)
    ctx.enqueue_function[r_qkv](Pq, Qkv2, Int32(1), Int32(CONV), grid_dim=ceildiv(CONV, 256), block_dim=256)
    ctx.enqueue_function[g_h](CurB2, Wz, Ph, Int32(1), Int32(H), Int32(H), grid_dim=(ceildiv(H, SBN), SPLITK), block_dim=SK_THREADS)
    ctx.enqueue_function[r_h](Ph, Z2, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.enqueue_function[g_ab](CurB2, Wa, Pab, Int32(1), Int32(NH_V), Int32(H), grid_dim=(1, SPLITK), block_dim=SK_THREADS)
    ctx.enqueue_function[r_ab](Pab, Araw2, Int32(1), Int32(NH_V), grid_dim=1, block_dim=256)
    ctx.enqueue_function[g_ab](CurB2, Wb, Pab, Int32(1), Int32(NH_V), Int32(H), grid_dim=(1, SPLITK), block_dim=SK_THREADS)
    ctx.enqueue_function[r_ab](Pab, Braw2, Int32(1), Int32(NH_V), grid_dim=1, block_dim=256)
    ctx.enqueue_function[gates_k](Araw1, Braw1, Eg, Beta, SsmA, DtB, grid_dim=1, block_dim=NH_V)
    ctx.enqueue_function[conv_k](Qkv1, Cs, Cw, Conv, Cs1, grid_dim=ceildiv(CONV, 256), block_dim=256)
    ctx.enqueue_function[l2_k](Conv, grid_dim=NH_V, block_dim=SSTATE)
    ctx.enqueue_function[delta_k](S0, S1t, Conv, Eg, Beta, O, grid_dim=NH_V, block_dim=SSTATE)
    ctx.enqueue_function[gated_k](O, Z1, Nw, Res1, grid_dim=NH_V, block_dim=SSTATE)
    ctx.enqueue_function[cast_k](Res1, ResB1, Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.enqueue_function[g_h](ResB2, Wo, Ph, Int32(1), Int32(H), Int32(H), grid_dim=(ceildiv(H, SBN), SPLITK), block_dim=SK_THREADS)
    ctx.enqueue_function[r_h](Ph, Out2, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.enqueue_function[add_k](X1, Out1, Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.synchronize()

    var y_got = ctx.enqueue_create_host_buffer[f32](H)
    var s1_got = ctx.enqueue_create_host_buffer[f32](NH_V * SSTATE * SSTATE)
    var cs1_got = ctx.enqueue_create_host_buffer[f32](3 * CONV)
    ctx.enqueue_copy(dst_buf=y_got, src_buf=out_d)
    ctx.enqueue_copy(dst_buf=s1_got, src_buf=s1_d)
    ctx.enqueue_copy(dst_buf=cs1_got, src_buf=cs1_d)
    ctx.synchronize()

    var o_ref = alloc[Float32](NH_V * SSTATE)
    var z_ref = alloc[Float32](H)
    var gated_ref = alloc[Float32](H)
    load_into(D + "ssm_o_ref.bin", o_ref.unsafe_bitcast[UInt8](), NH_V * SSTATE * 4)
    load_into(D + "ssm_z_ref.bin", z_ref.unsafe_bitcast[UInt8](), H * 4)
    load_into(D + "ssm_gated_ref.bin", gated_ref.unsafe_bitcast[UInt8](), H * 4)
    var o_got = ctx.enqueue_create_host_buffer[f32](NH_V * SSTATE)
    var z_got2 = ctx.enqueue_create_host_buffer[f32](H)
    var g_got = ctx.enqueue_create_host_buffer[f32](H)
    ctx.enqueue_copy(dst_buf=o_got, src_buf=o_d)
    ctx.enqueue_copy(dst_buf=z_got2, src_buf=z_d)
    ctx.enqueue_copy(dst_buf=g_got, src_buf=res_d)
    ctx.synchronize()
    check("z", z_got2.unsafe_ptr(), z_ref, H)
    check("o", o_got.unsafe_ptr(), o_ref, NH_V * SSTATE)
    check("gated", g_got.unsafe_ptr(), gated_ref, H, 5e-3)
    check("conv_state1", cs1_got.unsafe_ptr(), cs1_ref, 3 * CONV)
    check("S1", s1_got.unsafe_ptr(), s1_ref, NH_V * SSTATE * SSTATE)
    check("y", y_got.unsafe_ptr(), y_ref, H, 3e-2)
    print("PASS: qwen35 gated-delta-net block matches numpy reference")
