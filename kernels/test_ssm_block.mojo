"""Parity: one decode token through the qwen35 gated-delta-net block on GPU
vs the numpy reference (tools/ssm-ref.py implementing docs/qwen35-ssm-notes.md).

Gates: block output y, new recurrent state S1, and new conv window, all
rel < 1e-3 against the reference (shared bf16 weights, fp32 math both sides).
"""
from std.math import ceildiv
from std.memory import alloc
from std.sys import has_accelerator

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from elementwise import amar_rmsnorm
from matmul_skinny import amar_matmul_skinny, amar_skinny_reduce, SM, SBN, SPLITK, SK_THREADS
from ssm import (
    amar_cast_bf16, amar_residual_add, amar_ssm_gates, amar_ssm_conv, amar_ssm_qk_l2norm,
    amar_ssm_delta_step, amar_ssm_gated_out, CONV, NH_V, SSTATE,
)

comptime H = 4096
comptime bf16 = DType.bfloat16
comptime f32 = DType.float32

comptime x2_layout = row_major[1, H]()
comptime h_layout = row_major[H]()
comptime hb_layout = row_major[H]()
comptime cw_layout = row_major[CONV, 4]()
comptime o_layout = row_major[NH_V, SSTATE]()
comptime g_layout = row_major[NH_V]()
comptime n_layout = row_major[SSTATE]()
comptime convm1_layout = row_major[1, CONV]()
comptime g32m1_layout = row_major[1, NH_V]()
comptime om1_layout = row_major[1, NH_V, SSTATE]()
comptime csall_m1 = row_major[1, 1, 3, CONV]()
comptime ssall_m1 = row_major[1, 1, NH_V, SSTATE, SSTATE]()

comptime MROWS_T = 4
comptime SLOTS_T = MROWS_T + 1
comptime qkvm_t_layout = row_major[MROWS_T, CONV]()
comptime g32m_t_layout = row_major[MROWS_T, NH_V]()
comptime om_t_layout = row_major[MROWS_T, NH_V, SSTATE]()
comptime csall_t_layout = row_major[SLOTS_T, 1, 3, CONV]()
comptime ssall_t_layout = row_major[SLOTS_T, 1, NH_V, SSTATE, SSTATE]()
comptime seq_conv_layout = row_major[MROWS_T, CONV]()
comptime seq_o_layout = row_major[MROWS_T, NH_V, SSTATE]()

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


def rowview[
    LT: TensorLayout
](ctx: DeviceContext, b: DeviceBuffer[f32], o: Int, n: Int, lt: LT) -> TileTensor[f32, LT, MutAnyOrigin]:
    var s = DeviceBuffer[f32](ctx, b.unsafe_ptr() + o, n, owning=False)
    var t = TileTensor(s, lt)
    return rebind[TileTensor[f32, LT, MutAnyOrigin]](t)


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
    var s0_d = ctx.enqueue_create_buffer[f32](NH_V * SSTATE * SSTATE)
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
    var Cw = TileTensor(cw_d, cw_layout)
    var CsAll1 = TileTensor(cs_d, csall_m1)
    var SAll1 = TileTensor(s0_d, ssall_m1)
    var ConvM1 = TileTensor(conv_d, convm1_layout)
    var EgM1 = TileTensor(eg_d, g32m1_layout)
    var BetaM1 = TileTensor(beta_d, g32m1_layout)
    var OM1 = TileTensor(o_d, om1_layout)
    var O = TileTensor(o_d, o_layout)
    var Nw = TileTensor(nw_d, n_layout)
    var Res1 = TileTensor(res_d, h_layout)
    var ResB1 = TileTensor(resb_d, hb_layout)
    var ResB2 = TileTensor(resb_d, row_major[1, H]())
    var Out1 = TileTensor(out_d, h_layout)
    var Out2 = TileTensor(out_d, c_h)

    comptime rms_k = amar_rmsnorm[type_of(x2_layout), type_of(h_layout), type_of(x2_layout)]
    comptime cast_k = amar_cast_bf16[type_of(h_layout), type_of(hb_layout)]
    comptime g_qkv = amar_matmul_skinny[bf16, type_of(row_major[1, H]()), type_of(wqkv_layout), type_of(p_qkv)]
    comptime g_h = amar_matmul_skinny[bf16, type_of(row_major[1, H]()), type_of(wsq_layout), type_of(p_h)]
    comptime g_ab = amar_matmul_skinny[bf16, type_of(row_major[1, H]()), type_of(wab_layout), type_of(p_ab)]
    comptime r_qkv = amar_skinny_reduce[type_of(p_qkv), type_of(c_qkv)]
    comptime r_h = amar_skinny_reduce[type_of(p_h), type_of(c_h)]
    comptime r_ab = amar_skinny_reduce[type_of(p_ab), type_of(c_ab)]
    comptime gates_k = amar_ssm_gates[
        type_of(g_layout), type_of(g_layout), type_of(g_layout),
        type_of(g_layout), type_of(g_layout)
    ]
    comptime conv_k = amar_ssm_conv[
        type_of(c_qkv), type_of(csall_m1), type_of(cw_layout), type_of(convm1_layout)
    ]
    comptime l2_k = amar_ssm_qk_l2norm[type_of(convm1_layout)]
    comptime delta_k = amar_ssm_delta_step[
        1, type_of(ssall_m1), type_of(convm1_layout), type_of(g32m1_layout), type_of(om1_layout)
    ]
    comptime gated_k = amar_ssm_gated_out[
        type_of(o_layout), type_of(h_layout), type_of(n_layout), type_of(h_layout)
    ]
    comptime add_k = amar_residual_add[type_of(h_layout), type_of(h_layout)]

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
    ctx.enqueue_function[conv_k](Qkv2, CsAll1, Cw, ConvM1, Int32(0), Int32(0), Int32(1), Int32(1), grid_dim=ceildiv(CONV, 256), block_dim=256)
    ctx.enqueue_function[l2_k](ConvM1, Int32(1), grid_dim=NH_V, block_dim=SSTATE)
    ctx.enqueue_function[delta_k](SAll1, ConvM1, EgM1, BetaM1, OM1, Int32(0), Int32(0), Int32(1), grid_dim=NH_V, block_dim=SSTATE)
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
    ctx.enqueue_copy(dst_buf=s1_got, src_buf=s0_d)
    ctx.enqueue_copy(dst_buf=cs1_got, src_buf=cs_d)
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
    print("PASS: qwen35 gated-delta-net block matches numpy reference (m=1)")

    # -- M2 fold check: one m=4 folded call must be bit-identical to four
    # sequential m=1 calls walking the same ring one row at a time (the
    # per-row behavior the fold replaces). Reuses qkv_d/eg_d/beta_d, still
    # holding the m=1 pipeline's values (conv_k/l2_k/delta_k only read them).
    var qkvm_d = ctx.enqueue_create_buffer[f32](MROWS_T * CONV)
    var egm_d = ctx.enqueue_create_buffer[f32](MROWS_T * NH_V)
    var betam_d = ctx.enqueue_create_buffer[f32](MROWS_T * NH_V)
    var conv_seq_d = ctx.enqueue_create_buffer[f32](MROWS_T * CONV)
    var conv_fold_d = ctx.enqueue_create_buffer[f32](MROWS_T * CONV)
    var o_seq_d = ctx.enqueue_create_buffer[f32](MROWS_T * NH_V * SSTATE)
    var o_fold_d = ctx.enqueue_create_buffer[f32](MROWS_T * NH_V * SSTATE)
    var cs_seq_d = ctx.enqueue_create_buffer[f32](SLOTS_T * 3 * CONV)
    var cs_fold_d = ctx.enqueue_create_buffer[f32](SLOTS_T * 3 * CONV)
    var s0_seq_d = ctx.enqueue_create_buffer[f32](SLOTS_T * NH_V * SSTATE * SSTATE)
    var s0_fold_d = ctx.enqueue_create_buffer[f32](SLOTS_T * NH_V * SSTATE * SSTATE)
    ctx.enqueue_memset(cs_seq_d, 0)
    ctx.enqueue_memset(cs_fold_d, 0)
    ctx.enqueue_memset(s0_seq_d, 0)
    ctx.enqueue_memset(s0_fold_d, 0)
    ctx.synchronize()

    var cs_seq_slot0 = DeviceBuffer[f32](ctx, cs_seq_d.unsafe_ptr(), 3 * CONV, owning=False)
    var cs_fold_slot0 = DeviceBuffer[f32](ctx, cs_fold_d.unsafe_ptr(), 3 * CONV, owning=False)
    var s0_seq_slot0 = DeviceBuffer[f32](ctx, s0_seq_d.unsafe_ptr(), NH_V * SSTATE * SSTATE, owning=False)
    var s0_fold_slot0 = DeviceBuffer[f32](ctx, s0_fold_d.unsafe_ptr(), NH_V * SSTATE * SSTATE, owning=False)
    ctx.enqueue_copy(dst_buf=cs_seq_slot0, src_buf=cs_h)
    ctx.enqueue_copy(dst_buf=cs_fold_slot0, src_buf=cs_h)
    ctx.enqueue_copy(dst_buf=s0_seq_slot0, src_buf=s0_h)
    ctx.enqueue_copy(dst_buf=s0_fold_slot0, src_buf=s0_h)
    for r in range(MROWS_T):
        var qrow = DeviceBuffer[f32](ctx, qkvm_d.unsafe_ptr() + r * CONV, CONV, owning=False)
        var erow = DeviceBuffer[f32](ctx, egm_d.unsafe_ptr() + r * NH_V, NH_V, owning=False)
        var brow = DeviceBuffer[f32](ctx, betam_d.unsafe_ptr() + r * NH_V, NH_V, owning=False)
        ctx.enqueue_copy(dst_buf=qrow, src_buf=qkv_d)
        ctx.enqueue_copy(dst_buf=erow, src_buf=eg_d)
        ctx.enqueue_copy(dst_buf=brow, src_buf=beta_d)
    ctx.synchronize()

    var CsSeq = TileTensor(cs_seq_d, csall_t_layout)
    var CsFold = TileTensor(cs_fold_d, csall_t_layout)
    var S0Seq = TileTensor(s0_seq_d, ssall_t_layout)
    var S0Fold = TileTensor(s0_fold_d, ssall_t_layout)
    var ConvFold = TileTensor(conv_fold_d, qkvm_t_layout)
    var OFold = TileTensor(o_fold_d, om_t_layout)
    var Qkvm = TileTensor(qkvm_d, qkvm_t_layout)
    var EgmT = TileTensor(egm_d, g32m_t_layout)
    var BetamT = TileTensor(betam_d, g32m_t_layout)

    comptime conv_k1 = amar_ssm_conv[
        type_of(c_qkv), type_of(csall_t_layout), type_of(cw_layout), type_of(convm1_layout)
    ]
    comptime l2_k1 = amar_ssm_qk_l2norm[type_of(convm1_layout)]
    comptime delta_k1 = amar_ssm_delta_step[
        1, type_of(ssall_t_layout), type_of(convm1_layout), type_of(g32m1_layout), type_of(om1_layout)
    ]
    comptime conv_kM = amar_ssm_conv[
        type_of(qkvm_t_layout), type_of(csall_t_layout), type_of(cw_layout), type_of(qkvm_t_layout)
    ]
    comptime l2_kM = amar_ssm_qk_l2norm[type_of(qkvm_t_layout)]
    comptime delta_kM = amar_ssm_delta_step[
        MROWS_T, type_of(ssall_t_layout), type_of(qkvm_t_layout), type_of(g32m_t_layout), type_of(om_t_layout)
    ]

    for r in range(MROWS_T):
        var ConvRow = rowview(ctx, conv_seq_d, r * CONV, CONV, convm1_layout)
        var ORow = rowview(ctx, o_seq_d, r * NH_V * SSTATE, NH_V * SSTATE, om1_layout)
        ctx.enqueue_function[conv_k1](Qkv2, CsSeq, Cw, ConvRow, Int32(r), Int32(0), Int32(SLOTS_T), Int32(1), grid_dim=ceildiv(CONV, 256), block_dim=256)
        ctx.enqueue_function[l2_k1](ConvRow, Int32(1), grid_dim=NH_V, block_dim=SSTATE)
        ctx.enqueue_function[delta_k1](S0Seq, ConvRow, EgM1, BetaM1, ORow, Int32(r), Int32(0), Int32(SLOTS_T), grid_dim=NH_V, block_dim=SSTATE)

    ctx.enqueue_function[conv_kM](Qkvm, CsFold, Cw, ConvFold, Int32(0), Int32(0), Int32(SLOTS_T), Int32(MROWS_T), grid_dim=ceildiv(CONV, 256), block_dim=256)
    ctx.enqueue_function[l2_kM](ConvFold, Int32(MROWS_T), grid_dim=NH_V, block_dim=SSTATE)
    ctx.enqueue_function[delta_kM](S0Fold, ConvFold, EgmT, BetamT, OFold, Int32(0), Int32(0), Int32(SLOTS_T), grid_dim=NH_V, block_dim=SSTATE)
    ctx.synchronize()

    var conv_seq_got = ctx.enqueue_create_host_buffer[f32](MROWS_T * CONV)
    var conv_fold_got = ctx.enqueue_create_host_buffer[f32](MROWS_T * CONV)
    var o_seq_got = ctx.enqueue_create_host_buffer[f32](MROWS_T * NH_V * SSTATE)
    var o_fold_got = ctx.enqueue_create_host_buffer[f32](MROWS_T * NH_V * SSTATE)
    var cs_seq_got = ctx.enqueue_create_host_buffer[f32](SLOTS_T * 3 * CONV)
    var cs_fold_got = ctx.enqueue_create_host_buffer[f32](SLOTS_T * 3 * CONV)
    var s0_seq_got = ctx.enqueue_create_host_buffer[f32](SLOTS_T * NH_V * SSTATE * SSTATE)
    var s0_fold_got = ctx.enqueue_create_host_buffer[f32](SLOTS_T * NH_V * SSTATE * SSTATE)
    ctx.enqueue_copy(dst_buf=conv_seq_got, src_buf=conv_seq_d)
    ctx.enqueue_copy(dst_buf=conv_fold_got, src_buf=conv_fold_d)
    ctx.enqueue_copy(dst_buf=o_seq_got, src_buf=o_seq_d)
    ctx.enqueue_copy(dst_buf=o_fold_got, src_buf=o_fold_d)
    ctx.enqueue_copy(dst_buf=cs_seq_got, src_buf=cs_seq_d)
    ctx.enqueue_copy(dst_buf=cs_fold_got, src_buf=cs_fold_d)
    ctx.enqueue_copy(dst_buf=s0_seq_got, src_buf=s0_seq_d)
    ctx.enqueue_copy(dst_buf=s0_fold_got, src_buf=s0_fold_d)
    ctx.synchronize()

    check("m4_conv_out", conv_fold_got.unsafe_ptr(), conv_seq_got.unsafe_ptr(), MROWS_T * CONV, 0.0)
    check("m4_o_out", o_fold_got.unsafe_ptr(), o_seq_got.unsafe_ptr(), MROWS_T * NH_V * SSTATE, 0.0)
    check("m4_conv_ring", cs_fold_got.unsafe_ptr(), cs_seq_got.unsafe_ptr(), SLOTS_T * 3 * CONV, 0.0)
    check("m4_sstate_ring", s0_fold_got.unsafe_ptr(), s0_seq_got.unsafe_ptr(), SLOTS_T * NH_V * SSTATE * SSTATE, 0.0)
    print("PASS: m=4 folded SSM kernels bit-identical to 4x sequential per-row calls")
