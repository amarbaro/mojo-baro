from std.math import ceildiv
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major

from elementwise import (
    amar_rmsnorm, amar_rmsnorm_cast, amar_embed_lookup_pos, amar_argmax_pos, amar_tok_copy,
)
from matmul_skinny import (
    amar_matmul_skinny_q8row, amar_matmul_skinny_q4row, amar_skinny_reduce, amar_skinny_reduce_add,
    amar_skinny_reduce_swiglu_bf16, SM, SPLITK, ROW_WAVES, ROW_THREADS,
)
from ssm import (
    amar_ssm_reduce_gates, amar_ssm_conv, amar_ssm_qk_l2norm,
    amar_ssm_delta_step, amar_ssm_gated_out_bf16, amar_cast_bf16, CONV, NH_V, SSTATE,
)
from attn import (
    amar_head_rmsnorm, amar_attn_decode, amar_gate_mul_cast, amar_qgate_split, amar_rope_yarn, amar_kv_append,
    HD, NQH, NKVH,
)

comptime H = 4096
comptime FFN = 12288
comptime VOCAB = 248320
comptime QF = 2 * H
comptime KV = NKVH * HD
comptime N_LAYERS = 32
comptime N_SSM = 24
comptime N_ATT = 8
comptime TMAX = 128
comptime GEN_N = 64

comptime bf16 = DType.bfloat16
comptime f32 = DType.float32

comptime MROWS = SM
comptime KMAX = SM
comptime SLOTS = KMAX + 1
comptime CONV_SLOT = N_SSM * 3 * CONV
comptime SSM_SLOT = N_SSM * NH_V * SSTATE * SSTATE
comptime ATT32 = NKVH * TMAX * HD

comptime h_layout = row_major[H]()
comptime h2_layout = row_major[1, H]()
comptime xm_layout = row_major[MROWS, H]()
comptime xflat_layout = row_major[MROWS * H]()
comptime qfm_layout = row_major[MROWS, QF]()
comptime convm_layout = row_major[MROWS, CONV]()
comptime qm_layout = row_major[MROWS * NQH, HD]()
comptime kvm_layout = row_major[MROWS * NKVH, HD]()
comptime kvm_flat = row_major[MROWS, KV]()
comptime ffnm_layout = row_major[MROWS, FFN]()
comptime vm_layout = row_major[MROWS, VOCAB]()
comptime ffn_layout = row_major[FFN]()
comptime qf_layout = row_major[QF]()
comptime q_layout = row_major[NQH, HD]()
comptime kvh_layout = row_major[NKVH, HD]()
comptime kvflat_layout = row_major[KV]()
comptime cache_layout = row_major[NKVH, TMAX, HD]()
comptime hd_layout = row_major[HD]()
comptime conv_layout = row_major[CONV]()
comptime cs_layout = row_major[3, CONV]()
comptime cw_layout = row_major[CONV, 4]()
comptime s_layout = row_major[NH_V, SSTATE, SSTATE]()
comptime o_layout = row_major[NH_V, SSTATE]()
comptime g32_layout = row_major[NH_V]()
comptime g32m_layout = row_major[MROWS, NH_V]()
comptime om_layout = row_major[MROWS, NH_V, SSTATE]()
comptime csall_layout = row_major[SLOTS, N_SSM, 3, CONV]()
comptime ssall_layout = row_major[SLOTS, N_SSM, NH_V, SSTATE, SSTATE]()
comptime n128_layout = row_major[SSTATE]()
comptime emb_layout = row_major[VOCAB, H]()
comptime vrow_layout = row_major[1, VOCAB]()
comptime toks_layout = row_major[TMAX]()
comptime dtok_layout = row_major[KMAX + 1]()

comptime w_h_qf = row_major[H, QF]()
comptime w_h_h = row_major[H, H]()
comptime w_h_kv = row_major[H, KV]()
comptime w_h_32 = row_major[H, NH_V]()
comptime w_h_ffn = row_major[H, FFN]()
comptime w_ffn_h = row_major[FFN, H]()
comptime w_h_v = row_major[H, VOCAB]()
comptime w_qf_h = row_major[QF, H]()
comptime q_h_qf = row_major[QF, H]()
comptime s_h_qf = row_major[QF, H // 32]()
comptime q_h_h = row_major[H, H]()
comptime s_h_h = row_major[H, H // 32]()
comptime q_h_kv = row_major[KV, H]()
comptime s_h_kv = row_major[KV, H // 32]()
comptime q_h_32 = row_major[NH_V, H]()
comptime s_h_32 = row_major[NH_V, H // 32]()
comptime q_h_ffn = row_major[FFN, H]()
comptime s_h_ffn = row_major[FFN, H // 32]()
comptime q_ffn_h = row_major[H, FFN]()
comptime s_ffn_h = row_major[H, FFN // 32]()
comptime q_h_v = row_major[VOCAB, H]()
comptime s_h_v = row_major[VOCAB, H // 32]()
comptime q4_h_v = row_major[VOCAB, H // 2]()
comptime q_qf_h = row_major[H, QF]()
comptime s_qf_h = row_major[H, QF // 32]()

comptime p_qf = row_major[SPLITK, SM, QF]()
comptime p_h = row_major[SPLITK, SM, H]()
comptime p_kv = row_major[SPLITK, SM, KV]()
comptime p_32 = row_major[SPLITK, SM, NH_V]()
comptime p_ffn = row_major[SPLITK, SM, FFN]()
comptime p_v = row_major[SPLITK, SM, VOCAB]()
comptime c_qf = row_major[1, QF]()
comptime c_h = row_major[1, H]()
comptime c_kv = row_major[1, KV]()
comptime c_32 = row_major[1, NH_V]()
comptime c_ffn = row_major[1, FFN]()

comptime B2 = 2
comptime B4 = 4

comptime rmsc_k = amar_rmsnorm_cast[type_of(xm_layout), type_of(h_layout), type_of(xm_layout)]
comptime embed_k = amar_embed_lookup_pos[type_of(emb_layout), type_of(xm_layout), type_of(toks_layout)]
comptime argmax_k = amar_argmax_pos[type_of(vm_layout), type_of(toks_layout)]
comptime argmax_d = amar_argmax_pos[type_of(vm_layout), type_of(dtok_layout)]
comptime embed1_k = amar_embed_lookup_pos[type_of(emb_layout), type_of(h2_layout), type_of(dtok_layout)]
comptime rms_m = amar_rmsnorm[type_of(xm_layout), type_of(h_layout), type_of(xm_layout)]
comptime rms_h2 = amar_rmsnorm[type_of(h2_layout), type_of(h_layout), type_of(h2_layout)]
comptime rmsc_h2 = amar_rmsnorm_cast[type_of(h2_layout), type_of(h_layout), type_of(h2_layout)]
comptime cast_m = amar_cast_bf16[type_of(xflat_layout), type_of(xflat_layout)]
comptime cast_1 = amar_cast_bf16[type_of(h_layout), type_of(h_layout)]
comptime tokcp_k = amar_tok_copy[type_of(dtok_layout), type_of(toks_layout)]
comptime tokcp_b = amar_tok_copy[type_of(toks_layout), type_of(dtok_layout)]
comptime r_qf = amar_skinny_reduce[type_of(p_qf), type_of(qfm_layout), 1]
comptime r_h = amar_skinny_reduce[type_of(p_h), type_of(xm_layout), 1]
comptime r_kv = amar_skinny_reduce[type_of(p_kv), type_of(kvm_flat), 1]
comptime r_add = amar_skinny_reduce_add[type_of(p_h), type_of(xm_layout), 1]
comptime r_swiglu = amar_skinny_reduce_swiglu_bf16[type_of(p_ffn), type_of(ffnm_layout), 1]
comptime r_head = amar_skinny_reduce[type_of(p_v), type_of(vm_layout), 1]
comptime rgates_k = amar_ssm_reduce_gates[
    type_of(p_32), type_of(g32m_layout), type_of(g32_layout)
]
comptime conv_k = amar_ssm_conv[
    type_of(qfm_layout), type_of(csall_layout), type_of(cw_layout), type_of(convm_layout)
]
comptime l2_k = amar_ssm_qk_l2norm[type_of(convm_layout)]
comptime gated_k = amar_ssm_gated_out_bf16[
    type_of(om_layout), type_of(xm_layout), type_of(n128_layout), type_of(xm_layout)
]
comptime split_k = amar_qgate_split[type_of(qfm_layout), type_of(qm_layout), type_of(xflat_layout)]
comptime hrms_q = amar_head_rmsnorm[type_of(qm_layout), type_of(hd_layout)]
comptime hrms_kv = amar_head_rmsnorm[type_of(kvm_layout), type_of(hd_layout)]
comptime rope_q = amar_rope_yarn[type_of(qm_layout)]
comptime rope_k = amar_rope_yarn[type_of(kvm_layout)]
comptime append_k = amar_kv_append[type_of(cache_layout), type_of(kvm_layout)]
comptime att_k = amar_attn_decode[type_of(qm_layout), type_of(cache_layout), type_of(qm_layout)]
comptime gmul_k = amar_gate_mul_cast[type_of(xflat_layout), type_of(xflat_layout), type_of(xflat_layout)]


def delta_dispatch(
    ctx: DeviceContext,
    SAll: TileTensor[f32, type_of(ssall_layout), MutAnyOrigin],
    ConvOut: TileTensor[f32, type_of(convm_layout), MutAnyOrigin],
    Eg: TileTensor[f32, type_of(g32m_layout), MutAnyOrigin],
    Beta: TileTensor[f32, type_of(g32m_layout), MutAnyOrigin],
    O: TileTensor[f32, type_of(om_layout), MutAnyOrigin],
    ring: Int32, ssm_i: Int32, slots: Int32, m: Int,
) raises:
    comptime DL = type_of(ssall_layout)
    comptime CL = type_of(convm_layout)
    comptime GL = type_of(g32m_layout)
    comptime OL = type_of(om_layout)
    if m == 1:
        ctx.enqueue_function[amar_ssm_delta_step[1, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)
    elif m == 2:
        ctx.enqueue_function[amar_ssm_delta_step[2, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)
    elif m == 3:
        ctx.enqueue_function[amar_ssm_delta_step[3, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)
    elif m == 4:
        ctx.enqueue_function[amar_ssm_delta_step[4, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)
    elif m == 5:
        ctx.enqueue_function[amar_ssm_delta_step[5, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)
    elif m == 6:
        ctx.enqueue_function[amar_ssm_delta_step[6, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)
    elif m == 7:
        ctx.enqueue_function[amar_ssm_delta_step[7, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)
    else:
        ctx.enqueue_function[amar_ssm_delta_step[SM, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)


def gemm_q8[
    AL: TensorLayout, QL: TensorLayout, SL: TensorLayout, PL: TensorLayout
](
    ctx: DeviceContext,
    A: TileTensor[bf16, AL, MutAnyOrigin],
    Wq: TileTensor[DType.int8, QL, MutAnyOrigin],
    Ws: TileTensor[DType.float16, SL, MutAnyOrigin],
    P: TileTensor[f32, PL, MutAnyOrigin],
    m: Int, n: Int, k: Int,
) raises:
    if m == 1:
        ctx.enqueue_function[amar_matmul_skinny_q8row[4, 1, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif m == 2:
        ctx.enqueue_function[amar_matmul_skinny_q8row[4, 2, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif m == 3:
        ctx.enqueue_function[amar_matmul_skinny_q8row[4, 3, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif m <= 5:
        ctx.enqueue_function[amar_matmul_skinny_q8row[4, 5, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    else:
        ctx.enqueue_function[amar_matmul_skinny_q8row[4, SM, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )


def gemm_q4[
    AL: TensorLayout, QL: TensorLayout, SL: TensorLayout, PL: TensorLayout
](
    ctx: DeviceContext,
    A: TileTensor[bf16, AL, MutAnyOrigin],
    Wq: TileTensor[DType.uint8, QL, MutAnyOrigin],
    Ws: TileTensor[DType.float16, SL, MutAnyOrigin],
    P: TileTensor[f32, PL, MutAnyOrigin],
    m: Int, n: Int, k: Int,
) raises:
    if m == 1:
        ctx.enqueue_function[amar_matmul_skinny_q4row[2, 1, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif m == 2:
        ctx.enqueue_function[amar_matmul_skinny_q4row[2, 2, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif m == 3:
        ctx.enqueue_function[amar_matmul_skinny_q4row[2, 3, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif m <= 5:
        ctx.enqueue_function[amar_matmul_skinny_q4row[2, 5, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    else:
        ctx.enqueue_function[amar_matmul_skinny_q4row[2, SM, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
