"""baro engine: full-model greedy decode for qwen35 (Qwythos-9B), milestone 4.

Loads .work/engine-pack/ (fixed tensor order, 2D bf16 weights pre-transposed
to B-layout), reads prompt token ids from .work/engine-pack/prompt-tokens.txt,
runs the 32-block hybrid stack (24 gated-delta-net + 8 gated full-attention,
MTP block skipped) one token at a time, and prints greedy token ids.

Parity target: byte-identical token ids vs llama.cpp on the same GGUF.
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from elementwise import rmsnorm_cast, embed_lookup_pos, argmax_pos
from matmul_skinny import (
    matmul_skinny, skinny_reduce, skinny_reduce_add, skinny_reduce_swiglu_bf16,
    SM, SBN, SPLITK, SK_THREADS,
)
from ssm import (
    ssm_reduce_gates, ssm_conv, ssm_qk_l2norm,
    ssm_delta_step, ssm_gated_out_bf16, CONV, NH_V, SSTATE,
)
from attn import (
    head_rmsnorm, attn_decode, gate_mul_cast, qgate_split, rope_yarn, kv_append,
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
comptime GEN_N = 16

comptime bf16 = DType.bfloat16
comptime f32 = DType.float32

comptime h_layout = row_major[H]()
comptime h2_layout = row_major[1, H]()
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
comptime n128_layout = row_major[SSTATE]()
comptime emb_layout = row_major[VOCAB, H]()
comptime vrow_layout = row_major[1, VOCAB]()
comptime toks_layout = row_major[TMAX]()

comptime w_h_qf = row_major[H, QF]()
comptime w_h_h = row_major[H, H]()
comptime w_h_kv = row_major[H, KV]()
comptime w_h_32 = row_major[H, NH_V]()
comptime w_h_ffn = row_major[H, FFN]()
comptime w_ffn_h = row_major[FFN, H]()
comptime w_h_v = row_major[H, VOCAB]()

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


def is_attn(i: Int) -> Bool:
    return (i + 1) % 4 == 0


def push(mut off: List[Int], mut cursor: Int, n_bytes: Int):
    off.append(cursor)
    cursor += n_bytes


def wbf16(
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int
) -> DeviceBuffer[DType.bfloat16]:
    return DeviceBuffer[DType.bfloat16](
        ctx,
        (wbuf.unsafe_ptr() + o).unsafe_bitcast[Scalar[DType.bfloat16]](),
        n, owning=False,
    )


def wf32(
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int
) -> DeviceBuffer[DType.float32]:
    return DeviceBuffer[DType.float32](
        ctx,
        (wbuf.unsafe_ptr() + o).unsafe_bitcast[Scalar[DType.float32]](),
        n, owning=False,
    )


def tens_bf16[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.bfloat16, LT, MutAnyOrigin]:
    var b = wbf16(ctx, wbuf, o, n)
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.bfloat16, LT, MutAnyOrigin]](t)


def tens_f32[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.float32, LT, MutAnyOrigin]:
    var b = wf32(ctx, wbuf, o, n)
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.float32, LT, MutAnyOrigin]](t)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    comptime PACK = ".work/engine-pack/pack.bin"

    # --- offset table, mirroring tools/engine-pack.py order -----------------
    var off = List[Int]()
    var cursor = 0

    push(off, cursor, VOCAB * H * B2)
    for i in range(N_LAYERS):
        if is_attn(i):
            push(off, cursor, H * B4)
            push(off, cursor, H * QF * B2)
            push(off, cursor, H * KV * B2)
            push(off, cursor, H * KV * B2)
            push(off, cursor, HD * B4)
            push(off, cursor, HD * B4)
            push(off, cursor, H * H * B2)
        else:
            push(off, cursor, H * B4)
            push(off, cursor, H * CONV * B2)
            push(off, cursor, H * H * B2)
            push(off, cursor, H * NH_V * B2)
            push(off, cursor, H * NH_V * B2)
            push(off, cursor, CONV * 4 * B4)
            push(off, cursor, NH_V * B4)
            push(off, cursor, NH_V * B4)
            push(off, cursor, SSTATE * B4)
            push(off, cursor, H * H * B2)
        push(off, cursor, H * B4)
        push(off, cursor, H * FFN * B2)
        push(off, cursor, H * FFN * B2)
        push(off, cursor, FFN * H * B2)
    push(off, cursor, H * B4)
    push(off, cursor, H * VOCAB * B2)
    var total = cursor

    # --- load pack into one device buffer -----------------------------------
    print("loading pack:", total, "bytes")
    var wbuf = ctx.enqueue_create_buffer[DType.uint8](total)
    comptime CHUNK = 1 << 28
    var stage = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    ctx.synchronize()
    var t_load = perf_counter_ns()
    with open(PACK, "r") as f:
        var done = 0
        while done < total:
            var want = min(CHUNK, total - done)
            var data = f.read_bytes(want)
            if len(data) != want:
                raise Error("short read")
            var sp = stage.unsafe_ptr()
            for i in range(want):
                sp[unsafe_offset=i] = data[i]
            var dslice = DeviceBuffer[DType.uint8](
                ctx, wbuf.unsafe_ptr() + done, want, owning=False
            )
            var hslice = ctx.enqueue_create_host_buffer[DType.uint8](want) if want != CHUNK else stage
            if want != CHUNK:
                var hp = hslice.unsafe_ptr()
                for i in range(want):
                    hp[unsafe_offset=i] = data[i]
            ctx.enqueue_copy(dst_buf=dslice, src_buf=hslice)
            ctx.synchronize()
            done += want
    print("pack loaded in", Float64(perf_counter_ns() - t_load) / 1e9, "s")

    # --- prompt --------------------------------------------------------------
    var prompt = List[Int]()
    with open(".work/engine-pack/prompt-tokens.txt", "r") as f:
        var data = f.read_bytes()
        var val = 0
        var have = False
        for i in range(len(data)):
            var b = Int(data[i])
            if b >= 48 and b <= 57:
                val = val * 10 + (b - 48)
                have = True
            else:
                if have:
                    prompt.append(val)
                val = 0
                have = False
        if have:
            prompt.append(val)
    print("prompt tokens:", len(prompt))

    # --- activations / state -------------------------------------------------
    var x_d = ctx.enqueue_create_buffer[f32](H)
    var curb_d = ctx.enqueue_create_buffer[bf16](H)
    var qkv_d = ctx.enqueue_create_buffer[f32](CONV)
    var z_d = ctx.enqueue_create_buffer[f32](H)
    var eg_d = ctx.enqueue_create_buffer[f32](NH_V)
    var beta_d = ctx.enqueue_create_buffer[f32](NH_V)
    var conv_d = ctx.enqueue_create_buffer[f32](CONV)
    var so_d = ctx.enqueue_create_buffer[f32](NH_V * SSTATE)
    var resb_d = ctx.enqueue_create_buffer[bf16](H)
    var qf_d = ctx.enqueue_create_buffer[f32](QF)
    var q_d = ctx.enqueue_create_buffer[f32](NQH * HD)
    var gate_d = ctx.enqueue_create_buffer[f32](H)
    var k_d = ctx.enqueue_create_buffer[f32](KV)
    var v_d = ctx.enqueue_create_buffer[f32](KV)
    var ao_d = ctx.enqueue_create_buffer[f32](NQH * HD)
    var fgb_d = ctx.enqueue_create_buffer[bf16](FFN)
    var logits_d = ctx.enqueue_create_buffer[f32](VOCAB)
    var toks_d = ctx.enqueue_create_buffer[DType.int32](TMAX)

    var p_qf_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * QF)
    var p_h_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * H)
    var p_kv_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * KV)
    var p_32_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var p_32b_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var p_ffn_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * FFN)
    var p_ffn2_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * FFN)
    var p_v_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * VOCAB)

    var convstate_d = ctx.enqueue_create_buffer[f32](N_SSM * 3 * CONV)
    var sstate_d = ctx.enqueue_create_buffer[f32](N_SSM * NH_V * SSTATE * SSTATE)
    var kc_d = ctx.enqueue_create_buffer[f32](N_ATT * NKVH * TMAX * HD)
    var vc_d = ctx.enqueue_create_buffer[f32](N_ATT * NKVH * TMAX * HD)
    ctx.enqueue_memset(convstate_d, 0)
    ctx.enqueue_memset(sstate_d, 0)
    ctx.enqueue_memset(kc_d, 0)
    ctx.enqueue_memset(vc_d, 0)
    ctx.synchronize()

    # --- kernel bindings -----------------------------------------------------
    comptime rmsc_k = rmsnorm_cast[type_of(h2_layout), type_of(h_layout), type_of(h2_layout)]
    comptime embed_k = embed_lookup_pos[type_of(emb_layout), type_of(h2_layout), type_of(toks_layout)]
    comptime argmax_k = argmax_pos[type_of(vrow_layout), type_of(toks_layout)]

    comptime g_qf = matmul_skinny[bf16, type_of(h2_layout), type_of(w_h_qf), type_of(p_qf)]
    comptime g_h = matmul_skinny[bf16, type_of(h2_layout), type_of(w_h_h), type_of(p_h)]
    comptime g_kv = matmul_skinny[bf16, type_of(h2_layout), type_of(w_h_kv), type_of(p_kv)]
    comptime g_32 = matmul_skinny[bf16, type_of(h2_layout), type_of(w_h_32), type_of(p_32)]
    comptime g_ffn = matmul_skinny[bf16, type_of(h2_layout), type_of(w_h_ffn), type_of(p_ffn)]
    comptime g_down = matmul_skinny[bf16, type_of(row_major[1, FFN]()), type_of(w_ffn_h), type_of(p_h)]
    comptime g_head = matmul_skinny[bf16, type_of(h2_layout), type_of(w_h_v), type_of(p_v)]
    comptime r_qf = skinny_reduce[type_of(p_qf), type_of(c_qf)]
    comptime r_h = skinny_reduce[type_of(p_h), type_of(c_h)]
    comptime r_kv = skinny_reduce[type_of(p_kv), type_of(c_kv)]
    comptime r_add = skinny_reduce_add[type_of(p_h), type_of(h_layout)]
    comptime r_swiglu = skinny_reduce_swiglu_bf16[type_of(p_ffn), type_of(c_ffn)]
    comptime r_head = skinny_reduce[type_of(p_v), type_of(vrow_layout)]

    comptime rgates_k = ssm_reduce_gates[
        type_of(p_32), type_of(g32_layout), type_of(g32_layout)
    ]
    comptime conv_k = ssm_conv[
        type_of(conv_layout), type_of(cs_layout), type_of(cw_layout), type_of(conv_layout)
    ]
    comptime l2_k = ssm_qk_l2norm[type_of(conv_layout)]
    comptime delta_k = ssm_delta_step[
        type_of(s_layout), type_of(conv_layout), type_of(g32_layout), type_of(o_layout)
    ]
    comptime gated_k = ssm_gated_out_bf16[
        type_of(o_layout), type_of(h_layout), type_of(n128_layout), type_of(h_layout)
    ]
    comptime split_k = qgate_split[type_of(qf_layout), type_of(q_layout), type_of(h_layout)]
    comptime hrms_q = head_rmsnorm[type_of(q_layout), type_of(hd_layout)]
    comptime hrms_kv = head_rmsnorm[type_of(kvh_layout), type_of(hd_layout)]
    comptime rope_q = rope_yarn[type_of(q_layout)]
    comptime rope_k = rope_yarn[type_of(kvh_layout)]
    comptime append_k = kv_append[type_of(cache_layout), type_of(kvh_layout)]
    comptime att_k = attn_decode[type_of(q_layout), type_of(cache_layout), type_of(q_layout)]
    comptime gmul_k = gate_mul_cast[type_of(h_layout), type_of(h_layout), type_of(h_layout)]

    # --- decode loop ---------------------------------------------------------
    var X1 = TileTensor(x_d, h_layout)
    var X2 = TileTensor(x_d, h2_layout)
    var CurB2 = TileTensor(curb_d, h2_layout)
    var Logits2 = TileTensor(logits_d, vrow_layout)
    var Toks = TileTensor(toks_d, toks_layout)
    var Pv = TileTensor(p_v_d, p_v)

    var Embd = tens_bf16(ctx, wbuf, off[0], VOCAB * H, emb_layout)

    var n_total = len(prompt) + GEN_N
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](TMAX)
    ctx.synchronize()
    for i in range(TMAX):
        toks_h[i] = 0
    for i in range(len(prompt)):
        toks_h[i] = Int32(prompt[i])
    ctx.enqueue_copy(dst_buf=toks_d, src_buf=toks_h)
    ctx.synchronize()
    var t0 = perf_counter_ns()

    for pos in range(n_total - 1):
        ctx.enqueue_function[embed_k](
            Embd, X2, Toks, Int32(pos), Int32(H),
            grid_dim=(ceildiv(H, 256), 1), block_dim=256,
        )

        var w = 1
        var ssm_i = 0
        var att_i = 0
        for layer in range(N_LAYERS):
            # -- attention / ssm sub-block --
            ctx.enqueue_function[rmsc_k](
                X2, tens_f32(ctx, wbuf, off[w], H, h_layout), CurB2,
                Int32(H), Float32(1e-6), grid_dim=1, block_dim=256,
            )

            if is_attn(layer):
                var Wq = tens_bf16(ctx, wbuf, off[w + 1], H * QF, w_h_qf)
                var Wk = tens_bf16(ctx, wbuf, off[w + 2], H * KV, w_h_kv)
                var Wv = tens_bf16(ctx, wbuf, off[w + 3], H * KV, w_h_kv)
                var Qn = tens_f32(ctx, wbuf, off[w + 4], HD, hd_layout)
                var Kn = tens_f32(ctx, wbuf, off[w + 5], HD, hd_layout)
                var Wo = tens_bf16(ctx, wbuf, off[w + 6], H * H, w_h_h)
                var Pqf = TileTensor(p_qf_d, p_qf)
                var Pkv = TileTensor(p_kv_d, p_kv)
                var Ph = TileTensor(p_h_d, p_h)
                var Qf1 = TileTensor(qf_d, qf_layout)
                var Qf2 = TileTensor(qf_d, c_qf)
                var Q = TileTensor(q_d, q_layout)
                var Gate1 = TileTensor(gate_d, h_layout)
                var K2c = TileTensor(k_d, c_kv)
                var Khd = TileTensor(k_d, kvh_layout)
                var V2c = TileTensor(v_d, c_kv)
                var Vhd = TileTensor(v_d, kvh_layout)
                var kcb = DeviceBuffer[f32](
                    ctx, kc_d.unsafe_ptr() + att_i * NKVH * TMAX * HD,
                    NKVH * TMAX * HD, owning=False,
                )
                var vcb = DeviceBuffer[f32](
                    ctx, vc_d.unsafe_ptr() + att_i * NKVH * TMAX * HD,
                    NKVH * TMAX * HD, owning=False,
                )
                var Kc = TileTensor(kcb, cache_layout)
                var Vc = TileTensor(vcb, cache_layout)
                var Ao = TileTensor(ao_d, q_layout)
                var Ao1 = TileTensor(ao_d, h_layout)
                var AoB1 = TileTensor(resb_d, h_layout)
                var AoB2 = TileTensor(resb_d, h2_layout)

                ctx.enqueue_function[g_qf](CurB2, Wq, Pqf, Int32(1), Int32(QF), Int32(H), grid_dim=(ceildiv(QF, SBN), SPLITK), block_dim=SK_THREADS)
                ctx.enqueue_function[r_qf](Pqf, Qf2, Int32(1), Int32(QF), grid_dim=ceildiv(QF, 256), block_dim=256)
                ctx.enqueue_function[g_kv](CurB2, Wk, Pkv, Int32(1), Int32(KV), Int32(H), grid_dim=(ceildiv(KV, SBN), SPLITK), block_dim=SK_THREADS)
                ctx.enqueue_function[r_kv](Pkv, K2c, Int32(1), Int32(KV), grid_dim=ceildiv(KV, 256), block_dim=256)
                ctx.enqueue_function[g_kv](CurB2, Wv, Pkv, Int32(1), Int32(KV), Int32(H), grid_dim=(ceildiv(KV, SBN), SPLITK), block_dim=SK_THREADS)
                ctx.enqueue_function[r_kv](Pkv, V2c, Int32(1), Int32(KV), grid_dim=ceildiv(KV, 256), block_dim=256)
                ctx.enqueue_function[split_k](Qf1, Q, Gate1, grid_dim=NQH, block_dim=HD)
                ctx.enqueue_function[hrms_q](Q, Qn, Float32(1e-6), grid_dim=NQH, block_dim=HD)
                ctx.enqueue_function[hrms_kv](Khd, Kn, Float32(1e-6), grid_dim=NKVH, block_dim=HD)
                ctx.enqueue_function[rope_q](Q, Int32(pos), grid_dim=NQH, block_dim=32)
                ctx.enqueue_function[rope_k](Khd, Int32(pos), grid_dim=NKVH, block_dim=32)
                ctx.enqueue_function[append_k](Kc, Khd, Int32(pos), grid_dim=NKVH, block_dim=HD)
                ctx.enqueue_function[append_k](Vc, Vhd, Int32(pos), grid_dim=NKVH, block_dim=HD)
                ctx.enqueue_function[att_k](Q, Kc, Vc, Ao, Int32(pos + 1), Float32(0.0625), grid_dim=NQH, block_dim=HD)
                ctx.enqueue_function[gmul_k](Ao1, Gate1, AoB1, Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
                ctx.enqueue_function[g_h](AoB2, Wo, Ph, Int32(1), Int32(H), Int32(H), grid_dim=(ceildiv(H, SBN), SPLITK), block_dim=SK_THREADS)
                ctx.enqueue_function[r_add](Ph, X1, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
                att_i += 1
                w += 7
            else:
                var Wqkv = tens_bf16(ctx, wbuf, off[w + 1], H * CONV, w_h_qf)
                var Wz = tens_bf16(ctx, wbuf, off[w + 2], H * H, w_h_h)
                var Wa = tens_bf16(ctx, wbuf, off[w + 3], H * NH_V, w_h_32)
                var Wb = tens_bf16(ctx, wbuf, off[w + 4], H * NH_V, w_h_32)
                var Cw = tens_f32(ctx, wbuf, off[w + 5], CONV * 4, cw_layout)
                var SsmA = tens_f32(ctx, wbuf, off[w + 6], NH_V, g32_layout)
                var DtB = tens_f32(ctx, wbuf, off[w + 7], NH_V, g32_layout)
                var Nw = tens_f32(ctx, wbuf, off[w + 8], SSTATE, n128_layout)
                var Wsout = tens_bf16(ctx, wbuf, off[w + 9], H * H, w_h_h)
                var Pq = TileTensor(p_qf_d, p_qf)
                var Ph = TileTensor(p_h_d, p_h)
                var Pab = TileTensor(p_32_d, p_32)
                var Pab2 = TileTensor(p_32b_d, p_32)
                var Qkv1 = TileTensor(qkv_d, conv_layout)
                var Qkv2 = TileTensor(qkv_d, c_qf)
                var Z1 = TileTensor(z_d, h_layout)
                var Z2 = TileTensor(z_d, c_h)
                var Eg = TileTensor(eg_d, g32_layout)
                var Beta = TileTensor(beta_d, g32_layout)
                var csb = DeviceBuffer[f32](
                    ctx, convstate_d.unsafe_ptr() + ssm_i * 3 * CONV,
                    3 * CONV, owning=False,
                )
                var s0b = DeviceBuffer[f32](
                    ctx, sstate_d.unsafe_ptr() + ssm_i * NH_V * SSTATE * SSTATE,
                    NH_V * SSTATE * SSTATE, owning=False,
                )
                var csb_w = DeviceBuffer[f32](
                    ctx, convstate_d.unsafe_ptr() + ssm_i * 3 * CONV,
                    3 * CONV, owning=False,
                )
                var s0b_w = DeviceBuffer[f32](
                    ctx, sstate_d.unsafe_ptr() + ssm_i * NH_V * SSTATE * SSTATE,
                    NH_V * SSTATE * SSTATE, owning=False,
                )
                var Cs = TileTensor(csb, cs_layout)
                var Cs_w = TileTensor(csb_w, cs_layout)
                var S0 = TileTensor(s0b, s_layout)
                var S0_w = TileTensor(s0b_w, s_layout)
                var Conv = TileTensor(conv_d, conv_layout)
                var So = TileTensor(so_d, o_layout)
                var ResB1 = TileTensor(resb_d, h_layout)
                var ResB2 = TileTensor(resb_d, h2_layout)

                ctx.enqueue_function[g_qf](CurB2, Wqkv, Pq, Int32(1), Int32(CONV), Int32(H), grid_dim=(ceildiv(CONV, SBN), SPLITK), block_dim=SK_THREADS)
                ctx.enqueue_function[r_qf](Pq, Qkv2, Int32(1), Int32(CONV), grid_dim=ceildiv(CONV, 256), block_dim=256)
                ctx.enqueue_function[g_h](CurB2, Wz, Ph, Int32(1), Int32(H), Int32(H), grid_dim=(ceildiv(H, SBN), SPLITK), block_dim=SK_THREADS)
                ctx.enqueue_function[r_h](Ph, Z2, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
                ctx.enqueue_function[g_32](CurB2, Wa, Pab, Int32(1), Int32(NH_V), Int32(H), grid_dim=(1, SPLITK), block_dim=SK_THREADS)
                ctx.enqueue_function[g_32](CurB2, Wb, Pab2, Int32(1), Int32(NH_V), Int32(H), grid_dim=(1, SPLITK), block_dim=SK_THREADS)
                ctx.enqueue_function[rgates_k](Pab, Pab2, Eg, Beta, SsmA, DtB, grid_dim=1, block_dim=NH_V)
                ctx.enqueue_function[conv_k](Qkv1, Cs, Cw, Conv, Cs_w, grid_dim=ceildiv(CONV, 256), block_dim=256)
                ctx.enqueue_function[l2_k](Conv, grid_dim=NH_V, block_dim=SSTATE)
                ctx.enqueue_function[delta_k](S0, S0_w, Conv, Eg, Beta, So, grid_dim=NH_V, block_dim=SSTATE)
                ctx.enqueue_function[gated_k](So, Z1, Nw, ResB1, grid_dim=NH_V, block_dim=SSTATE)
                ctx.enqueue_function[g_h](ResB2, Wsout, Ph, Int32(1), Int32(H), Int32(H), grid_dim=(ceildiv(H, SBN), SPLITK), block_dim=SK_THREADS)
                ctx.enqueue_function[r_add](Ph, X1, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
                ssm_i += 1
                w += 10

            # -- ffn sub-block --
            ctx.enqueue_function[rmsc_k](
                X2, tens_f32(ctx, wbuf, off[w], H, h_layout), CurB2,
                Int32(H), Float32(1e-6), grid_dim=1, block_dim=256,
            )
            var Wfg = tens_bf16(ctx, wbuf, off[w + 1], H * FFN, w_h_ffn)
            var Wfu = tens_bf16(ctx, wbuf, off[w + 2], H * FFN, w_h_ffn)
            var Wfd = tens_bf16(ctx, wbuf, off[w + 3], FFN * H, w_ffn_h)
            var Pg = TileTensor(p_ffn_d, p_ffn)
            var Pu = TileTensor(p_ffn2_d, p_ffn)
            var Ph2 = TileTensor(p_h_d, p_h)
            var FgB2 = TileTensor(fgb_d, row_major[1, FFN]())
            ctx.enqueue_function[g_ffn](CurB2, Wfg, Pg, Int32(1), Int32(FFN), Int32(H), grid_dim=(ceildiv(FFN, SBN), SPLITK), block_dim=SK_THREADS)
            ctx.enqueue_function[g_ffn](CurB2, Wfu, Pu, Int32(1), Int32(FFN), Int32(H), grid_dim=(ceildiv(FFN, SBN), SPLITK), block_dim=SK_THREADS)
            ctx.enqueue_function[r_swiglu](Pg, Pu, FgB2, Int32(1), Int32(FFN), grid_dim=ceildiv(FFN, 256), block_dim=256)
            ctx.enqueue_function[g_down](FgB2, Wfd, Ph2, Int32(1), Int32(H), Int32(FFN), grid_dim=(ceildiv(H, SBN), SPLITK), block_dim=SK_THREADS)
            ctx.enqueue_function[r_add](Ph2, X1, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
            w += 4

            if pos == 0:
                var dbg = ctx.enqueue_create_host_buffer[f32](4)
                var db = DeviceBuffer[f32](ctx, x_d.unsafe_ptr(), 4, owning=False)
                ctx.enqueue_copy(dst_buf=dbg, src_buf=db)
                ctx.synchronize()
                print("dbg L", layer, ":", dbg[0], dbg[1], dbg[2], dbg[3])

        # -- head --
        if pos >= len(prompt) - 1:
            ctx.enqueue_function[rmsc_k](
                X2, tens_f32(ctx, wbuf, off[w], H, h_layout), CurB2,
                Int32(H), Float32(1e-6), grid_dim=1, block_dim=256,
            )
            var Whead = tens_bf16(ctx, wbuf, off[w + 1], H * VOCAB, w_h_v)
            ctx.enqueue_function[g_head](CurB2, Whead, Pv, Int32(1), Int32(VOCAB), Int32(H), grid_dim=(ceildiv(VOCAB, SBN), SPLITK), block_dim=SK_THREADS)
            ctx.enqueue_function[r_head](Pv, Logits2, Int32(1), Int32(VOCAB), grid_dim=ceildiv(VOCAB, 256), block_dim=256)
            ctx.enqueue_function[argmax_k](Logits2, Toks, Int32(VOCAB), Int32(pos + 1), grid_dim=1, block_dim=256)

    var t_host = Float64(perf_counter_ns() - t0) / 1e9
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    print("host_enqueue_s:", t_host, " gpu_total_s:", dt)
    ctx.enqueue_copy(dst_buf=toks_h, src_buf=toks_d)
    ctx.synchronize()
    var generated = List[Int]()
    for i in range(len(prompt), n_total):
        generated.append(Int(toks_h[i]))
        print("gen:", Int(toks_h[i]))
    print("tokens:", len(generated), " total_s:", dt, " tok/s:", Float64(n_total - 1) / dt)
    var line = String("")
    for i in range(len(generated)):
        line += String(generated[i]) + " "
    print("GENERATED:", line)
