"""baro engine: full-model greedy decode for qwen35 (Qwythos-9B), milestone 4.

Loads .work/engine-pack/ (fixed tensor order, 2D bf16 weights pre-transposed
to B-layout), reads prompt token ids from .work/engine-pack/prompt-tokens.txt,
runs the 32-block hybrid stack (24 gated-delta-net + 8 gated full-attention,
MTP block skipped) over a window of up to MROWS tokens at a time, and prints
greedy token ids.

Parity target: byte-identical token ids vs llama.cpp on the same GGUF.
"""
from std.math import ceildiv
from std.os import getenv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from elementwise import (
    amar_rmsnorm, amar_rmsnorm_cast, amar_embed_lookup_pos, amar_argmax_pos, amar_tok_copy,
)
from matmul_skinny import (
    amar_matmul_skinny_q8row, amar_skinny_reduce, amar_skinny_reduce_add,
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


def tens_q8q[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.int8, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.int8](
        ctx, (wbuf.unsafe_ptr() + o).unsafe_bitcast[Scalar[DType.int8]](), n, owning=False
    )
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.int8, LT, MutAnyOrigin]](t)


def tens_q8s[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.float16, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.float16](
        ctx, (wbuf.unsafe_ptr() + o + n).unsafe_bitcast[Scalar[DType.float16]](), n // 32, owning=False
    )
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.float16, LT, MutAnyOrigin]](t)


def tens_f32[
    LT: TensorLayout
](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT
) -> TileTensor[DType.float32, LT, MutAnyOrigin]:
    var b = wf32(ctx, wbuf, o, n)
    var t = TileTensor(b, lt)
    return rebind[TileTensor[DType.float32, LT, MutAnyOrigin]](t)


def row_f32[
    LT: TensorLayout
](
    ctx: DeviceContext, b: DeviceBuffer[f32], o: Int, n: Int, lt: LT
) -> TileTensor[f32, LT, MutAnyOrigin]:
    var s = DeviceBuffer[f32](ctx, b.unsafe_ptr() + o, n, owning=False)
    var t = TileTensor(s, lt)
    return rebind[TileTensor[f32, LT, MutAnyOrigin]](t)


def row_bf16[
    LT: TensorLayout
](
    ctx: DeviceContext, b: DeviceBuffer[bf16], o: Int, n: Int, lt: LT
) -> TileTensor[bf16, LT, MutAnyOrigin]:
    var s = DeviceBuffer[bf16](ctx, b.unsafe_ptr() + o, n, owning=False)
    var t = TileTensor(s, lt)
    return rebind[TileTensor[bf16, LT, MutAnyOrigin]](t)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var packdir = getenv("BARO_PACK", ".work/engine-pack-q8")
    var PACK = packdir + "/pack.bin"

    # --- offset table from the pack index (tools/engine-pack.py order) ----
    var off = List[Int]()
    var total = 0
    with open(packdir + "/index.txt", "r") as f:
        for line in f.read().splitlines():
            var parts = line.split(" ")
            if len(parts) < 4:
                continue
            var n = Int(parts[3])
            var dt = String(parts[1])
            off.append(Int(parts[2]))
            if dt == "bf16":
                total += n * B2
            elif dt == "f32":
                total += n * B4
            elif dt == "q8":
                total += n + (n // 32) * 2
            else:
                raise Error("unknown pack dtype " + dt)
    var e = len(off) - 15

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
    with open(packdir + "/prompt-tokens.txt", "r") as f:
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

    var kcfg = 4
    try:
        with open(packdir + "/spec-k.txt", "r") as f:
            var kd = f.read_bytes()
            var kv_ = 0
            var kh = False
            for i in range(len(kd)):
                var b = Int(kd[i])
                if b >= 48 and b <= 57:
                    kv_ = kv_ * 10 + (b - 48)
                    kh = True
                elif kh:
                    break
            if kh:
                kcfg = kv_
    except:
        kcfg = 4
    if kcfg < 1:
        kcfg = 1
    if kcfg > KMAX:
        kcfg = KMAX
    print("spec k:", kcfg)

    # --- activations / state -------------------------------------------------
    var x_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var curb_d = ctx.enqueue_create_buffer[bf16](MROWS * H)
    var qkv_d = ctx.enqueue_create_buffer[f32](MROWS * CONV)
    var z_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var eg_d = ctx.enqueue_create_buffer[f32](NH_V)
    var beta_d = ctx.enqueue_create_buffer[f32](NH_V)
    var conv_d = ctx.enqueue_create_buffer[f32](CONV)
    var so_d = ctx.enqueue_create_buffer[f32](NH_V * SSTATE)
    var resb_d = ctx.enqueue_create_buffer[bf16](MROWS * H)
    var qf_d = ctx.enqueue_create_buffer[f32](MROWS * QF)
    var q_d = ctx.enqueue_create_buffer[f32](MROWS * NQH * HD)
    var gate_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var k_d = ctx.enqueue_create_buffer[f32](MROWS * KV)
    var v_d = ctx.enqueue_create_buffer[f32](MROWS * KV)
    var ao_d = ctx.enqueue_create_buffer[f32](MROWS * NQH * HD)
    var fgb_d = ctx.enqueue_create_buffer[bf16](MROWS * FFN)
    var logits_d = ctx.enqueue_create_buffer[f32](MROWS * VOCAB)
    var toks_d = ctx.enqueue_create_buffer[DType.int32](TMAX)
    var hn_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var de_d = ctx.enqueue_create_buffer[f32](H)
    var mh_d = ctx.enqueue_create_buffer[f32](H)
    var cc_d = ctx.enqueue_create_buffer[bf16](MROWS * QF)
    var dtok_d = ctx.enqueue_create_buffer[DType.int32](KMAX + 1)

    var p_qf_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * QF)
    var p_h_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * H)
    var p_kv_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * KV)
    var p_32_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var p_32b_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var p_ffn_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * FFN)
    var p_ffn2_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * FFN)
    var p_v_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * VOCAB)

    var convstate_d = ctx.enqueue_create_buffer[f32](SLOTS * CONV_SLOT)
    var sstate_d = ctx.enqueue_create_buffer[f32](SLOTS * SSM_SLOT)
    var kc_d = ctx.enqueue_create_buffer[f32](N_ATT * ATT32)
    var vc_d = ctx.enqueue_create_buffer[f32](N_ATT * ATT32)
    var kc32_d = ctx.enqueue_create_buffer[f32](ATT32)
    var vc32_d = ctx.enqueue_create_buffer[f32](ATT32)
    ctx.enqueue_memset(convstate_d, 0)
    ctx.enqueue_memset(sstate_d, 0)
    ctx.enqueue_memset(kc_d, 0)
    ctx.enqueue_memset(vc_d, 0)
    ctx.enqueue_memset(kc32_d, 0)
    ctx.enqueue_memset(vc32_d, 0)
    ctx.synchronize()

    # --- kernel bindings -----------------------------------------------------
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
    comptime g_eh_1 = amar_matmul_skinny_q8row[4, 1, type_of(qfm_layout), type_of(q_qf_h), type_of(s_qf_h), type_of(p_h)]

    comptime g_qf_1 = amar_matmul_skinny_q8row[4, 1, type_of(xm_layout), type_of(q_h_qf), type_of(s_h_qf), type_of(p_qf)]
    comptime g_h_1 = amar_matmul_skinny_q8row[4, 1, type_of(xm_layout), type_of(q_h_h), type_of(s_h_h), type_of(p_h)]
    comptime g_kv_1 = amar_matmul_skinny_q8row[4, 1, type_of(xm_layout), type_of(q_h_kv), type_of(s_h_kv), type_of(p_kv)]
    comptime g_32_1 = amar_matmul_skinny_q8row[4, 1, type_of(xm_layout), type_of(q_h_32), type_of(s_h_32), type_of(p_32)]
    comptime g_ffn_1 = amar_matmul_skinny_q8row[4, 1, type_of(xm_layout), type_of(q_h_ffn), type_of(s_h_ffn), type_of(p_ffn)]
    comptime g_down_1 = amar_matmul_skinny_q8row[4, 1, type_of(ffnm_layout), type_of(q_ffn_h), type_of(s_ffn_h), type_of(p_h)]
    comptime g_head_1 = amar_matmul_skinny_q8row[4, 1, type_of(xm_layout), type_of(q_h_v), type_of(s_h_v), type_of(p_v)]

    comptime g_qf_v = amar_matmul_skinny_q8row[4, SM, type_of(xm_layout), type_of(q_h_qf), type_of(s_h_qf), type_of(p_qf)]
    comptime g_h_v = amar_matmul_skinny_q8row[4, SM, type_of(xm_layout), type_of(q_h_h), type_of(s_h_h), type_of(p_h)]
    comptime g_kv_v = amar_matmul_skinny_q8row[4, SM, type_of(xm_layout), type_of(q_h_kv), type_of(s_h_kv), type_of(p_kv)]
    comptime g_32_v = amar_matmul_skinny_q8row[4, SM, type_of(xm_layout), type_of(q_h_32), type_of(s_h_32), type_of(p_32)]
    comptime g_ffn_v = amar_matmul_skinny_q8row[4, SM, type_of(xm_layout), type_of(q_h_ffn), type_of(s_h_ffn), type_of(p_ffn)]
    comptime g_down_v = amar_matmul_skinny_q8row[4, SM, type_of(ffnm_layout), type_of(q_ffn_h), type_of(s_ffn_h), type_of(p_h)]
    comptime g_head_v = amar_matmul_skinny_q8row[4, SM, type_of(xm_layout), type_of(q_h_v), type_of(s_h_v), type_of(p_v)]

    comptime r_qf = amar_skinny_reduce[type_of(p_qf), type_of(qfm_layout), 1]
    comptime r_h = amar_skinny_reduce[type_of(p_h), type_of(xm_layout), 1]
    comptime r_kv = amar_skinny_reduce[type_of(p_kv), type_of(kvm_flat), 1]
    comptime r_add = amar_skinny_reduce_add[type_of(p_h), type_of(xm_layout), 1]
    comptime r_swiglu = amar_skinny_reduce_swiglu_bf16[type_of(p_ffn), type_of(ffnm_layout), 1]
    comptime r_head = amar_skinny_reduce[type_of(p_v), type_of(vm_layout), 1]

    comptime rgates_k = amar_ssm_reduce_gates[
        type_of(p_32), type_of(g32_layout), type_of(g32_layout)
    ]
    comptime conv_k = amar_ssm_conv[
        type_of(conv_layout), type_of(cs_layout), type_of(cw_layout), type_of(conv_layout)
    ]
    comptime l2_k = amar_ssm_qk_l2norm[type_of(conv_layout)]
    comptime delta_k = amar_ssm_delta_step[
        type_of(s_layout), type_of(conv_layout), type_of(g32_layout), type_of(o_layout)
    ]
    comptime gated_k = amar_ssm_gated_out_bf16[
        type_of(o_layout), type_of(h_layout), type_of(n128_layout), type_of(h_layout)
    ]
    comptime split_k = amar_qgate_split[type_of(qfm_layout), type_of(qm_layout), type_of(xflat_layout)]
    comptime hrms_q = amar_head_rmsnorm[type_of(qm_layout), type_of(hd_layout)]
    comptime hrms_kv = amar_head_rmsnorm[type_of(kvm_layout), type_of(hd_layout)]
    comptime rope_q = amar_rope_yarn[type_of(qm_layout)]
    comptime rope_k = amar_rope_yarn[type_of(kvm_layout)]
    comptime append_k = amar_kv_append[type_of(cache_layout), type_of(kvm_layout)]
    comptime att_k = amar_attn_decode[type_of(qm_layout), type_of(cache_layout), type_of(qm_layout)]
    comptime gmul_k = amar_gate_mul_cast[type_of(xflat_layout), type_of(xflat_layout), type_of(xflat_layout)]

    # --- decode loop ---------------------------------------------------------
    var Xm = TileTensor(x_d, xm_layout)
    var CurBm = TileTensor(curb_d, xm_layout)
    var Logitsm = TileTensor(logits_d, vm_layout)
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

    # SSM/conv state lives in a SLOTS-deep ring: row r of a window reads slot
    # (ring + r) and writes (ring + r + 1), so a k-token window never clobbers
    # the state a rejected token would have to roll back to.
    var ring = 0
    var pos = 0
    var t_prefill_end = t0
    # BARO_PROFILE=1: synchronize at sub-block boundaries and attribute GPU
    # time to attn / ssm / ffn / head. Off by default; the timed path is
    # untouched.
    var prof = getenv("BARO_PROFILE", "0") != "0"
    var pf2 = getenv("BARO_PROFILE", "0") == "2"
    var pc = [0, 0, 0, 0, 0, 0, 0, 0]
    var tq = t0
    var pf_att = 0
    var pf_ssm = 0
    var pf_ffn = 0
    var pf_head = 0
    var tp = t0
    var prefill_done = False
    while pos < n_total - 1:
        var m = 1
        if pos + 1 < len(prompt):
            m = min(MROWS, len(prompt) - 1 - pos)

        ctx.enqueue_function[embed_k](
            Embd, Xm, Toks, Int32(pos), Int32(H),
            grid_dim=(ceildiv(H, 256), m), block_dim=256,
        )

        var w = 1
        var ssm_i = 0
        var att_i = 0
        for layer in range(N_LAYERS):
            if prof:
                ctx.synchronize()
                tp = perf_counter_ns()
                tq = tp
            # -- attention / ssm sub-block --
            ctx.enqueue_function[rmsc_k](
                Xm, tens_f32(ctx, wbuf, off[w], H, h_layout), CurBm,
                Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
            )

            if is_attn(layer):
                var Wqq = tens_q8q(ctx, wbuf, off[w + 1], H * QF, q_h_qf)
                var Wqs = tens_q8s(ctx, wbuf, off[w + 1], H * QF, s_h_qf)
                var Wkq = tens_q8q(ctx, wbuf, off[w + 2], H * KV, q_h_kv)
                var Wks = tens_q8s(ctx, wbuf, off[w + 2], H * KV, s_h_kv)
                var Wvq = tens_q8q(ctx, wbuf, off[w + 3], H * KV, q_h_kv)
                var Wvs = tens_q8s(ctx, wbuf, off[w + 3], H * KV, s_h_kv)
                var Qn = tens_f32(ctx, wbuf, off[w + 4], HD, hd_layout)
                var Kn = tens_f32(ctx, wbuf, off[w + 5], HD, hd_layout)
                var Woq = tens_q8q(ctx, wbuf, off[w + 6], H * H, q_h_h)
                var Wos = tens_q8s(ctx, wbuf, off[w + 6], H * H, s_h_h)
                var Pqf = TileTensor(p_qf_d, p_qf)
                var Pkv = TileTensor(p_kv_d, p_kv)
                var Ph = TileTensor(p_h_d, p_h)
                var Qfm = TileTensor(qf_d, qfm_layout)
                var Q = TileTensor(q_d, qm_layout)
                var Gate = TileTensor(gate_d, xflat_layout)
                var Kflat = TileTensor(k_d, kvm_flat)
                var Khd = TileTensor(k_d, kvm_layout)
                var Vflat = TileTensor(v_d, kvm_flat)
                var Vhd = TileTensor(v_d, kvm_layout)
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
                var Ao = TileTensor(ao_d, qm_layout)
                var Aoflat = TileTensor(ao_d, xflat_layout)
                var AoB = TileTensor(resb_d, xflat_layout)
                var AoBm = TileTensor(resb_d, xm_layout)

                if m == 1:
                    ctx.enqueue_function[g_qf_1](CurBm, Wqq, Wqs, Pqf, Int32(m), Int32(QF), Int32(H), grid_dim=ceildiv(QF, ROW_WAVES), block_dim=ROW_THREADS)
                else:
                    ctx.enqueue_function[g_qf_v](CurBm, Wqq, Wqs, Pqf, Int32(m), Int32(QF), Int32(H), grid_dim=ceildiv(QF, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[r_qf](Pqf, Qfm, Int32(m), Int32(QF), grid_dim=ceildiv(m * QF, 256), block_dim=256)
                if m == 1:
                    ctx.enqueue_function[g_kv_1](CurBm, Wkq, Wks, Pkv, Int32(m), Int32(KV), Int32(H), grid_dim=ceildiv(KV, ROW_WAVES), block_dim=ROW_THREADS)
                else:
                    ctx.enqueue_function[g_kv_v](CurBm, Wkq, Wks, Pkv, Int32(m), Int32(KV), Int32(H), grid_dim=ceildiv(KV, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[r_kv](Pkv, Kflat, Int32(m), Int32(KV), grid_dim=ceildiv(m * KV, 256), block_dim=256)
                if m == 1:
                    ctx.enqueue_function[g_kv_1](CurBm, Wvq, Wvs, Pkv, Int32(m), Int32(KV), Int32(H), grid_dim=ceildiv(KV, ROW_WAVES), block_dim=ROW_THREADS)
                else:
                    ctx.enqueue_function[g_kv_v](CurBm, Wvq, Wvs, Pkv, Int32(m), Int32(KV), Int32(H), grid_dim=ceildiv(KV, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[r_kv](Pkv, Vflat, Int32(m), Int32(KV), grid_dim=ceildiv(m * KV, 256), block_dim=256)
                ctx.enqueue_function[split_k](Qfm, Q, Gate, grid_dim=(NQH, m), block_dim=HD)
                ctx.enqueue_function[hrms_q](Q, Qn, Float32(1e-6), grid_dim=m * NQH, block_dim=HD)
                ctx.enqueue_function[hrms_kv](Khd, Kn, Float32(1e-6), grid_dim=m * NKVH, block_dim=HD)
                ctx.enqueue_function[rope_q](Q, Int32(pos), Int32(NQH), grid_dim=(NQH, m), block_dim=32)
                ctx.enqueue_function[rope_k](Khd, Int32(pos), Int32(NKVH), grid_dim=(NKVH, m), block_dim=32)
                ctx.enqueue_function[append_k](Kc, Khd, Int32(pos), grid_dim=(NKVH, m), block_dim=HD)
                ctx.enqueue_function[append_k](Vc, Vhd, Int32(pos), grid_dim=(NKVH, m), block_dim=HD)
                ctx.enqueue_function[att_k](Q, Kc, Vc, Ao, Int32(pos + 1), Float32(0.0625), grid_dim=(NQH, m), block_dim=HD)
                ctx.enqueue_function[gmul_k](Aoflat, Gate, AoB, Int32(m * H), grid_dim=ceildiv(m * H, 256), block_dim=256)
                if m == 1:
                    ctx.enqueue_function[g_h_1](AoBm, Woq, Wos, Ph, Int32(m), Int32(H), Int32(H), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
                else:
                    ctx.enqueue_function[g_h_v](AoBm, Woq, Wos, Ph, Int32(m), Int32(H), Int32(H), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[r_add](Ph, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)
                att_i += 1
                w += 7
            else:
                var Wqkvq = tens_q8q(ctx, wbuf, off[w + 1], H * CONV, q_h_qf)
                var Wqkvs = tens_q8s(ctx, wbuf, off[w + 1], H * CONV, s_h_qf)
                var Wzq = tens_q8q(ctx, wbuf, off[w + 2], H * H, q_h_h)
                var Wzs = tens_q8s(ctx, wbuf, off[w + 2], H * H, s_h_h)
                var Waq = tens_q8q(ctx, wbuf, off[w + 3], H * NH_V, q_h_32)
                var Was = tens_q8s(ctx, wbuf, off[w + 3], H * NH_V, s_h_32)
                var Wbq = tens_q8q(ctx, wbuf, off[w + 4], H * NH_V, q_h_32)
                var Wbs = tens_q8s(ctx, wbuf, off[w + 4], H * NH_V, s_h_32)
                var Cw = tens_f32(ctx, wbuf, off[w + 5], CONV * 4, cw_layout)
                var SsmA = tens_f32(ctx, wbuf, off[w + 6], NH_V, g32_layout)
                var DtB = tens_f32(ctx, wbuf, off[w + 7], NH_V, g32_layout)
                var Nw = tens_f32(ctx, wbuf, off[w + 8], SSTATE, n128_layout)
                var Wsoutq = tens_q8q(ctx, wbuf, off[w + 9], H * H, q_h_h)
                var Wsouts = tens_q8s(ctx, wbuf, off[w + 9], H * H, s_h_h)
                var Pq = TileTensor(p_qf_d, p_qf)
                var Ph = TileTensor(p_h_d, p_h)
                var Pab = TileTensor(p_32_d, p_32)
                var Pab2 = TileTensor(p_32b_d, p_32)
                var Qkvm = TileTensor(qkv_d, qfm_layout)
                var Zm = TileTensor(z_d, xm_layout)
                var Eg = TileTensor(eg_d, g32_layout)
                var Beta = TileTensor(beta_d, g32_layout)
                var Conv = TileTensor(conv_d, conv_layout)
                var So = TileTensor(so_d, o_layout)
                var ResBm = TileTensor(resb_d, xm_layout)

                if m == 1:
                    ctx.enqueue_function[g_qf_1](CurBm, Wqkvq, Wqkvs, Pq, Int32(m), Int32(CONV), Int32(H), grid_dim=ceildiv(CONV, ROW_WAVES), block_dim=ROW_THREADS)
                    ctx.enqueue_function[g_h_1](CurBm, Wzq, Wzs, Ph, Int32(m), Int32(H), Int32(H), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
                    ctx.enqueue_function[g_32_1](CurBm, Waq, Was, Pab, Int32(m), Int32(NH_V), Int32(H), grid_dim=ceildiv(NH_V, ROW_WAVES), block_dim=ROW_THREADS)
                    ctx.enqueue_function[g_32_1](CurBm, Wbq, Wbs, Pab2, Int32(m), Int32(NH_V), Int32(H), grid_dim=ceildiv(NH_V, ROW_WAVES), block_dim=ROW_THREADS)
                else:
                    ctx.enqueue_function[g_qf_v](CurBm, Wqkvq, Wqkvs, Pq, Int32(m), Int32(CONV), Int32(H), grid_dim=ceildiv(CONV, ROW_WAVES), block_dim=ROW_THREADS)
                    ctx.enqueue_function[g_h_v](CurBm, Wzq, Wzs, Ph, Int32(m), Int32(H), Int32(H), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
                    ctx.enqueue_function[g_32_v](CurBm, Waq, Was, Pab, Int32(m), Int32(NH_V), Int32(H), grid_dim=ceildiv(NH_V, ROW_WAVES), block_dim=ROW_THREADS)
                    ctx.enqueue_function[g_32_v](CurBm, Wbq, Wbs, Pab2, Int32(m), Int32(NH_V), Int32(H), grid_dim=ceildiv(NH_V, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[r_qf](Pq, Qkvm, Int32(m), Int32(CONV), grid_dim=ceildiv(m * CONV, 256), block_dim=256)
                ctx.enqueue_function[r_h](Ph, Zm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)

                for r in range(m):
                    var rs = (ring + r) % SLOTS
                    var ws = (ring + r + 1) % SLOTS
                    var csb = DeviceBuffer[f32](
                        ctx,
                        convstate_d.unsafe_ptr() + rs * CONV_SLOT + ssm_i * 3 * CONV,
                        3 * CONV, owning=False,
                    )
                    var csb_w = DeviceBuffer[f32](
                        ctx,
                        convstate_d.unsafe_ptr() + ws * CONV_SLOT + ssm_i * 3 * CONV,
                        3 * CONV, owning=False,
                    )
                    var s0b = DeviceBuffer[f32](
                        ctx,
                        sstate_d.unsafe_ptr() + rs * SSM_SLOT
                        + ssm_i * NH_V * SSTATE * SSTATE,
                        NH_V * SSTATE * SSTATE, owning=False,
                    )
                    var s0b_w = DeviceBuffer[f32](
                        ctx,
                        sstate_d.unsafe_ptr() + ws * SSM_SLOT
                        + ssm_i * NH_V * SSTATE * SSTATE,
                        NH_V * SSTATE * SSTATE, owning=False,
                    )
                    var Cs = TileTensor(csb, cs_layout)
                    var Cs_w = TileTensor(csb_w, cs_layout)
                    var S0 = TileTensor(s0b, s_layout)
                    var S0_w = TileTensor(s0b_w, s_layout)
                    var Qkv1 = row_f32(ctx, qkv_d, r * CONV, CONV, conv_layout)
                    var Z1 = row_f32(ctx, z_d, r * H, H, h_layout)
                    var ResB1 = row_bf16(ctx, resb_d, r * H, H, h_layout)
                    if pf2:
                        ctx.synchronize()
                        var nw = perf_counter_ns()
                        pc[0] += Int(nw - tq)
                        tq = nw
                    ctx.enqueue_function[rgates_k](Pab, Pab2, Eg, Beta, SsmA, DtB, Int32(r), grid_dim=1, block_dim=NH_V)
                    if pf2:
                        ctx.synchronize()
                        var nw = perf_counter_ns()
                        pc[1] += Int(nw - tq)
                        tq = nw
                    ctx.enqueue_function[conv_k](Qkv1, Cs, Cw, Conv, Cs_w, grid_dim=ceildiv(CONV, 256), block_dim=256)
                    if pf2:
                        ctx.synchronize()
                        var nw = perf_counter_ns()
                        pc[2] += Int(nw - tq)
                        tq = nw
                    ctx.enqueue_function[l2_k](Conv, grid_dim=NH_V, block_dim=SSTATE)
                    if pf2:
                        ctx.synchronize()
                        var nw = perf_counter_ns()
                        pc[3] += Int(nw - tq)
                        tq = nw
                    ctx.enqueue_function[delta_k](S0, S0_w, Conv, Eg, Beta, So, grid_dim=NH_V, block_dim=SSTATE)
                    if pf2:
                        ctx.synchronize()
                        var nw = perf_counter_ns()
                        pc[4] += Int(nw - tq)
                        tq = nw
                    ctx.enqueue_function[gated_k](So, Z1, Nw, ResB1, grid_dim=NH_V, block_dim=SSTATE)

                if pf2:
                    ctx.synchronize()
                    var nw = perf_counter_ns()
                    pc[5] += Int(nw - tq)
                    tq = nw
                if m == 1:
                    ctx.enqueue_function[g_h_1](ResBm, Wsoutq, Wsouts, Ph, Int32(m), Int32(H), Int32(H), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
                else:
                    ctx.enqueue_function[g_h_v](ResBm, Wsoutq, Wsouts, Ph, Int32(m), Int32(H), Int32(H), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[r_add](Ph, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)
                ssm_i += 1
                w += 10

            if prof:
                ctx.synchronize()
                var now = perf_counter_ns()
                if is_attn(layer):
                    pf_att += Int(now - tp)
                else:
                    pf_ssm += Int(now - tp)
                    if pf2:
                        pc[6] += Int(now - tq)
                tp = now
                tq = now
            # -- ffn sub-block --
            ctx.enqueue_function[rmsc_k](
                Xm, tens_f32(ctx, wbuf, off[w], H, h_layout), CurBm,
                Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
            )
            var Wfgq = tens_q8q(ctx, wbuf, off[w + 1], H * FFN, q_h_ffn)
            var Wfgs = tens_q8s(ctx, wbuf, off[w + 1], H * FFN, s_h_ffn)
            var Wfuq = tens_q8q(ctx, wbuf, off[w + 2], H * FFN, q_h_ffn)
            var Wfus = tens_q8s(ctx, wbuf, off[w + 2], H * FFN, s_h_ffn)
            var Wfdq = tens_q8q(ctx, wbuf, off[w + 3], FFN * H, q_ffn_h)
            var Wfds = tens_q8s(ctx, wbuf, off[w + 3], FFN * H, s_ffn_h)
            var Pg = TileTensor(p_ffn_d, p_ffn)
            var Pu = TileTensor(p_ffn2_d, p_ffn)
            var Ph2 = TileTensor(p_h_d, p_h)
            var FgBm = TileTensor(fgb_d, ffnm_layout)
            if m == 1:
                ctx.enqueue_function[g_ffn_1](CurBm, Wfgq, Wfgs, Pg, Int32(1), Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[g_ffn_1](CurBm, Wfuq, Wfus, Pu, Int32(1), Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
            else:
                ctx.enqueue_function[g_ffn_v](CurBm, Wfgq, Wfgs, Pg, Int32(m), Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
                ctx.enqueue_function[g_ffn_v](CurBm, Wfuq, Wfus, Pu, Int32(m), Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[r_swiglu](Pg, Pu, FgBm, Int32(m), Int32(FFN), grid_dim=ceildiv(m * FFN, 256), block_dim=256)
            if m == 1:
                ctx.enqueue_function[g_down_1](FgBm, Wfdq, Wfds, Ph2, Int32(m), Int32(H), Int32(FFN), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
            else:
                ctx.enqueue_function[g_down_v](FgBm, Wfdq, Wfds, Ph2, Int32(m), Int32(H), Int32(FFN), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[r_add](Ph2, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)
            w += 4
            if prof:
                ctx.synchronize()
                var now = perf_counter_ns()
                pf_ffn += Int(now - tp)
                tp = now

        # -- head --
        if prof:
            ctx.synchronize()
            tp = perf_counter_ns()
        if pos + m >= len(prompt):
            ctx.enqueue_function[rmsc_k](
                Xm, tens_f32(ctx, wbuf, off[w], H, h_layout), CurBm,
                Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
            )
            # f32 copy of the post-final-norm hidden state (pre-LM-head), for
            # the MTP draft head below — only the last window's last row
            # survives to be used, since MTP drafts off the final trunk token.
            var Hnm = TileTensor(hn_d, xm_layout)
            ctx.enqueue_function[rms_m](
                Xm, tens_f32(ctx, wbuf, off[w], H, h_layout), Hnm,
                Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
            )
            var Wheadq = tens_q8q(ctx, wbuf, off[w + 1], H * VOCAB, q_h_v)
            var Wheads = tens_q8s(ctx, wbuf, off[w + 1], H * VOCAB, s_h_v)
            if m == 1:
                ctx.enqueue_function[g_head_1](CurBm, Wheadq, Wheads, Pv, Int32(m), Int32(VOCAB), Int32(H), grid_dim=ceildiv(VOCAB, ROW_WAVES), block_dim=ROW_THREADS)
            else:
                ctx.enqueue_function[g_head_v](CurBm, Wheadq, Wheads, Pv, Int32(m), Int32(VOCAB), Int32(H), grid_dim=ceildiv(VOCAB, ROW_WAVES), block_dim=ROW_THREADS)
            ctx.enqueue_function[r_head](Pv, Logitsm, Int32(m), Int32(VOCAB), grid_dim=ceildiv(m * VOCAB, 256), block_dim=256)
            ctx.enqueue_function[argmax_k](Logitsm, Toks, Int32(VOCAB), Int32(pos + 1), grid_dim=m, block_dim=256)
            if not prefill_done:
                ctx.synchronize()
                t_prefill_end = perf_counter_ns()
                prefill_done = True
            if prof:
                ctx.synchronize()
                pf_head += Int(perf_counter_ns() - tp)

        # advance by the window width; once verify lands this becomes the
        # accepted-token count, which is what makes rollback free.
        ring = (ring + m) % SLOTS
        pos += m

    var t_host = Float64(perf_counter_ns() - t0) / 1e9
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    print("host_enqueue_s:", t_host, " gpu_total_s:", dt)
    if prof:
        var tot = Float64(pf_att + pf_ssm + pf_ffn + pf_head)
        print("profile: attn", Float64(pf_att) / 1e9, Float64(pf_att) / tot)
        print("profile: ssm", Float64(pf_ssm) / 1e9, Float64(pf_ssm) / tot)
        print("profile: ffn", Float64(pf_ffn) / 1e9, Float64(pf_ffn) / tot)
        print("profile: head", Float64(pf_head) / 1e9, Float64(pf_head) / tot)
    if pf2:
        var names = ["gemm4+reduce2", "rgates", "conv", "l2", "delta", "gated", "out_gemm+add", "-"]
        var st = Float64(pc[0] + pc[1] + pc[2] + pc[3] + pc[4] + pc[5] + pc[6])
        for i in range(7):
            print("ssm-kernel:", names[i], Float64(pc[i]) / 1e9, Float64(pc[i]) / st)
    ctx.enqueue_copy(dst_buf=toks_h, src_buf=toks_d)
    ctx.synchronize()
    var generated = List[Int]()
    for i in range(len(prompt), n_total):
        generated.append(Int(toks_h[i]))
    var prefill_s = Float64(t_prefill_end - t0) / 1e9
    var decode_s = dt - prefill_s
    # tok/s_gen is the only number comparable to llama.cpp: it divides
    # GEN_N - 1 by decode time alone, matching timings.predicted_per_second.
    # tok/s_total includes prefill and is reported for completeness only --
    # it is not the engine's throughput against any external baseline.
    print("tokens:", len(generated), " prefill_s:", prefill_s, " decode_s:", decode_s)
    print("tok/s_total:", Float64(GEN_N) / dt, " tok/s_gen:", Float64(GEN_N - 1) / decode_s)
    var line = String("")
    for i in range(len(generated)):
        line += String(generated[i]) + " "
    print("GENERATED:", line)

    # =========================================================================
    # MTP (NextN) draft head, blk.32 — computed and dumped, never consumed.
    # Trunk decode is finished and its output already copied to host above,
    # so every scratch buffer below (Xm, CurBm, qf/q/k/v/ao/resb/fgb, p_*_d,
    # logits_d, cc_d, hn_d, de_d, dtok_d) is free to reuse. Only kc32_d/vc32_d
    # (blk.32's own, still-zeroed KV cache) are written; trunk's kc_d/vc_d,
    # convstate_d, sstate_d and toks_d are left untouched.
    # =========================================================================
    var last_tok = generated[len(generated) - 1]
    var Dtok = TileTensor(dtok_d, dtok_layout)
    ctx.enqueue_function[tokcp_b](
        Toks, Dtok, Int32(n_total - 1), Int32(0), Int32(1),
        grid_dim=1, block_dim=32,
    )

    # embed(last generated token) via the trunk's token_embd.weight — blk.32
    # has no embed_tokens tensor of its own (absent from GGUF).
    var TokEmb = row_f32(ctx, de_d, 0, H, h2_layout)
    ctx.enqueue_function[embed1_k](
        Embd, TokEmb, Dtok, Int32(0), Int32(H),
        grid_dim=(ceildiv(H, 256), 1), block_dim=256,
    )

    # concat [enorm(tok_embd) | hnorm(h_nextn)] into cc_d, embedding half
    # first, per docs/mtp-notes.md §2 step 4.
    var Enorm = tens_f32(ctx, wbuf, off[e + 12], H, h_layout)
    var Hnorm = tens_f32(ctx, wbuf, off[e + 13], H, h_layout)
    var CcEmbed = row_bf16(ctx, cc_d, 0, H, h2_layout)
    var CcH = row_bf16(ctx, cc_d, H, H, h2_layout)
    var HnRow = row_f32(ctx, hn_d, 0, H, h2_layout)
    ctx.enqueue_function[rmsc_h2](
        TokEmb, Enorm, CcEmbed, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256
    )
    ctx.enqueue_function[rmsc_h2](
        HnRow, Hnorm, CcH, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256
    )

    # eh_proj: [QF] -> [H], gives the residual stream blk.32 runs on.
    var CcM = TileTensor(cc_d, qfm_layout)
    var Wehq = tens_q8q(ctx, wbuf, off[e + 11], QF * H, q_qf_h)
    var Wehs = tens_q8s(ctx, wbuf, off[e + 11], QF * H, s_qf_h)
    var PhEh = TileTensor(p_h_d, p_h)
    ctx.enqueue_function[g_eh_1](
        CcM, Wehq, Wehs, PhEh, Int32(1), Int32(H), Int32(QF),
        grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS,
    )
    ctx.enqueue_function[r_h](PhEh, Xm, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)

    # blk.32 full-attention sub-block (single self-attended token, pos 0,
    # its own dedicated kc32_d/vc32_d cache).
    var AttNorm32 = tens_f32(ctx, wbuf, off[e + 0], H, h_layout)
    ctx.enqueue_function[rmsc_k](Xm, AttNorm32, CurBm, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)

    var Wq32q = tens_q8q(ctx, wbuf, off[e + 1], H * QF, q_h_qf)
    var Wq32s = tens_q8s(ctx, wbuf, off[e + 1], H * QF, s_h_qf)
    var Wk32q = tens_q8q(ctx, wbuf, off[e + 2], H * KV, q_h_kv)
    var Wk32s = tens_q8s(ctx, wbuf, off[e + 2], H * KV, s_h_kv)
    var Wv32q = tens_q8q(ctx, wbuf, off[e + 3], H * KV, q_h_kv)
    var Wv32s = tens_q8s(ctx, wbuf, off[e + 3], H * KV, s_h_kv)
    var Qn32 = tens_f32(ctx, wbuf, off[e + 4], HD, hd_layout)
    var Kn32 = tens_f32(ctx, wbuf, off[e + 5], HD, hd_layout)
    var Wo32q = tens_q8q(ctx, wbuf, off[e + 6], H * H, q_h_h)
    var Wo32s = tens_q8s(ctx, wbuf, off[e + 6], H * H, s_h_h)
    var Pqf32 = TileTensor(p_qf_d, p_qf)
    var Pkv32 = TileTensor(p_kv_d, p_kv)
    var Ph32 = TileTensor(p_h_d, p_h)
    var Qfm32 = TileTensor(qf_d, qfm_layout)
    var Q32 = TileTensor(q_d, qm_layout)
    var Gate32 = TileTensor(gate_d, xflat_layout)
    var Kflat32 = TileTensor(k_d, kvm_flat)
    var Khd32 = TileTensor(k_d, kvm_layout)
    var Vflat32 = TileTensor(v_d, kvm_flat)
    var Vhd32 = TileTensor(v_d, kvm_layout)
    var Kc32 = TileTensor(kc32_d, cache_layout)
    var Vc32 = TileTensor(vc32_d, cache_layout)
    var Ao32 = TileTensor(ao_d, qm_layout)
    var Aoflat32 = TileTensor(ao_d, xflat_layout)
    var AoB32 = TileTensor(resb_d, xflat_layout)
    var AoBm32 = TileTensor(resb_d, xm_layout)

    ctx.enqueue_function[g_qf_1](CurBm, Wq32q, Wq32s, Pqf32, Int32(1), Int32(QF), Int32(H), grid_dim=ceildiv(QF, ROW_WAVES), block_dim=ROW_THREADS)
    ctx.enqueue_function[r_qf](Pqf32, Qfm32, Int32(1), Int32(QF), grid_dim=ceildiv(QF, 256), block_dim=256)
    ctx.enqueue_function[g_kv_1](CurBm, Wk32q, Wk32s, Pkv32, Int32(1), Int32(KV), Int32(H), grid_dim=ceildiv(KV, ROW_WAVES), block_dim=ROW_THREADS)
    ctx.enqueue_function[r_kv](Pkv32, Kflat32, Int32(1), Int32(KV), grid_dim=ceildiv(KV, 256), block_dim=256)
    ctx.enqueue_function[g_kv_1](CurBm, Wv32q, Wv32s, Pkv32, Int32(1), Int32(KV), Int32(H), grid_dim=ceildiv(KV, ROW_WAVES), block_dim=ROW_THREADS)
    ctx.enqueue_function[r_kv](Pkv32, Vflat32, Int32(1), Int32(KV), grid_dim=ceildiv(KV, 256), block_dim=256)
    ctx.enqueue_function[split_k](Qfm32, Q32, Gate32, grid_dim=(NQH, 1), block_dim=HD)
    ctx.enqueue_function[hrms_q](Q32, Qn32, Float32(1e-6), grid_dim=NQH, block_dim=HD)
    ctx.enqueue_function[hrms_kv](Khd32, Kn32, Float32(1e-6), grid_dim=NKVH, block_dim=HD)
    ctx.enqueue_function[rope_q](Q32, Int32(0), Int32(NQH), grid_dim=(NQH, 1), block_dim=32)
    ctx.enqueue_function[rope_k](Khd32, Int32(0), Int32(NKVH), grid_dim=(NKVH, 1), block_dim=32)
    ctx.enqueue_function[append_k](Kc32, Khd32, Int32(0), grid_dim=(NKVH, 1), block_dim=HD)
    ctx.enqueue_function[append_k](Vc32, Vhd32, Int32(0), grid_dim=(NKVH, 1), block_dim=HD)
    ctx.enqueue_function[att_k](Q32, Kc32, Vc32, Ao32, Int32(1), Float32(0.0625), grid_dim=(NQH, 1), block_dim=HD)
    ctx.enqueue_function[gmul_k](Aoflat32, Gate32, AoB32, Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.enqueue_function[g_h_1](AoBm32, Wo32q, Wo32s, Ph32, Int32(1), Int32(H), Int32(H), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
    ctx.enqueue_function[r_add](Ph32, Xm, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)

    # blk.32 dense SwiGLU FFN sub-block.
    var PostAttnNorm32 = tens_f32(ctx, wbuf, off[e + 7], H, h_layout)
    ctx.enqueue_function[rmsc_k](Xm, PostAttnNorm32, CurBm, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
    var Wfg32q = tens_q8q(ctx, wbuf, off[e + 8], H * FFN, q_h_ffn)
    var Wfg32s = tens_q8s(ctx, wbuf, off[e + 8], H * FFN, s_h_ffn)
    var Wfu32q = tens_q8q(ctx, wbuf, off[e + 9], H * FFN, q_h_ffn)
    var Wfu32s = tens_q8s(ctx, wbuf, off[e + 9], H * FFN, s_h_ffn)
    var Wfd32q = tens_q8q(ctx, wbuf, off[e + 10], FFN * H, q_ffn_h)
    var Wfd32s = tens_q8s(ctx, wbuf, off[e + 10], FFN * H, s_ffn_h)
    var Pg32 = TileTensor(p_ffn_d, p_ffn)
    var Pu32 = TileTensor(p_ffn2_d, p_ffn)
    var Ph2_32 = TileTensor(p_h_d, p_h)
    var FgBm32 = TileTensor(fgb_d, ffnm_layout)
    ctx.enqueue_function[g_ffn_1](CurBm, Wfg32q, Wfg32s, Pg32, Int32(1), Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
    ctx.enqueue_function[g_ffn_1](CurBm, Wfu32q, Wfu32s, Pu32, Int32(1), Int32(FFN), Int32(H), grid_dim=ceildiv(FFN, ROW_WAVES), block_dim=ROW_THREADS)
    ctx.enqueue_function[r_swiglu](Pg32, Pu32, FgBm32, Int32(1), Int32(FFN), grid_dim=ceildiv(FFN, 256), block_dim=256)
    ctx.enqueue_function[g_down_1](FgBm32, Wfd32q, Wfd32s, Ph2_32, Int32(1), Int32(H), Int32(FFN), grid_dim=ceildiv(H, ROW_WAVES), block_dim=ROW_THREADS)
    ctx.enqueue_function[r_add](Ph2_32, Xm, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)

    # shared_head_norm (blk.32's own) then trunk's LM head (blk.32 has no
    # shared_head_head tensor of its own — absent from GGUF).
    var SharedHeadNorm = tens_f32(ctx, wbuf, off[e + 14], H, h_layout)
    ctx.enqueue_function[rmsc_k](Xm, SharedHeadNorm, CurBm, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
    var WheadTrunkq = tens_q8q(ctx, wbuf, off[e - 1], H * VOCAB, q_h_v)
    var WheadTrunks = tens_q8s(ctx, wbuf, off[e - 1], H * VOCAB, s_h_v)
    ctx.enqueue_function[g_head_1](CurBm, WheadTrunkq, WheadTrunks, Pv, Int32(1), Int32(VOCAB), Int32(H), grid_dim=ceildiv(VOCAB, ROW_WAVES), block_dim=ROW_THREADS)
    ctx.enqueue_function[r_head](Pv, Logitsm, Int32(1), Int32(VOCAB), grid_dim=ceildiv(VOCAB, 256), block_dim=256)
    ctx.enqueue_function[argmax_d](Logitsm, Dtok, Int32(VOCAB), Int32(0), grid_dim=1, block_dim=256)

    ctx.synchronize()
    var draft_logits_h = ctx.enqueue_create_host_buffer[f32](VOCAB)
    ctx.enqueue_copy(
        dst_buf=draft_logits_h,
        src_buf=DeviceBuffer[f32](ctx, logits_d.unsafe_ptr(), VOCAB, owning=False),
    )
    # h_nextn is dumped too: tools/draft-ref.py's numpy reference needs the
    # EXACT f32 hidden state the GPU consumed, or a mismatch says nothing about
    # correctness -- it would just mean the two sides were fed different inputs.
    var hn_h = ctx.enqueue_create_host_buffer[f32](H)
    ctx.enqueue_copy(
        dst_buf=hn_h,
        src_buf=DeviceBuffer[f32](ctx, hn_d.unsafe_ptr(), H, owning=False),
    )
    var dtok_h = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(
        dst_buf=dtok_h,
        src_buf=DeviceBuffer[DType.int32](ctx, dtok_d.unsafe_ptr(), 1, owning=False),
    )
    ctx.synchronize()

    with open(".work/draft-logits.bin", "w") as df:
        var dp = draft_logits_h.unsafe_ptr().unsafe_bitcast[UInt8]()
        var dspan = Span[UInt8](unsafe_ptr=dp, length=VOCAB * 4)
        df.write_bytes(dspan)

    with open(".work/draft-hn.bin", "w") as hf:
        var hp = hn_h.unsafe_ptr().unsafe_bitcast[UInt8]()
        var hspan = Span[UInt8](unsafe_ptr=hp, length=H * 4)
        hf.write_bytes(hspan)

    print("DRAFT: from_token", last_tok, "draft_argmax", Int(dtok_h[0]))
