"""baro engine: full-model greedy decode for qwen35 (Qwythos-9B), milestone 4.

Loads .work/engine-pack/ (fixed tensor order, 2D bf16 weights pre-transposed
to B-layout), reads prompt token ids from .work/engine-pack/prompt-tokens.txt,
runs the 32-block hybrid stack (24 gated-delta-net + 8 gated full-attention,
MTP block skipped) over a window of up to MROWS tokens at a time, and prints
greedy token ids.

Parity target: byte-identical token ids vs llama.cpp on the same GGUF.
"""
from std.math import ceildiv
from std.memory import memcpy
from std.os import getenv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major
from registry import *



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


def blk32_forward(
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], off: List[Int], e: Int,
    m: Int, pos: Int, tok_pos: Int, do_head: Bool,
    hsrc: DeviceBuffer[f32], mut x_d: DeviceBuffer[f32], mut curb_d: DeviceBuffer[bf16],
    mut qf_d: DeviceBuffer[f32], mut q_d: DeviceBuffer[f32], mut k_d: DeviceBuffer[f32],
    mut v_d: DeviceBuffer[f32], mut gate_d: DeviceBuffer[f32], mut ao_d: DeviceBuffer[f32],
    mut resb_d: DeviceBuffer[bf16], mut fgb_d: DeviceBuffer[bf16],
    mut p_qf_d: DeviceBuffer[f32], mut p_kv_d: DeviceBuffer[f32], mut p_h_d: DeviceBuffer[f32],
    mut p_ffn_d: DeviceBuffer[f32], mut p_ffn2_d: DeviceBuffer[f32], mut p_v_d: DeviceBuffer[f32],
    mut logits_d: DeviceBuffer[f32], mut cc_d: DeviceBuffer[bf16], mut de_d: DeviceBuffer[f32],
    mut hd_d: DeviceBuffer[f32], mut kc32_d: DeviceBuffer[f32], mut vc32_d: DeviceBuffer[f32],
    mut toks_d: DeviceBuffer[DType.int32], mut dtok_d: DeviceBuffer[DType.int32],
) raises:
    # blk.32 (NextN) draft head over m rows: row r is token Toks[tok_pos + r]
    # at sequence position pos + r, paired with hidden row r of hsrc
    # (docs/mtp-notes.md: h from BEFORE that token). Writes the draft's own
    # post-shared-head-norm hidden into hd_d rows and, if do_head, the
    # argmax of every row into dtok_d[r].
    var Embd = tens_bf16(ctx, wbuf, off[0], VOCAB * H, emb_layout)
    var Toks = TileTensor(toks_d, toks_layout)
    var Dtok = TileTensor(dtok_d, dtok_layout)
    var Xm = TileTensor(x_d, xm_layout)
    var CurBm = TileTensor(curb_d, xm_layout)
    var Logitsm = TileTensor(logits_d, vm_layout)
    var DeM = TileTensor(de_d, xm_layout)
    ctx.enqueue_function[embed_k](
        Embd, DeM, Toks, Int32(tok_pos), Int32(H),
        grid_dim=(ceildiv(H, 256), m), block_dim=256,
    )
    var Enorm = tens_f32(ctx, wbuf, off[e + 12], H, h_layout)
    var Hnorm = tens_f32(ctx, wbuf, off[e + 13], H, h_layout)
    for r in range(m):
        var TokEmb = row_f32(ctx, de_d, r * H, H, h2_layout)
        var HnRow = row_f32(ctx, hsrc, r * H, H, h2_layout)
        var CcEmbed = row_bf16(ctx, cc_d, r * QF, H, h2_layout)
        var CcH = row_bf16(ctx, cc_d, r * QF + H, H, h2_layout)
        ctx.enqueue_function[rmsc_h2](TokEmb, Enorm, CcEmbed, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
        ctx.enqueue_function[rmsc_h2](HnRow, Hnorm, CcH, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
    var CcM = TileTensor(cc_d, qfm_layout)
    var Wehq = tens_q8q(ctx, wbuf, off[e + 11], QF * H, q_qf_h)
    var Wehs = tens_q8s(ctx, wbuf, off[e + 11], QF * H, s_qf_h)
    var PhEh = TileTensor(p_h_d, p_h)
    gemm_q8(ctx, CcM, Wehq, Wehs, PhEh, m, H, QF)
    ctx.enqueue_function[r_h](PhEh, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)

    var AttNorm32 = tens_f32(ctx, wbuf, off[e + 0], H, h_layout)
    ctx.enqueue_function[rmsc_k](Xm, AttNorm32, CurBm, Int32(H), Float32(1e-6), grid_dim=m, block_dim=256)
    var Wqq = tens_q8q(ctx, wbuf, off[e + 1], H * QF, q_h_qf)
    var Wqs = tens_q8s(ctx, wbuf, off[e + 1], H * QF, s_h_qf)
    var Wkq = tens_q8q(ctx, wbuf, off[e + 2], H * KV, q_h_kv)
    var Wks = tens_q8s(ctx, wbuf, off[e + 2], H * KV, s_h_kv)
    var Wvq = tens_q8q(ctx, wbuf, off[e + 3], H * KV, q_h_kv)
    var Wvs = tens_q8s(ctx, wbuf, off[e + 3], H * KV, s_h_kv)
    var Qn = tens_f32(ctx, wbuf, off[e + 4], HD, hd_layout)
    var Kn = tens_f32(ctx, wbuf, off[e + 5], HD, hd_layout)
    var Woq = tens_q8q(ctx, wbuf, off[e + 6], H * H, q_h_h)
    var Wos = tens_q8s(ctx, wbuf, off[e + 6], H * H, s_h_h)
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
    var Kc = TileTensor(kc32_d, cache_layout)
    var Vc = TileTensor(vc32_d, cache_layout)
    var Ao = TileTensor(ao_d, qm_layout)
    var Aoflat = TileTensor(ao_d, xflat_layout)
    var AoB = TileTensor(resb_d, xflat_layout)
    var AoBm = TileTensor(resb_d, xm_layout)
    gemm_q8(ctx, CurBm, Wqq, Wqs, Pqf, m, QF, H)
    ctx.enqueue_function[r_qf](Pqf, Qfm, Int32(m), Int32(QF), grid_dim=ceildiv(m * QF, 256), block_dim=256)
    gemm_q8(ctx, CurBm, Wkq, Wks, Pkv, m, KV, H)
    ctx.enqueue_function[r_kv](Pkv, Kflat, Int32(m), Int32(KV), grid_dim=ceildiv(m * KV, 256), block_dim=256)
    gemm_q8(ctx, CurBm, Wvq, Wvs, Pkv, m, KV, H)
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
    gemm_q8(ctx, AoBm, Woq, Wos, Ph, m, H, H)
    ctx.enqueue_function[r_add](Ph, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)

    var PostAttnNorm = tens_f32(ctx, wbuf, off[e + 7], H, h_layout)
    ctx.enqueue_function[rmsc_k](Xm, PostAttnNorm, CurBm, Int32(H), Float32(1e-6), grid_dim=m, block_dim=256)
    var Wfgq = tens_q8q(ctx, wbuf, off[e + 8], H * FFN, q_h_ffn)
    var Wfgs = tens_q8s(ctx, wbuf, off[e + 8], H * FFN, s_h_ffn)
    var Wfuq = tens_q8q(ctx, wbuf, off[e + 9], H * FFN, q_h_ffn)
    var Wfus = tens_q8s(ctx, wbuf, off[e + 9], H * FFN, s_h_ffn)
    var Wfdq = tens_q8q(ctx, wbuf, off[e + 10], FFN * H, q_ffn_h)
    var Wfds = tens_q8s(ctx, wbuf, off[e + 10], FFN * H, s_ffn_h)
    var Pg = TileTensor(p_ffn_d, p_ffn)
    var Pu = TileTensor(p_ffn2_d, p_ffn)
    var Ph2 = TileTensor(p_h_d, p_h)
    var FgBm = TileTensor(fgb_d, ffnm_layout)
    gemm_q8(ctx, CurBm, Wfgq, Wfgs, Pg, m, FFN, H)
    gemm_q8(ctx, CurBm, Wfuq, Wfus, Pu, m, FFN, H)
    ctx.enqueue_function[r_swiglu](Pg, Pu, FgBm, Int32(m), Int32(FFN), grid_dim=ceildiv(m * FFN, 256), block_dim=256)
    gemm_q8(ctx, FgBm, Wfdq, Wfds, Ph2, m, H, FFN)
    ctx.enqueue_function[r_add](Ph2, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)

    var SharedHeadNorm = tens_f32(ctx, wbuf, off[e + 14], H, h_layout)
    var HdM = TileTensor(hd_d, xm_layout)
    ctx.enqueue_function[rms_m](Xm, SharedHeadNorm, HdM, Int32(H), Float32(1e-6), grid_dim=m, block_dim=256)
    if do_head:
        ctx.enqueue_function[rmsc_k](Xm, SharedHeadNorm, CurBm, Int32(H), Float32(1e-6), grid_dim=m, block_dim=256)
        var Wheadq = tens_q8q(ctx, wbuf, off[e - 1], H * VOCAB, q_h_v)
        var Wheads = tens_q8s(ctx, wbuf, off[e - 1], H * VOCAB, s_h_v)
        var Pv = TileTensor(p_v_d, p_v)
        gemm_q8(ctx, CurBm, Wheadq, Wheads, Pv, m, VOCAB, H)
        ctx.enqueue_function[r_head](Pv, Logitsm, Int32(m), Int32(VOCAB), grid_dim=ceildiv(m * VOCAB, 256), block_dim=256)
        ctx.enqueue_function[argmax_d](Logitsm, Dtok, Int32(VOCAB), Int32(0), grid_dim=m, block_dim=256)


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
            var dslice = DeviceBuffer[DType.uint8](
                ctx, wbuf.unsafe_ptr() + done, want, owning=False
            )
            var hslice = ctx.enqueue_create_host_buffer[DType.uint8](want) if want != CHUNK else stage
            memcpy(dest=hslice.unsafe_ptr(), src=data.unsafe_ptr(), count=want)
            ctx.enqueue_copy(dst_buf=dslice, src_buf=hslice)
            ctx.synchronize()
            done += want
    print("pack loaded in", Float64(perf_counter_ns() - t_load) / 1e9, "s")

    # --- prompt --------------------------------------------------------------
    var prompt = List[Int]()
    var prompt_path = getenv("BARO_PROMPT", packdir + "/prompt-tokens.txt")
    print("prompt file:", prompt_path)
    with open(prompt_path, "r") as f:
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

    var kcfg = 2
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
        kcfg = 2
    var kenv = getenv("BARO_SPEC_K", "")
    if kenv != "":
        kcfg = atol(kenv)
    if kcfg < 1:
        kcfg = 1
    if kcfg > KMAX:
        kcfg = KMAX
    print("spec k:", kcfg)
    var spec = getenv("BARO_SPEC", "0") == "1"
    print("BARO_SPEC:", spec)
    var spec_dbg = getenv("BARO_SPEC_DBG", "0") == "1"
    var n_drafted = 0
    var n_accepted = 0
    var pos_prev = 0
    var dtok_h = ctx.enqueue_create_host_buffer[DType.int32](KMAX + 1)
    var win_h = ctx.enqueue_create_host_buffer[DType.int32](KMAX + 1)

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
    var de_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var hd_d = ctx.enqueue_create_buffer[f32](MROWS * H)
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
    var pf_proc = 0
    var pf_draft = 0
    var tp = t0
    var prefill_done = False
    while pos < n_total - 1:
        var m = 1
        var win_spec = False
        if pos + 1 < len(prompt):
            m = min(MROWS, len(prompt) - 1 - pos)
        elif spec:
            # process: tokens pos_prev+1..pos through the draft head with the
            # trunk's h rows (row r = h(pos_prev + r)); last row = draft step 0.
            var nproc = pos - pos_prev
            if prof:
                ctx.synchronize()
                tp = perf_counter_ns()
            var hn_rows = DeviceBuffer[f32](ctx, hn_d.unsafe_ptr(), nproc * H, owning=False)
            blk32_forward(ctx, wbuf, off, e, nproc, pos_prev + 1, pos_prev + 1, True, hn_rows,
                x_d, curb_d, qf_d, q_d, k_d, v_d, gate_d, ao_d, resb_d, fgb_d, p_qf_d, p_kv_d, p_h_d,
                p_ffn_d, p_ffn2_d, p_v_d, logits_d, cc_d, de_d, hd_d, kc32_d, vc32_d, toks_d, dtok_d)
            var Dtok = TileTensor(dtok_d, dtok_layout)
            ctx.enqueue_function[tokcp_k](Dtok, Toks, Int32(nproc - 1), Int32(pos + 1), Int32(1), grid_dim=1, block_dim=32)
            m = min(kcfg + 1, n_total - 1 - pos)
            if prof:
                ctx.synchronize()
                pf_proc += Int(perf_counter_ns() - tp)
                tp = perf_counter_ns()
            var hrow = nproc - 1
            for j in range(1, m - 1):
                var hd_row = DeviceBuffer[f32](ctx, hd_d.unsafe_ptr() + hrow * H, H, owning=False)
                blk32_forward(ctx, wbuf, off, e, 1, pos + j, pos + j, True, hd_row,
                    x_d, curb_d, qf_d, q_d, k_d, v_d, gate_d, ao_d, resb_d, fgb_d, p_qf_d, p_kv_d, p_h_d,
                    p_ffn_d, p_ffn2_d, p_v_d, logits_d, cc_d, de_d, hd_d, kc32_d, vc32_d, toks_d, dtok_d)
                ctx.enqueue_function[tokcp_k](Dtok, Toks, Int32(0), Int32(pos + j + 1), Int32(1), grid_dim=1, block_dim=32)
                hrow = 0
            n_drafted += m - 1
            win_spec = True
            if prof:
                ctx.synchronize()
                pf_draft += Int(perf_counter_ns() - tp)

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

                gemm_q8(ctx, CurBm, Wqq, Wqs, Pqf, m, QF, H)
                ctx.enqueue_function[r_qf](Pqf, Qfm, Int32(m), Int32(QF), grid_dim=ceildiv(m * QF, 256), block_dim=256)
                gemm_q8(ctx, CurBm, Wkq, Wks, Pkv, m, KV, H)
                ctx.enqueue_function[r_kv](Pkv, Kflat, Int32(m), Int32(KV), grid_dim=ceildiv(m * KV, 256), block_dim=256)
                gemm_q8(ctx, CurBm, Wvq, Wvs, Pkv, m, KV, H)
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
                gemm_q8(ctx, AoBm, Woq, Wos, Ph, m, H, H)
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

                gemm_q8(ctx, CurBm, Wqkvq, Wqkvs, Pq, m, CONV, H)
                gemm_q8(ctx, CurBm, Wzq, Wzs, Ph, m, H, H)
                gemm_q8(ctx, CurBm, Waq, Was, Pab, m, NH_V, H)
                gemm_q8(ctx, CurBm, Wbq, Wbs, Pab2, m, NH_V, H)
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
                gemm_q8(ctx, ResBm, Wsoutq, Wsouts, Ph, m, H, H)
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
            gemm_q8(ctx, CurBm, Wfgq, Wfgs, Pg, m, FFN, H)
            gemm_q8(ctx, CurBm, Wfuq, Wfus, Pu, m, FFN, H)
            ctx.enqueue_function[r_swiglu](Pg, Pu, FgBm, Int32(m), Int32(FFN), grid_dim=ceildiv(m * FFN, 256), block_dim=256)
            gemm_q8(ctx, FgBm, Wfdq, Wfds, Ph2, m, H, FFN)
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
        # f32 copy of the post-final-norm hidden state (pre-LM-head): row r is
        # h(pos + r), what the MTP draft head pairs with token pos + r + 1.
        var Hnm = TileTensor(hn_d, xm_layout)
        ctx.enqueue_function[rms_m](
            Xm, tens_f32(ctx, wbuf, off[w], H, h_layout), Hnm,
            Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
        )
        if pos + m >= len(prompt):
            ctx.enqueue_function[rmsc_k](
                Xm, tens_f32(ctx, wbuf, off[w], H, h_layout), CurBm,
                Int32(H), Float32(1e-6), grid_dim=m, block_dim=256,
            )
            var Wheadq = tens_q8q(ctx, wbuf, off[w + 1], H * VOCAB, q_h_v)
            var Wheads = tens_q8s(ctx, wbuf, off[w + 1], H * VOCAB, s_h_v)
            gemm_q8(ctx, CurBm, Wheadq, Wheads, Pv, m, VOCAB, H)
            ctx.enqueue_function[r_head](Pv, Logitsm, Int32(m), Int32(VOCAB), grid_dim=ceildiv(m * VOCAB, 256), block_dim=256)
            if win_spec:
                var Dtok = TileTensor(dtok_d, dtok_layout)
                ctx.enqueue_function[argmax_d](Logitsm, Dtok, Int32(VOCAB), Int32(0), grid_dim=m, block_dim=256)
                ctx.enqueue_copy(dst_buf=dtok_h, src_buf=DeviceBuffer[DType.int32](ctx, dtok_d.unsafe_ptr(), KMAX + 1, owning=False))
                ctx.enqueue_copy(dst_buf=win_h, src_buf=DeviceBuffer[DType.int32](ctx, toks_d.unsafe_ptr() + pos + 1, KMAX + 1, owning=False))
                ctx.synchronize()
                var n_acc = 0
                while n_acc < m - 1 and dtok_h[n_acc] == win_h[n_acc]:
                    n_acc += 1
                n_accepted += n_acc
                if spec_dbg:
                    var line = String("win pos=") + String(pos) + " m=" + String(m) + " n_acc=" + String(n_acc) + " toks:"
                    for i in range(m):
                        line += " " + String(win_h[i]) + "/" + String(dtok_h[i])
                    print(line)
                ctx.enqueue_function[tokcp_k](Dtok, Toks, Int32(n_acc), Int32(pos + n_acc + 1), Int32(1), grid_dim=1, block_dim=32)
                m = n_acc + 1
            else:
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
        pos_prev = pos
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
        print("profile: mtp_proc", Float64(pf_proc) / 1e9, " mtp_draft", Float64(pf_draft) / 1e9)
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
    if spec:
        print("mtp: drafted", n_drafted, " accepted", n_accepted, " k", kcfg)

    # MTP (NextN) draft head receipt, blk.32: last generated token paired with
    # the last trunk hidden row, attended at position 0 (arm A: empty draft
    # KV, identical to the 2026-09-01 validation dump; arm B: draft KV holds
    # the run, so only arm A's DRAFT line is the receipt).
    var hn_last = DeviceBuffer[f32](ctx, hn_d.unsafe_ptr(), H, owning=False)
    blk32_forward(ctx, wbuf, off, e, 1, 0, n_total - 1, True, hn_last,
        x_d, curb_d, qf_d, q_d, k_d, v_d, gate_d, ao_d, resb_d, fgb_d, p_qf_d, p_kv_d, p_h_d,
        p_ffn_d, p_ffn2_d, p_v_d, logits_d, cc_d, de_d, hd_d, kc32_d, vc32_d, toks_d, dtok_d)
    var last_tok = generated[len(generated) - 1]
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
    var dtok1_h = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(
        dst_buf=dtok1_h,
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

    print("DRAFT: from_token", last_tok, "draft_argmax", Int(dtok1_h[0]))
