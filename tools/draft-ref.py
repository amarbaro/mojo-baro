#!/usr/bin/env python3
"""Numpy f32 reference + comparator for the blk.32 NextN/MTP draft head.

Implements docs/mtp-notes.md §2 steps 3-7 (qwen35, Qwythos-9B). This is the
GATE the draft head must pass before speculative decoding is wired up in
serve/engine.mojo — the draft head does not exist yet; this file exists so
the gate is ready the moment it does.

Follows tools/model-ref.py / tools/ssm-ref.py / tools/attn-ref.py
conventions: `amar_rmsnorm`, `rne` (bf16 round-to-nearest-even truncation of
intermediate activations), bf16 tensor decode via uint16<<16 view, and
tools/gguf-extract.py's `parse()` for reading raw tensor bytes out of the
real GGUF (no separate .work/gguf dump needed — this reads the model file
directly since blk.32 tensors were never extracted by the existing tools).

blk.32 tensor inventory (docs/mtp-notes.md §1) — 15 tensors, all present in
this GGUF:
  blk.32.attn_norm.weight, .attn_q.weight, .attn_q_norm.weight, .attn_k.weight,
  .attn_k_norm.weight, .attn_v.weight, .attn_output.weight,
  .post_attention_norm.weight, .ffn_gate.weight, .ffn_up.weight,
  .ffn_down.weight, .nextn.eh_proj.weight, .nextn.enorm.weight,
  .nextn.hnorm.weight, .nextn.shared_head_norm.weight.
Reused from the TRUNK (blk.32.nextn.embed_tokens.weight and
blk.32.nextn.shared_head_head.weight are ABSENT from this GGUF — §2 step 3
and step 7): `token_embd.weight` for the token-embedding lookup, and
`output.weight` for the final logits matmul.

qwen35.cpp line references cited throughout are from mtp-notes.md, which was
generated against /home/mario/llama.cpp's src/models/qwen35.cpp.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "gguf_extract", Path(__file__).resolve().parent / "gguf-extract.py"
)
_gguf_extract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gguf_extract)
gguf_parse = _gguf_extract.parse
GGML_BYTES = _gguf_extract.GGML_BYTES

DEFAULT_MODEL = (
    "/home/mario/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/"
    "Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16.gguf"
)

H = 4096          # n_embd
NQH = 16          # n_head
NKVH = 4          # n_head_kv
HD = 256          # head_dim
NROT = 64         # partial-rotary dims (matches tools/attn-ref.py's trunk RoPE)
FFN = 12288       # n_ff
EPS = 1e-6
IL = 32           # blk.32 = the single NextN layer (n_layer_nextn == 1)

# YaRN RoPE params — same as tools/attn-ref.py's trunk full-attention block.
# mtp-notes.md §2 step 6 / §4 describe the MTP block's RoPE as
# `ggml_rope_multi` with mrope sections [11,11,10,0], "the same mrope
# sections as the trunk's full-attention layers" — but the ALREADY-COMMITTED
# trunk reference (tools/attn-ref.py) implements plain single-section
# partial-rotary NEOX YaRN (NROT=64, no section splitting), not mrope-multi.
# JUDGMENT CALL: for text-only decode (no image/video tokens) all 3 mrope
# position axes carry identical position ids, so section-split mrope
# degenerates to ordinary 1D rope over the same NROT=64 span — this is the
# working assumption, matching what serve/engine.mojo's existing trunk RoPE
# kernel actually computes (mtp-notes.md §4: "reuse mojo-baro's existing
# trunk full-attention RoPE kernel rather than writing a new one"). NOT
# proven against llama.cpp's actual mrope codepath; flagged in the report.
FREQ_BASE = 1e7
FREQ_SCALE = 0.25       # yarn factor=4 -> scale=1/4, matches attn-ref.py
N_CTX_ORIG = 262144
BETA_FAST, BETA_SLOW = 32.0, 1.0
EXT_FACTOR, ATTN_FACTOR = 1.0, 1.0


def amar_rmsnorm(x, w, axis=-1):
    return x / np.sqrt(np.mean(x * x, axis=axis, keepdims=True) + EPS) * w


def rne(x):
    """Round-to-nearest-even truncation to bf16, kept as f32 (matches
    tools/model-ref.py / tools/attn-ref.py's `rne`)."""
    u = np.ascontiguousarray(x).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)


def silu(x):
    return x / (1 + np.exp(-x))


def yarn_cos_sin(pos):
    def corr_dim(beta):
        return NROT * np.log(N_CTX_ORIG / (beta * 2 * np.pi)) / (2 * np.log(FREQ_BASE))

    low = np.clip(np.floor(corr_dim(BETA_FAST)), 0, NROT - 1)
    high = np.clip(np.ceil(corr_dim(BETA_SLOW)), 0, NROT - 1)
    j = np.arange(NROT // 2, dtype=np.float32)
    theta_extrap = pos * FREQ_BASE ** (-2.0 * j / NROT)
    theta_interp = FREQ_SCALE * theta_extrap
    ramp = np.clip((j - low) / max(high - low, 0.001), 0, 1) * EXT_FACTOR
    theta = theta_interp * (1 - ramp) + theta_extrap * ramp
    mscale = ATTN_FACTOR * (1 + 0.1 * np.log(1 / FREQ_SCALE)) if EXT_FACTOR != 0 else ATTN_FACTOR
    return (np.cos(theta) * mscale).astype(np.float32), (np.sin(theta) * mscale).astype(np.float32)


def rope(x, pos):
    y = x.copy()
    c, s = yarn_cos_sin(pos)
    half = NROT // 2
    x0 = x[..., :half]
    x1 = x[..., half:NROT]
    y[..., :half] = x0 * c - x1 * s
    y[..., half:NROT] = x0 * s + x1 * c
    return y


class GGUFTensors:
    """Thin lazy-load wrapper over tools/gguf-extract.py's parse() output."""

    def __init__(self, path):
        self.f, self.infos, self.data_start, self.kv = gguf_parse(path)

    def get(self, name):
        dims, ttype, offset = self.infos[name]
        tname, esize = GGML_BYTES[ttype]
        n_elem = int(np.prod(dims))
        self.f.seek(self.data_start + offset)
        raw = self.f.read(n_elem * esize)
        shape = tuple(reversed(dims))  # GGUF dims are innermost-first
        if tname == "bf16":
            w16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            return (w16.astype(np.uint32) << 16).view(np.float32)
        if tname == "f32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        raise ValueError(f"unsupported dtype {tname} for {name}")

    def close(self):
        self.f.close()


def draft_forward(gt: GGUFTensors, tok: int, h_in: np.ndarray, pos: int = 0) -> np.ndarray:
    """One blk.32 NextN forward pass. Returns logits, shape [n_vocab].

    tok   : int token id being drafted (§2 step 3 — embedding lookup).
    h_in  : f32 [n_embd] — the trunk's post-output-norm hidden state
            ("h_nextn") for the PRIOR position (§2 step 2/4, §3's shift-by-one).
    pos   : RoPE position for this single-token self-attention (no KV cache —
            see selftest()'s docstring for why this only proves shapes/load).
    """
    b = f"blk.{IL}."

    # §2 step 3: token embedding via the TRUNK's token_embd.weight —
    # blk.32.nextn.embed_tokens.weight is ABSENT from this GGUF.
    tok_embd_w = gt.get("token_embd.weight")  # [n_vocab, n_embd]
    tok_embd = tok_embd_w[tok].astype(np.float32).copy()

    # §2 step 4: enorm/hnorm + concat, EMBEDDING HALF FIRST.
    h_norm = amar_rmsnorm(h_in, gt.get(b + "nextn.hnorm.weight"))
    e_norm = amar_rmsnorm(tok_embd, gt.get(b + "nextn.enorm.weight"))
    # llama.cpp: ggml_concat(e_norm, h_norm, dim=0) — embedding first, hidden
    # state second (mtp-notes.md §2 step 4 / §4: "swapping halves would
    # silently produce wrong (but plausibly-shaped) results"). DO NOT swap.
    concat = np.concatenate([e_norm, h_norm])  # [2*n_embd] = [8192]

    # §2 step 5: eh_proj -> this is inpSA, the residual base for attention
    # (NOT h_in itself).
    eh_proj_w = gt.get(b + "nextn.eh_proj.weight")  # [8192, 4096] (out,in)
    eh = eh_proj_w @ concat  # [4096]

    # §2 step 6: one full-attention block, structurally identical to a trunk
    # full-attention layer (mirrors tools/attn-ref.py's attn_layer body).
    cur = rne(amar_rmsnorm(eh, gt.get(b + "attn_norm.weight")))

    wq = gt.get(b + "attn_q.weight")            # [8192, 4096] (out,in): joint Q+gate
    wk = gt.get(b + "attn_k.weight")             # [1024, 4096]
    wv = gt.get(b + "attn_v.weight")             # [1024, 4096]
    wo = gt.get(b + "attn_output.weight")        # [4096, 4096]
    qn = gt.get(b + "attn_q_norm.weight")        # [256]
    kn = gt.get(b + "attn_k_norm.weight")        # [256]

    qf = (cur @ wq.T).reshape(NQH, 2 * HD)
    q, gate = qf[:, :HD], qf[:, HD:].reshape(H)
    k = (cur @ wk.T).reshape(NKVH, HD)
    v = (cur @ wv.T).reshape(NKVH, HD)

    q = rope(amar_rmsnorm(q, qn), pos)
    k = rope(amar_rmsnorm(k, kn), pos)

    # Single-position self-attention (no KV cache): the only key/value is
    # this token's own k/v, so softmax collapses to weight 1.0 per head.
    # This exercises tensor shapes/wiring, not the multi-position RoPE path.
    o = np.zeros((NQH, HD), dtype=np.float32)
    for h in range(NQH):
        o[h] = v[h // (NQH // NKVH)]

    att = rne((o.reshape(H)) * (1 / (1 + np.exp(-gate))))
    attn_out = att @ wo.T

    # Residual against the EH_PROJ OUTPUT, not raw h/embd (§2 step 6).
    resid1 = eh + attn_out

    cur2 = rne(amar_rmsnorm(resid1, gt.get(b + "post_attention_norm.weight")))
    gt_w = gt.get(b + "ffn_gate.weight")
    up_w = gt.get(b + "ffn_up.weight")
    down_w = gt.get(b + "ffn_down.weight")
    gate_v = cur2 @ gt_w.T
    up_v = cur2 @ up_w.T
    act = rne(silu(gate_v) * up_v)
    ffn_out = act @ down_w.T

    resid2 = resid1 + ffn_out

    # §2 step 7: final RMSNorm with blk.32's OWN shared_head_norm (present in
    # this GGUF — used instead of model.output_norm), then logits via the
    # TRUNK's output.weight (blk.32.nextn.shared_head_head.weight is ABSENT).
    normed = amar_rmsnorm(resid2, gt.get(b + "nextn.shared_head_norm.weight"))
    output_w = gt.get("output.weight")  # [n_vocab, n_embd]
    logits = normed @ output_w.T
    return logits


def compare(ref_logits: np.ndarray, engine_logits_path: str,
            cos_thresh: float = 0.999, rel_err_thresh: float = 1e-2) -> bool:
    """Report top-1 agreement, cosine similarity, and max relative error.

    All THREE are always reported — cosine alone is the known silent-
    corruption signature for this class of bug (high cosine, differing
    argmax), so a top-1 mismatch fails the gate even when cosine is high.
    """
    eng_logits = np.fromfile(engine_logits_path, dtype=np.float32)
    if eng_logits.shape != ref_logits.shape:
        print(f"SHAPE MISMATCH: ref={ref_logits.shape} engine={eng_logits.shape}")
        print("GATE: FAIL")
        return False

    top1_ref = int(np.argmax(ref_logits))
    top1_eng = int(np.argmax(eng_logits))
    top1_match = top1_ref == top1_eng

    denom = np.linalg.norm(ref_logits) * np.linalg.norm(eng_logits)
    cosine = float(np.dot(ref_logits, eng_logits) / denom) if denom > 0 else 0.0

    rel_eps = 1e-6
    max_rel_err = float(np.max(np.abs(ref_logits - eng_logits) / (np.abs(ref_logits) + rel_eps)))

    print(f"top1_ref={top1_ref} top1_engine={top1_eng} top1_match={top1_match}")
    print(f"cosine_similarity={cosine:.8f}")
    print(f"max_relative_error={max_rel_err:.6e}")

    # PROVISIONAL thresholds — not yet calibrated against a real engine dump
    # (the engine cannot emit draft logits yet). Recalibrate once a known-
    # good bf16-accumulation-noise baseline exists.
    passed = top1_match and cosine >= cos_thresh and max_rel_err <= rel_err_thresh
    print(f"GATE: {'PASS' if passed else 'FAIL'}")
    return passed


def selftest(model_path: str = DEFAULT_MODEL, tok: int = 760, pos: int = 0):
    """Runs draft_forward() on REAL blk.32 tensors, proving tensor loading
    and shapes are correct.

    No real trunk h_nextn export exists yet (the engine doesn't compute
    speculative decoding or export h_nextn). h_in is therefore a zero vector
    here — this proves shape/load correctness end-to-end, NOT numerical
    correctness against llama.cpp. True numerical validation is blocked on
    the engine producing a real (tok, h_in) -> logits dump; see compare().
    """
    gt = GGUFTensors(model_path)
    h_in = np.zeros(H, dtype=np.float32)
    logits = draft_forward(gt, tok, h_in, pos=pos)
    gt.close()
    print(f"logits.shape={logits.shape} dtype={logits.dtype}")
    print(f"logits[:5]={logits[:5]}")
    print(f"argmax={int(np.argmax(logits))} max={float(logits.max()):.6f} "
          f"min={float(logits.min()):.6f}")
    return logits


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] == "selftest":
        tok = int(argv[argv.index("--tok") + 1]) if "--tok" in argv else 760
        pos = int(argv[argv.index("--pos") + 1]) if "--pos" in argv else 0
        selftest(tok=tok, pos=pos)
        return
    if argv[0] == "compare":
        # compare REF_TOK REF_H_BIN ENGINE_LOGITS_BIN [MODEL_PATH]
        ref_tok = int(argv[1])
        h_in = np.fromfile(argv[2], dtype=np.float32)
        engine_logits_path = argv[3]
        model_path = argv[4] if len(argv) > 4 else DEFAULT_MODEL
        gt = GGUFTensors(model_path)
        ref_logits = draft_forward(gt, ref_tok, h_in)
        gt.close()
        ok = compare(ref_logits, engine_logits_path)
        sys.exit(0 if ok else 1)
    print(f"unknown command: {argv[0]}", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
