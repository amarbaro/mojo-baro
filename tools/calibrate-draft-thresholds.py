#!/usr/bin/env python3
"""Calibration experiment: measure bf16-vs-f32 representational gap for the
blk.32 NextN draft head, on REAL model weights, to set draft-ref.py's gate
thresholds from data instead of guesswork.

Two forward-pass variants of the same math:
  - f32_full:      no bf16 rounding anywhere (pure f32 reference)
  - bf16_emulated: bf16 round-trip (rne) applied at every point the GPU
                    kernel actually casts to bf16 before a GEMM/write:
                    after each RMSNorm (amar_rmsnorm_cast writes bf16),
                    after silu*up (elementwise write before down-proj GEMM),
                    after the attn output gate multiply, AND at the two
                    extra boundaries draft-ref.py's existing rne() calls
                    don't cover: the enorm/hnorm concat (GEMM input to
                    eh_proj) and the final shared_head_norm output (GEMM
                    input to the output projection).

The gap between these two variants, across many (token, h_in) samples, IS
the bf16 accumulation noise the gate must tolerate.

Falsification check: swap the eh_proj concat halves (known silent-corruption
mode, docs/mtp-notes.md §2 step 4) and confirm the calibrated thresholds
reject it.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/mario/Projects/mojo-baro/tools")
import importlib.util
_spec = importlib.util.spec_from_file_location("draft_ref", "/home/mario/Projects/mojo-baro/tools/draft-ref.py")
dr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dr)

H, NQH, NKVH, HD = dr.H, dr.NQH, dr.NKVH, dr.HD

def identity(x):
    return x

def draft_forward_variant(gt, tok, h_in, pos, cast, swap_concat=False):
    """Reimplementation of dr.draft_forward with a pluggable bf16-cast fn
    `cast` applied at every GEMM-input boundary. cast=dr.rne -> bf16_emulated,
    cast=identity -> f32_full."""
    b = f"blk.{dr.IL}."
    tok_embd_w = gt.get("token_embd.weight")
    tok_embd = tok_embd_w[tok].astype(np.float32).copy()

    h_norm = dr.amar_rmsnorm(h_in, gt.get(b + "nextn.hnorm.weight"))
    e_norm = dr.amar_rmsnorm(tok_embd, gt.get(b + "nextn.enorm.weight"))
    if swap_concat:
        concat = np.concatenate([h_norm, e_norm])
    else:
        concat = np.concatenate([e_norm, h_norm])
    concat = cast(concat)  # GEMM input to eh_proj

    eh_proj_w = gt.get(b + "nextn.eh_proj.weight")
    eh = eh_proj_w @ concat

    cur = cast(dr.amar_rmsnorm(eh, gt.get(b + "attn_norm.weight")))

    wq = gt.get(b + "attn_q.weight")
    wk = gt.get(b + "attn_k.weight")
    wv = gt.get(b + "attn_v.weight")
    wo = gt.get(b + "attn_output.weight")
    qn = gt.get(b + "attn_q_norm.weight")
    kn = gt.get(b + "attn_k_norm.weight")

    qf = (cur @ wq.T).reshape(NQH, 2 * HD)
    q, gate = qf[:, :HD], qf[:, HD:].reshape(H)
    k = (cur @ wk.T).reshape(NKVH, HD)
    v = (cur @ wv.T).reshape(NKVH, HD)

    q = dr.rope(cast(dr.amar_rmsnorm(q, qn)), pos)
    k = dr.rope(cast(dr.amar_rmsnorm(k, kn)), pos)

    o = np.zeros((NQH, HD), dtype=np.float32)
    for h in range(NQH):
        o[h] = v[h // (NQH // NKVH)]

    att = cast((o.reshape(H)) * (1 / (1 + np.exp(-gate))))
    attn_out = att @ wo.T

    resid1 = eh + attn_out

    cur2 = cast(dr.amar_rmsnorm(resid1, gt.get(b + "post_attention_norm.weight")))
    gt_w = gt.get(b + "ffn_gate.weight")
    up_w = gt.get(b + "ffn_up.weight")
    down_w = gt.get(b + "ffn_down.weight")
    gate_v = cur2 @ gt_w.T
    up_v = cur2 @ up_w.T
    act = cast(dr.silu(gate_v) * up_v)
    ffn_out = act @ down_w.T

    resid2 = resid1 + ffn_out

    normed = cast(dr.amar_rmsnorm(resid2, gt.get(b + "nextn.shared_head_norm.weight")))
    output_w = gt.get("output.weight")
    logits = normed @ output_w.T
    return logits


def metrics(ref, eng):
    """top-1, cosine, max ABS error, and top-K relative error.

    Full-vector relative error is meaningless on a logit vector: entries near
    zero blow the ratio up to ~1e3 under clean bf16 noise. Two replacements:
      - max_abs_err over the whole vector (logits are bounded, ~+/-12 here)
      - max_rel_err restricted to the top-K logits by |ref|, which are the only
        entries that decide argmax or survive sampling, and whose magnitudes
        make the ratio well-conditioned.
    """
    top1_ref, top1_eng = int(np.argmax(ref)), int(np.argmax(eng))
    denom = float(np.linalg.norm(ref) * np.linalg.norm(eng))
    cosine = float(np.dot(ref, eng) / denom) if denom > 0 else 0.0
    cosine = min(1.0, max(-1.0, cosine))          # clamp fp drift past +/-1
    max_abs_err = float(np.max(np.abs(ref - eng)))
    K = 64
    idx = np.argsort(np.abs(ref))[-K:]
    max_rel_topk = float(np.max(np.abs(ref[idx] - eng[idx]) / np.abs(ref[idx])))
    return top1_ref == top1_eng, cosine, max_abs_err, max_rel_topk


def main():
    gt = dr.GGUFTensors(str(Path(dr.DEFAULT_MODEL).expanduser()))
    rng = np.random.default_rng(1234)

    tokens = [1, 42, 100, 760, 2048, 5000, 10000, 32000, 60000, 90000]
    results = []
    for i, tok in enumerate(tokens):
        # Vary h_in: realistic-scale random hidden state (not all-zero, since
        # selftest's zero vector wouldn't exercise the RMSNorm/GEMM bf16
        # rounding paths meaningfully — real h_nextn has nonzero variance).
        h_in = rng.normal(0, 1.0, size=H).astype(np.float32)
        pos = i * 7

        ref = draft_forward_variant(gt, tok, h_in, pos, cast=identity)
        eng = draft_forward_variant(gt, tok, h_in, pos, cast=dr.rne)
        match, cos, abserr, relerr = metrics(ref, eng)
        results.append((tok, match, cos, relerr, abserr))
        print(f"tok={tok:6d} pos={pos:3d} top1_match={match} cosine={cos:.8f} max_abs_err={abserr:.6e} rel_topk={relerr:.6e}")

    cosines = [r[2] for r in results]
    relerrs = [r[3] for r in results]
    abserrs = [r[4] for r in results]
    matches = [r[1] for r in results]

    print()
    print(f"top1_match: all={all(matches)} ({sum(matches)}/{len(matches)})")
    print(f"cosine: min={min(cosines):.8f} median={sorted(cosines)[len(cosines)//2]:.8f} max={max(cosines):.8f}")
    print(f"abs_err: min={min(abserrs):.6e} median={sorted(abserrs)[len(abserrs)//2]:.6e} max={max(abserrs):.6e}")
    print(f"rel_topk: min={min(relerrs):.6e} median={sorted(relerrs)[len(relerrs)//2]:.6e} max={max(relerrs):.6e}")

    # Falsification: swap concat halves for one sample, check thresholds reject it.
    print()
    print("=== falsification: eh_proj concat-swap corruption ===")
    tok, h_in, pos = 760, rng.normal(0, 1.0, size=H).astype(np.float32), 0
    ref = draft_forward_variant(gt, tok, h_in, pos, cast=dr.rne, swap_concat=False)
    corrupted = draft_forward_variant(gt, tok, h_in, pos, cast=dr.rne, swap_concat=True)
    match, cos, abserr, relerr = metrics(ref, corrupted)
    print(f"top1_match={match} cosine={cos:.8f} max_abs_err={abserr:.6e} rel_topk={relerr:.6e}")

    gt.close()

main()
