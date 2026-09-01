#!/usr/bin/env python3
"""Numpy reference for one decode token through the qwen35 full-attention block
(docs/qwen35-ssm-notes.md §7b). Text-only IMROPE = partial-rotary NEOX YaRN over
the first 64 of 256 head dims; ext_factor=1 assumed (scaling.type=yarn).
"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent.parent / ".work/gguf"
H = 4096
HD = 256
NQH = 16
NKVH = 4
NROT = 64
T_PRE = 7
POS = T_PRE
EPS = 1e-6
FREQ_BASE = 1e7
FREQ_SCALE = 0.25
N_CTX_ORIG = 262144
BETA_FAST, BETA_SLOW = 32.0, 1.0
EXT_FACTOR, ATTN_FACTOR = 1.0, 1.0


def bf16(path, shape):
    w = np.fromfile(D / path, dtype=np.uint16).reshape(shape)
    return (w.astype(np.uint32) << 16).view(np.float32)


def f32(path, shape):
    return np.fromfile(D / path, dtype=np.float32).reshape(shape)


def amar_rmsnorm(x, w, axis=-1):
    return x / np.sqrt(np.mean(x * x, axis=axis, keepdims=True) + EPS) * w


def rne(x):
    u = x.view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)


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


def main():
    wq = bf16("blk_3_attn_q_weight.bin", (2 * H, H))
    wk = bf16("blk_3_attn_k_weight.bin", (NKVH * HD, H))
    wv = bf16("blk_3_attn_v_weight.bin", (NKVH * HD, H))
    wo = bf16("blk_3_attn_output_weight.bin", (H, H))
    qn = f32("blk_3_attn_q_norm_weight.bin", (HD,))
    kn = f32("blk_3_attn_k_norm_weight.bin", (HD,))
    an = f32("blk_3_attn_norm_weight.bin", (H,))

    rng = np.random.default_rng(23)
    x = rng.standard_normal(H).astype(np.float32) * 0.5
    kcache = rng.standard_normal((NKVH, T_PRE, HD)).astype(np.float32) * 0.3
    vcache = rng.standard_normal((NKVH, T_PRE, HD)).astype(np.float32) * 0.3

    cur = rne(amar_rmsnorm(x, an))

    qfull = (cur @ wq.T).reshape(NQH, 2 * HD)
    q = qfull[:, :HD]
    gate = qfull[:, HD:].reshape(H)
    k = (cur @ wk.T).reshape(NKVH, HD)
    v = (cur @ wv.T).reshape(NKVH, HD)

    q = amar_rmsnorm(q, qn)
    k = amar_rmsnorm(k, kn)
    q = rope(q, POS)
    k = rope(k, POS)

    kc = np.concatenate([kcache, k[:, None, :]], axis=1)
    vc = np.concatenate([vcache, v[:, None, :]], axis=1)

    scale = 1.0 / np.sqrt(HD)
    o = np.zeros((NQH, HD), dtype=np.float32)
    for h in range(NQH):
        kvh = h // (NQH // NKVH)
        s = kc[kvh] @ q[h] * scale
        p = np.exp(s - s.max())
        p /= p.sum()
        o[h] = p @ vc[kvh]

    att = o.reshape(H) / (1 + np.exp(-gate))
    att = rne(att)
    y = x + att @ wo.T

    x.tofile(D / "attn_x.bin")
    kcache.tofile(D / "attn_kcache.bin")
    vcache.tofile(D / "attn_vcache.bin")
    y.tofile(D / "attn_y_ref.bin")
    kc[:, T_PRE].tofile(D / "attn_knew_ref.bin")
    vc[:, T_PRE].tofile(D / "attn_vnew_ref.bin")
    print(json.dumps({"y_mean_abs": float(np.abs(y).mean()), "pos": POS}))


if __name__ == "__main__":
    main()
