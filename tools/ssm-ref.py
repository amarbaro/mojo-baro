#!/usr/bin/env python3
"""Numpy reference for one decode token through a qwen35 gated-delta-net block.

Implements docs/qwen35-ssm-notes.md exactly (llama.cpp op order: state decay
BEFORE the delta error term). Consumes blk.0 tensors from .work/gguf/, invents
a fixed-seed hidden state + conv window + recurrent state, and writes every
input and the expected outputs for kernels/test_ssm_block.mojo.
"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent.parent / ".work/gguf"
H = 4096
CONV = 8192
KDIM = 2048
NH_K = 16
NH_V = 32
S = 128
EPS = 1e-6


def bf16(path, shape):
    w = np.fromfile(D / path, dtype=np.uint16).reshape(shape)
    return (w.astype(np.uint32) << 16).view(np.float32)


def f32(path, shape):
    return np.fromfile(D / path, dtype=np.float32).reshape(shape)


def rmsnorm(x, w, axis=-1):
    return x / np.sqrt(np.mean(x * x, axis=axis, keepdims=True) + EPS) * w


def silu(x):
    return x / (1 + np.exp(-x))


def main():
    wqkv = bf16("blk_0_attn_qkv_weight.bin", (CONV, H))
    wz = bf16("blk_0_attn_gate_weight.bin", (H, H))
    walpha = bf16("blk_0_ssm_alpha_weight.bin", (NH_V, H))
    wbeta = bf16("blk_0_ssm_beta_weight.bin", (NH_V, H))
    wout = bf16("blk_0_ssm_out_weight.bin", (H, H))
    conv_w = f32("blk_0_ssm_conv1d_weight.bin", (CONV, 4))
    ssm_a = f32("blk_0_ssm_a.bin", (NH_V,))
    dt_bias = f32("blk_0_ssm_dt_bias.bin", (NH_V,))
    norm_w = f32("blk_0_ssm_norm_weight.bin", (S,))
    attn_norm = f32("blk_0_attn_norm_weight.bin", (H,))

    rng = np.random.default_rng(11)
    x = rng.standard_normal(H).astype(np.float32) * 0.5
    conv_state = rng.standard_normal((3, CONV)).astype(np.float32) * 0.1
    S0 = rng.standard_normal((NH_V, S, S)).astype(np.float32) * 0.05

    cur = rmsnorm(x, attn_norm)
    u = cur.view(np.uint32)
    cur = (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)

    qkv = cur @ wqkv.T
    z = cur @ wz.T
    beta = 1 / (1 + np.exp(-(cur @ wbeta.T)))
    a_sp = np.log1p(np.exp(cur @ walpha.T + dt_bias))
    g = a_sp * ssm_a

    win = np.concatenate([conv_state, qkv[None, :]], axis=0)
    conv_out = np.einsum("tc,ct->c", win, conv_w)
    conv_out = silu(conv_out)
    new_conv_state = win[1:]

    q = conv_out[:KDIM].reshape(NH_K, S)
    k = conv_out[KDIM : 2 * KDIM].reshape(NH_K, S)
    v = conv_out[2 * KDIM :].reshape(NH_V, S)
    q = q / np.sqrt(np.sum(q * q, axis=1, keepdims=True) + EPS)
    k = k / np.sqrt(np.sum(k * k, axis=1, keepdims=True) + EPS)
    q = np.repeat(q, NH_V // NH_K, axis=0) / np.sqrt(S)
    k = np.repeat(k, NH_V // NH_K, axis=0)

    o = np.zeros((NH_V, S), dtype=np.float32)
    S1 = np.zeros_like(S0)
    for h in range(NH_V):
        Sh = S0[h] * np.exp(g[h])
        sk = Sh.T @ k[h]
        d = (v[h] - sk) * beta[h]
        Sh = Sh + np.outer(k[h], d)
        o[h] = Sh.T @ q[h]
        S1[h] = Sh

    gated = (rmsnorm(o, norm_w) * silu(z.reshape(NH_V, S))).reshape(H)
    ug = gated.view(np.uint32)
    gated = (((ug + 0x7FFF + ((ug >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)
    out = gated @ wout.T
    y = x + out

    o.tofile(D / "ssm_o_ref.bin")
    z.tofile(D / "ssm_z_ref.bin")
    gated.tofile(D / "ssm_gated_ref.bin")
    x.tofile(D / "ssm_x.bin")
    conv_state.tofile(D / "ssm_conv_state.bin")
    S0.tofile(D / "ssm_s0.bin")
    y.tofile(D / "ssm_y_ref.bin")
    S1.tofile(D / "ssm_s1_ref.bin")
    new_conv_state.tofile(D / "ssm_conv_state1_ref.bin")
    print(json.dumps({"y_mean_abs": float(np.abs(y).mean()),
                      "s1_mean_abs": float(np.abs(S1).mean())}))


if __name__ == "__main__":
    main()
