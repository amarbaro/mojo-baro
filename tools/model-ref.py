#!/usr/bin/env python3
"""Streaming numpy reference for the qwen35 stack over the engine pack.

Runs the first N layers for token POS=0 (zero states) and prints the same
per-layer probes serve/engine.mojo emits, to bisect chain divergence.
Reads tensors lazily from .work/engine-pack/pack.bin via the index.
"""
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent.parent / ".work/engine-pack"
H, FFN, CONV, KDIM = 4096, 12288, 8192, 2048
NH_K, NH_V, S = 16, 32, 128
HD, NQH, NKVH, NROT = 256, 16, 4, 64
EPS = 1e-6

idx = {}
for line in (D / "index.txt").read_text().splitlines():
    name, dt, off, n = line.split()
    idx[name] = (dt, int(off), int(n))
pack = np.memmap(D / "pack.bin", dtype=np.uint8, mode="r")


def T(name, shape):
    dt, off, n = idx[name]
    if dt == "bf16":
        raw = pack[off : off + n * 2].view(np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    return pack[off : off + n * 4].view(np.float32).reshape(shape).copy()


def amar_rmsnorm(x, w, axis=-1):
    return x / np.sqrt(np.mean(x * x, axis=axis, keepdims=True) + EPS) * w


def rne(x):
    u = np.ascontiguousarray(x).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)


def silu(x):
    return x / (1 + np.exp(-x))


def yarn_rope(x, pos):
    FREQ_BASE, FREQ_SCALE, N_CTX = 1e7, 0.25, 262144
    def corr(beta):
        return NROT * np.log(N_CTX / (beta * 2 * np.pi)) / (2 * np.log(FREQ_BASE))
    low = np.clip(np.floor(corr(32.0)), 0, NROT - 1)
    high = np.clip(np.ceil(corr(1.0)), 0, NROT - 1)
    j = np.arange(NROT // 2, dtype=np.float32)
    te = pos * FREQ_BASE ** (-2.0 * j / NROT)
    ti = FREQ_SCALE * te
    ramp = np.clip((j - low) / max(high - low, 0.001), 0, 1)
    th = ti * (1 - ramp) + te * ramp
    m = 1 + 0.1 * np.log(1 / FREQ_SCALE)
    c, s = np.cos(th) * m, np.sin(th) * m
    y = x.copy()
    x0, x1 = x[..., : NROT // 2], x[..., NROT // 2 : NROT]
    y[..., : NROT // 2] = x0 * c - x1 * s
    y[..., NROT // 2 : NROT] = x0 * s + x1 * c
    return y


CONV_STATE = {}
S_STATE = {}
KC = {}
VC = {}


def ssm_layer(i, x, pos):
    b = f"blk.{i}."
    cur = rne(amar_rmsnorm(x, T(b + "attn_norm.weight", (H,))))
    qkv = cur @ T(b + "attn_qkv.weight", (H, CONV))
    z = cur @ T(b + "attn_gate.weight", (H, H))
    beta = 1 / (1 + np.exp(-(cur @ T(b + "ssm_alpha.weight", (H, NH_V)) * 0 + cur @ T(b + "ssm_beta.weight", (H, NH_V)))))
    a_sp = np.log1p(np.exp(cur @ T(b + "ssm_alpha.weight", (H, NH_V)) + T(b + "ssm_dt.bias", (NH_V,))))
    g = a_sp * T(b + "ssm_a", (NH_V,))
    conv_w = T(b + "ssm_conv1d.weight", (CONV, 4))
    st = CONV_STATE.setdefault(i, np.zeros((3, CONV), dtype=np.float32))
    win = np.concatenate([st, qkv[None, :]], axis=0)
    CONV_STATE[i] = win[1:].copy()
    conv_out = silu(np.einsum("tc,ct->c", win, conv_w))
    q = conv_out[:KDIM].reshape(NH_K, S)
    k = conv_out[KDIM : 2 * KDIM].reshape(NH_K, S)
    v = conv_out[2 * KDIM :].reshape(NH_V, S)
    q = q / np.sqrt(np.sum(q * q, axis=1, keepdims=True) + EPS)
    k = k / np.sqrt(np.sum(k * k, axis=1, keepdims=True) + EPS)
    q = np.tile(q, (2, 1)) / np.sqrt(S)
    k = np.tile(k, (2, 1))
    o = np.zeros((NH_V, S), dtype=np.float32)
    SS = S_STATE.setdefault(i, np.zeros((NH_V, S, S), dtype=np.float32))
    for h in range(NH_V):
        Sh = SS[h] * np.exp(g[h])
        d = (v[h] - Sh.T @ k[h]) * beta[h]
        Sh = Sh + np.outer(k[h], d)
        o[h] = Sh.T @ q[h]
        SS[h] = Sh
    gated = rne((amar_rmsnorm(o, T(b + "ssm_norm.weight", (S,))) * silu(z.reshape(NH_V, S))).reshape(H))
    return x + gated @ T(b + "ssm_out.weight", (H, H))


def attn_layer(i, x, pos):
    b = f"blk.{i}."
    cur = rne(amar_rmsnorm(x, T(b + "attn_norm.weight", (H,))))
    qf = (cur @ T(b + "attn_q.weight", (H, 2 * H))).reshape(NQH, 2 * HD)
    q, gate = qf[:, :HD], qf[:, HD:].reshape(H)
    k = (cur @ T(b + "attn_k.weight", (H, NKVH * HD))).reshape(NKVH, HD)
    v = (cur @ T(b + "attn_v.weight", (H, NKVH * HD))).reshape(NKVH, HD)
    q = yarn_rope(amar_rmsnorm(q, T(b + "attn_q_norm.weight", (HD,))), pos)
    k = yarn_rope(amar_rmsnorm(k, T(b + "attn_k_norm.weight", (HD,))), pos)
    kc = KC.setdefault(i, np.zeros((NKVH, 0, HD), dtype=np.float32))
    vc = VC.setdefault(i, np.zeros((NKVH, 0, HD), dtype=np.float32))
    kc = np.concatenate([kc, k[:, None, :]], axis=1)
    vc = np.concatenate([vc, v[:, None, :]], axis=1)
    KC[i], VC[i] = kc, vc
    o = np.zeros((NQH, HD), dtype=np.float32)
    scale = 1.0 / np.sqrt(HD)
    for h in range(NQH):
        sc = kc[h // 4] @ q[h] * scale
        pr = np.exp(sc - sc.max()); pr /= pr.sum()
        o[h] = pr @ vc[h // 4]
    att = rne(o.reshape(H) / (1 + np.exp(-gate)))
    return x + att @ T(b + "attn_output.weight", (H, H))


def full_decode(n_gen):
    prompt = [int(t) for t in (D / "prompt-tokens.txt").read_text().split()]
    emb = T("token_embd.weight", (248320, H))
    on = T("output_norm.weight", (H,))
    generated = []
    tok = prompt[0]
    n_total = len(prompt) + n_gen
    for pos in range(n_total - 1):
        tok = prompt[pos] if pos < len(prompt) else nxt
        x = emb[tok].astype(np.float32).copy()
        for i in range(32):
            x = attn_layer(i, x, pos) if (i + 1) % 4 == 0 else ssm_layer(i, x, pos)
            cur = rne(amar_rmsnorm(x, T(f"blk.{i}.post_attention_norm.weight", (H,))))
            gt = cur @ T(f"blk.{i}.ffn_gate.weight", (H, FFN))
            up = cur @ T(f"blk.{i}.ffn_up.weight", (H, FFN))
            act = rne(silu(gt) * up)
            x = x + act @ T(f"blk.{i}.ffn_down.weight", (FFN, H))
        logits = rne(amar_rmsnorm(x, on)) @ T("output.weight", (H, 248320))
        nxt = int(np.argmax(logits))
        if pos >= len(prompt) - 1:
            generated.append(nxt)
            print("gen:", nxt, flush=True)
    print("GENERATED:", generated)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "decode":
        full_decode(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
        return
    n_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    tok = 760
    emb = T("token_embd.weight", (248320, H))
    x = emb[tok].astype(np.float32).copy()
    for i in range(n_layers):
        if (i + 1) % 4 == 0:
            x = attn_layer(i, x, 0)
        else:
            x = ssm_layer(i, x, 0)
        # ffn
        cur = rne(amar_rmsnorm(x, T(f"blk.{i}.post_attention_norm.weight", (H,))))
        gt = cur @ T(f"blk.{i}.ffn_gate.weight", (H, FFN))
        up = cur @ T(f"blk.{i}.ffn_up.weight", (H, FFN))
        act = rne(silu(gt) * up)
        x = x + act @ T(f"blk.{i}.ffn_down.weight", (FFN, H))
        print(f"dbg L {i} : {x[0]:.8f} {x[1]:.8f} {x[2]:.8f} {x[3]:.8f}")


if __name__ == "__main__":
    main()
