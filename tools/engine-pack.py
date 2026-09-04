#!/usr/bin/env python3
"""Build the engine weight pack from a GGUF: one flat binary + text index.

2D bf16 weights are stored TRANSPOSED (B-layout [in, out]) for the skinny
GEMM; f32 tensors (norms, conv, ssm scalars) as-is. Emits, per line:
  name dtype offset_bytes n_elem
in a fixed, engine-known order. The blk.32 NextN draft head is appended
after output.weight, so every trunk offset is unchanged by its presence.

Usage: tools/engine-pack.py MODEL.gguf OUTDIR [--q8]

--q8: every 2D bf16 weight except token_embd is stored int8 in weight-native
[out, in] layout followed by fp16 block scales [out, in/32], ggml q8_0
rounding (d = amax/127 in fp32, q = roundf(x * (1/d)), d stored as fp16).
Index dtype is "q8"; n_elem counts weights, the scales follow at
offset + n_elem bytes.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("ge", Path(__file__).parent / "gguf-extract.py")
ge = module_from_spec(spec)
spec.loader.exec_module(ge)

N_LAYERS = 32


def is_attn(i):
    return (i + 1) % 4 == 0


def mtp_names():
    base = "blk.32."
    core = ["attn_norm.weight", "attn_q.weight", "attn_k.weight",
            "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight",
            "attn_output.weight", "post_attention_norm.weight",
            "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
            "nextn.eh_proj.weight", "nextn.enorm.weight",
            "nextn.hnorm.weight", "nextn.shared_head_norm.weight"]
    return [base + n for n in core]


def layer_names(i):
    base = f"blk.{i}."
    if is_attn(i):
        core = ["attn_norm.weight", "attn_q.weight", "attn_k.weight",
                "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                "attn_output.weight"]
    else:
        core = ["attn_norm.weight", "attn_qkv.weight", "attn_gate.weight",
                "ssm_alpha.weight", "ssm_beta.weight", "ssm_conv1d.weight",
                "ssm_a", "ssm_dt.bias", "ssm_norm.weight", "ssm_out.weight"]
    ffn = ["post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight",
           "ffn_down.weight"]
    return [base + n for n in core + ffn]


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def quantize_q8_0(w16):
    x = bf16_to_f32(w16).reshape(w16.shape[0], -1, 32)
    amax = np.abs(x).max(axis=2)
    d = (amax / 127.0).astype(np.float32)
    idv = np.where(d != 0, np.float32(1.0) / d, np.float32(0)).astype(np.float32)
    x0 = (x * idv[:, :, None]).astype(np.float32)
    q = np.sign(x0) * np.floor(np.abs(x0) + np.float32(0.5))
    q = np.clip(q, -128, 127).astype(np.int8)
    return q.reshape(w16.shape[0], -1), d.astype(np.float16)


def main():
    q8 = "--q8" in sys.argv
    if q8:
        sys.argv.remove("--q8")
    model, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    f, infos, data_start, kv = ge.parse(model)

    order = ["token_embd.weight"]
    for i in range(N_LAYERS):
        order += layer_names(i)
    order += ["output_norm.weight", "output.weight"]
    order += mtp_names()

    idx_lines = []
    off = 0
    with open(outdir / "pack.bin", "wb") as out:
        for name in order:
            dims, ttype, toff = infos[name]
            tname, esize = ge.GGML_BYTES[ttype]
            n_elem = int(np.prod(dims))
            f.seek(data_start + toff)
            raw = f.read(n_elem * esize)
            shape = list(reversed(dims))
            if tname == "bf16" and len(shape) == 2 and name != "token_embd.weight":
                w = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                if q8:
                    q, d = quantize_q8_0(w)
                    raw = q.tobytes() + d.tobytes()
                    tname = "q8"
                else:
                    raw = np.ascontiguousarray(w.T).tobytes()
            out.write(raw)
            idx_lines.append(f"{name} {tname} {off} {n_elem}")
            off += len(raw)
    (outdir / "index.txt").write_text("\n".join(idx_lines) + "\n")
    print(f"packed {len(order)} tensors, {off/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
