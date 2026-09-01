#!/usr/bin/env python3
"""Build the engine weight pack from a GGUF: one flat binary + text index.

2D bf16 weights are stored TRANSPOSED (B-layout [in, out]) for the skinny
GEMM; f32 tensors (norms, conv, ssm scalars) as-is. Emits, per line:
  name dtype offset_bytes n_elem
in a fixed, engine-known order. MTP block 32 is skipped.

Usage: tools/engine-pack.py MODEL.gguf OUTDIR
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


def main():
    model, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    f, infos, data_start, kv = ge.parse(model)

    order = ["token_embd.weight"]
    for i in range(N_LAYERS):
        order += layer_names(i)
    order += ["output_norm.weight", "output.weight"]

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
                raw = np.ascontiguousarray(w.T).tobytes()
            out.write(raw)
            idx_lines.append(f"{name} {tname} {off} {n_elem}")
            off += len(raw)
    (outdir / "index.txt").write_text("\n".join(idx_lines) + "\n")
    print(f"packed {len(order)} tensors, {off/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
