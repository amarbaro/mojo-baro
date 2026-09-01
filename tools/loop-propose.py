#!/usr/bin/env python3
"""Proposer step of the self-optimising loop: the served model reads the engine
sources embedded in its own gguf plus the engine's BARO_PROFILE shares, and
proposes ONE unified diff per identity framing.

Usage: tools/loop-propose.py MODEL.gguf ITER [--n 4] [--region auto|attn|ssm|ffn|head]
         [--profile .work/profile-run.log] [--endpoint http://127.0.0.1:8083]
Writes .work/loop/<ITER>/{meta.json,FILES,src/,prompt.md,cand-<i>.raw.md,cand-<i>.diff,cand-<i>.predict}
"""
import argparse, json, re, subprocess, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IDENT = Path.home() / "Brain/identities/identities-index.md"
MARK = {"attn": ("# -- attention / ssm sub-block --", "# -- ffn sub-block --"),
        "ssm": ("# -- attention / ssm sub-block --", "# -- ffn sub-block --"),
        "ffn": ("# -- ffn sub-block --", "# -- head --"),
        "head": ("# -- head --", "# advance by the window")}
KFILES = {"attn": ["attn.mojo", "elementwise.mojo", "matmul_skinny.mojo"],
          "ssm": ["ssm.mojo", "elementwise.mojo", "matmul_skinny.mojo"],
          "ffn": ["elementwise.mojo", "matmul_skinny.mojo"],
          "head": ["elementwise.mojo", "matmul_skinny.mojo"]}

RULES = """You are optimising the inference engine embedded inside your own model file
(Qwythos-9B, AMD RX 7900 XTX gfx1100, Mojo). Decode is GPU-bound and
bandwidth-bound: single-token decode reads every weight once, so wins come from
fewer bytes moved, fewer kernel launches, and fused math -- not from FLOPs.
Propose exactly ONE change, one concern, as a unified diff against the files
shown (paths relative to the source dir, e.g. `--- a/engine.mojo`). Rules:
- Only edit the files shown. Never touch anything else.
- Kernel files carry zero comments and zero docstrings.
- Output must stay bit-identical: the gate compares 64 greedy tokens.
- Do not change timing, printing, or profiling code.
- End with one line `PREDICT: <signed percent>` = your predicted change in tok/s_gen.
Reply with a short rationale (<= 8 lines), then the diff in a ```diff fence, then PREDICT."""


def identities():
    rows = []
    for line in IDENT.read_text().splitlines():
        m = re.match(r"\| (\d\d) \| \[\[([^\]]+)\]\] \| [^|]+ \| (.+?) \|\s*$", line)
        if m:
            rows.append((m.group(2), m.group(3).strip()))
    return rows


def slice_region(engine_src, region):
    a, b = MARK[region]
    lines = engine_src.splitlines()
    ia = next(i for i, l in enumerate(lines) if a in l)
    ib = next(i for i, l in enumerate(lines) if b in l and i > ia)
    ka = next(i for i, l in enumerate(lines) if "# --- kernel bindings" in l)
    kb = next(i for i, l in enumerate(lines) if "# --- decode loop" in l)
    return (f"engine.mojo lines {ka+1}-{kb} (kernel bindings):\n" + "\n".join(lines[ka:kb]) +
            f"\n\nengine.mojo lines {ia+1}-{ib} (target region `{region}`):\n" + "\n".join(lines[ia:ib]))


def profile_shares(path):
    shares = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"profile: (\w+) ([\d.]+) ([\d.]+)", line)
        if m:
            shares[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return shares


def ask(endpoint, prompt, identity_line):
    body = {"messages": [{"role": "system", "content": RULES + "\n\nFraming for this attempt: " + identity_line},
                         {"role": "user", "content": prompt}],
            "temperature": 0.6, "max_tokens": 4096, "cache_prompt": True}
    req = urllib.request.Request(endpoint + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("iter")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--region", default="auto")
    ap.add_argument("--profile", default=".work/profile-run.log")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8083")
    ap.add_argument("--start", type=int, default=0, help="first identity index")
    a = ap.parse_args()
    out = ROOT / ".work/loop" / a.iter
    (out / "src").mkdir(parents=True, exist_ok=True)
    meta = json.loads(subprocess.run([str(ROOT / ".venv/bin/python3"), str(ROOT / "tools/gguf-extract.py"),
                                      a.model, "--meta"], capture_output=True, text=True, check=True).stdout)
    files = meta["baro.kernel.files"].split(",")
    (out / "FILES").write_text("\n".join(files) + "\n")
    (out / "meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k.startswith("baro.")}, indent=1))
    for f in files:
        (out / "src" / f).write_text(meta[f"baro.kernel.src.{f}"])
    shares = profile_shares(a.profile)
    region = a.region if a.region != "auto" else max(shares, key=lambda k: shares[k][1])
    prof_txt = "\n".join(f"  {k}: {v[0]*1000:.1f} ms  ({v[1]*100:.1f}%)" for k, v in shares.items())
    prompt = (f"Engine source commit in this gguf: {meta['baro.kernel.commit']}\n"
              f"GPU time per decode run, by sub-block (BARO_PROFILE=1):\n{prof_txt}\n"
              f"Target region: `{region}` (largest share).\n\n"
              + slice_region((out / "src/engine.mojo").read_text(), region))
    for f in KFILES[region]:
        prompt += f"\n\n===== {f} =====\n" + (out / "src" / f).read_text()
    (out / "prompt.md").write_text(prompt)
    ids = identities()[a.start:a.start + a.n]
    print(f"region={region} prompt_chars={len(prompt)} identities={[i[0] for i in ids]}", flush=True)
    for i, (name, line) in enumerate(ids):
        raw = ask(a.endpoint, prompt, line)
        (out / f"cand-{i}.raw.md").write_text(f"identity: {name}\n\n" + raw)
        body = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        m = re.search(r"```diff\n(.*?)```", body, flags=re.S)
        p = re.search(r"PREDICT:\s*([+-]?\d+(?:\.\d+)?)\s*%?", body)
        (out / f"cand-{i}.diff").write_text(m.group(1) if m else "")
        (out / f"cand-{i}.predict").write_text((p.group(1) if p else "none") + "\n")
        print(f"cand-{i} [{name}] diff={'yes' if m else 'NO'} lines={m.group(1).count(chr(10)) if m else 0} "
              f"predict={p.group(1) if p else 'none'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
