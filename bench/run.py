#!/usr/bin/env python3
"""Build, run, and log the GEMM benchmarks.

Every run appends to bench/log.jsonl with the commit it measured, so a number
is always traceable to the code that produced it. Prints each variant against
the best previously recorded result for the same shape.
"""
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = ROOT / "bench" / "log.jsonl"
BIN = ROOT / ".work" / "bench"
SHIM = ROOT / ".work" / "shim-build"


def sh(*cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, **kw)


def commit():
    r = sh("git", "rev-parse", "--short", "HEAD")
    dirty = sh("git", "status", "--porcelain").stdout.strip()
    return (r.stdout.strip() or "unknown") + ("-dirty" if dirty else "")


def gpu_name():
    r = sh("rocminfo")
    for line in r.stdout.splitlines():
        if "gfx" in line and "Name:" in line:
            return line.split(":")[-1].strip()
    return "unknown"


def history():
    if not LOG.exists():
        return []
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]


def main():
    build = sh(
        str(ROOT / ".venv/bin/mojo"), "build", "bench/bench.mojo",
        "-o", str(BIN), "-I", "kernels",
        "-Xlinker", f"-L{SHIM}", "-Xlinker", "-lbaro_shim",
        "-Xlinker", "-rpath", "-Xlinker", str(SHIM),
    )
    if build.returncode:
        print(build.stderr, file=sys.stderr)
        return 1

    run = subprocess.run([str(BIN)], cwd=ROOT, text=True, capture_output=True)
    if run.returncode:
        print(run.stderr, file=sys.stderr)
        return 1

    prior = history()
    sha, gpu, ts = commit(), gpu_name(), time.time()
    rows, failed = [], False

    with LOG.open("a") as fh:
        for line in run.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            rec = json.loads(line)
            rec.update(commit=sha, gpu=gpu, ts=ts)
            fh.write(json.dumps(rec) + "\n")
            rows.append(rec)
            failed |= not rec["correct"]

    print(f"{'variant':<10} {'ms':>9} {'GFLOP/s':>10} {'vs best':>9}  correct")
    for r in rows:
        same = [
            p for p in prior
            if p["variant"] == r["variant"]
            and (p["m"], p["n"], p["k"]) == (r["m"], r["n"], r["k"])
            and p.get("correct")
        ]
        best = max((p["gflops"] for p in same), default=None)
        delta = "  —" if best is None else f"{r['gflops'] / best:.2f}x"
        mark = "ok" if r["correct"] else f"WRONG (err {r['max_err']:.3g})"
        print(f"{r['variant']:<10} {r['ms']:>9.4f} {r['gflops']:>10.1f} {delta:>9}  {mark}")

    if failed:
        print("\nFAIL: a variant produced incorrect results", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
