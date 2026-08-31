#!/usr/bin/env python3
"""Build, run, and log the GEMM benchmarks.

Results go into the AMDHQ experiment ledger (~/AMDHQ/runs/runs.jsonl +
runs.sqlite), not a private log -- this box already has one record of ROCm
experiments and a second would fragment it. Each run carries the commit it
measured, so a number is always traceable to the code that produced it, and
prints against the best previously recorded result for the same shape.
"""
import json
import pathlib
import os
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BIN = ROOT / ".work" / "bench"
AMDHQ = pathlib.Path("/home/mario/AMDHQ")

sys.path.insert(0, str(AMDHQ))
from tools.ledger import ExperimentLedger  # noqa: E402

LEDGER = ExperimentLedger()
LOG = pathlib.Path(LEDGER.jsonl_path)
SHIM = ROOT / ".work" / "shim-build"


# MAX's device pool otherwise claims ~90% of VRAM on DeviceContext creation.
# At 100% hipBLASLt cannot allocate its handle and the process segfaults; the
# cap costs nothing here and measured slightly faster.
ROLE_KEY = "mojo-baro-gemm"

ENV = {**os.environ}
ENV.setdefault("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT", "10")


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
    """Prior mojo-baro GEMM runs, flattened back to the shape the table wants."""
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("role_key") != ROLE_KEY:
            continue
        cfg, met = rec.get("config", {}), rec.get("metrics", {})
        out.append({
            "variant": rec.get("backend", ""),
            "m": cfg.get("m"), "n": cfg.get("n"), "k": cfg.get("k"),
            "gflops": met.get("gflops"),
            "correct": rec.get("status") == "SUCCESS",
        })
    return out


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

    run = subprocess.run([str(BIN)], cwd=ROOT, text=True, capture_output=True, env=ENV)
    if run.returncode:
        print(run.stderr, file=sys.stderr)
        return 1

    prior = history()
    sha, gpu = commit(), gpu_name()
    rows, failed = [], False

    for line in run.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        LEDGER.record_run(
            target_node=gpu,
            role_key=ROLE_KEY,
            model_name=f"gemm-{rec['m']}x{rec['n']}x{rec['k']}-{rec['dtype']}",
            backend=rec["variant"],
            config={k: rec[k] for k in ("m", "n", "k", "iters", "tile", "dtype")},
            metrics={"gflops": rec["gflops"], "ms": rec["ms"],
                     "max_err": rec["max_err"]},
            status="SUCCESS" if rec["correct"] else "FAIL",
            notes=f"commit {sha}",
        )
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
