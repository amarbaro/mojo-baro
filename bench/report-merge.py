#!/usr/bin/env python3
"""Merge the two fp16 arms plus the environment into one receipt.

Called by bench/report.sh, which supplies the environment through env vars.
Refuses to write a receipt whose numbers cannot be trusted: a wrong or
short-warmed number that looks plausible is worse than no number, because
it enters the comparison table and nobody can tell it apart later.
"""
import json, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
WARMUP_FLOOR = 10.0


def load(tag):
    p = ROOT / ".work" / f"fp16-{tag}.jsonl"
    rows = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            j = json.loads(line)
            rows[j["m"]] = j
    return rows


ours, lt = load("rpt_wmma"), load("rpt_lt")
sizes = sorted(set(ours) & set(lt))
if not sizes:
    sys.exit("report-merge: no size produced both arms")

missing = sorted(set(int(s) for s in os.environ["SIZES"].split()) - set(sizes))
problems = []
if missing:
    problems.append(f"no paired result at {missing} (arm crashed or ran out of memory)")
for s in sizes:
    for tag, row in (("ours", ours[s]), ("hipBLASLt", lt[s])):
        if not row.get("correct"):
            problems.append(f"{tag} numerically INCORRECT at {s}^3 (max_err {row['max_err']:.2e})")
        if float(row.get("warmup_s", 0)) < WARMUP_FLOOR:
            problems.append(
                f"{tag} at {s}^3 warmed only {row['warmup_s']}s; below {WARMUP_FLOOR}s the "
                "clocks have not settled and the number reads low")

env = json.loads(os.environ["GPUJSON"])
report = {
    "schema": "mojo-baro/fp16-gemm-report/1",
    "commit": os.environ["COMMIT"],
    "dirty": os.environ["DIRTY"].strip(),
    "mojo": os.environ["MOJOVER"],
    "env": env,
    "state_after": json.loads(os.environ["STATE_AFTER"]),
    "valid": not problems,
    "problems": problems,
    "sizes": [
        {
            "size": s,
            "ours_gflops": ours[s]["gflops"],
            "hipblaslt_gflops": lt[s]["gflops"],
            "ratio": ours[s]["gflops"] / lt[s]["gflops"],
            "ours": ours[s],
            "hipblaslt": lt[s],
        }
        for s in sizes
    ],
}

out = pathlib.Path(os.environ["OUT"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n")

cu = env.get("compute_units")
print()
print(f"## {env['gpu']} ({env['gfx']}, {cu} CU)")
print()
print(f"- commit `{report['commit']}`" + (f" **dirty:** `{report['dirty']}`" if report["dirty"] else ""))
print(f"- ROCm {env['rocm']}, {env['hipblaslt']}, {report['mojo']}")
print(f"- clocks after the run: {report['state_after']}")
print()
print("| size | ours GFLOP/s | hipBLASLt GFLOP/s | ratio |")
print("|---|---|---|---|")
for r in report["sizes"]:
    print(f"| {r['size']}³ | {r['ours_gflops']:.0f} | {r['hipblaslt_gflops']:.0f} | {r['ratio']:.2f} |")
print()
if problems:
    print("**This receipt is NOT valid.** Please still open the issue -- a failure")
    print("on hardware we do not have is a useful result. Reasons:")
    for p in problems:
        print(f"- {p}")
else:
    print("All arms numerically correct, warm-up satisfied.")
print()
print(f"Full receipt: `{out.relative_to(ROOT) if out.is_absolute() else out}` (attach it to the issue)")
