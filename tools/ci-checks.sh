#!/usr/bin/env bash
# usage: tools/ci-checks.sh
# Every invariant that can be checked without a GPU or the Mojo toolchain.
# Runs in CI and locally. Kernel work still needs ./run-tests.sh on the card.
set -u
cd "$(dirname "$0")/.."
fails=0
step() { printf '\n== %s\n' "$1"; }
ok()   { echo "  OK  $1"; }
bad()  { echo "  FAIL $1"; fails=$((fails + 1)); }

step "kernel census (no orphaned kernels)"
if python3 tools/kernel-census.py --check; then ok "every amar_* kernel is reachable"
else bad "orphaned kernel: it is in kernels/ but no engine, bench or test calls it"; fi

step "docs/KERNELS.md is current"
cp docs/KERNELS.md .ci-kernels-before.md
python3 tools/kernel-census.py >/dev/null
if diff -q .ci-kernels-before.md docs/KERNELS.md >/dev/null; then ok "generated census matches the committed file"
else bad "docs/KERNELS.md is stale; regenerate with tools/kernel-census.py"; diff -u .ci-kernels-before.md docs/KERNELS.md | head -20; fi
mv .ci-kernels-before.md docs/KERNELS.md

step "python sources parse"
pyfiles=$(git ls-files '*.py')
if python3 -m py_compile $pyfiles 2>&1; then ok "$(echo "$pyfiles" | wc -l) files"
else bad "python syntax error"; fi

step "shell scripts parse"
for f in $(git ls-files '*.sh'); do
  bash -n "$f" 2>/dev/null || bad "$f"
done
[ "$fails" = 0 ] && ok "$(git ls-files '*.sh' | wc -l) scripts"

step "issue templates are valid yaml"
if python3 - <<'PY'
import sys, yaml, pathlib
for p in sorted(pathlib.Path(".github").rglob("*.yml")):
    try:
        yaml.safe_load(p.read_text())
    except Exception as e:
        print(f"  {p}: {e}"); sys.exit(1)
PY
then ok "all .github yaml loads"; else bad "invalid issue template yaml"; fi

step "protocol rules P1-P6 present"
missing=""
for n in 1 2 3 4 5 6; do
  grep -qE "^## P$n\." bench/PROTOCOL-RULES.md || missing="$missing P$n"
done
if [ -z "$missing" ]; then ok "P1-P6 all present"
else bad "PROTOCOL-RULES.md lost rules:$missing"; fi

step "referenced protocol and doc files exist"
if python3 - <<'PY'
import pathlib, re, sys
# Repo-relative paths only: a match preceded by "/" or a word character belongs
# to a URL (github.com/ggml-org/ggml/blob/master/docs/gguf.md) and is not ours.
PAT = re.compile(r"(?<![\w/])((?:bench|docs|tools|kernels|serve|shim)/[A-Za-z0-9_.-]+\.(?:md|py|sh|mojo))")
docs = ["README.md", "CONTRIBUTING.md"] + [str(p) for p in sorted(pathlib.Path("docs").glob("*.md"))] \
     + [str(p) for p in sorted(pathlib.Path("bench").glob("*.md"))]
refs, bad = set(), 0
for d in docs:
    for m in PAT.finditer(pathlib.Path(d).read_text()):
        refs.add((m.group(1), d))
for r, d in sorted(refs):
    if not pathlib.Path(r).exists():
        print(f"  {d} references {r}, which does not exist"); bad = 1
print(f"  {len(refs)} referenced paths")
sys.exit(bad)
PY
then ok "every referenced repo path resolves"; else bad "dangling reference in the docs"; fi

step "hardware receipts are well-formed"
if python3 - <<'PY'
import json, pathlib, sys
bad = 0
for p in sorted(pathlib.Path("results").glob("*.json")):
    d = json.loads(p.read_text())
    for k in ("schema", "commit", "env", "valid", "problems", "sizes"):
        if k not in d:
            print(f"  {p.name}: missing '{k}'"); bad = 1
    if d.get("valid") != (not d.get("problems")):
        print(f"  {p.name}: valid flag disagrees with problems list"); bad = 1
    for s in d.get("sizes", []):
        if abs(s["ratio"] - s["ours_gflops"] / s["hipblaslt_gflops"]) > 1e-6:
            print(f"  {p.name}: ratio at {s['size']} does not match its own arms"); bad = 1
sys.exit(bad)
PY
then ok "$(ls results/*.json 2>/dev/null | wc -l) receipts consistent"; else bad "malformed receipt"; fi

printf '\n'
if [ "$fails" = 0 ]; then echo "all non-GPU checks passed"; else echo "$fails check(s) failed"; fi
exit $((fails > 0))
