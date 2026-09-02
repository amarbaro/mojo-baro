#!/usr/bin/env bash
# usage: bench/fp16-templates.sh [LOG]
# Ten square fp16 size templates, ours (WMMA) vs hipBLASLt, one build+run per size.
# Stops llama-server first (restores it on exit). Every arm-defining parameter is
# read back from each binary's own JSON output and printed in the table (PROTOCOL-RULES P1).
set -eu; cd "$(dirname "$0")/.."
LOG=${1:-.work/fp16-templates-$(date +%Y%m%d-%H%M).log}; exec > >(tee -a "$LOG") 2>&1
SIZES="256 512 768 1024 1536 2048 2560 3072 3584 4096"
P=$(pgrep -f '^/home/mario/llama.cpp/build/bin/llama-server' || true); [ -n "$P" ] && { echo "killed llama-server pid $P"; kill $P; sleep 3; }
trap 'cmd=$(cat ~/Brain/mojo-baro/llama-server-cmdline.txt); (setsid nohup bash -c "$cmd" > .work/llama-server.log 2>&1 < /dev/null &); echo "server restart issued"' EXIT
echo "commit: $(git rev-parse --short HEAD)  dirty: [$(git status --short | tr '\n' ' ')]  gpu: $(rocm-smi --showproductname 2>/dev/null | grep -m1 'Card series' | sed 's/.*: //')  mojo: $(./.venv/bin/mojo --version | head -1)"
echo "env: MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10  sizes: $SIZES  iters + warmup_s per arm: read back from each binary JSON"
rm -f .work/fp16-tpl_wmma.jsonl .work/fp16-tpl_lt.jsonl
bench/fp16-sizes.sh bench/bench_fp16_pipe.mojo tpl_wmma $SIZES | grep '^{' >/dev/null
bench/fp16-sizes.sh bench/bench_fp16_lt.mojo tpl_lt $SIZES | grep '^{' >/dev/null
python3 - <<'PY'
import json
w={j["m"]:j for j in map(json.loads,open(".work/fp16-tpl_wmma.jsonl"))}
l={j["m"]:j for j in map(json.loads,open(".work/fp16-tpl_lt.jsonl"))}
print(f"{'size':>5} | {'ours gflops':>11} {'ok':>5} {'max_err':>9} {'iters':>5} {'warm':>5} {'pgr':>3} {'lb':>2} {'blk':>12} {'warps':>6} {'wtile':>6} {'grid':>8} | {'hipBLASLt':>10} {'iters':>5} {'warm':>5} {'ok':>5} {'max_err':>8} {'algo':>4} {'splitk':>6} | {'ratio':>6}")
for s in sorted(w):
    a,b=w[s],l[s]
    print(f"{s:>5} | {a['gflops']:>11.0f} {str(a['correct']):>5} {a['max_err']:>9.2e} {a['iters']:>5} {a['warmup_s']:>5.0f} {a['pgr']:>3} {a['lb']:>2} {str(a['blk']):>12} {str(a['warps']):>6} {str(a['wtile']):>6} {str(a['grid']):>8} | {b['gflops']:>10.0f} {b['iters']:>5} {b['warmup_s']:>5.0f} {str(b['correct']):>5} {b['max_err']:>8.2e} {b['algo_chosen']:>4} {b['splitk']:>6} | {a['gflops']/b['gflops']:>6.3f}")
bad=[s for s in w if not w[s]['correct']]+[s for s in l if not l[s]['correct']]
print("ALL CORRECT" if not bad else f"INCORRECT at {bad}")
PY
