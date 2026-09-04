#!/usr/bin/env bash
# usage: tools/llama-mtp-prompts.sh MODEL.gguf OUTDIR [SPEC_N_MAX] [PORT]
# llama.cpp bar on bench/mtp-prompts/: one server pass with MTP
# (--spec-type draft-mtp --spec-draft-n-max N), one without; per prompt one
# greedy 64-token completion. Records predicted_per_second, draft_n,
# draft_n_accepted and the token ids. Engine must not be running (GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
model=$1; out=$2; nmax=${3:-4}; port=${4:-8097}
mkdir -p "$out"
run_pass() {  # $1 = tag, rest = extra server flags
  tag=$1; shift
  ~/llama.cpp/build/bin/llama-server -m "$model" -c 8192 -ngl 99 -fa on -b 2048 -ub 512 -t 8 \
    --host 127.0.0.1 --port "$port" -ctk q8_0 -ctv q8_0 "$@" > "$out/server-$tag.log" 2>&1 &
  pid=$!
  for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done
  curl -s "http://127.0.0.1:$port/props" > "$out/props-$tag.json"
  for tf in bench/mtp-prompts/p*.tokens; do
    p=$(basename "$tf" .tokens); prompt=$(tr -s ' \n' ',,' < "$tf" | sed 's/,$//')
    curl -s "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
      -d "{\"prompt\": [$prompt], \"n_predict\": 64, \"temperature\": 0, \"top_k\": 1, \"cache_prompt\": false, \"return_tokens\": true}" \
      > "$out/$p.$tag.json"
  done
  kill $pid; wait $pid 2>/dev/null || true
  sleep 2
}
run_pass mtp --spec-type draft-mtp --spec-draft-n-max "$nmax"
run_pass nospec
python3 - "$out" <<'PY'
import sys,json,glob,os
out=sys.argv[1]
print("prompt nospec_tps mtp_tps ratio draft_n accepted acc identity")
for f in sorted(glob.glob(f"{out}/p*.mtp.json")):
    p=os.path.basename(f).split('.')[0]
    m=json.load(open(f)); n=json.load(open(f"{out}/{p}.nospec.json"))
    tm,tn=m["timings"],n["timings"]
    idn="PASS" if m["tokens"]==n["tokens"] else "FAIL"
    d=tm.get("draft_n",0); a=tm.get("draft_n_accepted",0)
    print(p, f'{tn["predicted_per_second"]:.1f}', f'{tm["predicted_per_second"]:.1f}', f'{tm["predicted_per_second"]/tn["predicted_per_second"]:.2f}', d, a, f'{(a/d*100 if d else 0):.0f}%', idn)
PY
