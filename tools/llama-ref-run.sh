#!/usr/bin/env bash
# usage: tools/llama-ref-run.sh MODEL.gguf OUTDIR [PORT] [N_RUNS]
# Starts llama-server (no speculative flags, greedy), posts the pack's prompt
# as TOKEN IDS, writes ref-tokens-64-llama.txt (run 1), timings.jsonl (all
# runs, predicted_per_second read back from the response), props.json (P1
# receipt), then stops the server.
set -euo pipefail
model=$1; out=$2; port=${3:-8097}; runs=${4:-5}
mkdir -p "$out"
prompt=$(tr -s ' \n' ',,' < .work/engine-pack/prompt-tokens.txt | sed 's/,$//')
~/llama.cpp/build/bin/llama-server -m "$model" -c 8192 -ngl 99 -fa on -b 2048 -ub 512 -t 8 \
  --host 127.0.0.1 --port "$port" -ctk q8_0 -ctv q8_0 > "$out/server.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done
curl -s "http://127.0.0.1:$port/props" > "$out/props.json"
: > "$out/timings.jsonl"
for r in $(seq 1 "$runs"); do
  resp=$(curl -s "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\": [$prompt], \"n_predict\": 64, \"temperature\": 0, \"top_k\": 1, \"cache_prompt\": false, \"return_tokens\": true}")
  echo "$resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); t=d.get("timings",{}); print(json.dumps({"run": '"$r"', "predicted_n": t.get("predicted_n"), "predicted_per_second": t.get("predicted_per_second"), "draft_n": t.get("draft_n"), "prompt_n": t.get("prompt_n")}))' >> "$out/timings.jsonl"
  if [ "$r" = 1 ]; then
    echo "$resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("\n".join(str(t) for t in d["tokens"]))' > "$out/ref-tokens-64-llama.txt"
  fi
done
cat "$out/timings.jsonl"
