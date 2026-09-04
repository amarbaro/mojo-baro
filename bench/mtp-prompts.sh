#!/usr/bin/env bash
# usage: bench/mtp-prompts.sh ENGINE OUTDIR [K...]   (default K=4)
# For every bench/mtp-prompts/*.tokens: arm A (no spec) once, arm B per K once.
# Gate: arm B GENERATED == arm A GENERATED (greedy identity). Records tok/s_gen,
# drafted/accepted. llama-server must be stopped (engine needs the GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1; out=$2; shift 2; ks=${*:-4}
mkdir -p "$out"
echo "prompt n_prompt arm k tok_s_gen drafted accepted identity" > "$out/results.txt"
for tf in bench/mtp-prompts/p*.tokens; do
  p=$(basename "$tf" .tokens); n=$(wc -w < "$tf")
  BARO_PROMPT="$tf" "$eng" > "$out/$p.A.log" 2>&1
  ga=$(grep '^GENERATED' "$out/$p.A.log"); ta=$(grep -oE 'tok/s_gen: [0-9.]+' "$out/$p.A.log" | cut -d' ' -f2)
  echo "$p $n A 0 $ta 0 0 ref" >> "$out/results.txt"
  for k in $ks; do
    BARO_PROMPT="$tf" BARO_SPEC=1 BARO_SPEC_K=$k "$eng" > "$out/$p.B$k.log" 2>&1
    gb=$(grep '^GENERATED' "$out/$p.B$k.log"); tb=$(grep -oE 'tok/s_gen: [0-9.]+' "$out/$p.B$k.log" | cut -d' ' -f2)
    read -r _ d _ a _ _ <<< "$(grep '^mtp:' "$out/$p.B$k.log" | sed 's/mtp: //')"
    [ "$ga" = "$gb" ] && id=PASS || id=FAIL
    echo "$p $n B $k $tb $d $a $id" >> "$out/results.txt"
  done
done
column -t "$out/results.txt"
