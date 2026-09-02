#!/usr/bin/env bash
# usage: bench/race-fp16.sh <rounds> <label=binary>...
# Interleaves the given bench binaries round-robin for <rounds> rounds (same thermal
# window for every arm), prints every gflops sample and the per-arm median.
set -eu; cd "$(dirname "$0")/.."
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10
rounds=$1; shift
declare -A samples
for ((r=1; r<=rounds; r++)); do
  for arm in "$@"; do
    label=${arm%%=*}; bin=${arm#*=}
    out=$($bin); g=$(grep -oE '"gflops": [0-9.]+' <<<"$out" | grep -oE '[0-9.]+$'); ok=$(grep -oE '"correct": [a-z]+' <<<"$out")
    printf "round %d  %-10s %8.0f  %s\n" "$r" "$label" "$g" "$ok"
    samples[$label]+="$g "
  done
done
echo "--- medians over $rounds rounds"
for arm in "$@"; do
  label=${arm%%=*}
  printf "%-10s median %8.0f  (min %.0f max %.0f)\n" "$label" $(tr ' ' '\n' <<<"${samples[$label]}" | grep . | sort -n | awk '{a[NR]=$1} END{m=(NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2; print m, a[1], a[NR]}')
done
