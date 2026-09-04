#!/usr/bin/env bash
# usage: bench/race-fp16.sh <rounds> <label=binary>...
# Interleaves the given bench binaries round-robin for <rounds> rounds (same thermal
# window for every arm), prints every gflops sample and the per-arm median.
# PROBE=1: wrap every sample in bench/clock-probe.sh (byte-for-byte, same script,
# same output), read back its sclk_med and print FLOP/clk/CU alongside gflops.
set -eu; cd "$(dirname "$0")/.."
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10
rounds=$1; shift
probe=${PROBE:-0}
declare -A samples
for ((r=1; r<=rounds; r++)); do
  for arm in "$@"; do
    label=${arm%%=*}; bin=${arm#*=}
    if [ "$probe" = "1" ]; then
      out=$(bench/clock-probe.sh "$bin")
    else
      out=$($bin)
    fi
    g=$(grep -oE '"gflops": [0-9.]+' <<<"$out" | grep -oE '[0-9.]+$'); ok=$(grep -oE '"correct": [a-z]+' <<<"$out")
    if [ "$probe" = "1" ]; then
      sclk_med=$(grep -oE 'sclk min/med/max [0-9]+/[0-9]+/[0-9]+' <<<"$out" | awk '{split($NF,a,"/"); print a[2]}')
      fpc=$(awk -v g="$g" -v s="$sclk_med" 'BEGIN{ if (s>0) printf "%.1f", (g*1.0e9)/(96*s*1.0e6); else print "NA" }')
      printf "round %d  %-10s %8.0f  sclk_med %sMHz  flop/clk/cu %s  %s\n" "$r" "$label" "$g" "$sclk_med" "$fpc" "$ok"
      grep -E "^clock-probe:" <<<"$out"
    else
      printf "round %d  %-10s %8.0f  %s\n" "$r" "$label" "$g" "$ok"
    fi
    samples[$label]+="$g "
  done
done
echo "--- medians over $rounds rounds"
for arm in "$@"; do
  label=${arm%%=*}
  printf "%-10s median %8.0f  (min %.0f max %.0f)\n" "$label" $(tr ' ' '\n' <<<"${samples[$label]}" | grep . | sort -n | awk '{a[NR]=$1} END{m=(NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2; print m, a[1], a[NR]}')
done
