#!/usr/bin/env bash
# usage: bench/wmma-peak.sh [NUM_CU]
# WMMA-only roofline. The grid is sized to fill the card, so the compute-unit
# count is arm-defining: bench_wmma_peak.mojo hardcodes 96 (this box's XTX) and
# a wrong value silently rescales the roofline on any other card. Reads the CU
# count from the running GPU (bench/gpu-info.py), patches the comptime knob,
# builds, runs, restores the source. Pass NUM_CU to override.
set -eu
cd "$(dirname "$0")/.."
src=bench/bench_wmma_peak.mojo

if [ $# -ge 1 ]; then
  CU=$1
  origin="override"
else
  info=$(./bench/gpu-info.py) || exit 1
  CU=$(printf '%s' "$info" | python3 -c 'import json,sys; print(json.load(sys.stdin)["compute_units"] or "")')
  [ -n "$CU" ] || { echo "wmma-peak.sh: rocminfo did not report a compute-unit count; pass it explicitly: bench/wmma-peak.sh <NUM_CU>" >&2; exit 1; }
  origin=$(printf '%s' "$info" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["gpu"]+" ("+d["gfx"]+")")')
fi

orig=$(mktemp); cp "$src" "$orig"; trap 'cp "$orig" "$src"; rm -f "$orig"' EXIT
sed -i -E "s/^comptime NUM_CU = [0-9]+$/comptime NUM_CU = $CU/" "$src"
grep -qE "^comptime NUM_CU = $CU$" "$src" || { echo "wmma-peak.sh: failed to patch NUM_CU in $src" >&2; exit 1; }

echo "num_cu: $CU  from: $origin"
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10
./.venv/bin/mojo build "$src" -o .work/bench_wmma_peak -I kernels
.work/bench_wmma_peak
