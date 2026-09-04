#!/usr/bin/env bash
# usage: bench/build-variant.sh <tag> <size> [NAME=VALUE ...]
# SRC=<file> overrides the bench source (default bench/bench_fp16_pipe.mojo).
# Copies kernels/matmul_wmma_pipe.mojo and bench/bench_fp16_pipe.mojo aside,
# patches comptime M/N/K in the bench and every comptime NAME=VALUE given
# (kernel first, then bench if NAME is not in the kernel), builds to
# .work/fp16_<tag>_<size>, restores both sources, prints the binary path.
set -eu
cd "$(dirname "$0")/.."
tag=$1; size=$2; shift 2
kernel=kernels/matmul_wmma_pipe.mojo
bench=${SRC:-bench/bench_fp16_pipe.mojo}
S=$PWD/.work/shim-build
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10

korig=$(mktemp); cp "$kernel" "$korig"
borig=$(mktemp); cp "$bench" "$borig"
trap 'cp "$korig" "$kernel"; cp "$borig" "$bench"; rm -f "$korig" "$borig"' EXIT

sed -i -E "s/^comptime (M|N|K) = [0-9]+$/comptime \1 = $size/" "$bench"

for kv in "$@"; do
  name=${kv%%=*}
  value=${kv#*=}
  if grep -qE "^comptime $name = " "$kernel"; then
    sed -i -E "s/^comptime $name = .*\$/comptime $name = $value/" "$kernel"
  elif grep -qE "^comptime $name = " "$bench"; then
    sed -i -E "s/^comptime $name = .*\$/comptime $name = $value/" "$bench"
  else
    echo "build-variant.sh: NAME '$name' matches no comptime knob in $kernel or $bench" >&2
    exit 1
  fi
done

out=.work/fp16_${tag}_${size}
./.venv/bin/mojo build "$bench" -o "$out" -I kernels \
  -Xlinker -L"$S" -Xlinker -lamarbaro_shim -Xlinker -rpath -Xlinker "$S"
echo "$out"
