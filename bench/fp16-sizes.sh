#!/usr/bin/env bash
# usage: bench/fp16-sizes.sh <bench.mojo> <tag> [sizes...]
# Patches comptime M/N/K in <bench.mojo>, builds+runs per size, appends JSON to .work/fp16-<tag>.jsonl, restores the source.
set -eu; cd "$(dirname "$0")/.."
src=$1; tag=$2; shift 2; sizes=${@:-512 2048 4096}
S=$PWD/.work/shim-build
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10
orig=$(mktemp); cp "$src" "$orig"; trap 'cp "$orig" "$src"; rm -f "$orig"' EXIT
for s in $sizes; do
  sed -i -E "s/^comptime (M|N|K) = [0-9]+$/comptime \1 = $s/" "$src"
  ./.venv/bin/mojo build "$src" -o .work/fp16_${tag}_$s -I kernels \
    -Xlinker -L"$S" -Xlinker -lamarbaro_shim -Xlinker -rpath -Xlinker "$S" 2>&1 | grep -E "error" -A3 || true
  ./.work/fp16_${tag}_$s | tee -a .work/fp16-$tag.jsonl
done
