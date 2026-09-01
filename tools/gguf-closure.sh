#!/usr/bin/env bash
# Strict identity closure: build the engine FROM the sources embedded in a BARO
# gguf (baro.kernel.src.* KVs), run it on .work/engine-pack/, gate on ref tokens.
# usage: tools/gguf-closure.sh MODEL-BARO.gguf [ref-tokens-file] [outdir]
set -euo pipefail
cd "$(dirname "$0")/.."
model=$1; ref=${2:-.work/engine-pack/ref-tokens-64.txt}; out=${3:-.work/gguf-src}
rm -rf "$out"; mkdir -p "$out"
./.venv/bin/python3 tools/gguf-extract.py "$model" --meta > "$out/meta.json"
jq -r '.["baro.kernel.files"]' "$out/meta.json" | tr ',' '\n' > "$out/FILES"
while read -r f; do mkdir -p "$out/$(dirname "$f")"; jq -r --arg k "baro.kernel.src.$f" '.[$k]' "$out/meta.json" > "$out/$f"; done < "$out/FILES"
jq -r '"commit: " + .["baro.kernel.commit"] + "  arch: " + .["baro.kernel.arch"] + "  files: " + .["baro.kernel.files"]' "$out/meta.json"
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10
# vendor arm: if the gguf carries the hipBLASLt shim sources, build them too
if [ -f "$out/shim/CMakeLists.txt" ]; then
  cmake -S "$out/shim" -B "$out/shim-build" -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build "$out/shim-build" -j"$(nproc)" >/dev/null \
    && echo "shim built from gguf: $(ls "$out"/shim-build/*.so)" || { echo "shim build FAILED"; exit 1; }
fi
./.venv/bin/mojo build "$out/engine.mojo" -I "$out" -o .work/engine-closure 2>&1 | grep -E "error" -A3 && exit 1 || true
./.work/engine-closure > "$out/run.log"
grep -E "tok/s|host_enqueue" "$out/run.log"
tools/check-tokens.sh "$ref" "$out/run.log"
