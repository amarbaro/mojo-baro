#!/usr/bin/env bash
# Re-embed the COMMITTED repo sources into a new gguf with lineage; never overwrites.
# usage: tools/loop-embed-winner.sh SRC-BARO.gguf ITER
set -euo pipefail
cd "$(dirname "$0")/.."
src=$1; iter=$2
dst="${src%.gguf}-loop-$iter.gguf"
[ ! -e "$dst" ] || { echo "refusing to overwrite $dst"; exit 1; }
parent=$(./.venv/bin/python3 tools/gguf-extract.py "$src" --meta | jq -r '.["baro.kernel.commit"]')
files=$(./.venv/bin/python3 tools/gguf-extract.py "$src" --meta | jq -r '.["baro.kernel.files"]' | tr ',' '\n' | sed 's#^engine.mojo$#serve/engine.mojo#; s#^\([a-z_]*\.mojo\)$#kernels/\1#')
BARO_KERNEL_PARENT="$parent" ./.venv/bin/python3 tools/gguf-embed.py "$src" "$dst" $files
echo "$dst" > .work/loop/CHAMPION
echo "wrote $dst (parent $parent); CHAMPION updated"
