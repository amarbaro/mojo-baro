#!/usr/bin/env bash
# usage: bench/report.sh [--check] [--quick] [--sizes "512 2048"] [--out FILE]
#
# Produce one portable receipt for the fp16 WMMA GEMM kernel on this machine:
# our kernel vs hipBLASLt at each square size, plus the card, the toolchain and
# the clock state the numbers were taken under. Writes results/report-*.json and
# prints a markdown block to paste into a GitHub issue.
#
# Needs no model weights. --check runs the preflight only.
set -eu
cd "$(dirname "$0")/.."

SIZES="256 512 768 1024 1536 2048 2560 3072 3584 4096"
CHECK_ONLY=0
OUT=""
while [ $# -gt 0 ]; do
  case $1 in
    --check) CHECK_ONLY=1 ;;
    --quick) SIZES="512 2048 4096" ;;
    --sizes) SIZES=$2; shift ;;
    --out) OUT=$2; shift ;;
    -h|--help) sed -n '2,9p' "$0" | cut -c3-; exit 0 ;;
    *) echo "report.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
  shift
done

fail() { echo "report.sh: $*" >&2; exit 1; }

# --- preflight: fail before a ten-minute build, not after -------------------
command -v python3 >/dev/null || fail "python3 not found"
command -v cmake   >/dev/null || fail "cmake not found; the C++ shim needs it"
[ -x ./.venv/bin/mojo ] || fail "./.venv/bin/mojo missing. Run 'uv sync' first (see README)."

GPUJSON=$(./bench/gpu-info.py) || exit 1
GFX=$(printf '%s' "$GPUJSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["gfx"])')
GPUNAME=$(printf '%s' "$GPUJSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["gpu"])')
ROCM=$(printf '%s' "$GPUJSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["rocm"])')
HIPBLASLT=$(printf '%s' "$GPUJSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["hipblaslt"])')

[ -n "$HIPBLASLT" ] || fail "hipBLASLt not found. It is the comparison baseline; install rocm-libs."
case "$GFX" in
  gfx11*) ;;
  *) echo "report.sh: WARNING $GFX is not RDNA3. The kernel targets gfx11xx;"
     echo "           it may not build, and any number it produces is untuned."
     echo "           Continuing -- a failure on $GFX is itself a useful report." ;;
esac

COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
DIRTY=$(git status --porcelain --untracked-files=no 2>/dev/null | tr '\n' ' ')
MOJOVER=$(./.venv/bin/mojo --version 2>/dev/null | head -1)

echo "gpu:       $GPUNAME ($GFX)"
echo "rocm:      $ROCM   hipblaslt: $HIPBLASLT"
echo "mojo:      $MOJOVER"
echo "commit:    $COMMIT${DIRTY:+  (dirty: $DIRTY)}"
echo "sizes:     $SIZES"
if [ "$CHECK_ONLY" = 1 ]; then echo "preflight OK"; exit 0; fi
[ -z "$DIRTY" ] || echo "NOTE working tree is dirty; the receipt records this and the numbers are not attributable to $COMMIT alone."

# --- build the shim, then run both arms at every size -----------------------
mkdir -p .work results
S="$PWD/.work/shim-build"
cmake -S shim -B "$S" -DCMAKE_BUILD_TYPE=Release >/dev/null || fail "shim cmake configure failed"
cmake --build "$S" -j"$(nproc)" >/dev/null || fail "shim build failed"

rm -f .work/fp16-rpt_wmma.jsonl .work/fp16-rpt_lt.jsonl
echo "running ours (wmma pipe) -- 10 s clock warm-up per size, this is slow on purpose"
bench/fp16-sizes.sh bench/bench_fp16_pipe.mojo rpt_wmma $SIZES >/dev/null
echo "running hipBLASLt baseline"
bench/fp16-sizes.sh bench/bench_fp16_lt.mojo rpt_lt $SIZES >/dev/null

[ -s .work/fp16-rpt_wmma.jsonl ] || fail "our kernel produced no output; the build likely failed above"
[ -s .work/fp16-rpt_lt.jsonl ]   || fail "hipBLASLt arm produced no output"

STATE_AFTER=$(./bench/gpu-info.py | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["state"]))')

OUT=${OUT:-results/report-$GFX-$(printf '%s' "$GPUNAME" | tr ' ' '-' | tr -cd 'A-Za-z0-9-')-$COMMIT.json}
export GPUJSON COMMIT DIRTY MOJOVER OUT STATE_AFTER SIZES
python3 bench/report-merge.py
