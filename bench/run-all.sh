#!/usr/bin/env bash
# Everything, logged. usage: bench/run-all.sh [LOG]   (stops llama-server, restores it at the end)
cd "$(dirname "$0")/.."
LOG=${1:-.work/runall-$(date +%Y%m%d-%H%M).log}; exec > >(tee -a "$LOG") 2>&1
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10
S=$PWD/.work/shim-build; L="-I kernels -Xlinker -L$S -Xlinker -lamarbaro_shim -Xlinker -rpath -Xlinker $S"
sec() { echo; echo "##### $1  ($(date +%H:%M:%S))"; }
P=$(pgrep -f '^/home/mario/llama.cpp/build/bin/llama-server' || true); [ -n "$P" ] && { kill $P; sleep 3; }
trap 'cmd=$(cat ~/Brain/mojo-baro/llama-server-cmdline.txt); (setsid nohup bash -c "$cmd" > .work/llama-server.log 2>&1 < /dev/null &); echo "server restart issued"' EXIT
echo "commit: $(git rev-parse --short HEAD)  dirty: $(git status --short | tr '\n' ' ')  gpu: $(rocm-smi --showproductname 2>/dev/null | grep -m1 'Card series' | sed 's/.*: //')"
sec "run-tests.sh (shim + fp32 GEMM parity)"; ./run-tests.sh 2>&1 | grep -viE "crashpad|warning: doc" | tail -8
sec "bench/run.py fp32 512^3"; ./bench/run.py 2>&1 | grep -viE crashpad | tail -8
sec "fp16 WMMA ours 512/2048/4096"; bench/fp16-sizes.sh bench/bench_fp16.mojo wmma 512 2048 4096 2>&1 | grep '^{'
sec "hipBLASLt fp16 2048/4096"; bench/fp16-sizes.sh bench/bench_fp16_lt.mojo lt 2048 4096 2>&1 | grep '^{'
sec "kernel parity tests"
for t in test_elementwise test_ssm_block test_attn_block test_q8_gemm test_gguf_gemm; do
  echo "--- $t"; ./.venv/bin/mojo build kernels/$t.mojo -o .work/$t $L 2>&1 | grep -E "error" -A3 | head -6; ./.work/$t 2>&1 | grep -viE crashpad | tail -4
done
sec "engine: repo build vs closure build, interleaved x3"
./.venv/bin/mojo build serve/engine.mojo -I kernels -o .work/engine 2>&1 | grep -E "error" -A3
for i in 1 2 3; do ./.work/engine > .work/ra-repo$i.log; echo "repo    $(grep -oE 'tok/s_gen: [0-9.]+' .work/ra-repo$i.log)"; ./.work/engine-closure > .work/ra-clos$i.log; echo "closure $(grep -oE 'tok/s_gen: [0-9.]+' .work/ra-clos$i.log)"; done
tools/check-tokens.sh .work/engine-pack/ref-tokens-64.txt .work/ra-repo1.log; tools/check-tokens.sh .work/engine-pack/ref-tokens-64.txt .work/ra-clos1.log
cmp -s .work/gguf-src/engine.mojo serve/engine.mojo && echo "closure engine.mojo == repo" || echo "closure engine.mojo DIFFERS from repo"
sec "engine profile"; BARO_PROFILE=1 ./.work/engine | grep -E "profile:|tok/s_gen"
sec "done"
