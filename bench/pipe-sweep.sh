#!/usr/bin/env bash
# usage: bench/pipe-sweep.sh <size> <cfg>...   cfg = WM,WN,TM,TN,BK,PADA,PADB,TRANSB[,C_F16[,PGR[,LB]]]
# Builds kernels/matmul_wmma_pipe.mojo variants in parallel into .work/pipe-sweep/, runs them
# sequentially, appends JSON + ISA receipt (vgpr/spill/lds) to .work/pipe-sweep.jsonl.
set -eu; cd "$(dirname "$0")/.."
size=$1; shift
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10
W=.work/pipe-sweep; mkdir -p $W
K=kernels/matmul_wmma_pipe.mojo; B=bench/bench_fp16_pipe.mojo
pids=(); tags=()
for cfg in "$@"; do
  IFS=, read wm wn tm tn bk pa pb tb cf pg lb <<<"$cfg"; cf=${cf:-1}; pg=${pg:-1}; lb=${lb:-1}
  tag="w${wm}x${wn}_t${tm}x${tn}_k${bk}_p${pa}${pb}_tb${tb}_c${cf}_g${pg}_lb${lb}_$size"; tags+=("$tag")
  d=$W/$tag; mkdir -p $d/kernels; cp kernels/*.mojo $d/kernels/
  sed -i -E "s/^comptime WARPS_M = [0-9]+/comptime WARPS_M = $wm/; s/^comptime WARPS_N = [0-9]+/comptime WARPS_N = $wn/; s/^comptime WTILE_M = [0-9]+/comptime WTILE_M = $tm/; s/^comptime WTILE_N = [0-9]+/comptime WTILE_N = $tn/; s/^comptime BLK_K = [0-9]+/comptime BLK_K = $bk/; s/^comptime PAD_A = [0-9]+/comptime PAD_A = $pa/; s/^comptime PAD_B = [0-9]+/comptime PAD_B = $pb/; s/^comptime TRANS_B = [0-9]+/comptime TRANS_B = $tb/; s/^comptime C_F16 = [0-9]+/comptime C_F16 = $cf/; s/^comptime PGR = [0-9]+/comptime PGR = ${pg:-1}/; s/^comptime LB = [0-9]+/comptime LB = ${lb:-1}/" $d/kernels/matmul_wmma_pipe.mojo
  sed -E "s/^comptime (M|N|K) = [0-9]+$/comptime \1 = $size/" $B > $d/bench.mojo
  ( ./.venv/bin/mojo build $d/bench.mojo -o $d/bin -I $d/kernels > $d/build.log 2>&1 || echo "BUILD FAIL $tag" ) &
  pids+=($!)
done
wait
for tag in "${tags[@]}"; do
  d=$W/$tag; [ -x $d/bin ] || { echo "{\"tag\": \"$tag\", \"status\": \"build-fail\"}" | tee -a $W.jsonl; grep -m3 error $d/build.log; continue; }
  isa=$(python3 tools/isa-receipt.py $d/bin --hist 0 | grep -oE "(vgpr_count|vgpr_spill_count|group_segment_fixed_size|private_segment_fixed_size)=[0-9]+" | tr '\n' ' ')
  out=$($d/bin)
  echo "${out%\}}, \"tag\": \"$tag\", \"isa\": \"$isa\"}" | tee -a $W.jsonl
done
