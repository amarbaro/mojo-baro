#!/usr/bin/env bash
# usage: bench/clock-probe.sh <cmd...>
# Runs <cmd> while sampling GPU 0 every ~0.2 s, then prints the busy-window (GPU use
# >= 90%) receipt: sclk min/median/max, power range, peak junction. Samples land in
# .work/clock-probe-<ts>.log. The sustained clock is an arm-defining parameter
# (PROTOCOL-RULES P1): peak FLOP/s scales with it, and WMMA load holds a lower
# clock than the fp32 kernels do at the same package power.
set -u; cd "$(dirname "$0")/.."
mkdir -p .work; SMI=.work/clock-probe-$(date +%Y%m%d-%H%M%S).log; : > "$SMI"
( while :; do
    l=$(rocm-smi -d 0 --showuse --showclocks --showpower --showtemp 2>/dev/null \
        | grep -E "GPU use|sclk clock level|Graphics Package Power|Sensor junction" \
        | sed -E 's/^GPU\[0\][[:space:]]*: //' | tr '\n' '|')
    echo "$(date +%s.%N)|$l" >> "$SMI"; sleep 0.2
  done ) & S=$!
"$@"; rc=$?
kill $S 2>/dev/null; wait $S 2>/dev/null
awk -F'|' -v f="$SMI" '
  { u=s=p=t=""
    for (i=2; i<=NF; i++) {
      if ($i ~ /GPU use/)        { sub(/.*: /, "", $i); u=$i+0 }
      if ($i ~ /sclk/)           { match($i, /[0-9]+Mhz/); s=substr($i, RSTART, RLENGTH-3)+0 }
      if ($i ~ /Package Power/)  { sub(/.*: /, "", $i); p=$i+0 }
      if ($i ~ /junction/)       { sub(/.*: /, "", $i); t=$i+0 } }
    n++
    if (u >= 90 && s > 0) { b++; sc[b]=s; if (p>pmax) pmax=p; if (pmin=="" || p<pmin) pmin=p; if (t>tmax) tmax=t } }
  END {
    if (!b) { printf "clock-probe: %d samples, none busy (GPU use >= 90%%) -- %s\n", n, f; exit }
    asort(sc)
    printf "clock-probe: busy %d/%d samples  sclk min/med/max %d/%d/%d MHz  power %.0f-%.0f W  junction max %.0f C  -- %s\n", b, n, sc[1], sc[int((b+1)/2)], sc[b], pmin, pmax, tmax, f }' "$SMI"
exit $rc
