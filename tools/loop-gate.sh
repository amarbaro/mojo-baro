#!/usr/bin/env bash
# Mechanical gate ladder for loop candidates. Serial, fail-closed, one receipt per candidate.
# usage: tools/loop-gate.sh ITER CHAMPION_TOKPS   (llama-server must be stopped: engine needs the GPU)
set -uo pipefail
cd "$(dirname "$0")/.."
iter=$1; champ=$2; dir=.work/loop/$iter
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10
: > "$dir/SURVIVORS.md"
for d in "$dir"/cand-*.diff; do
  c=$(basename "$d" .diff); r="$dir/$c.receipt.json"; work="$dir/$c"; rm -rf "$work"; mkdir -p "$work"
  fail() { echo "{\"cand\":\"$c\",\"stage\":\"$1\",\"result\":\"FAIL\",\"why\":\"$2\"}" > "$r"; echo "$c: FAIL $1: $2"; return 1; }
  [ -s "$d" ] || { fail parse "no diff fence"; continue; }
  # stage 0: scope -- only files listed in the gguf, no gate/timing/profiling edits
  touched=$(grep -E '^\+\+\+ ' "$d" | sed -E 's#^\+\+\+ (b/)?##; s/\t.*//' | sort -u)
  bad=""; for f in $touched; do grep -qx "$(basename "$f")" "$dir/FILES" || bad="$bad $f"; done
  [ -z "$bad" ] || { fail scope "files outside gguf list:$bad"; continue; }
  grep -E '^[+-].*(BARO_PROFILE|perf_counter_ns|tok/s|check-tokens|ref-tokens|getenv|print\()' "$d" >/dev/null && { fail scope "touches timing/print/profile code"; continue; }
  # stage 1: apply + compile on a copy of the gguf sources
  cp -r "$dir/src" "$work/src"
  patch -p1 -d "$work/src" --dry-run -s < "$d" >/dev/null 2>&1 || patch -p0 -d "$work/src" --dry-run -s < "$d" >/dev/null 2>&1 || { fail apply "patch does not apply"; continue; }
  patch -p1 -d "$work/src" -s < "$d" >/dev/null 2>&1 || patch -p0 -d "$work/src" -s < "$d" >/dev/null 2>&1
  ./.venv/bin/mojo build "$work/src/engine.mojo" -I "$work/src" -o "$work/engine" > "$work/build.log" 2>&1 || { fail compile "$(grep -m1 error: "$work/build.log" | cut -c1-160)"; continue; }
  # stage 2: token identity at 64
  ./"$work/engine" > "$work/run0.log" 2>&1 || { fail run "engine exited $?"; continue; }
  tools/check-tokens.sh .work/engine-pack/ref-tokens-64.txt "$work/run0.log" > "$work/gate.log" 2>&1 || { fail identity "$(head -1 "$work/gate.log")"; continue; }
  # stage 3: preregistered perf -- candidate's own PREDICT is the preregistration; read tok/s_gen back
  pred=$(cat "$dir/$c.predict"); t=()
  for k in 1 2 3; do ./"$work/engine" > "$work/run$k.log" 2>&1; t+=("$(grep -oE 'tok/s_gen: [0-9.]+' "$work/run$k.log" | awk '{print $2}')"); done
  med=$(printf '%s\n' "${t[@]}" | sort -n | sed -n 2p); lo=$(printf '%s\n' "${t[@]}" | sort -n | head -1); hi=$(printf '%s\n' "${t[@]}" | sort -n | tail -1)
  spread=$(awk -v l="$lo" -v h="$hi" 'BEGIN{printf "%.3f", (h-l)/l}')
  ok=$(awk -v m="$med" -v c="$champ" -v s="$spread" 'BEGIN{print (m >= c*1.02 && s < 0.05) ? 1 : 0}')
  [ "$ok" = 1 ] || { fail perf "median $med vs champion $champ (need +2%), spread $spread"; continue; }
  # stage 4: ISA sanity -- no scratch, no spills in any kernel of the built binary
  python3 - "$work/engine" "$work" <<'PY'
import struct,sys,subprocess
d=open(sys.argv[1],'rb').read(); i=0; n=0; bad=[]
while True:
    i=d.find(b'\x7fELF',i)
    if i<0: break
    if struct.unpack_from('<H',d,i+18)[0]==224:
        shoff,=struct.unpack_from('<Q',d,i+40); se,sn=struct.unpack_from('<HH',d,i+58)
        p=f"{sys.argv[2]}/co{n}.co"; open(p,'wb').write(d[i:i+shoff+se*sn]); n+=1
        notes=subprocess.run(['/opt/rocm/llvm/bin/llvm-readelf','--notes',p],capture_output=True,text=True).stdout
        for line in notes.splitlines():
            if ('.private_segment_fixed_size:' in line or 'spill_count:' in line) and line.split(':')[1].strip()!='0':
                bad.append(line.strip())
    i+=4
print(f"code_objects={n} bad={bad}")
sys.exit(1 if bad or n==0 else 0)
PY
  [ $? = 0 ] || { fail isa "scratch or spills"; continue; }
  echo "{\"cand\":\"$c\",\"stage\":\"all\",\"result\":\"PASS\",\"predict_pct\":\"$pred\",\"tokps\":[${t[0]},${t[1]},${t[2]}],\"median\":$med,\"champion\":$champ,\"spread\":$spread}" > "$r"
  echo "$c: PASS median $med vs $champ (predict $pred%)"; { echo "## $c  median $med vs champion $champ (predict $pred%)"; echo '```diff'; cat "$d"; echo '```'; } >> "$dir/SURVIVORS.md"
done
echo "survivors: $(grep -c '^## ' "$dir/SURVIVORS.md")"
