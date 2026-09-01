#!/usr/bin/env bash
set -euo pipefail

ref_file="${1:-.work/engine-pack/ref-tokens-64.txt}"
input="${2:-/dev/stdin}"

if [[ ! -f "$ref_file" ]]; then
    echo "FAIL: ref file not found: $ref_file"
    exit 1
fi

gen_line=$(grep -m1 '^GENERATED:' "$input" || true)
if [[ -z "$gen_line" ]]; then
    echo "FAIL: no GENERATED line found in input"
    exit 1
fi

gen_ids=(${gen_line#GENERATED:})
mapfile -t ref_ids < "$ref_file"

n_ref=${#ref_ids[@]}
n_gen=${#gen_ids[@]}
n=$(( n_ref < n_gen ? n_ref : n_gen ))

for ((i = 0; i < n; i++)); do
    if [[ "${gen_ids[$i]}" != "${ref_ids[$i]}" ]]; then
        echo "FAIL: mismatch at position $((i + 1)): expected ${ref_ids[$i]}, got ${gen_ids[$i]}"
        exit 1
    fi
done

if [[ "$n_gen" -ne "$n_ref" ]]; then
    echo "FAIL: length mismatch: expected $n_ref ids, got $n_gen"
    exit 1
fi

echo "PASS: $n_ref tokens match"
exit 0
