#!/usr/bin/env bash
# Build the C++ shim, then build and run the Mojo verification tests.
set -euo pipefail
cd "$(dirname "$0")"
S="$PWD/.work/shim-build"
cmake -S shim -B "$S" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$S" -j"$(nproc)" >/dev/null
./.venv/bin/mojo build kernels/test_gemm.mojo -o .work/test_gemm -I kernels \
  -Xlinker -L"$S" -Xlinker -lbaro_shim -Xlinker -rpath -Xlinker "$S"
./.work/test_gemm
