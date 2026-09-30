#!/usr/bin/env bash
# poc/sycl/run.sh -- build + test the SYCL Phase A PoC.
# One command, clean state: rm -rf build && ./run.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# Only source setvars when the oneAPI env is absent: a subshell that already
# inherited a sourced env makes setvars.sh refuse to re-run (exit 3), and
# there is nothing to gain from re-sourcing anyway.
if [ -z "${ONEAPI_ROOT:-}" ]; then
  # shellcheck disable=SC1091
  source "$HOME/intel/oneapi/setvars.sh"
  [ -n "${ONEAPI_ROOT:-}" ] || { echo "run.sh: setvars.sh did not set ONEAPI_ROOT" >&2; exit 1; }
fi
cd "$ROOT"
cmake -S . -B build \
      -DCMAKE_CXX_COMPILER="$(command -v icpx)" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
bash "$ROOT/check_mirrors.sh"
