#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUTLASS_DIR="${1:-${CUTLASS_DIR:-}}"
if [[ -z "$CUTLASS_DIR" || ! -f "$CUTLASS_DIR/include/cutlass/cutlass.h" ]]; then
  echo "usage: CUTLASS_DIR=/path/to/cutlass $0" >&2
  exit 2
fi
cmake -S "$ROOT/cuda" -B "$ROOT/build/sm100" -DCUTLASS_DIR="$CUTLASS_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build/sm100" --parallel
