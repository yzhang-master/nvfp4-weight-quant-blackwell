#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUTLASS_DIR="${CUTLASS_DIR:-}" "$ROOT/scripts/build_sm100.sh"
"$ROOT/build/sm100/nvfp4_sm100" --m=128 --n=1024 --k=1024 --warmup=10 --iterations=50 "$@"
