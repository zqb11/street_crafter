#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
CONFIG="${1:-configs/nio_efa3fa0a_video.yaml}"
if [[ $# -gt 0 ]]; then shift; fi

DATASET_TYPE="$("$PYTHON_BIN" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get("task", ""))' "$CONFIG")"
if [[ "$DATASET_TYPE" == "nio" ]]; then
    exec "$PYTHON_BIN" -m video_diffusion.sample_nio --config "$CONFIG" "$@"
else
    exec "$PYTHON_BIN" render.py --config "$CONFIG" mode diffusion "$@"
fi
