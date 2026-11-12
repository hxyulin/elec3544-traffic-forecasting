#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
trap 'status=$?; echo EXIT_STATUS=$status; date -u +%FT%TZ' EXIT
for seed in 42 43 44; do
  for size in current wide; do
    for variant in one two; do
      echo "RUN_START seed=$seed size=$size variant=$variant"
      timeout --signal=TERM --kill-after=30 900 uv run --extra neural scripts/try_capacity.py --seed "$seed" --size "$size" --variant "$variant"
    done
  done
done
echo CAPACITY_COMPLETED
