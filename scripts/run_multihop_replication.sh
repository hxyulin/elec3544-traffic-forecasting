#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
trap 'status=$?; echo EXIT_STATUS=$status; date -u +%FT%TZ' EXIT
date -u +%FT%TZ
for seed in 42 43 44; do
  for variant in local one two geometry; do
    echo "RUN_START seed=$seed variant=$variant"
    timeout --signal=TERM --kill-after=30 900 uv run --extra neural scripts/try_multihop_replication.py --seed "$seed" --variant "$variant"
    echo "RUN_COMPLETED seed=$seed variant=$variant"
  done
done
echo REPLICATION_COMPLETED
