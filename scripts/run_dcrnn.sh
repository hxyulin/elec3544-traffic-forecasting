#!/usr/bin/env bash
# Run from a prepared Linux study checkout, after tests and --mode pilot/prepare.
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for seed in 42 43 44; do
  timeout 1800 uv run python scripts/try_dcrnn.py --mode fit --seed "$seed"
  timeout 1800 uv run python scripts/try_dcrnn.py --mode fit --seed "$seed" --disabled
done
printf 'ALL_SIX_COMPLETED\n'
