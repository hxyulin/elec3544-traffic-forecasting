#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
trap 'code=$?; printf "EXIT %s %s\n" "$code" "$(date -u +%FT%TZ)"' EXIT
printf 'START %s\n' "$(date -u +%FT%TZ)"
for seed in 42 43 44; do
  for schedule in A B; do
    timeout 1800 uv run python scripts/try_teacher_schedule.py --mode fit --seed "$seed" --schedule "$schedule"
    timeout 1800 uv run python scripts/try_teacher_schedule.py --mode fit --seed "$seed" --schedule "$schedule" --disabled
  done
done
printf 'ALL_TWELVE_COMPLETED\n'
