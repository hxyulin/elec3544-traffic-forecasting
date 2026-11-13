#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for study in spatial temporal; do
  uv run python scripts/try_attention_${study}.py --prepare
 done
for study in spatial temporal; do
  if [[ $study == spatial ]]; then variants='uniform content bias combined'; else variants='sinusoidal rope'; fi
  for seed in 42 43 44; do
    for variant in $variants; do
      timeout 900 uv run python scripts/try_attention_${study}.py --seed "$seed" --variant "$variant"
    done
  done
  uv run python scripts/summarize_attention_${study}.py
 done
