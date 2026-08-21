#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
FINAL_ROUND=5

echo "=== rebuild pooled tables $(date -Is) ==="

if [[ ! -f results/seven_axis_baseline_replay_split/seed0/eval_train/eval_results.json ]]; then
  echo "--- pooled baseline train eval ---"
  "${PY}" -m infl_ens.evaluation \
    --config configs/evaluation/seven_axis_split_eval_train.yaml \
    -- run_dir=results/seven_axis_baseline_replay_split/seed0 \
       output_dir=results/seven_axis_baseline_replay_split/seed0/eval_train \
       rounds="[${FINAL_ROUND}]" \
       agents='["pooled-baseline"]'
fi

echo "--- comparison tables ---"
"${PY}" scripts/build_seven_axis_split_tables.py

echo "=== done $(date -Is) ==="
