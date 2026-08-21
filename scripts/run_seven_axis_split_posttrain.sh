#!/usr/bin/env bash
# Finish post-training for the 6-round split run (eval + tables).
# Use after closed loop + baseline replay complete.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/seven_axis_split_posttrain.log}"
FINAL_ROUND=5

exec > >(tee -a "$LOG") 2>&1
echo "=== split posttrain started $(date -Is) ==="

if [[ ! -f results/seven_axis_baseline_replay_split/seed0/history.json ]]; then
  echo "--- pooled baseline replay ---"
  "${PY}" -m infl_ens.training \
    --config configs/benchmark/router/seven_axis_baseline_replay_split.yaml
fi

echo "--- merge eval (train partition, round ${FINAL_ROUND}) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/seven_axis_split_eval_train.yaml \
  -- rounds="[${FINAL_ROUND}]"

echo "--- merge eval (test partition, round ${FINAL_ROUND}) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/seven_axis_split_eval_test.yaml \
  -- rounds="[${FINAL_ROUND}]"

echo "--- pooled baseline eval (train partition) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/seven_axis_split_eval_train.yaml \
  -- run_dir=results/seven_axis_baseline_replay_split/seed0 \
     output_dir=results/seven_axis_baseline_replay_split/seed0/eval_train \
     rounds="[${FINAL_ROUND}]" \
     agents='["pooled-baseline"]'

echo "--- pooled baseline eval (test partition) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/seven_axis_split_eval_test.yaml \
  -- run_dir=results/seven_axis_baseline_replay_split/seed0 \
     output_dir=results/seven_axis_baseline_replay_split/seed0/eval_test \
     rounds="[${FINAL_ROUND}]" \
     agents='["pooled-baseline"]'

echo "--- comparison tables ---"
"${PY}" scripts/build_seven_axis_split_tables.py

echo "=== split posttrain finished $(date -Is) ==="
