#!/usr/bin/env bash
# Post-training for seven-axis pair-merge: pooled baseline replay + final-round eval.
#
#   bash scripts/run_seven_axis_posttrain.sh
#
# Run on doob after ``seven_axis_pair_merge_r40`` closed loop completes.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/seven_axis_posttrain.log}"

exec > >(tee -a "$LOG") 2>&1

echo "=== seven_axis posttrain started $(date -Is) ==="

echo "--- baseline replay ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/seven_axis_baseline_replay_r40.yaml

echo "--- merge adapter eval (round 39) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/seven_axis_run_eval.yaml

echo "--- pooled baseline eval (round 39) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/seven_axis_run_eval.yaml \
  -- run_dir=results/seven_axis_baseline_replay_r40/seed0 \
     output_dir=results/seven_axis_baseline_replay_r40/seed0/eval_final_round \
     rounds=[39]

echo "=== seven_axis posttrain finished $(date -Is) ==="
