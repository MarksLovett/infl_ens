#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/seven_axis_posttrain.log}"

{
  echo "=== merge eval started $(date -Is) ==="
  "${PY}" -m infl_ens.evaluation \
    --config configs/evaluation/seven_axis_run_eval.yaml
  echo "=== baseline eval started $(date -Is) ==="
  "${PY}" -m infl_ens.evaluation \
    --config configs/evaluation/seven_axis_run_eval.yaml \
    -- run_dir=results/seven_axis_baseline_replay_r40/seed0 \
       output_dir=results/seven_axis_baseline_replay_r40/seed0/eval_final_round \
       rounds=[39]
  echo "=== posttrain finished $(date -Is) ==="
} >> "$LOG" 2>&1
