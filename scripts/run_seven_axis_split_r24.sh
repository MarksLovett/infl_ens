#!/usr/bin/env bash
# 24-round split experiment (~840 batch, exact train coverage with remainder).
# Set RUN_POSTTRAIN_FIRST=1 to finish 6-round tables before training.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/seven_axis_split_r24_pipeline.log}"

exec > >(tee -a "$LOG") 2>&1
echo "=== seven_axis split r24 pipeline started $(date -Is) ==="

if [[ "${RUN_POSTTRAIN_FIRST:-0}" == "1" ]]; then
  bash scripts/run_seven_axis_split_posttrain.sh
fi

echo "--- closed-loop training (24 rounds, ~840 batch) ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/seven_axis_pair_merge_split_r24.yaml

FINAL_ROUND="$("${PY}" - <<'PY'
import json
from pathlib import Path
h = json.loads(Path("results/seven_axis_pair_merge_split_r24/seed0/history.json").read_text())
print(int(h[-1]["round"]))
PY
)"
echo "final round index: ${FINAL_ROUND}"

echo "--- pooled baseline replay ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/seven_axis_baseline_replay_split_r24.yaml

echo "--- pooled baseline eval (train) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/seven_axis_split_eval_train.yaml \
  -- run_dir=results/seven_axis_baseline_replay_split_r24/seed0 \
     output_dir=results/seven_axis_baseline_replay_split_r24/seed0/eval_train \
     rounds="[${FINAL_ROUND}]" \
     agents='["pooled-baseline"]'

echo "--- pooled baseline eval (test) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/seven_axis_split_eval_test.yaml \
  -- run_dir=results/seven_axis_baseline_replay_split_r24/seed0 \
     output_dir=results/seven_axis_baseline_replay_split_r24/seed0/eval_test \
     rounds="[${FINAL_ROUND}]" \
     agents='["pooled-baseline"]'

echo "--- merge eval (train) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/seven_axis_split_eval_train.yaml \
  -- run_dir=results/seven_axis_pair_merge_split_r24/seed0 \
     output_dir=results/seven_axis_pair_merge_split_r24/seed0/eval_train \
     rounds="[${FINAL_ROUND}]"

echo "--- merge eval (test) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/seven_axis_split_eval_test.yaml \
  -- run_dir=results/seven_axis_pair_merge_split_r24/seed0 \
     output_dir=results/seven_axis_pair_merge_split_r24/seed0/eval_test \
     rounds="[${FINAL_ROUND}]"

echo "--- comparison tables ---"
"${PY}" scripts/build_seven_axis_split_tables.py \
  --run-dir results/seven_axis_pair_merge_split_r24/seed0 \
  --baseline-run-dir results/seven_axis_baseline_replay_split_r24/seed0

echo "=== seven_axis split r24 pipeline finished $(date -Is) ==="
