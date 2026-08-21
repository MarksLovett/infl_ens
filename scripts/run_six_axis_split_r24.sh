#!/usr/bin/env bash
# 24-round six-axis split experiment (6 merge pairs, 12 agents).

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/six_axis_split_r24_pipeline.log}"

exec > >(tee -a "$LOG") 2>&1
echo "=== six_axis split r24 pipeline started $(date -Is) ==="

echo "--- closed-loop training (24 rounds) ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/six_axis_pair_merge_split_r24.yaml

FINAL_ROUND="$("${PY}" - <<'PY'
import json
from pathlib import Path
h = json.loads(Path("results/six_axis_pair_merge_split_r24/seed0/history.json").read_text())
print(int(h[-1]["round"]))
PY
)"
echo "final round index: ${FINAL_ROUND}"

echo "--- pooled baseline replay ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/six_axis_baseline_replay_split_r24.yaml

echo "--- pooled baseline eval (train) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/six_axis_split_eval_train.yaml \
  -- run_dir=results/six_axis_baseline_replay_split_r24/seed0 \
     output_dir=results/six_axis_baseline_replay_split_r24/seed0/eval_train \
     rounds="[${FINAL_ROUND}]" \
     agents='["pooled-baseline"]'

echo "--- pooled baseline eval (test) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/six_axis_split_eval_test.yaml \
  -- run_dir=results/six_axis_baseline_replay_split_r24/seed0 \
     output_dir=results/six_axis_baseline_replay_split_r24/seed0/eval_test \
     rounds="[${FINAL_ROUND}]" \
     agents='["pooled-baseline"]'

echo "--- merge eval (train) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/six_axis_split_eval_train.yaml \
  -- run_dir=results/six_axis_pair_merge_split_r24/seed0 \
     output_dir=results/six_axis_pair_merge_split_r24/seed0/eval_train \
     rounds="[${FINAL_ROUND}]"

echo "--- merge eval (test) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/six_axis_split_eval_test.yaml \
  -- run_dir=results/six_axis_pair_merge_split_r24/seed0 \
     output_dir=results/six_axis_pair_merge_split_r24/seed0/eval_test \
     rounds="[${FINAL_ROUND}]"

echo "--- comparison tables ---"
"${PY}" scripts/build_six_axis_split_tables.py \
  --run-dir results/six_axis_pair_merge_split_r24/seed0 \
  --baseline-run-dir results/six_axis_baseline_replay_split_r24/seed0

echo "=== six_axis split r24 pipeline finished $(date -Is) ==="
