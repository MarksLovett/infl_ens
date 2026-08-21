#!/usr/bin/env bash
# 12-round split experiment (batch ~1681, exact train coverage).
# Chains after run_seven_axis_split_posttrain.sh if RUN_POSTTRAIN_FIRST=1.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/seven_axis_split_r12_pipeline.log}"

exec > >(tee -a "$LOG") 2>&1
echo "=== seven_axis split r12 pipeline started $(date -Is) ==="

if [[ "${RUN_POSTTRAIN_FIRST:-0}" == "1" ]]; then
  bash scripts/run_seven_axis_split_posttrain.sh
fi

echo "--- closed-loop training (12 rounds, smaller batch) ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/seven_axis_pair_merge_split_r12.yaml

FINAL_ROUND="$("${PY}" - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("data/splits/seven_axis_seed0.json").read_text())
print(int(m["meta"].get("n_rounds", 12)) - 1)
PY
)"
# Re-read from r12 history if manifest meta still says 6
if [[ -f results/seven_axis_pair_merge_split_r12/seed0/history.json ]]; then
  FINAL_ROUND="$("${PY}" - <<'PY'
import json
from pathlib import Path
h = json.loads(Path("results/seven_axis_pair_merge_split_r12/seed0/history.json").read_text())
print(int(h[-1]["round"]))
PY
)"
fi
echo "final round index: ${FINAL_ROUND}"

echo "--- pooled baseline replay ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/seven_axis_baseline_replay_split_r12.yaml

echo "--- merge eval (train) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/seven_axis_split_eval_train.yaml \
  -- run_dir=results/seven_axis_pair_merge_split_r12/seed0 \
     output_dir=results/seven_axis_pair_merge_split_r12/seed0/eval_train \
     rounds="[${FINAL_ROUND}]"

echo "--- merge eval (test) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/seven_axis_split_eval_test.yaml \
  -- run_dir=results/seven_axis_pair_merge_split_r12/seed0 \
     output_dir=results/seven_axis_pair_merge_split_r12/seed0/eval_test \
     rounds="[${FINAL_ROUND}]"

echo "--- pooled baseline eval (test) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/seven_axis_split_eval_test.yaml \
  -- run_dir=results/seven_axis_baseline_replay_split_r12/seed0 \
     output_dir=results/seven_axis_baseline_replay_split_r12/seed0/eval_test \
     rounds="[${FINAL_ROUND}]" \
     agents='["pooled-baseline"]'

echo "--- comparison tables ---"
"${PY}" scripts/build_seven_axis_split_tables.py \
  --run-dir results/seven_axis_pair_merge_split_r12/seed0 \
  --baseline-run-dir results/seven_axis_baseline_replay_split_r12/seed0

echo "=== seven_axis split r12 pipeline finished $(date -Is) ==="
