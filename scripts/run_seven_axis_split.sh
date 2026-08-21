#!/usr/bin/env bash
# Full seven-axis split experiment: manifest → train → baseline → eval → tables.
#
#   bash scripts/run_seven_axis_split.sh
#
# Run on doob after syncing the repo.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/seven_axis_split_pipeline.log}"
MANIFEST="${MANIFEST:-data/splits/seven_axis_seed0.json}"

exec > >(tee -a "$LOG") 2>&1

echo "=== seven_axis split pipeline started $(date -Is) ==="

echo "--- build split manifest ---"
"${PY}" scripts/build_seven_axis_split.py

FINAL_ROUND="$("${PY}" - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("data/splits/seven_axis_seed0.json").read_text())
print(int(m["meta"]["n_rounds"]) - 1)
PY
)"
echo "final round index: ${FINAL_ROUND}"

echo "--- closed-loop training (exact train coverage) ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/seven_axis_pair_merge_split.yaml

echo "--- pooled baseline replay ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/seven_axis_baseline_replay_split.yaml

echo "--- merge eval (train partition) ---"
"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/seven_axis_split_eval_train.yaml \
  -- rounds="[${FINAL_ROUND}]"

echo "--- merge eval (test partition) ---"
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

echo "=== seven_axis split pipeline finished $(date -Is) ==="
