#!/usr/bin/env bash
# Five-axis collapse experiment: split → init → train → baseline → routing gate.
#
#   bash scripts/run_collapse_experiment.sh
#
# Run on doob after syncing the repo.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/collapse_experiment.log}"

exec > >(tee -a "$LOG") 2>&1

echo "=== collapse experiment started $(date -Is) ==="

echo "--- build five-axis split manifest ---"
"${PY}" scripts/build_five_axis_split.py

echo "--- build 10-clone / 5-axis init positions ---"
"${PY}" scripts/build_five_axis_collapse_init.py

FINAL_ROUND="$("${PY}" - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("data/splits/five_axis_seed0.json").read_text())
print(int(m["meta"]["n_rounds"]) - 1)
PY
)"
echo "final round index: ${FINAL_ROUND}"

echo "--- closed-loop training (5 axes, 10 clones) ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/seven_axis_collapse_dead_axes.yaml

echo "--- pooled baseline replay ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/seven_axis_collapse_baseline_replay.yaml

echo "--- flat routing comparison (collapse test pool) ---"
"${PY}" scripts/compare_routing_weights.py \
  --router-config configs/benchmark/router/seven_axis_collapse_dead_axes.yaml \
  --history results/seven_axis_collapse_dead_axes/seed0/history.json \
  --merge-run-dir results/seven_axis_collapse_dead_axes/seed0 \
  --baseline-run-dir results/seven_axis_collapse_dead_axes_baseline/seed0 \
  --partition test --max-eval-records 1000 \
  --save-merge-nll-cache results/seven_axis_collapse_dead_axes/seed0/merge_nll_test.npy \
  --output-json results/seven_axis_collapse_dead_axes/seed0/routing_weight_comparison.json

echo "--- pair occupancy analysis ---"
"${PY}" scripts/analyze_pair_occupancy.py \
  --routing-json results/seven_axis_collapse_dead_axes/seed0/routing_weight_comparison.json \
  --router-config configs/benchmark/router/seven_axis_collapse_dead_axes.yaml \
  --output-json results/seven_axis_collapse_dead_axes/seed0/pair_occupancy.json

echo "=== collapse experiment finished $(date -Is) ==="
