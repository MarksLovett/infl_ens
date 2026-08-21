#!/usr/bin/env bash
# Hypercube-GA collapse: theory pre + SFT-only pair merge + routing gate.
#
#   bash scripts/run_hypercube_collapse_experiment.sh
#
# Requires results/hypercube_edge_gradient_ascent/fixed_positions.json on doob.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/hypercube_collapse_experiment.log}"

exec > >(tee -a "$LOG") 2>&1

echo "=== hypercube collapse experiment started $(date -Is) ==="

if [[ ! -f results/hypercube_edge_gradient_ascent/fixed_positions.json ]]; then
  echo "--- generate hypercube-edge GA positions ---"
  "${PY}" scripts/run_merge_near_gradient_ascent.py --init hypercube_edges
fi

if [[ ! -f data/splits/five_axis_seed0.json ]]; then
  echo "--- build five-axis split manifest ---"
  "${PY}" scripts/build_five_axis_split.py
fi

echo "--- closed-loop training (hypercube GA start, SFT-only merge) ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/seven_axis_collapse_hypercube_ga.yaml

echo "--- pooled baseline replay ---"
"${PY}" -m infl_ens.training \
  --config configs/benchmark/router/seven_axis_collapse_hypercube_ga_baseline_replay.yaml

echo "--- flat routing comparison (collapse test pool) ---"
"${PY}" scripts/compare_routing_weights.py \
  --router-config configs/benchmark/router/seven_axis_collapse_hypercube_ga.yaml \
  --history results/seven_axis_collapse_hypercube_ga/seed0/history.json \
  --merge-run-dir results/seven_axis_collapse_hypercube_ga/seed0 \
  --baseline-run-dir results/seven_axis_collapse_hypercube_ga_baseline/seed0 \
  --partition test --max-eval-records 1000 \
  --save-merge-nll-cache results/seven_axis_collapse_hypercube_ga/seed0/merge_nll_test.npy \
  --output-json results/seven_axis_collapse_hypercube_ga/seed0/routing_weight_comparison.json

echo "--- pair occupancy analysis ---"
"${PY}" scripts/analyze_pair_occupancy.py \
  --routing-json results/seven_axis_collapse_hypercube_ga/seed0/routing_weight_comparison.json \
  --router-config configs/benchmark/router/seven_axis_collapse_hypercube_ga.yaml \
  --output-json results/seven_axis_collapse_hypercube_ga/seed0/pair_occupancy.json

echo "=== hypercube collapse experiment finished $(date -Is) ==="
