#!/usr/bin/env bash
# 2x2 attribution sweep at seed 0: init (GA vs random) x theory_pre (on vs off).
#
#   bash scripts/run_attribution_2x2.sh
#
# Routing gate only; reuses existing pooled baseline for NLL reference.
# Skips cells whose history.json already exists unless FORCE=1.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/attribution_2x2/experiment.log}"
BASELINE_DIR="${BASELINE_DIR:-results/seven_axis_collapse_hypercube_ga_baseline/seed0}"
FORCE="${FORCE:-0}"

mkdir -p results/attribution_2x2
exec > >(tee -a "$LOG") 2>&1

echo "=== attribution 2x2 started $(date -Is) ==="

if [[ ! -f results/hypercube_edge_gradient_ascent/fixed_positions.json ]]; then
  echo "--- generate shared GA fixed_positions ---"
  "${PY}" scripts/run_merge_near_gradient_ascent.py --init hypercube_edges
fi

if [[ ! -f data/splits/five_axis_seed0.json ]]; then
  echo "--- build five-axis split manifest ---"
  "${PY}" scripts/build_five_axis_split.py
fi

# Reuse completed GA+pre run if present (same config modulo output path).
if [[ ! -f results/attribution_2x2/ga_theory_pre/seed0/history.json \
  && -f results/seven_axis_collapse_hypercube_ga/seed0/history.json ]]; then
  echo "--- seed ga_theory_pre from seven_axis_collapse_hypercube_ga/seed0 ---"
  mkdir -p results/attribution_2x2/ga_theory_pre
  cp -a results/seven_axis_collapse_hypercube_ga/seed0 \
    results/attribution_2x2/ga_theory_pre/seed0
fi

CELLS=(
  ga_theory_pre
  ga_no_theory_pre
  random_theory_pre
  random_no_theory_pre
)

for cell in "${CELLS[@]}"; do
  cfg="configs/benchmark/router/attribution_2x2/${cell}.yaml"
  run_dir="results/attribution_2x2/${cell}/seed0"
  hist="${run_dir}/history.json"

  if [[ -f "$hist" && "$FORCE" != "1" ]]; then
    echo "--- skip ${cell} (history exists; set FORCE=1 to rerun) ---"
  else
    echo "--- closed-loop training: ${cell} ---"
    "${PY}" -m infl_ens.training --config "$cfg"
  fi

  echo "--- routing gate: ${cell} ---"
  "${PY}" scripts/compare_routing_weights.py \
    --router-config "$cfg" \
    --history "$hist" \
    --merge-run-dir "$run_dir" \
    --baseline-run-dir "$BASELINE_DIR" \
    --partition test --max-eval-records 1000 \
    --save-merge-nll-cache "${run_dir}/merge_nll_test.npy" \
    --output-json "${run_dir}/routing_weight_comparison.json"

  echo "--- pair occupancy: ${cell} ---"
  "${PY}" scripts/analyze_pair_occupancy.py \
    --routing-json "${run_dir}/routing_weight_comparison.json" \
    --router-config "$cfg" \
    --output-json "${run_dir}/pair_occupancy.json"
done

echo "--- attribution summary ---"
"${PY}" scripts/summarize_attribution_2x2.py \
  --root results/attribution_2x2 \
  --output-json results/attribution_2x2/summary.json

echo "=== attribution 2x2 finished $(date -Is) ==="
