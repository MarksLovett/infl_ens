#!/usr/bin/env bash
# Within-merge spread: oracle k=2 aligned vs misaligned control.
# Fixed 5 merges, seed-0 split, per-round within_merge in history.
#
#   bash scripts/run_within_merge_spread.sh

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/within_merge_spread/experiment.log}"
BASELINE_DIR="${BASELINE_DIR:-results/seven_axis_collapse_hypercube_ga_baseline/seed0}"
FORCE="${FORCE:-0}"

mkdir -p results/within_merge_spread
exec > >(tee -a "$LOG") 2>&1

echo "=== within-merge spread started $(date -Is) ==="

"${PY}" scripts/build_oracle_spread_positions.py
"${PY}" scripts/build_within_merge_spread_configs.py > results/within_merge_spread/config_manifest.json

CONFIGS=(
  oracle_k2_aligned
  oracle_k2_misaligned
)

for cfg_name in "${CONFIGS[@]}"; do
  cfg="configs/benchmark/router/within_merge_spread/${cfg_name}.yaml"
  run_dir="results/within_merge_spread/${cfg_name}"
  hist="${run_dir}/history.json"

  if [[ -f "$hist" && "$FORCE" != "1" ]]; then
    echo "--- skip ${cfg_name} (history exists) ---"
  else
    echo "--- closed-loop training: ${cfg_name} ---"
    "${PY}" -m infl_ens.training --config "$cfg"
  fi

  echo "--- routing gate: ${cfg_name} ---"
  "${PY}" scripts/compare_routing_weights.py \
    --router-config "$cfg" \
    --history "$hist" \
    --merge-run-dir "$run_dir" \
    --baseline-run-dir "$BASELINE_DIR" \
    --partition test --max-eval-records 1000 \
    --save-merge-nll-cache "${run_dir}/merge_nll_test.npy" \
    --output-json "${run_dir}/routing_weight_comparison.json"
done

echo "--- summary ---"
"${PY}" scripts/summarize_within_merge_spread.py \
  --root results/within_merge_spread \
  --output-json results/within_merge_spread/summary.json

echo "=== within-merge spread finished $(date -Is) ==="
