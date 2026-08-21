#!/usr/bin/env bash
# Seed-0 split isolation: GA no-theory-pre, fixed five_axis_seed0.json,
# training seeds 1/2/3. Tests whether +0.009 Δ vs pooled reproduces.
#
#   bash scripts/run_seed0_isolation.sh

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/seed0_isolation/experiment.log}"
BASELINE_DIR="${BASELINE_DIR:-results/seven_axis_collapse_hypercube_ga_baseline/seed0}"
FORCE="${FORCE:-0}"

mkdir -p results/seed0_isolation
exec > >(tee -a "$LOG") 2>&1

echo "=== seed-0 isolation started $(date -Is) ==="

"${PY}" scripts/build_seed0_isolation_configs.py > results/seed0_isolation/config_manifest.json

if [[ ! -f "data/splits/five_axis_seed0.json" ]]; then
  echo "--- build split manifest seed 0 ---"
  "${PY}" scripts/build_five_axis_split.py \
    --config configs/benchmark/router/seven_axis_collapse_hypercube_ga.yaml \
    --output "data/splits/five_axis_seed0.json" \
    --seed 0
fi

CONFIGS=(
  ga_no_theory_pre_train1
  ga_no_theory_pre_train2
  ga_no_theory_pre_train3
)

for cfg_name in "${CONFIGS[@]}"; do
  cfg="configs/benchmark/router/seed0_isolation/${cfg_name}.yaml"
  run_dir="results/seed0_isolation/${cfg_name}"
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
"${PY}" scripts/summarize_seed0_isolation.py \
  --root results/seed0_isolation \
  --output-json results/seed0_isolation/summary.json

echo "=== seed-0 isolation finished $(date -Is) ==="
