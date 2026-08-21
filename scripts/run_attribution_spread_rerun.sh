#!/usr/bin/env bash
# Spread-calibrated attribution re-run.
#
#   bash scripts/run_attribution_spread_rerun.sh
#
# Random arms: matched (~0.9) and moderate (~0.45) spread, 3 seeds.
# GA reproducibility: ga_no_theory_pre seeds 1-4 + ga_theory_pre seed 1 spot-check.
# Skips cells with history.json unless FORCE=1.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/attribution_spread_rerun/experiment.log}"
BASELINE_DIR="${BASELINE_DIR:-results/seven_axis_collapse_hypercube_ga_baseline/seed0}"
FORCE="${FORCE:-0}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"

mkdir -p results/attribution_spread_rerun
exec > >(tee -a "$LOG") 2>&1

echo "=== attribution spread re-run started $(date -Is) ==="

"${PY}" scripts/build_attribution_spread_rerun_configs.py > results/attribution_spread_rerun/config_manifest.json

for seed in 0 1 2 3 4; do
  if [[ ! -f "data/splits/five_axis_seed${seed}.json" ]]; then
    echo "--- build split manifest seed ${seed} ---"
    "${PY}" scripts/build_five_axis_split.py \
      --config configs/benchmark/router/seven_axis_collapse_hypercube_ga.yaml \
      --output "data/splits/five_axis_seed${seed}.json" \
      --seed "${seed}"
  fi
done

CONFIGS=(
  ga_no_theory_pre_seed1
  ga_no_theory_pre_seed2
  ga_no_theory_pre_seed3
  ga_no_theory_pre_seed4
  ga_theory_pre_seed1
  random_s09_theory_pre_seed0
  random_s09_theory_pre_seed1
  random_s09_theory_pre_seed2
  random_s09_no_theory_pre_seed0
  random_s09_no_theory_pre_seed1
  random_s09_no_theory_pre_seed2
  random_s045_theory_pre_seed0
  random_s045_theory_pre_seed1
  random_s045_theory_pre_seed2
)

echo "--- verify init spread (random cells) ---"
for cfg_name in "${CONFIGS[@]}"; do
  if [[ "${cfg_name}" == random_* ]]; then
  cfg="configs/benchmark/router/attribution_spread_rerun/${cfg_name}.yaml"
    "${PY}" scripts/verify_init_spread.py --config "$cfg"
  fi
done

if [[ "$VERIFY_ONLY" == "1" ]]; then
  echo "VERIFY_ONLY=1; stopping before training."
  exit 0
fi

for cfg_name in "${CONFIGS[@]}"; do
  cfg="configs/benchmark/router/attribution_spread_rerun/${cfg_name}.yaml"
  run_dir="results/attribution_spread_rerun/${cfg_name}"
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

  echo "--- pair occupancy: ${cfg_name} ---"
  "${PY}" scripts/analyze_pair_occupancy.py \
    --routing-json "${run_dir}/routing_weight_comparison.json" \
    --router-config "$cfg" \
    --output-json "${run_dir}/pair_occupancy.json"
done

echo "--- summary ---"
"${PY}" scripts/summarize_attribution_spread_rerun.py \
  --root results/attribution_spread_rerun \
  --output-json results/attribution_spread_rerun/summary.json

echo "=== attribution spread re-run finished $(date -Is) ==="
