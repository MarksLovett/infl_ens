#!/usr/bin/env bash
# Trait-space whitening: baseline / standardize / whiten on seed-0 eval.
#
#   VERIFY_ONLY=1 bash scripts/run_trait_whitening.sh
#   bash scripts/run_trait_whitening.sh

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/trait_whitening/experiment.log}"
REF_CONFIG="${REF_CONFIG:-configs/benchmark/router/attribution_2x2/ga_theory_pre.yaml}"
BASELINE_DIR="${BASELINE_DIR:-results/seven_axis_collapse_hypercube_ga_baseline/seed0}"
FORCE="${FORCE:-0}"
VERIFY_ONLY="${VERIFY_ONLY:-1}"
RESUME="${RESUME:-0}"

mkdir -p results/trait_whitening
exec > >(tee -a "$LOG") 2>&1

echo "=== trait whitening started $(date -Is) ==="

if [[ "$RESUME" != "1" ]]; then
if [[ ! -f data/splits/five_axis_seed1.json ]]; then
  echo "--- build seed-1 split manifest ---"
  "${PY}" scripts/build_five_axis_split.py \
    --config "$REF_CONFIG" \
    --output data/splits/five_axis_seed1.json \
    --seed 1
fi

echo "--- fit transforms on seed-1 (unsupervised) ---"
"${PY}" scripts/fit_whitening_transform.py --router-config "$REF_CONFIG"

echo "--- build arm configs ---"
"${PY}" scripts/build_trait_whitening_configs.py > results/trait_whitening/config_manifest.json

ALLOWLIST_EXTRA="trait_space.linear_transform"
for arm in baseline standardize whiten; do
  cfg="configs/benchmark/router/trait_whitening/${arm}.yaml"
  echo "--- config diff: ${arm} vs reference ---"
  EXTRA=()
  if [[ "$arm" != "baseline" ]]; then
    EXTRA=(--allow trait_space.linear_transform)
  fi
  "${PY}" scripts/diff_router_configs.py \
    --reference "$REF_CONFIG" \
    --candidate "$cfg" \
    "${EXTRA[@]}"
done

echo "--- sigma resolution per arm (stability_fraction confound check) ---"
"${PY}" scripts/report_whitening_sigma.py \
  --output-json results/trait_whitening/sigma_per_arm.json

echo "--- VERIFY_ONLY: seed-0 moment check ---"
"${PY}" scripts/verify_whitening_transform.py \
  --output-json results/trait_whitening/verify_seed0_moments.json

if [[ "$VERIFY_ONLY" == "1" ]]; then
  echo "VERIFY_ONLY=1; stopping before training."
  exit 0
fi
fi

ARMS=(baseline standardize whiten)
for arm in "${ARMS[@]}"; do
  cfg="configs/benchmark/router/trait_whitening/${arm}.yaml"
  run_dir="results/trait_whitening/${arm}/seed0"
  hist="${run_dir}/history.json"

  if [[ -f "$hist" && "$FORCE" != "1" ]]; then
    echo "--- skip ${arm} training (history exists) ---"
  else
    echo "--- closed-loop training: ${arm} ---"
    "${PY}" -m infl_ens.training --config "$cfg"
  fi

  echo "--- routing gate: ${arm} ---"
  if [[ -f "${run_dir}/routing_weight_comparison.json" && "$FORCE" != "1" ]]; then
    echo "--- skip ${arm} routing (results exist) ---"
  else
  "${PY}" scripts/compare_routing_weights.py \
    --router-config "$cfg" \
    --history "$hist" \
    --merge-run-dir "$run_dir" \
    --baseline-run-dir "$BASELINE_DIR" \
    --partition test --max-eval-records 1000 \
    --save-merge-nll-cache "${run_dir}/merge_nll_test.npy" \
    --output-json "${run_dir}/routing_weight_comparison.json"
  fi
done

echo "--- evaluate all arms ---"
"${PY}" scripts/evaluate_whitening.py \
  --output-json results/trait_whitening/summary.json

echo "=== trait whitening finished $(date -Is) ==="
