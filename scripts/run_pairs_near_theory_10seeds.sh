#!/usr/bin/env bash
# Initialize clones in pairs near per-sigma (2,2) theory endpoints; 10 seeds × 2 σ.
#
# Uses expected_pool + blend 0.5 + init_noise jitter around reference finals
# (default: baseline_blend05 / pool_and_noise seed0 at each σ).
#
# Usage:
#   bash scripts/run_pairs_near_theory_10seeds.sh
#   AGGREGATE_ONLY=1 bash scripts/run_pairs_near_theory_10seeds.sh

set -euo pipefail

CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml}"
SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.75}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
N_ROUNDS="${N_ROUNDS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
INIT_NOISE="${INIT_NOISE:-0.01}"
SWEEP_ROOT="${SWEEP_ROOT:-results/pairs_near_theory_10seeds}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"
PY="${PY:-.venv/bin/python}"

mkdir -p "${SWEEP_ROOT}"

if [[ "${AGGREGATE_ONLY}" != "1" ]]; then
  echo "================================================================"
  echo "  pairs_near_theory: expected_pool + init near (2,2) reference"
  echo "  init_noise=${INIT_NOISE}  seeds=${SEEDS}  sigmas=${SIGMA_VALUES}"
  echo "================================================================"

  for S in ${SIGMA_VALUES}; do
  REF="${THEORY_REF_ROOT:-results/theory_match_fixes/baseline_blend05}/sigma${S}/seed0/history.json"
  if [[ ! -f "${REF}" ]]; then
    REF="results/pool_and_noise_10seeds/sigma${S}/seed0/history.json"
  fi
  if [[ ! -f "${REF}" ]]; then
    echo "warning: no theory ref at ${REF}; sim will use gradient fallback" >&2
    REF_ARG=()
  else
    echo "  sigma=${S} theory_ref=${REF}"
    REF_ARG=(--theory-ref "${REF}")
  fi

    for seed in ${SEEDS}; do
      run_dir="${SWEEP_ROOT}/sigma${S}/seed${seed}"
      mkdir -p "${run_dir}"
      if [[ -f "${run_dir}/history.json" ]]; then
        echo "[skip] ${run_dir}/history.json"
        continue
      fi
      echo "[run] sigma=${S} seed=${seed}"
      ${PY} scripts/simulate_position_only_loop.py \
        --config "${CONFIG}" \
        --output-dir "${run_dir}" \
        --sigma-fraction "${S}" \
        --seed "${seed}" \
        --n-rounds "${N_ROUNDS}" \
        --batch-size "${BATCH_SIZE}" \
        --blend 0.5 \
        --centroid-mode expected_pool \
        --init-mode pairs_near_theory \
        --init-noise "${INIT_NOISE}" \
        --position-step-mode static \
        "${REF_ARG[@]}"
    done
  done
fi

echo
${PY} scripts/summarize_pairs_near_theory.py --root "${SWEEP_ROOT}"
