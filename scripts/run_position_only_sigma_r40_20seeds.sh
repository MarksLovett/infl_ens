#!/usr/bin/env bash
# Position-only sigma grid (fast): 20 seeds × 5 σ × 40 rounds — same grid as SFT sweep.
#
#   results/position_only_sigma_r40_20seeds/sigma*/seed*/
#
# Usage:
#   bash scripts/run_position_only_sigma_r40_20seeds.sh
#   INIT_MODE=pairs_near_theory bash scripts/run_position_only_sigma_r40_20seeds.sh

set -euo pipefail

CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml}"
SWEEP_ROOT="${SWEEP_ROOT:-results/position_only_sigma_r40_20seeds}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.5 0.75 1 1.5}"
N_ROUNDS="${N_ROUNDS:-40}"
INIT_MODE="${INIT_MODE:-theory_gradient}"
INIT_NOISE="${INIT_NOISE:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PY="${PY:-.venv/bin/python}"

mkdir -p "${SWEEP_ROOT}"

echo "================================================================"
echo "  position-only sigma grid  init=${INIT_MODE}  rounds=${N_ROUNDS}"
echo "  seeds=${SEEDS}"
echo "  sigmas=${SIGMA_VALUES}"
echo "  root=${SWEEP_ROOT}"
echo "================================================================"

for S in ${SIGMA_VALUES}; do
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
      --init-mode "${INIT_MODE}" \
      --init-noise "${INIT_NOISE}" \
      --position-step-mode static
  done
done

echo
${PY} scripts/count_equilibrium_types.py --root "${SWEEP_ROOT}"
