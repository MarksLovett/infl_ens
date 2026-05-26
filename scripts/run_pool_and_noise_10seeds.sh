#!/usr/bin/env bash
# scripts/run_pool_and_noise_10seeds.sh
#
# Both fixes together over 10 seeds:
#   centroid_mode: expected_pool
#   init_noise: 0.01
#   batch_size: 256 (routing / sim batch path)
#
# Then aggregate mean ± std of final positions.
#
# Usage:
#   bash scripts/run_pool_and_noise_10seeds.sh
#   AGGREGATE_ONLY=1 bash scripts/run_pool_and_noise_10seeds.sh

set -euo pipefail

CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml}"
SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.75}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
N_ROUNDS="${N_ROUNDS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
INIT_NOISE="${INIT_NOISE:-0.01}"
SWEEP_ROOT="${SWEEP_ROOT:-results/pool_and_noise_10seeds}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"
PYTHON_PREFIX="${PYTHON_PREFIX:-}"
PY="${PYTHON_PREFIX:+env ${PYTHON_PREFIX}} ${PY:-.venv/bin/python}"

mkdir -p "${SWEEP_ROOT}"

if [[ "${AGGREGATE_ONLY}" != "1" ]]; then
    echo "================================================================"
    echo "  pool_and_noise: expected_pool + init_noise=${INIT_NOISE}"
    echo "  seeds=${SEEDS}  sigmas=${SIGMA_VALUES}  rounds=${N_ROUNDS}"
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
                --init-noise "${INIT_NOISE}" \
                --position-step-mode static
        done
    done
fi

echo
${PY} scripts/aggregate_final_positions.py \
    --root "${SWEEP_ROOT}" \
    --output-json "${SWEEP_ROOT}/final_positions_stats.json"
