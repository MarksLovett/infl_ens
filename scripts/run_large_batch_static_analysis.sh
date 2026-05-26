#!/usr/bin/env bash
# scripts/run_large_batch_static_analysis.sh
#
# Static / large-batch position-only analysis at sigma 0.25 and 0.75.
# No SFT — compares how batch variance affects bifurcation:
#
#   batch_256       — default training batch (stochastic routing)
#   full_pool       — entire corpus (~10k prompts) per round
#   expected_pool   — deterministic G(1-G) pool centroid (M -> inf limit)
#
# Results:
#   results/large_batch_static_analysis/<mode>/sigma<S>/seed<K>/history.json
#   scripts/figures/large_batch_static_analysis/
#
# Usage:
#   bash scripts/run_large_batch_static_analysis.sh
#   COMPARE_ONLY=1 bash scripts/run_large_batch_static_analysis.sh

set -euo pipefail

CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml}"
SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.75}"
SEEDS="${SEEDS:-0 1 2 3 4}"
N_ROUNDS="${N_ROUNDS:-20}"
# Optional explicit large batch (ignored when mode is expected_pool or full_pool).
BATCH_SIZE_LARGE="${BATCH_SIZE_LARGE:-10000}"
SWEEP_ROOT="${SWEEP_ROOT:-results/large_batch_static_analysis}"
FIG_ROOT="${FIG_ROOT:-scripts/figures/large_batch_static_analysis}"
BLEND_MAX="${BLEND_MAX:-0.5}"
COMPARE_ONLY="${COMPARE_ONLY:-0}"
PYTHON_PREFIX="${PYTHON_PREFIX:-}"
PY="${PYTHON_PREFIX:+env ${PYTHON_PREFIX}} ${PY:-.venv/bin/python}"

mkdir -p "${SWEEP_ROOT}" "${FIG_ROOT}"

run_cell() {
    local mode_slug="$1"
    local centroid_mode="$2"
    local batch_arg="$3"
    local seed="$4"
    local sigma_frac="$5"

    local run_dir="${SWEEP_ROOT}/${mode_slug}/sigma${sigma_frac}/seed${seed}"
    mkdir -p "${run_dir}"

    if [[ -f "${run_dir}/history.json" ]]; then
        echo "[skip] ${run_dir}/history.json"
        return 0
    fi

    echo "[run] ${mode_slug} sigma=${sigma_frac} seed=${seed}"
    ${PY} scripts/simulate_position_only_loop.py \
        --config "${CONFIG}" \
        --mode simulate \
        --output-dir "${run_dir}" \
        --sigma-fraction "${sigma_frac}" \
        --seed "${seed}" \
        --n-rounds "${N_ROUNDS}" \
        --blend "${BLEND_MAX}" \
        --position-step-mode static \
        --centroid-mode "${centroid_mode}" \
        ${batch_arg} \
        2>&1 | tee "${run_dir}/run.log"
}

if [[ "${COMPARE_ONLY}" != "1" ]]; then
    echo "================================================================"
    echo "  large-batch static position analysis (no SFT)"
    echo "  sigmas : ${SIGMA_VALUES}"
    echo "  seeds  : ${SEEDS}"
    echo "  rounds : ${N_ROUNDS}"
    echo "================================================================"

    for S in ${SIGMA_VALUES}; do
        for seed in ${SEEDS}; do
            run_cell "batch_256" "batch" "--batch-size 256" "${seed}" "${S}"
            run_cell "batch_large" "batch" "--batch-size ${BATCH_SIZE_LARGE}" "${seed}" "${S}"
            run_cell "full_pool" "full_pool" "" "${seed}" "${S}"
            run_cell "expected_pool" "expected_pool" "" "${seed}" "${S}"
        done
    done
fi

echo
${PY} scripts/compare_batch_size_static.py \
    --root "${SWEEP_ROOT}" \
    --figure-root "${FIG_ROOT}" \
    --reference-root results/position_step_stability_test/mode_static

echo "done. figures: ${FIG_ROOT}/"
