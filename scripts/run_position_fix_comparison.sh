#!/usr/bin/env bash
# scripts/run_position_fix_comparison.sh
#
# Compare stability fixes at sigma 0.25 and 0.75 with batch_size=256 (SFT
# scale in sim: routed minibatch for training path; position update varies).
#
# Variants:
#   baseline          — batch centroid, init_noise=1e-4 (current default)
#   expected_pool     — deterministic pool centroid for position update
#   init_noise_1e-2   — stronger symmetry breaking, batch centroid
#   pool_and_noise    — expected_pool + init_noise=1e-2
#
# Fast: uses simulate_position_only_loop.py (no LoRA). Resumable.
#
# Usage:
#   bash scripts/run_position_fix_comparison.sh
#   COMPARE_ONLY=1 bash scripts/run_position_fix_comparison.sh

set -euo pipefail

CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml}"
SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.75}"
SEEDS="${SEEDS:-0 1 2 3 4}"
N_ROUNDS="${N_ROUNDS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SWEEP_ROOT="${SWEEP_ROOT:-results/position_fix_comparison}"
FIG_ROOT="${FIG_ROOT:-scripts/figures/position_fix_comparison}"
BLEND_MAX="${BLEND_MAX:-0.5}"
COMPARE_ONLY="${COMPARE_ONLY:-0}"
PYTHON_PREFIX="${PYTHON_PREFIX:-}"
PY="${PYTHON_PREFIX:+env ${PYTHON_PREFIX}} ${PY:-.venv/bin/python}"

mkdir -p "${SWEEP_ROOT}" "${FIG_ROOT}"

run_variant() {
    local slug="$1"
    local centroid_mode="$2"
    local init_noise="$3"
    local seed="$4"
    local sigma_frac="$5"

    local run_dir="${SWEEP_ROOT}/${slug}/sigma${sigma_frac}/seed${seed}"
    mkdir -p "${run_dir}"

    if [[ -f "${run_dir}/history.json" ]]; then
        echo "[skip] ${run_dir}/history.json"
        return 0
    fi

    echo "[run] ${slug} sigma=${sigma_frac} seed=${seed}"
    ${PY} scripts/simulate_position_only_loop.py \
        --config "${CONFIG}" \
        --output-dir "${run_dir}" \
        --sigma-fraction "${sigma_frac}" \
        --seed "${seed}" \
        --n-rounds "${N_ROUNDS}" \
        --batch-size "${BATCH_SIZE}" \
        --blend "${BLEND_MAX}" \
        --centroid-mode "${centroid_mode}" \
        --init-noise "${init_noise}" \
        --position-step-mode static \
        2>&1 | tee "${run_dir}/run.log"
}

if [[ "${COMPARE_ONLY}" != "1" ]]; then
    echo "================================================================"
    echo "  position fix comparison (batch_size=${BATCH_SIZE}, no SFT)"
    echo "  sigmas : ${SIGMA_VALUES}"
    echo "  seeds  : ${SEEDS}"
    echo "================================================================"

    for S in ${SIGMA_VALUES}; do
        for seed in ${SEEDS}; do
            run_variant "baseline" "batch" "1e-4" "${seed}" "${S}"
            run_variant "expected_pool" "expected_pool" "1e-4" "${seed}" "${S}"
            run_variant "init_noise_1e-2" "batch" "0.01" "${seed}" "${S}"
            run_variant "pool_and_noise" "expected_pool" "0.01" "${seed}" "${S}"
        done
    done
fi

echo
${PY} scripts/compare_batch_size_static.py \
    --root "${SWEEP_ROOT}" \
    --figure-root "${FIG_ROOT}"

echo
echo "If a variant wins, enable in training YAML or overrides:"
echo "  closed_loop.centroid_mode=expected_pool"
echo "  closed_loop.init_noise=0.01"
echo "Then: bash scripts/run_position_only_seed_sigma_sweep.sh"
