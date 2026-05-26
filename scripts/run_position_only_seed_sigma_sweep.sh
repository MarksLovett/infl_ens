#!/usr/bin/env bash
# scripts/run_position_only_seed_sigma_sweep.sh
#
# Seed × sigma grid for the matched position_only cumulative config.
# Each cell is an independent closed-loop run:
#
#   results/<SWEEP_NAME>/sigma<S>/seed<SEED>/
#   scripts/figures/<SWEEP_NAME>/per_run/sigma<S>/seed<SEED>/
#
# Resumable: training is skipped when history.json exists. Per-run figures
# are regenerated on each invocation. After training (or when all cells
# exist), calls scripts/aggregate_seed_sigma_sweep.py for mean ± std plots.
#
# Extend the grid by editing SEEDS or SIGMA_VALUES (whitespace-separated).
#
# Run scripts/run_position_step_stability_test.sh first (fast, no SFT —
# position updates are independent of LoRA). Pick POSITION_STEP_MODE from
# the winner, then launch this full sweep.
#
# Usage:
#   nohup bash scripts/run_position_only_seed_sigma_sweep.sh \
#       > results/position_only_cum_seed_sigma_sweep/launch.log 2>&1 &
#
# Train only (skip aggregation):
#   SKIP_AGGREGATE=1 bash scripts/run_position_only_seed_sigma_sweep.sh
#
# Aggregate only (all histories must exist):
#   TRAIN_ONLY=0 AGGREGATE_ONLY=1 bash scripts/run_position_only_seed_sigma_sweep.sh

set -euo pipefail

CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml}"
# Whitespace-separated lists — add seeds or sigmas here.
SEEDS="${SEEDS:-0 1 2 3 4}"
SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.5 0.75 1 1.5}"
SWEEP_NAME="${SWEEP_NAME:-position_only_cum_seed_sigma_sweep}"
N_ROUNDS="${N_ROUNDS:-20}"
TRAIN_ONLY="${TRAIN_ONLY:-1}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"
# Adaptive position step (empty POSITION_STEP_MODE = static blend only).
POSITION_STEP_MODE="${POSITION_STEP_MODE:-}"
POSITION_STEP_CAP="${POSITION_STEP_CAP:-0.05}"
BLEND_MAX="${BLEND_MAX:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CENTROID_MODE="${CENTROID_MODE:-}"
INIT_NOISE_OVERRIDE="${INIT_NOISE_OVERRIDE:-}"
PYTHON_PREFIX="${PYTHON_PREFIX:-}"
PY="${PYTHON_PREFIX:+env ${PYTHON_PREFIX}} ${PY:-.venv/bin/python}"

RESULTS_ROOT="results/${SWEEP_NAME}"
FIG_ROOT="scripts/figures/${SWEEP_NAME}"
PER_RUN_FIG="${FIG_ROOT}/per_run"

if [[ ! -f "${CONFIG}" ]]; then
    echo "CONFIG not found: ${CONFIG}" >&2
    exit 1
fi

mkdir -p "${RESULTS_ROOT}" "${FIG_ROOT}"

run_one() {
    local run_dir="$1"
    local fig_dir="$2"
    local title="$3"
    local seed="$4"
    local sigma_frac="$5"

    mkdir -p "${run_dir}" "${fig_dir}"

    local train_extra=(
        "closed_loop.blend=${BLEND_MAX}"
        "closed_loop.batch_size=${BATCH_SIZE}"
    )
    if [[ -n "${CENTROID_MODE}" ]]; then
        train_extra+=("closed_loop.centroid_mode=${CENTROID_MODE}")
    fi
    if [[ -n "${INIT_NOISE_OVERRIDE}" ]]; then
        train_extra+=("closed_loop.init_noise=${INIT_NOISE_OVERRIDE}")
    fi
    if [[ -n "${POSITION_STEP_MODE}" ]]; then
        train_extra+=(
            "closed_loop.position_step.mode=${POSITION_STEP_MODE}"
            "closed_loop.position_step.step_cap=${POSITION_STEP_CAP}"
            "closed_loop.position_step.blend_max=${BLEND_MAX}"
        )
    fi

    if [[ -f "${run_dir}/history.json" ]]; then
        echo "[skip-train] ${run_dir}/history.json"
    else
        echo "[train] ${run_dir}  (seed=${seed}, sigma_fraction=${sigma_frac})"
        ${PY} -m infl_ens.training \
            --config "${CONFIG}" \
            "output_dir=${run_dir}" \
            "closed_loop.sft.output_dir=${run_dir}/agents" \
            "seed=${seed}" \
            "closed_loop.sft.seed=${seed}" \
            "sigma_fraction=${sigma_frac}" \
            "closed_loop.n_rounds=${N_ROUNDS}" \
            "${train_extra[@]}" \
            2>&1 | tee "${run_dir}/training.log"
    fi

    echo "[plot] ${fig_dir}/trajectory"
    ${PY} scripts/plot_closed_loop_history.py \
        --history "${run_dir}/history.json" \
        --axis-labels harm hallucination \
        --title "${title}" \
        --output-stem "${fig_dir}/trajectory" \
        > /dev/null

    echo "[plot] ${fig_dir}/theory_vs_sft"
    ${PY} scripts/compare_theory_vs_sft.py \
        --config "${CONFIG}" \
        --history "${run_dir}/history.json" \
        --axis-labels harm hallucination \
        --title "${title}  theory vs SFT" \
        --output-stem "${fig_dir}/theory_vs_sft" \
        --summary-json "${run_dir}/theory_vs_sft.json" \
        --sigma-fraction-override "${sigma_frac}" \
        > /dev/null

    echo "[plot] ${fig_dir}/probe"
    ${PY} scripts/probe_sft_capability.py \
        --run-dir "${run_dir}" \
        --output-stem "${fig_dir}/probe" \
        > /dev/null
}

echo "================================================================"
echo "  position_only seed × sigma sweep"
echo "  config       : ${CONFIG}"
echo "  seeds        : ${SEEDS}"
echo "  sigma values : ${SIGMA_VALUES}"
echo "  n_rounds     : ${N_ROUNDS}"
echo "  batch_size     : ${BATCH_SIZE}"
echo "  centroid_mode  : ${CENTROID_MODE:-batch}"
echo "  init_noise     : ${INIT_NOISE_OVERRIDE:-from config}"
echo "  position_step  : ${POSITION_STEP_MODE:-static (blend=${BLEND_MAX})}"
if [[ -n "${POSITION_STEP_MODE}" ]]; then
    echo "  step_cap       : ${POSITION_STEP_CAP}"
fi
echo "  results      : ${RESULTS_ROOT}/"
echo "  figures      : ${FIG_ROOT}/"
echo "================================================================"

if [[ "${AGGREGATE_ONLY}" != "1" && "${TRAIN_ONLY}" != "0" ]]; then
    for S in ${SIGMA_VALUES}; do
        sigma_slug="sigma${S}"
        for seed in ${SEEDS}; do
            seed_slug="seed${seed}"
            run_dir="${RESULTS_ROOT}/${sigma_slug}/${seed_slug}"
            fig_dir="${PER_RUN_FIG}/${sigma_slug}/${seed_slug}"
            title="position_only ${sigma_slug} ${seed_slug}"
            run_one "${run_dir}" "${fig_dir}" "${title}" "${seed}" "${S}"
        done
    done
fi

if [[ "${TRAIN_ONLY}" == "1" && "${SKIP_AGGREGATE}" == "1" ]]; then
    echo "training complete (aggregation skipped)."
    exit 0
fi

if [[ "${TRAIN_ONLY}" == "1" && "${AGGREGATE_ONLY}" == "1" ]]; then
    echo "error: TRAIN_ONLY=1 and AGGREGATE_ONLY=1 are mutually exclusive" >&2
    exit 1
fi

echo
echo "[aggregate] seed-mean statistics -> ${FIG_ROOT}/aggregate/"
${PY} scripts/aggregate_seed_sigma_sweep.py \
    --root "${RESULTS_ROOT}" \
    --figure-root "${FIG_ROOT}" \
    --config "${CONFIG}" \
    --axis-labels harm hallucination \
    --title "position_only seed×sigma sweep (mean ± std over seeds)"

echo
echo "done."
echo "  per-run figures : ${PER_RUN_FIG}/sigma*/seed*/"
echo "  aggregates      : ${FIG_ROOT}/aggregate/"
