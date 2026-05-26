#!/usr/bin/env bash
# scripts/run_position_only_cum_sweeps.sh
#
# Two-pass sweep over the matched position_only cumulative config:
#   - loss_reweight: position_only  ((1-G) on centroid only; unit-weight SFT)
#   - batch_size: 256               (same as loss_reweight_cum baseline)
#   - cumulative LoRA
#
#   PASS 1 — round sweep  (n_rounds ∈ {10, 20, 40}; sigma from config)
#   PASS 2 — sigma sweep  (sigma_fraction ∈ {0.25, 0.5, 0.75, 1, 1.5};
#                          n_rounds = SIGMA_SWEEP_N_ROUNDS, default 20)
#
# Layout mirrors run_loss_reweight_cum_sweeps.sh for side-by-side comparison:
#
#   results/position_only_cum_round_sweep/r{10,20,40}/
#   results/position_only_cum_sigma_sweep/sigma{0.25,0.5,0.75,1,1.5}/
#   scripts/figures/position_only_cum_*_sweep/<slug>/
#
# Resumable: training skipped when history.json exists; figures always
# regenerated.
#
# Usage:
#   nohup bash scripts/run_position_only_cum_sweeps.sh \
#       > results/position_only_cum_sweeps.log 2>&1 &
#
# Environment variables: same as run_loss_reweight_cum_sweeps.sh except
# defaults point at the position_only_cum config and sweep directory names.

set -euo pipefail

CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml}"
ROUND_VALUES="${ROUND_VALUES:-10 20 40}"
SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.5 0.75 1 1.5}"
ROUND_SWEEP_NAME="${ROUND_SWEEP_NAME:-position_only_cum_round_sweep}"
SIGMA_SWEEP_NAME="${SIGMA_SWEEP_NAME:-position_only_cum_sigma_sweep}"
SIGMA_SWEEP_N_ROUNDS="${SIGMA_SWEEP_N_ROUNDS:-20}"
# Set REDO_SIGMA_SWEEP=1 to delete existing sigma-sweep runs before pass 2
# (e.g. after adding init_noise). Round sweep is never auto-deleted.
REDO_SIGMA_SWEEP="${REDO_SIGMA_SWEEP:-0}"
PYTHON_PREFIX="${PYTHON_PREFIX:-}"
PY="${PYTHON_PREFIX:+env ${PYTHON_PREFIX}} ${PY:-.venv/bin/python}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "CONFIG not found: ${CONFIG}" >&2
    exit 1
fi

run_one() {
    local run_dir="$1"
    local fig_dir="$2"
    local title="$3"
    local config="$4"
    shift 4

    mkdir -p "${run_dir}" "${fig_dir}"

    if [[ -f "${run_dir}/history.json" ]]; then
        echo "[skip-train] ${run_dir}/history.json already exists"
    else
        echo "[train] ${run_dir}"
        ${PY} -m infl_ens.training \
            --config "${config}" \
            "output_dir=${run_dir}" \
            "closed_loop.sft.output_dir=${run_dir}/agents" \
            "$@" \
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
    theo_extra=()
    if [[ -n "${THEO_SIGMA_FRAC:-}" ]]; then
        theo_extra=(--sigma-fraction-override "${THEO_SIGMA_FRAC}")
    fi
    ${PY} scripts/compare_theory_vs_sft.py \
        --config "${config}" \
        --history "${run_dir}/history.json" \
        --axis-labels harm hallucination \
        --title "${title}  theory vs SFT" \
        --output-stem "${fig_dir}/theory_vs_sft" \
        --summary-json "${run_dir}/theory_vs_sft.json" \
        "${theo_extra[@]}" \
        > /dev/null

    echo "[plot] ${fig_dir}/probe"
    ${PY} scripts/probe_sft_capability.py \
        --run-dir "${run_dir}" \
        --output-stem "${fig_dir}/probe" \
        > /dev/null
}

aggregate_one() {
    local sweep_root="$1"
    local fig_dir="$2"
    local mode="$3"
    mkdir -p "${fig_dir}"
    echo "[aggregate] ${sweep_root}  --mode ${mode}"
    if ${PY} scripts/plot_sweep.py \
            --root "${sweep_root}" \
            --mode "${mode}" \
            --output-stem "${fig_dir}/sweep" \
            2>&1 | tee "${fig_dir}/aggregate.log"; then
        echo "[aggregate] ok -> ${fig_dir}/sweep.{pdf,png}"
    else
        echo "[aggregate] plot_sweep.py --mode ${mode} failed; per-run figures still valid."
    fi
}

echo "================================================================"
echo "  position_only cumulative sweeps (matched config, batch=256)"
echo "  config              : ${CONFIG}"
echo "  round values        : ${ROUND_VALUES}"
echo "  sigma values        : ${SIGMA_VALUES}"
echo "  sigma-sweep n_rounds: ${SIGMA_SWEEP_N_ROUNDS}"
echo "  round sweep root    : results/${ROUND_SWEEP_NAME}/"
echo "  sigma sweep root    : results/${SIGMA_SWEEP_NAME}/"
echo "================================================================"

if [[ "${SKIP_ROUND_SWEEP:-0}" != "1" ]]; then
echo
echo "================================================================"
echo "  PASS 1: round sweep"
echo "================================================================"
for R in ${ROUND_VALUES}; do
    slug="r${R}"
    run_dir="results/${ROUND_SWEEP_NAME}/${slug}"
    fig_dir="scripts/figures/${ROUND_SWEEP_NAME}/${slug}"
    run_one "${run_dir}" "${fig_dir}" "position_only ${slug}" "${CONFIG}" \
        "closed_loop.n_rounds=${R}"
done

aggregate_one \
    "results/${ROUND_SWEEP_NAME}" \
    "scripts/figures/${ROUND_SWEEP_NAME}/aggregate" \
    "rounds"
fi

if [[ "${REDO_SIGMA_SWEEP}" == "1" ]]; then
    echo "[redo] removing prior sigma sweep under results/${SIGMA_SWEEP_NAME}/"
    rm -rf "results/${SIGMA_SWEEP_NAME}"
    rm -rf "scripts/figures/${SIGMA_SWEEP_NAME}"
fi

echo
echo "================================================================"
echo "  PASS 2: sigma sweep"
echo "================================================================"
for S in ${SIGMA_VALUES}; do
    slug="sigma${S}"
    run_dir="results/${SIGMA_SWEEP_NAME}/${slug}"
    fig_dir="scripts/figures/${SIGMA_SWEEP_NAME}/${slug}"
    THEO_SIGMA_FRAC="${S}" run_one "${run_dir}" "${fig_dir}" "position_only ${slug}" "${CONFIG}" \
        "sigma_fraction=${S}" \
        "closed_loop.n_rounds=${SIGMA_SWEEP_N_ROUNDS}"
    unset THEO_SIGMA_FRAC
done

aggregate_one \
    "results/${SIGMA_SWEEP_NAME}" \
    "scripts/figures/${SIGMA_SWEEP_NAME}/aggregate" \
    "sigma"

echo
echo "================================================================"
echo "sweeps complete."
echo "  round : scripts/figures/${ROUND_SWEEP_NAME}/"
echo "  sigma : scripts/figures/${SIGMA_SWEEP_NAME}/"
echo "================================================================"
