#!/usr/bin/env bash
# Full SFT sweeps (default init: theory_gradient â€” GA from separated start).
#
# PASS 1 â€” round sweep: 10 seeds Ã— n_rounds âˆˆ {10, 20, 40}
#   Ïƒ from config (default sigma_fraction=0.5)
#
# PASS 2 â€” sigma sweep: 10 seeds Ã— Ïƒ/Ïƒâ‚€* âˆˆ {0.25, 0.5, 0.75, 1, 1.5}, 20 rounds
#
# Per cell: train â†’ trajectory / theory_vs_sft / probe plots under
#   scripts/figures/<SWEEP_NAME>/per_run/...
# Then aggregate_seed_sigma_sweep.py â†’
#   scripts/figures/<SWEEP_NAME>/aggregate/...
#
# Usage:
#   nohup bash scripts/run_pairs_near_eq_sweeps.sh \
#       > results/pairs_near_eq_sweeps.log 2>&1 &
#
#   SKIP_ROUND_SWEEP=1 bash scripts/run_pairs_near_eq_sweeps.sh   # sigma only
#   SKIP_SIGMA_SWEEP=1 bash scripts/run_pairs_near_eq_sweeps.sh   # rounds only
#   AGGREGATE_ONLY=1 ROUND_SWEEP_NAME=... bash ...               # re-aggregate

set -euo pipefail

CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
ROUND_VALUES="${ROUND_VALUES:-10 20 40}"
SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.5 0.75 1 1.5}"
ROUND_SWEEP_NAME="${ROUND_SWEEP_NAME:-pairs_near_eq_round_sweep}"
SIGMA_SWEEP_NAME="${SIGMA_SWEEP_NAME:-pairs_near_eq_sigma_sweep}"
SIGMA_SWEEP_N_ROUNDS="${SIGMA_SWEEP_N_ROUNDS:-20}"
INIT_NOISE="${INIT_NOISE:-0}"
THEORY_REF_ROOT="${THEORY_REF_ROOT:-results/theory_match_fixes/baseline_blend05}"
SKIP_ROUND_SWEEP="${SKIP_ROUND_SWEEP:-0}"
# Sigma pass disabled by default until re-enabled explicitly.
SKIP_SIGMA_SWEEP="${SKIP_SIGMA_SWEEP:-1}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"
PYTHON_PREFIX="${PYTHON_PREFIX:-}"
PY="${PYTHON_PREFIX:+env ${PYTHON_PREFIX}} ${PY:-.venv/bin/python}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "CONFIG not found: ${CONFIG}" >&2
    exit 1
fi

# Shared training overrides (pairs near equilibrium + expected_pool).
INIT_MODE="${INIT_MODE:-theory_gradient}"

TRAIN_BASE=(
    "closed_loop.init_mode=${INIT_MODE}"
    "closed_loop.init_noise=${INIT_NOISE}"
    "closed_loop.centroid_mode=expected_pool"
    "closed_loop.batch_size=256"
    "closed_loop.blend=0.5"
)

# Resolve a (2,2) reference history for pairs_near_theory (high-Ïƒ may lack per-Ïƒ files).
resolve_theory_ref_path() {
    local S="$1"
    local p
    for p in \
        "${THEORY_REF_ROOT}/sigma${S}/seed0/history.json" \
        "results/${SIGMA_SWEEP_NAME}/sigma${S}/seed0/history.json" \
        "results/theory_match_fixes/baseline_blend05/sigma${S}/seed0/history.json" \
        "results/pool_and_noise_10seeds/sigma${S}/seed0/history.json"
    do
        if [[ -f "${p}" ]]; then
            echo "${p}"
            return 0
        fi
    done
    for p in \
        "results/${SIGMA_SWEEP_NAME}/sigma0.25/seed0/history.json" \
        "results/${SIGMA_SWEEP_NAME}/sigma0.5/seed0/history.json" \
        "results/theory_match_fixes/baseline_blend05/sigma0.25/seed0/history.json" \
        "results/theory_match_fixes/baseline_blend05/sigma0.5/seed0/history.json"
    do
        if [[ -f "${p}" ]]; then
            echo "${p}"
            return 0
        fi
    done
    return 1
}

run_one() {
    local run_dir="$1"
    local fig_dir="$2"
    local title="$3"
    local seed="$4"
    shift 4

    mkdir -p "${run_dir}" "${fig_dir}"

    if [[ "${AGGREGATE_ONLY}" == "1" ]]; then
        return 0
    fi

    if [[ -f "${run_dir}/history.json" ]]; then
        echo "[skip-train] ${run_dir}/history.json"
    else
        echo "[train] ${run_dir}  seed=${seed}"
        train_extra=("$@")
        if [[ "${INIT_MODE}" == "pairs_near_theory" && -n "${THEORY_REF_PATH:-}" ]]; then
            train_extra=("closed_loop.theory_ref=${THEORY_REF_PATH}" "${train_extra[@]}")
        fi
        ${PY} -m infl_ens.training \
            --config "${CONFIG}" \
            "output_dir=${run_dir}" \
            "closed_loop.sft.output_dir=${run_dir}/agents" \
            "seed=${seed}" \
            "closed_loop.sft.seed=${seed}" \
            "${TRAIN_BASE[@]}" \
            "${train_extra[@]}" \
            2>&1 | tee "${run_dir}/training.log"
    fi

    echo "[plot] ${fig_dir}"
    ${PY} scripts/plot_closed_loop_history.py \
        --history "${run_dir}/history.json" \
        --axis-labels harm hallucination \
        --title "${title}" \
        --output-stem "${fig_dir}/trajectory" \
        > /dev/null

    theo_extra=()
    if [[ -n "${THEO_SIGMA_FRAC:-}" ]]; then
        theo_extra=(--sigma-fraction-override "${THEO_SIGMA_FRAC}")
    fi
    ${PY} scripts/compare_theory_vs_sft.py \
        --config "${CONFIG}" \
        --history "${run_dir}/history.json" \
        --axis-labels harm hallucination \
        --title "${title}  theory vs SFT" \
        --output-stem "${fig_dir}/theory_vs_sft" \
        --summary-json "${run_dir}/theory_vs_sft.json" \
        "${theo_extra[@]}" \
        > /dev/null

    ${PY} scripts/probe_sft_capability.py \
        --run-dir "${run_dir}" \
        --output-stem "${fig_dir}/probe" \
        > /dev/null
}

aggregate_pass() {
    local results_root="$1"
    local fig_root="$2"
    local layout="$3"
    local agg_title="$4"

    echo "[aggregate] ${fig_root}/aggregate/  (layout=${layout})"
    ${PY} scripts/aggregate_seed_sigma_sweep.py \
        --root "${results_root}" \
        --figure-root "${fig_root}" \
        --layout "${layout}" \
        --axis-labels harm hallucination \
        --title "${agg_title}"
}

echo "================================================================"
echo "  SFT sweeps  init_mode=${INIT_MODE}"
echo "  config         : ${CONFIG}"
echo "  seeds          : ${SEEDS}"
echo "  round values   : ${ROUND_VALUES}"
echo "  sigma values   : ${SIGMA_VALUES}"
echo "  theory ref root: ${THEORY_REF_ROOT}"
echo "================================================================"

ROUND_RESULTS="results/${ROUND_SWEEP_NAME}"
ROUND_FIG="scripts/figures/${ROUND_SWEEP_NAME}"
SIGMA_RESULTS="results/${SIGMA_SWEEP_NAME}"
SIGMA_FIG="scripts/figures/${SIGMA_SWEEP_NAME}"

if [[ "${SKIP_ROUND_SWEEP}" != "1" ]]; then
    echo
    echo "======== PASS 1: round sweep (seeds Ã— rounds) ========"
    if [[ "${AGGREGATE_ONLY}" != "1" ]]; then
        for R in ${ROUND_VALUES}; do
            slug="r${R}"
            for seed in ${SEEDS}; do
                run_dir="${ROUND_RESULTS}/${slug}/seed${seed}"
                fig_dir="${ROUND_FIG}/per_run/${slug}/seed${seed}"
                title="${INIT_MODE} ${slug} seed${seed}"
                run_one "${run_dir}" "${fig_dir}" "${title}" "${seed}" \
                    "closed_loop.n_rounds=${R}"
            done
        done
    fi
    aggregate_pass "${ROUND_RESULTS}" "${ROUND_FIG}" "round_seed" \
        "${INIT_MODE} round sweep (mean Â± std over seeds)"
fi

if [[ "${SKIP_SIGMA_SWEEP}" != "1" ]]; then
    echo
    echo "======== PASS 2: sigma sweep (seeds Ã— Ïƒ) ========"
    if [[ "${AGGREGATE_ONLY}" != "1" ]]; then
        for S in ${SIGMA_VALUES}; do
            slug="sigma${S}"
            THEORY_REF_PATH=""
            if [[ "${INIT_MODE}" == "pairs_near_theory" ]]; then
                if THEORY_REF_PATH="$(resolve_theory_ref_path "${S}")"; then
                    echo "  sigma=${S} theory_ref=${THEORY_REF_PATH}"
                else
                    echo "  sigma=${S} theory_ref=<Python resolve_theory_22_reference>" >&2
                fi
            fi
            for seed in ${SEEDS}; do
                run_dir="${SIGMA_RESULTS}/${slug}/seed${seed}"
                fig_dir="${SIGMA_FIG}/per_run/${slug}/seed${seed}"
                title="${INIT_MODE} ${slug} seed${seed}"
                THEO_SIGMA_FRAC="${S}" run_one "${run_dir}" "${fig_dir}" "${title}" "${seed}" \
                    "sigma_fraction=${S}" \
                    "closed_loop.n_rounds=${SIGMA_SWEEP_N_ROUNDS}"
                unset THEO_SIGMA_FRAC
            done
            unset THEORY_REF_PATH
        done
    fi
    aggregate_pass "${SIGMA_RESULTS}" "${SIGMA_FIG}" "sigma_seed" \
        "${INIT_MODE} sigma sweep (mean Â± std over seeds)"
fi

echo
echo "================================================================"
echo "done."
echo "  round results : ${ROUND_RESULTS}/r*/seed*/"
echo "  round figures : ${ROUND_FIG}/per_run/  +  ${ROUND_FIG}/aggregate/"
echo "  sigma results : ${SIGMA_RESULTS}/sigma*/seed*/"
echo "  sigma figures : ${SIGMA_FIG}/per_run/  +  ${SIGMA_FIG}/aggregate/"
echo "================================================================"
