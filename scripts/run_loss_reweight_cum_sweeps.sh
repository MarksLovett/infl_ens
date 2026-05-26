#!/usr/bin/env bash
# scripts/run_loss_reweight_cum_sweeps.sh
#
# Two-pass sweep over the loss_reweight cumulative config
# (canonical routing + (1-G) loss reweight + cumulative LoRA):
#
#   PASS 1 — round sweep  (n_rounds varies; sigma_fraction taken from config)
#   PASS 2 — sigma sweep  (sigma_fraction varies; n_rounds = SIGMA_SWEEP_N_ROUNDS)
#
# Default sweep grid:
#
#   round values            : 10  20  40
#   sigma values            : 0.25  0.5  0.75  1  1.5
#   sigma-sweep n_rounds    : 20
#
# Sigma values 0.25 / 0.5 / 0.75 are below the stability threshold
# (sigma_0*) and should bifurcate; 1.0 and 1.5 are at/above threshold
# and serve as the negative control — the symmetric Nash equilibrium
# should remain locally stable there, so the SFT trajectories should
# stay near the resource-weighted mean rather than splitting into
# specialists.
#
# Per AGENTS.md §4 rule 5, every run goes to its own results/<run_id>/
# directory; this script lays them out as:
#
#   results/<SWEEP_NAME>/<slug>/                  # per-run training output
#       history.json
#       theory_vs_sft.json
#       training.log
#       agents/clone-<i>/round-NN/                # per-round LoRA adapters
#   scripts/figures/<SWEEP_NAME>/<slug>/          # per-run figure stems
#       trajectory.{pdf,png}
#       theory_vs_sft.{pdf,png}
#       probe.{pdf,png}                          (or probe_*.{pdf,png})
#   scripts/figures/<SWEEP_NAME>/aggregate/       # plot_sweep.py output
#       sweep.{pdf,png}
#       sweep.csv
#
# Resumable: a run is skipped if its history.json already exists;
# delete that file to force a re-run. Figures are always re-generated
# because they're cheap.
#
# Usage:
#   bash scripts/run_loss_reweight_cum_sweeps.sh                # defaults above
#   ROUND_VALUES="10 20" SIGMA_VALUES="0.5 0.75" \
#       bash scripts/run_loss_reweight_cum_sweeps.sh
#
# Environment variables:
#   CONFIG               Base YAML (default: the loss_reweight_cum config).
#   ROUND_VALUES         Whitespace-separated n_rounds values for pass 1.
#                        Default: "10 20 40".
#   SIGMA_VALUES         Whitespace-separated sigma_fraction values for pass 2.
#                        Default: "0.25 0.5 0.75 1 1.5".
#   ROUND_SWEEP_NAME     Slug for the round sweep root.
#                        Default: "loss_reweight_cum_round_sweep".
#   SIGMA_SWEEP_NAME     Slug for the sigma sweep root.
#                        Default: "loss_reweight_cum_sigma_sweep".
#   SIGMA_SWEEP_N_ROUNDS n_rounds value used during the sigma sweep
#                        (default: 20).
#   PYTHON_PREFIX        Optional prefix on each python call, e.g.
#                        "PYTHONPATH=src" if the package is not installed
#                        via `pip install -e .`. Default: empty.

set -euo pipefail

# ----------------------------------------------------------------------
# Configurable knobs (env-var overridable)
# ----------------------------------------------------------------------
CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r10_loss_reweight_cum.yaml}"
ROUND_VALUES="${ROUND_VALUES:-10 20 40}"
SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.5 0.75 1 1.5}"
ROUND_SWEEP_NAME="${ROUND_SWEEP_NAME:-loss_reweight_cum_round_sweep}"
SIGMA_SWEEP_NAME="${SIGMA_SWEEP_NAME:-loss_reweight_cum_sigma_sweep}"
SIGMA_SWEEP_N_ROUNDS="${SIGMA_SWEEP_N_ROUNDS:-20}"
PYTHON_PREFIX="${PYTHON_PREFIX:-}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "CONFIG not found: ${CONFIG}" >&2
    exit 1
fi

# Build the python launcher; honour PYTHONPATH=src style prefixes.
PY="${PYTHON_PREFIX:+env ${PYTHON_PREFIX}} python"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

# run_one : train one config + run all three analysis plots into a
# dedicated per-run figure directory.
#
# Args:
#   $1 = run_dir       results path (e.g. results/<sweep>/<slug>)
#   $2 = fig_dir       figure path  (e.g. scripts/figures/<sweep>/<slug>)
#   $3 = title         human-readable title used in figure titles
#   $4 = config        YAML config path
#   $5+= extra training overrides (KEY=VAL ...)
run_one() {
    local run_dir="$1"
    local fig_dir="$2"
    local title="$3"
    local config="$4"
    shift 4

    mkdir -p "${run_dir}" "${fig_dir}"

    # --- Training (skip if history.json exists) ---
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

    # --- Per-run figures ---
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

# aggregate_one : run plot_sweep.py over a sweep root. plot_sweep.py
# may not support every mode we pass; we tee + fall through on failure
# so the rest of the script continues even if the aggregate plot can't
# be built.
#
# Args:
#   $1 = sweep_root    e.g. results/<sweep>
#   $2 = fig_dir       e.g. scripts/figures/<sweep>/aggregate
#   $3 = mode          plot_sweep.py mode ('sigma' / 'rounds' / 'seeds' / 'kde')
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
        echo "[aggregate] plot_sweep.py --mode ${mode} failed; the per-run figures under ${sweep_root%/}/ are still valid."
    fi
}

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
echo "================================================================"
echo "  loss_reweight cumulative sweeps"
echo "  config              : ${CONFIG}"
echo "  round values        : ${ROUND_VALUES}"
echo "  sigma values        : ${SIGMA_VALUES}"
echo "  sigma-sweep n_rounds: ${SIGMA_SWEEP_N_ROUNDS}"
echo "  round sweep root    : results/${ROUND_SWEEP_NAME}/"
echo "  sigma sweep root    : results/${SIGMA_SWEEP_NAME}/"
echo "  python prefix       : ${PYTHON_PREFIX:-<none>}"
echo "================================================================"

# ----------------------------------------------------------------------
# PASS 1 — round sweep
# ----------------------------------------------------------------------
echo
echo "================================================================"
echo "  PASS 1: round sweep"
echo "  (n_rounds varies; sigma_fraction taken from ${CONFIG})"
echo "================================================================"
for R in ${ROUND_VALUES}; do
    slug="r${R}"
    run_dir="results/${ROUND_SWEEP_NAME}/${slug}"
    fig_dir="scripts/figures/${ROUND_SWEEP_NAME}/${slug}"
    run_one "${run_dir}" "${fig_dir}" "round_sweep ${slug}" "${CONFIG}" \
        "closed_loop.n_rounds=${R}"
done

aggregate_one \
    "results/${ROUND_SWEEP_NAME}" \
    "scripts/figures/${ROUND_SWEEP_NAME}/aggregate" \
    "rounds"

# ----------------------------------------------------------------------
# PASS 2 — sigma sweep
# ----------------------------------------------------------------------
echo
echo "================================================================"
echo "  PASS 2: sigma sweep"
echo "  (sigma_fraction varies; n_rounds = ${SIGMA_SWEEP_N_ROUNDS})"
echo "================================================================"
for S in ${SIGMA_VALUES}; do
    slug="sigma${S}"
    run_dir="results/${SIGMA_SWEEP_NAME}/${slug}"
    fig_dir="scripts/figures/${SIGMA_SWEEP_NAME}/${slug}"
    THEO_SIGMA_FRAC="${S}" run_one "${run_dir}" "${fig_dir}" "sigma_sweep ${slug}" "${CONFIG}" \
        "sigma_fraction=${S}" \
        "closed_loop.n_rounds=${SIGMA_SWEEP_N_ROUNDS}"
    unset THEO_SIGMA_FRAC
done

aggregate_one \
    "results/${SIGMA_SWEEP_NAME}" \
    "scripts/figures/${SIGMA_SWEEP_NAME}/aggregate" \
    "sigma"

# ----------------------------------------------------------------------
# Wrap up
# ----------------------------------------------------------------------
echo
echo "================================================================"
echo "sweeps complete."
echo
echo "  round sweep figures : scripts/figures/${ROUND_SWEEP_NAME}/"
echo "    - per-run         : scripts/figures/${ROUND_SWEEP_NAME}/r{10,20,40}/"
echo "    - aggregate       : scripts/figures/${ROUND_SWEEP_NAME}/aggregate/"
echo
echo "  sigma sweep figures : scripts/figures/${SIGMA_SWEEP_NAME}/"
echo "    - per-run         : scripts/figures/${SIGMA_SWEEP_NAME}/sigma{0.25,0.5,0.75,1,1.5}/"
echo "    - aggregate       : scripts/figures/${SIGMA_SWEEP_NAME}/aggregate/"
echo
echo "  per-run JSON summary lives next to each run's history.json under"
echo "  results/<sweep>/<slug>/."
echo "================================================================"
