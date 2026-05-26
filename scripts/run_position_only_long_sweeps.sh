#!/usr/bin/env bash
# scripts/run_position_only_long_sweeps.sh
#
# Two-pass sweep over the long-SFT + position-only-reweight + cumulative
# LoRA closed-loop config:
#
#   PASS 1 — round sweep  (n_rounds ∈ {10, 20, 40};
#                          sigma_fraction = 0.5 from the YAML)
#   PASS 2 — sigma sweep  (sigma_fraction ∈ {0.25, 0.5, 0.75, 1, 1.5};
#                          n_rounds = SIGMA_SWEEP_N_ROUNDS, default 20)
#
# Sigma values 0.25 / 0.5 / 0.75 are below the stability threshold
# (sigma_0*) and should bifurcate; 1.0 and 1.5 are at/above threshold
# and serve as the negative control — the symmetric Nash equilibrium
# should remain locally stable there.
#
# Compute estimate: the long-SFT config trains ~6× more queries per
# round and uses a 2× larger micro-batch than the default budget, for
# ~3× more optimizer steps per agent per round. Across the default
# grid this works out to roughly
#
#     3 × (10 + 20 + 40) + 3 × (5 sigmas × 20 rounds)
#     = 210 + 300
#     = 510 (default-budget round-equivalents)
#
# i.e. about 17× the wall time of one default-budget r10 run. Start it
# under nohup (see the bottom of this header) and check in later.
#
# Per AGENTS.md §4 rule 5, every run goes to its own results/<run_id>/
# directory:
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
#       ess_compare.{pdf,png}
#   scripts/figures/<SWEEP_NAME>/aggregate/       # plot_sweep.py output
#       sweep.{pdf,png}
#       sweep.csv
#
# Resumable: a run is skipped if its history.json already exists;
# delete that file to force a re-run. Figures are always re-rendered.
#
# Usage:
#   bash scripts/run_position_only_long_sweeps.sh             # defaults above
#   ROUND_VALUES="10 20" SIGMA_VALUES="0.5" \
#       bash scripts/run_position_only_long_sweeps.sh         # smaller sweep
#
# Run in the background:
#   nohup bash scripts/run_position_only_long_sweeps.sh \
#       > results/position_only_long_sweeps.log 2>&1 &
#   disown
#   tail -f results/position_only_long_sweeps.log
#
# Environment variables:
#   CONFIG               Base YAML (default: the position_only_long_cum config).
#   ROUND_VALUES         Whitespace-separated n_rounds values for pass 1.
#                        Default: "10 20 40".
#   SIGMA_VALUES         Whitespace-separated sigma_fraction values for pass 2.
#                        Default: "0.25 0.5 0.75 1 1.5".
#   ROUND_SWEEP_NAME     Slug for the round sweep root.
#                        Default: "position_only_long_round_sweep".
#   SIGMA_SWEEP_NAME     Slug for the sigma sweep root.
#                        Default: "position_only_long_sigma_sweep".
#   SIGMA_SWEEP_N_ROUNDS n_rounds used during the sigma sweep
#                        (default: 20).
#   PYTHON_PREFIX        Optional prefix on each python call, e.g.
#                        "PYTHONPATH=src" if the package is not installed
#                        via `pip install -e .`. Default: empty.

set -euo pipefail

CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r10_position_only_long_cum.yaml}"
ROUND_VALUES="${ROUND_VALUES:-10 20 40}"
SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.5 0.75 1 1.5}"
ROUND_SWEEP_NAME="${ROUND_SWEEP_NAME:-position_only_long_round_sweep}"
SIGMA_SWEEP_NAME="${SIGMA_SWEEP_NAME:-position_only_long_sigma_sweep}"
SIGMA_SWEEP_N_ROUNDS="${SIGMA_SWEEP_N_ROUNDS:-20}"
PYTHON_PREFIX="${PYTHON_PREFIX:-}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "CONFIG not found: ${CONFIG}" >&2
    exit 1
fi

PY="${PYTHON_PREFIX:+env ${PYTHON_PREFIX}} python"

# ----------------------------------------------------------------------
# Per-run helper: train + four analysis plots into a per-run figure dir.
# ----------------------------------------------------------------------
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
    ${PY} scripts/compare_theory_vs_sft.py \
        --config "${config}" \
        --history "${run_dir}/history.json" \
        --axis-labels harm hallucination \
        --title "${title}  theory vs SFT" \
        --output-stem "${fig_dir}/theory_vs_sft" \
        --summary-json "${run_dir}/theory_vs_sft.json" \
        > /dev/null

    echo "[plot] ${fig_dir}/probe"
    ${PY} scripts/probe_sft_capability.py \
        --run-dir "${run_dir}" \
        --output-stem "${fig_dir}/probe" \
        > /dev/null

    # ESS diagnostic — quantifies what the alternative reweight modes
    # *would have* done at each round's positions. Useful for the
    # position_only vs one_minus_G A/B since the metric only depends on
    # positions and sigma, not on the actual rule that ran.
    echo "[plot] ${fig_dir}/ess_compare"
    ${PY} scripts/compare_routing_ess.py \
        --config "${config}" \
        --history "${run_dir}/history.json" \
        --output-stem "${fig_dir}/ess_compare" \
        --summary-json "${run_dir}/routing_ess.json" \
        > /dev/null || \
        echo "[plot] compare_routing_ess.py not present or failed; per-run figures still valid"
}

# ----------------------------------------------------------------------
# Aggregate helper.
# ----------------------------------------------------------------------
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
echo "  position-only reweight + long SFT sweeps"
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
    run_one "${run_dir}" "${fig_dir}" "position_only_long ${slug}" "${CONFIG}" \
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
    run_one "${run_dir}" "${fig_dir}" "position_only_long ${slug}" "${CONFIG}" \
        "sigma_fraction=${S}" \
        "closed_loop.n_rounds=${SIGMA_SWEEP_N_ROUNDS}"
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
