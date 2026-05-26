#!/usr/bin/env bash
# Generate per-run (+ optional aggregate) figures for a sigmaÃ—seed results tree.
#
# Usage:
#   bash scripts/plot_sweep_figures.sh results/position_only_sigma_r40_20seeds
#   bash scripts/plot_sweep_figures.sh results/sigma_r40_20seeds scripts/figures/sigma_r40_20seeds
#   SKIP_PROBE=0 AGGREGATE=1 bash scripts/plot_sweep_figures.sh results/sigma_r40_20seeds
#
# Environment:
#   SKIP_PROBE=1   Skip probe_sft_capability (no agents/ for position-only).
#   AGGREGATE=1    Run aggregate_seed_sigma_sweep.py after per-run plots.
#   FORCE=1        Replot even if trajectory.pdf exists.

set -euo pipefail

RESULTS_ROOT="${1:?results root required}"
FIG_ROOT="${2:-scripts/figures/$(basename "${RESULTS_ROOT}")}"
CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml}"
SKIP_PROBE="${SKIP_PROBE:-1}"
AGGREGATE="${AGGREGATE:-1}"
FORCE="${FORCE:-0}"
PY="${PY:-.venv/bin/python}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "CONFIG not found: ${CONFIG}" >&2
    exit 1
fi

plot_cell() {
    local run_dir="$1"
    local fig_dir="$2"
    local title="$3"
    local sigma_frac="$4"

    mkdir -p "${fig_dir}"
    if [[ "${FORCE}" != "1" && -f "${fig_dir}/trajectory.pdf" ]]; then
        echo "[skip-plot] ${fig_dir}/trajectory.pdf"
        return 0
    fi

    echo "[plot] ${run_dir} -> ${fig_dir}"
    ${PY} scripts/plot_closed_loop_history.py \
        --history "${run_dir}/history.json" \
        --axis-labels harm hallucination \
        --title "${title}" \
        --output-stem "${fig_dir}/trajectory" \
        > /dev/null

    theo_extra=()
    if [[ -n "${sigma_frac}" ]]; then
        theo_extra=(--sigma-fraction-override "${sigma_frac}")
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

    if [[ "${SKIP_PROBE}" != "1" && -d "${run_dir}/agents" ]]; then
        ${PY} scripts/probe_sft_capability.py \
            --run-dir "${run_dir}" \
            --output-stem "${fig_dir}/probe" \
            > /dev/null
    fi
}

echo "================================================================"
echo "  plot sweep figures"
echo "  results : ${RESULTS_ROOT}"
echo "  figures : ${FIG_ROOT}"
echo "  skip_probe=${SKIP_PROBE}  aggregate=${AGGREGATE}"
echo "================================================================"

n=0
for sigma_dir in "${RESULTS_ROOT}"/sigma*; do
    [[ -d "${sigma_dir}" ]] || continue
    slug="${sigma_dir##*/}"
    sigma_frac="${slug#sigma}"
    for seed_dir in "${sigma_dir}"/seed*; do
        [[ -d "${seed_dir}" ]] || continue
        [[ -f "${seed_dir}/history.json" ]] || continue
        seed="${seed_dir##*/seed}"
        fig_dir="${FIG_ROOT}/per_run/${slug}/${seed_dir##*/}"
        title="${slug} seed${seed}"
        plot_cell "${seed_dir}" "${fig_dir}" "${title}" "${sigma_frac}"
        n=$((n + 1))
    done
done

echo "plotted ${n} cells"

if [[ "${AGGREGATE}" == "1" && "${n}" -gt 0 ]]; then
    echo "[aggregate] ${FIG_ROOT}/aggregate/"
    ${PY} scripts/aggregate_seed_sigma_sweep.py \
        --root "${RESULTS_ROOT}" \
        --figure-root "${FIG_ROOT}" \
        --layout sigma_seed \
        --axis-labels harm hallucination \
        --title "$(basename "${RESULTS_ROOT}") sigma sweep (mean Â± std over seeds)"
fi

echo "done. figures under ${FIG_ROOT}/"
