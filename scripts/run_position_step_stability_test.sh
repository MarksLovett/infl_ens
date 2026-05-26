#!/usr/bin/env bash
# scripts/run_position_step_stability_test.sh
#
# Fast pre-sweep check at sigma 0.25 and 0.75: compares position-step
# policies using simulate_position_only_loop.py (routing + weighted
# centroid only — NO SFT). Position dynamics are independent of LoRA.
#
# Layout:
#   results/position_step_stability_test/mode_<slug>/sigma<S>/seed<K>/history.json
#   scripts/figures/position_step_stability_test/
#
# Modes (edit STEP_MODES):
#   static          — fixed blend=0.5
#   cap_linf_0.05   — L∞ step cap 0.05
#   cap_linf_0.03   — tighter cap
#   trust_box_0.05  — L∞ cap + stay in [0,1]^2
#
# Optional replay on a full run (same routed corpora, new step policy):
#   REPLAY_SOURCE=results/.../history.json bash scripts/run_position_step_stability_test.sh
#
# Usage:
#   bash scripts/run_position_step_stability_test.sh
#   COMPARE_ONLY=1 bash scripts/run_position_step_stability_test.sh

set -euo pipefail

CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml}"
SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.75}"
SEEDS="${SEEDS:-0 1 2 3 4}"
N_ROUNDS="${N_ROUNDS:-20}"
STEP_MODES="${STEP_MODES:-static:static:0 cap_linf_0.05:cap_linf:0.05 cap_linf_0.03:cap_linf:0.03 trust_box_0.05:trust_box:0.05}"
SWEEP_ROOT="${SWEEP_ROOT:-results/position_step_stability_test}"
FIG_ROOT="${FIG_ROOT:-scripts/figures/position_step_stability_test}"
BLEND_MAX="${BLEND_MAX:-0.5}"
COMPARE_ONLY="${COMPARE_ONLY:-0}"
REPLAY_SOURCE="${REPLAY_SOURCE:-}"
PYTHON_PREFIX="${PYTHON_PREFIX:-}"
PY="${PYTHON_PREFIX:+env ${PYTHON_PREFIX}} ${PY:-.venv/bin/python}"

mkdir -p "${SWEEP_ROOT}" "${FIG_ROOT}"

run_sim_cell() {
    local mode_slug="$1"
    local step_mode="$2"
    local step_cap="$3"
    local seed="$4"
    local sigma_frac="$5"

    local run_dir="${SWEEP_ROOT}/mode_${mode_slug}/sigma${sigma_frac}/seed${seed}"
    mkdir -p "${run_dir}"

    if [[ -f "${run_dir}/history.json" ]]; then
        echo "[skip] ${run_dir}/history.json"
        return 0
    fi

    if [[ -n "${REPLAY_SOURCE}" ]]; then
        echo "[replay] mode=${mode_slug} sigma=${sigma_frac} seed=${seed}"
        ${PY} scripts/simulate_position_only_loop.py \
            --config "${CONFIG}" \
            --mode replay \
            --history "${REPLAY_SOURCE}" \
            --output-dir "${run_dir}" \
            --blend "${BLEND_MAX}" \
            --position-step-mode "${step_mode}" \
            --step-cap "${step_cap}"
    else
        echo "[simulate] mode=${mode_slug} sigma=${sigma_frac} seed=${seed}"
        ${PY} scripts/simulate_position_only_loop.py \
            --config "${CONFIG}" \
            --mode simulate \
            --output-dir "${run_dir}" \
            --sigma-fraction "${sigma_frac}" \
            --seed "${seed}" \
            --n-rounds "${N_ROUNDS}" \
            --blend "${BLEND_MAX}" \
            --position-step-mode "${step_mode}" \
            --step-cap "${step_cap}"
    fi
}

if [[ "${COMPARE_ONLY}" != "1" ]]; then
    echo "================================================================"
    echo "  position-step stability test (no SFT)"
    echo "  sigmas : ${SIGMA_VALUES}"
    echo "  seeds  : ${SEEDS}"
    echo "  rounds : ${N_ROUNDS}"
    echo "  modes  : ${STEP_MODES}"
    if [[ -n "${REPLAY_SOURCE}" ]]; then
        echo "  replay : ${REPLAY_SOURCE}"
    fi
    echo "================================================================"

    for spec in ${STEP_MODES}; do
        IFS=':' read -r mode_slug step_mode step_cap <<< "${spec}"
        for S in ${SIGMA_VALUES}; do
            for seed in ${SEEDS}; do
                run_sim_cell "${mode_slug}" "${step_mode}" "${step_cap}" "${seed}" "${S}"
            done
        done
    done
fi

echo
echo "[compare] ${FIG_ROOT}/"
${PY} scripts/compare_position_step_modes.py \
    --root "${SWEEP_ROOT}" \
    --figure-root "${FIG_ROOT}"

echo
echo "Pick the winning mode, then launch the seed×sigma sweep with:"
echo "  POSITION_STEP_MODE=cap_linf POSITION_STEP_CAP=0.05 \\"
echo "    bash scripts/run_position_only_seed_sigma_sweep.sh"
