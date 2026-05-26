#!/usr/bin/env bash
# scripts/run_position_only_cum_r10.sh
#
# Single 10-round closed-loop run with the *matched* position_only config:
#   - loss_reweight: position_only  (unit-weight SFT, (1-G) centroid)
#   - batch_size: 256               (same as loss_reweight_cum baseline)
#   - cumulative LoRA, sigma_fraction 0.5
#
# Layout mirrors loss_reweight_cum_round_sweep/r10 for fair comparison:
#
#   results/position_only_cum_round_sweep/r10/
#   scripts/figures/position_only_cum_round_sweep/r10/
#
# Usage (on doob, from repo root):
#   nohup bash scripts/run_position_only_cum_r10.sh \
#       > results/position_only_cum_round_sweep/r10/launch.log 2>&1 &

set -euo pipefail

CONFIG="configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml"
RUN_DIR="results/position_only_cum_round_sweep/r10"
FIG_DIR="scripts/figures/position_only_cum_round_sweep/r10"
TITLE="position_only matched r10"

PY="${PYTHON_PREFIX:+env ${PYTHON_PREFIX}} ${PY:-.venv/bin/python}"

mkdir -p "${RUN_DIR}" "${FIG_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "CONFIG not found: ${CONFIG}" >&2
    exit 1
fi

if [[ -f "${RUN_DIR}/history.json" ]]; then
    echo "[skip-train] ${RUN_DIR}/history.json already exists"
else
    echo "[train] ${RUN_DIR}"
    ${PY} -m infl_ens.training \
        --config "${CONFIG}" \
        "output_dir=${RUN_DIR}" \
        "closed_loop.sft.output_dir=${RUN_DIR}/agents" \
        "closed_loop.n_rounds=10" \
        2>&1 | tee "${RUN_DIR}/training.log"
fi

echo "[plot] ${FIG_DIR}/trajectory"
${PY} scripts/plot_closed_loop_history.py \
    --history "${RUN_DIR}/history.json" \
    --axis-labels harm hallucination \
    --title "${TITLE}" \
    --output-stem "${FIG_DIR}/trajectory"

echo "[plot] ${FIG_DIR}/theory_vs_sft"
${PY} scripts/compare_theory_vs_sft.py \
    --config "${CONFIG}" \
    --history "${RUN_DIR}/history.json" \
    --axis-labels harm hallucination \
    --title "${TITLE}  theory vs SFT" \
    --output-stem "${FIG_DIR}/theory_vs_sft" \
    --summary-json "${RUN_DIR}/theory_vs_sft.json"

echo "[plot] ${FIG_DIR}/probe"
${PY} scripts/probe_sft_capability.py \
    --run-dir "${RUN_DIR}" \
    --output-stem "${FIG_DIR}/probe"

echo "done: ${RUN_DIR}"
echo "figures: ${FIG_DIR}/"
