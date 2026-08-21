#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
SPEC_ROOT="${SPEC_ROOT:-results/ai4privacy_fixed_theory_specialists_r40}"
OUT_ROOT="${OUT_ROOT:-results/ai4privacy_fixed_theory_generalist_r40}"
FIG_ROOT="${FIG_ROOT:-scripts/figures/ai4privacy_fixed_theory_generalist_r40}"
CONFIG="${CONFIG:-configs/benchmark/router/ai4privacy_fixed_theory_generalist_replay_r40.yaml}"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${FIG_ROOT}"

for SEED in ${SEEDS}; do
  HIST="${SPEC_ROOT}/seed${SEED}/history.json"
  OUT="${OUT_ROOT}/seed${SEED}"
  LOG="${LOG_DIR}/seed${SEED}.log"
  STEM="${FIG_ROOT}/seed${SEED}/generalist_centroid"

  if [[ ! -f "${HIST}" ]]; then
    echo "seed ${SEED}: missing ${HIST}, skipping" | tee -a "${LOG_DIR}/run_all.log"
    continue
  fi
  if [[ -f "${OUT}/replay_summary.json" ]]; then
    echo "seed ${SEED}: replay_summary exists, skipping" | tee -a "${LOG_DIR}/run_all.log"
    continue
  fi

  echo "seed ${SEED}: start $(date)" | tee -a "${LOG_DIR}/run_all.log"
  mkdir -p "${OUT}" "$(dirname "${STEM}")"
  "${PY}" -m infl_ens.training \
    --config "${CONFIG}" \
    "history_path=${HIST}" \
    "output_dir=${OUT}" \
    "sft.output_dir=${OUT}/agents" \
    "seed=${SEED}" \
    "sft.seed=${SEED}" \
    > "${LOG}" 2>&1
  code=$?
  echo "seed ${SEED}: exit ${code} $(date)" | tee -a "${LOG_DIR}/run_all.log"
  if [[ "${code}" -ne 0 ]]; then
    exit "${code}"
  fi

  "${PY}" scripts/plot_pairwise_position_updates.py \
    --history "${OUT}/history.json" \
    --axis-labels harm hallucination privacy \
    --title "AI4Privacy fixed-theory generalist seed ${SEED}" \
    --output-stem "${STEM}" \
    > "${LOG_DIR}/plot_seed${SEED}.log" 2>&1
done

echo "all generalist seeds done $(date)" | tee -a "${LOG_DIR}/run_all.log"
