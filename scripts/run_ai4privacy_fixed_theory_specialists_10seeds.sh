#!/usr/bin/env bash
set -euo pipefail

ROOT="results/ai4privacy_fixed_theory_specialists_r40"
CONFIG="${ROOT}/config.yaml"

cd "$(dirname "$0")/.."
mkdir -p "${ROOT}/logs"

for SEED in 0 1 2 3 4 5 6 7 8 9; do
  OUT="${ROOT}/seed${SEED}"
  LOG="${ROOT}/logs/seed${SEED}.log"
  if [[ -f "${OUT}/history.json" ]]; then
    echo "seed ${SEED}: history exists, skipping" | tee -a "${ROOT}/logs/run_all.log"
    continue
  fi

  echo "seed ${SEED}: start $(date)" | tee -a "${ROOT}/logs/run_all.log"
  .venv/bin/python -m infl_ens.training \
    --config "${CONFIG}" \
    "seed=${SEED}" \
    "output_dir=${OUT}" \
    "closed_loop.sft.output_dir=${OUT}/agents" \
    "closed_loop.sft.seed=${SEED}" \
    > "${LOG}" 2>&1
  code=$?
  echo "seed ${SEED}: exit ${code} $(date)" | tee -a "${ROOT}/logs/run_all.log"
  if [[ "${code}" -ne 0 ]]; then
    exit "${code}"
  fi
done

echo "all seeds done $(date)" | tee -a "${ROOT}/logs/run_all.log"
