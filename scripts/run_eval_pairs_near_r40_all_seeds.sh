#!/usr/bin/env bash
# Final-round eval for seeds 0..9 (skip cells that already have eval_results.json).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
SWEEP_ROOT="${SWEEP_ROOT:-results/pairs_near_eq_round_sweep/r40}"

for seed in ${SEEDS}; do
  RUN_DIR="${SWEEP_ROOT}/seed${seed}"
  OUT_DIR="results/eval_pairs_near_eq_r40_seed${seed}_final"
  if [[ -f "${OUT_DIR}/eval_results.json" ]]; then
    echo "[skip] seed${seed} ${OUT_DIR}/eval_results.json"
    continue
  fi
  if [[ ! -f "${RUN_DIR}/history.json" ]]; then
    echo "[skip] seed${seed} missing ${RUN_DIR}/history.json"
    continue
  fi
  echo "[eval] seed${seed}"
  mkdir -p "${OUT_DIR}"
  "${PY}" -m infl_ens.evaluation \
    --config configs/evaluation/run_final_round.yaml \
    "run_dir=${RUN_DIR}" \
    "output_dir=${OUT_DIR}" \
    "seed=${seed}" \
    2>&1 | tee "${OUT_DIR}/eval.log"
done
echo "all seeds done."
