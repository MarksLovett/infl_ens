#!/usr/bin/env bash
# Evaluate final-round (round-39) adapters for all 4 agents on both benchmarks.
#
# Default run: pairs_near_eq position-only cum, 40 rounds, seed 0.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
RUN_DIR="${RUN_DIR:-results/pairs_near_eq_round_sweep/r40/seed0}"
OUT_DIR="${OUT_DIR:-results/eval_pairs_near_eq_r40_seed0_final}"

mkdir -p "${OUT_DIR}"
echo "run_dir=${RUN_DIR} output=${OUT_DIR} (round 39, clones 0-3)"

"${PY}" -m infl_ens.evaluation \
  --config configs/evaluation/run_final_round.yaml \
  "run_dir=${RUN_DIR}" \
  "output_dir=${OUT_DIR}" \
  2>&1 | tee "${OUT_DIR}/eval.log"

echo "done: ${OUT_DIR}/eval_results.json"
