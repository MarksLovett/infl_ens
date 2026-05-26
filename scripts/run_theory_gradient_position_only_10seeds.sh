#!/usr/bin/env bash
# Position-only loop (no SFT): theory_gradient init, 10 seeds, theory vs sim plots.
#
#   results/theory_grad_position_only_10seeds/seed{0..9}/history.json
#   scripts/figures/theory_grad_position_only_10seeds/seed{0..9}/theory_vs_sft.{pdf,png}
#
# Usage:
#   bash scripts/run_theory_gradient_position_only_10seeds.sh
#   PLOT_ONLY=1 bash scripts/run_theory_gradient_position_only_10seeds.sh

set -euo pipefail

CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
SIGMA_FRACTION="${SIGMA_FRACTION:-0.5}"
N_ROUNDS="${N_ROUNDS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
INIT_NOISE="${INIT_NOISE:-0}"
SWEEP_ROOT="${SWEEP_ROOT:-results/theory_grad_position_only_10seeds}"
FIG_ROOT="${FIG_ROOT:-scripts/figures/theory_grad_position_only_10seeds}"
PLOT_ONLY="${PLOT_ONLY:-0}"
PY="${PY:-.venv/bin/python}"

mkdir -p "${SWEEP_ROOT}" "${FIG_ROOT}"

echo "================================================================"
echo "  position-only + theory_gradient init"
echo "  seeds=${SEEDS}  sigma_fraction=${SIGMA_FRACTION}  rounds=${N_ROUNDS}"
echo "  results: ${SWEEP_ROOT}/"
echo "  figures: ${FIG_ROOT}/"
echo "================================================================"

for seed in ${SEEDS}; do
  run_dir="${SWEEP_ROOT}/seed${seed}"
  fig_dir="${FIG_ROOT}/seed${seed}"
  mkdir -p "${run_dir}" "${fig_dir}"

  if [[ "${PLOT_ONLY}" != "1" ]]; then
    if [[ -f "${run_dir}/history.json" ]]; then
      echo "[skip-sim] ${run_dir}/history.json"
    else
      echo "[sim] seed=${seed}"
      ${PY} scripts/simulate_position_only_loop.py \
        --config "${CONFIG}" \
        --output-dir "${run_dir}" \
        --sigma-fraction "${SIGMA_FRACTION}" \
        --seed "${seed}" \
        --n-rounds "${N_ROUNDS}" \
        --batch-size "${BATCH_SIZE}" \
        --blend 0.5 \
        --centroid-mode expected_pool \
        --init-mode theory_gradient \
        --init-noise "${INIT_NOISE}" \
        --position-step-mode static
    fi
  fi

  if [[ ! -f "${run_dir}/history.json" ]]; then
    echo "missing ${run_dir}/history.json — skip plot" >&2
    continue
  fi

  echo "[plot] seed=${seed} -> ${fig_dir}/theory_vs_sft"
  ${PY} scripts/compare_theory_vs_sft.py \
    --config "${CONFIG}" \
    --history "${run_dir}/history.json" \
    --sigma-fraction-override "${SIGMA_FRACTION}" \
    --seed "${seed}" \
    --axis-labels harm hallucination \
    --title "theory_gradient position-only seed${seed} (σ=${SIGMA_FRACTION}×σ₀*)" \
    --output-stem "${fig_dir}/theory_vs_sft" \
    --summary-json "${run_dir}/theory_vs_sft.json" \
    > /dev/null
done

echo
echo "done. figures under ${FIG_ROOT}/seed*/theory_vs_sft.{pdf,png}"
