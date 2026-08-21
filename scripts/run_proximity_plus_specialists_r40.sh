#!/usr/bin/env bash
# 40-round proximity merge + per-clone specialists (same init as pairs_near_eq).

set -euo pipefail

CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r40_proximity_plus_specialists_cum.yaml}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
SWEEP_NAME="${SWEEP_NAME:-proximity_plus_specialists_r40}"
N_ROUNDS="${N_ROUNDS:-40}"
PY="${PY:-.venv/bin/python}"

for seed in ${SEEDS}; do
    run_dir="results/${SWEEP_NAME}/seed${seed}"
    mkdir -p "${run_dir}"
    if [[ -f "${run_dir}/history.json" ]]; then
        echo "[skip] ${run_dir}/history.json"
        continue
    fi
    echo "[train] ${run_dir} seed=${seed}"
    ${PY} -m infl_ens.training \
        --config "${CONFIG}" \
        "output_dir=${run_dir}" \
        "closed_loop.sft.output_dir=${run_dir}/agents" \
        "seed=${seed}" \
        "closed_loop.sft.seed=${seed}" \
        "closed_loop.n_rounds=${N_ROUNDS}" \
        2>&1 | tee "${run_dir}/training.log"
done

echo "proximity+specialists sweep done."
