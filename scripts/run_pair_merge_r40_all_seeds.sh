#!/usr/bin/env bash
# 40-round pair-merge closed loop: 10 seeds (theory_gradient + expected_pool).
#
# Usage:
#   nohup bash scripts/run_pair_merge_r40_all_seeds.sh \
#       > results/pair_merge_r40_sweep.log 2>&1 &

set -euo pipefail

CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r40_pair_merge_cum.yaml}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
SWEEP_NAME="${SWEEP_NAME:-pair_merge_round_sweep}"
N_ROUNDS="${N_ROUNDS:-40}"
PY="${PY:-.venv/bin/python}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "CONFIG not found: ${CONFIG}" >&2
    exit 1
fi

for seed in ${SEEDS}; do
    run_dir="results/${SWEEP_NAME}/r${N_ROUNDS}/seed${seed}"
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

echo "pair-merge training sweep done."
