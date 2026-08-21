#!/usr/bin/env bash
# Run four-way comparison per seed, then aggregate.

set -euo pipefail

PY="${PY:-.venv/bin/python}"
ROUND="${ROUND:-39}"
SPECIALIST_ROOT="${SPECIALIST_ROOT:-results/pairs_near_eq_round_sweep/r40}"
BASELINE_ROOT="${BASELINE_ROOT:-results/baseline_replay_r40}"
MERGE_ROOT="${MERGE_ROOT:-results/proximity_merge_round_sweep/r40}"
BASE_EVAL="${BASE_EVAL:-results/base_model_eval_qwen2_5_1_5b/base_eval.json}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"

for seed in ${SEEDS}; do
    merge_dir="${MERGE_ROOT}/seed${seed}"
    if ! compgen -G "${merge_dir}/agents/merge-*/round-${ROUND}" > /dev/null; then
        echo "[skip] missing merge adapters for seed${seed}" >&2
        continue
    fi
    echo "[compare] seed${seed}"
    ${PY} scripts/compare_all_r40_models.py \
        --specialist-dir "${SPECIALIST_ROOT}/seed${seed}" \
        --baseline-dir "${BASELINE_ROOT}/seed${seed}" \
        --merge-dir "${merge_dir}" \
        --base-eval-json "${BASE_EVAL}" \
        --round "${ROUND}" \
        --seed "${seed}"
done

${PY} scripts/aggregate_compare_all_seeds.py \
    --glob "${MERGE_ROOT}/seed*/compare_all_round${ROUND}.json" \
    --output "results/compare_all_r40_round${ROUND}_aggregate.json"
