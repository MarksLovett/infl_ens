#!/usr/bin/env bash
# Ablation: one LoRA per corner on merged pair batch (same prompts as merge).
# Replays SFT from proximity_plus_specialists histories, then evaluates.

set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-results/proximity_plus_specialists_r40}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
ROUND="${ROUND:-39}"
PY="${PY:-.venv/bin/python}"

for seed in ${SEEDS}; do
    run_dir="${SOURCE_ROOT}/seed${seed}"
    hist="${run_dir}/history.json"
    if [[ ! -f "${hist}" ]]; then
        echo "[skip] missing ${hist}" >&2
        continue
    fi
    out_agents="${run_dir}/agents"
    marker="${out_agents}/single-pool-low/round-${ROUND}"
    if [[ -d "${marker}" ]]; then
        echo "[skip-replay] seed${seed} single-pool adapters exist"
    else
        echo "[replay] seed${seed} corner pooled single trainer"
        ${PY} scripts/replay_corner_single_pool_ablation.py \
            --history "${hist}" \
            --output-dir "${out_agents}" \
            --seed "${seed}" \
            2>&1 | tee "${run_dir}/single_pool_replay.log"
    fi
done

echo "[eval] aggregate comparison"
${PY} scripts/replay_corner_single_pool_ablation.py \
    --compare-only \
    --source-root "${SOURCE_ROOT}" \
    --round "${ROUND}" \
    --output "results/corner_single_pool_ablation_compare.json"

echo "done."
