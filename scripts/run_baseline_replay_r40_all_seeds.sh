#!/usr/bin/env bash
# Pooled baseline replay for seeds 0..9 using logged batches from r40 runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
HIST_ROOT="${HIST_ROOT:-results/pairs_near_eq_round_sweep/r40}"
OUT_ROOT="${OUT_ROOT:-results/baseline_replay_r40}"

for seed in ${SEEDS}; do
  hist="${HIST_ROOT}/seed${seed}/history.json"
  out="${OUT_ROOT}/seed${seed}"
  if [[ ! -f "${hist}" ]]; then
    echo "[skip] missing ${hist}"
    continue
  fi
  if [[ -f "${out}/replay_summary.json" ]]; then
    echo "[skip] ${out}/replay_summary.json exists"
    continue
  fi
  echo "[replay] seed${seed}"
  mkdir -p "${out}"
  ${PY} -m infl_ens.training \
    --config configs/benchmark/router/baseline_replay_r40.yaml \
    "history_path=${hist}" \
    "output_dir=${out}" \
    "sft.output_dir=${out}/agents" \
    "seed=${seed}" \
    "sft.seed=${seed}" \
    2>&1 | tee "${out}/training.log"
done
echo "done."
