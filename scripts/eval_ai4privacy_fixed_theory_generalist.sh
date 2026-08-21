#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
ROOT="${ROOT:-results/ai4privacy_fixed_theory_generalist_r40}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
LOG_DIR="${ROOT}/logs"

mkdir -p "${LOG_DIR}"

for SEED in ${SEEDS}; do
  RUN="${ROOT}/seed${SEED}"
  OUT="${RUN}/eval_final_round"
  CFG="${ROOT}/eval_seed${SEED}.yaml"

  if [[ ! -d "${RUN}/agents/generalist/round-39" ]]; then
    echo "seed ${SEED}: missing ${RUN}/agents/generalist/round-39, skipping" \
      | tee -a "${LOG_DIR}/eval_all.log"
    continue
  fi
  if [[ -f "${OUT}/eval_results.json" ]]; then
    echo "seed ${SEED}: eval exists, skipping" | tee -a "${LOG_DIR}/eval_all.log"
    continue
  fi

  echo "seed ${SEED}: eval start $(date)" | tee -a "${LOG_DIR}/eval_all.log"
  cat > "${CFG}" <<YAML
task: run_eval
seed: ${SEED}
output_dir: ${OUT}
base_model: Qwen/Qwen2.5-1.5B-Instruct
run_dir: ${RUN}
agents: [generalist]
rounds: [39]
benchmarks:
  - kind: beavertails
    path: data/beavertails/30k_train.jsonl
    max_records: 5000
  - kind: halueval
    path: data/halueval
    tasks: [qa, dialogue]
    max_records: 5000
  - kind: toxicchat
    path: data/toxicchat
    score_mode: jailbreaking
    human_annotated_only: false
    max_records: 5000
  - kind: ai4privacy
    path: data/ai4privacy
    score_mode: density
    english_only: true
    max_records: 5000
eval:
  max_seq_length: 1024
  forward_batch_size: 8
  max_eval_records: 256
YAML

  "${PY}" -m infl_ens.evaluation --config "${CFG}" \
    > "${LOG_DIR}/eval_seed${SEED}.log" 2>&1
  echo "seed ${SEED}: eval done $(date)" | tee -a "${LOG_DIR}/eval_all.log"
done

echo "all generalist evals done $(date)" | tee -a "${LOG_DIR}/eval_all.log"
