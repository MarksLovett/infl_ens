#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="results/ai4privacy_fixed_theory_specialists_r40"
FIG_ROOT="scripts/figures/ai4privacy_fixed_theory_specialists_r40"
LOG_DIR="${ROOT}/logs"
mkdir -p "${FIG_ROOT}" "${LOG_DIR}"

echo "postprocess: waiting for 10 histories at $(date)" | tee -a "${LOG_DIR}/postprocess.log"
while true; do
  done_count=0
  for SEED in 0 1 2 3 4 5 6 7 8 9; do
    if [[ -f "${ROOT}/seed${SEED}/history.json" ]]; then
      done_count=$((done_count + 1))
    fi
  done
  echo "postprocess: ${done_count}/10 histories ready at $(date)" | tee -a "${LOG_DIR}/postprocess.log"
  if [[ "${done_count}" -eq 10 ]]; then
    break
  fi
  sleep 300
done

cat > "${ROOT}/eval_final_round.yaml" <<'YAML'
task: run_eval
seed: 0
output_dir: results/ai4privacy_fixed_theory_specialists_r40/EVAL_OUT
base_model: Qwen/Qwen2.5-1.5B-Instruct
run_dir: results/ai4privacy_fixed_theory_specialists_r40/RUN_DIR
agents: [clone-0, clone-1, clone-2, clone-3]
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

for SEED in 0 1 2 3 4 5 6 7 8 9; do
  RUN="${ROOT}/seed${SEED}"
  STEM="${FIG_ROOT}/seed${SEED}"
  mkdir -p "${STEM}"

  echo "postprocess: plotting positions seed ${SEED}" | tee -a "${LOG_DIR}/postprocess.log"
  .venv/bin/python scripts/plot_pairwise_position_updates.py \
    --history "${RUN}/history.json" \
    --axis-labels harm hallucination privacy \
    --title "AI4Privacy fixed-theory specialists seed ${SEED}" \
    --output-stem "${STEM}/position_updates" \
    > "${LOG_DIR}/plot_seed${SEED}.log" 2>&1

  echo "postprocess: final-round benchmark eval seed ${SEED}" | tee -a "${LOG_DIR}/postprocess.log"
  sed \
    -e "s#results/ai4privacy_fixed_theory_specialists_r40/EVAL_OUT#${RUN}/eval_final_round#g" \
    -e "s#results/ai4privacy_fixed_theory_specialists_r40/RUN_DIR#${RUN}#g" \
    "${ROOT}/eval_final_round.yaml" > "${ROOT}/eval_seed${SEED}.yaml"
  .venv/bin/python -m infl_ens.evaluation \
    --config "${ROOT}/eval_seed${SEED}.yaml" \
    > "${LOG_DIR}/eval_seed${SEED}.log" 2>&1

  echo "postprocess: specialization probe seed ${SEED}" | tee -a "${LOG_DIR}/postprocess.log"
  .venv/bin/python scripts/probe_sft_capability.py \
    --run-dir "${RUN}" \
    --rounds 0 9 19 29 39 \
    --max-prompts 64 \
    --max-seq-length 1024 \
    --forward-batch-size 8 \
    --output-stem "${STEM}/specialization_probe" \
    > "${LOG_DIR}/probe_seed${SEED}.log" 2>&1
done

echo "postprocess: done at $(date)" | tee -a "${LOG_DIR}/postprocess.log"
