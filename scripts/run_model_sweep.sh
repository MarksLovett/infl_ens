#!/usr/bin/env bash
# scripts/run_model_sweep.sh
#
# Model scale-family sweep: run the SAME closed-loop training pipeline on
# 9 cells (3 families x 3 scales), changing ONLY the agent base model.
#
# HANDOFF MODEL (git, not scp): this script is meant to be run *on doob*
# after `git pull` of the denislim/model-sweep branch. Unlike
# run_soft_smoke_on_doob.sh, it does NOT scp/ssh from a laptop -- it assumes
# it is already executing inside the repo on the GPU host. Per AGENTS.md
# rule 9, GPU training runs on mlovett@doob.dartmouth.edu; the datasets and
# the fingerprinted trait-space cache already live there under data/.
#
# Prerequisites on doob:
#   * `git checkout denislim/model-sweep && git pull`
#   * `huggingface-cli login` with a token that has accepted the Llama
#     3.1/3.2 and Gemma 2/3 licenses (Qwen is open). Otherwise the gated
#     from_pretrained calls 401.
#   * The .venv used elsewhere in the repo (transformers/peft/trl/datasets).
#
# Usage:
#   nohup bash scripts/run_model_sweep.sh > model_sweep.log 2>&1 &
#   CELLS="qwen_1b llama_1b gemma_1b" bash scripts/run_model_sweep.sh   # subset
#   SKIP_EVAL=1 bash scripts/run_model_sweep.sh                          # train only
#
# Environment variables (all optional):
#   GPU        CUDA device to pin (default: 0).
#   PY         Python interpreter (default: .venv/bin/python).
#   SEEDS      space-separated seeds (default: "0").
#   CELLS      space-separated cell keys to run (default: all 9).
#   RESULTS_ROOT  sweep results root (default: results/model_sweep).
#   SKIP_EVAL  set to 1 to skip the per-cell final-round eval.
#   SKIP_EXISTING  set to 0 to force retrain cells with a history.json
#                  (default: 1 -- resumable, skip completed cells).

set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
PY="${PY:-.venv/bin/python}"
SEEDS="${SEEDS:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-results/model_sweep}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

export PYTHONPATH="${PYTHONPATH:-src}"
export CUDA_VISIBLE_DEVICES="${GPU}"

BASE_CFG="configs/benchmark/router/model_sweep_base.yaml"
EVAL_CFG="configs/evaluation/run_final_round.yaml"

# --- The 9 cells: cell-key -> HuggingFace repo id --------------------------
# Cell keys are "<family>_<tier>" where tier in {1b,3b,8b}. The tier suffix
# selects the per-device micro-batch below (memory budget), NOT the exact
# parameter count.
declare -A REPOS=(
  [qwen_1b]="Qwen/Qwen2.5-1.5B-Instruct"
  [qwen_3b]="Qwen/Qwen2.5-3B-Instruct"
  [qwen_8b]="Qwen/Qwen2.5-7B-Instruct"
  [llama_1b]="meta-llama/Llama-3.2-1B-Instruct"
  [llama_3b]="meta-llama/Llama-3.2-3B-Instruct"
  [llama_8b]="meta-llama/Llama-3.1-8B-Instruct"
  [gemma_1b]="google/gemma-3-1b-it"
  [gemma_3b]="google/gemma-3-4b-it"
  [gemma_8b]="google/gemma-2-9b-it"
)

# Per-tier per-device micro-batch, sized to fit alongside the resident 8B
# INT4 trait-space embedder on a 40GB card.
declare -A PDBS_BY_TIER=( [1b]=16 [3b]=8 [8b]=4 )

# Deterministic cell order (small tiers first: cheapest, surfaces plumbing
# bugs -- gated auth, chat templates -- before the multi-hour large tier).
ALL_CELLS=(
  qwen_1b llama_1b gemma_1b
  qwen_3b llama_3b gemma_3b
  qwen_8b llama_8b gemma_8b
)
CELLS="${CELLS:-${ALL_CELLS[*]}}"

echo "[sweep] GPU=${GPU} PY=${PY} SEEDS='${SEEDS}' cells='${CELLS}'"

for cell in ${CELLS}; do
  repo="${REPOS[$cell]:-}"
  if [[ -z "${repo}" ]]; then
    echo "[sweep] WARN unknown cell '${cell}', skipping" >&2
    continue
  fi
  tier="${cell##*_}"
  pdbs="${PDBS_BY_TIER[$tier]:-8}"

  for seed in ${SEEDS}; do
    out="${RESULTS_ROOT}/${cell}/seed${seed}"
    hist="${out}/history.json"

    if [[ "${SKIP_EXISTING}" == "1" && -f "${hist}" ]]; then
      echo "[sweep] skip ${cell} seed${seed} (history.json exists)"
    else
      echo "[sweep] TRAIN ${cell} seed${seed} model=${repo} pdbs=${pdbs}"
      ${PY} -m infl_ens.training --config "${BASE_CFG}" \
        closed_loop.sft.base_model="${repo}" \
        closed_loop.sft.per_device_batch_size="${pdbs}" \
        output_dir="${out}" \
        closed_loop.sft.output_dir="${out}/agents" \
        seed="${seed}" \
        closed_loop.sft.seed="${seed}"
    fi

    if [[ "${SKIP_EVAL}" != "1" ]]; then
      eval_out="${out}/eval_final"
      if [[ -f "${eval_out}/eval_results.json" ]]; then
        echo "[sweep] skip eval ${cell} seed${seed} (eval_results.json exists)"
      else
        echo "[sweep] EVAL ${cell} seed${seed}"
        ${PY} -m infl_ens.evaluation --config "${EVAL_CFG}" \
          base_model="${repo}" \
          run_dir="${out}" \
          output_dir="${eval_out}"
      fi
    fi
  done
done

# --- Roll the per-cell eval JSONs into one family x scale table ------------
if [[ "${SKIP_EVAL}" != "1" ]]; then
  echo "[sweep] summarise"
  ${PY} scripts/summarize_model_sweep.py \
    --results-root "${RESULTS_ROOT}" \
    --seeds ${SEEDS}
fi

echo "[sweep] done. Results under ${RESULTS_ROOT}/"
