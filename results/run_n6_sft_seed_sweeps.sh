#!/usr/bin/env bash
set -euo pipefail

cd /home/mlovett/infl_ens

PY="${PY:-.venv/bin/python}"
CONFIG="configs/benchmark/router/safety_truth_ai4privacy_n6_theory_only_sigma04.yaml"
GEN_CONFIG="configs/benchmark/router/ai4privacy_fixed_theory_generalist_replay_r40.yaml"
FIG_ROOT="scripts/figures/ai4privacy_n6_sft_sigma05_stretch_h25_p25"
BASE_EVAL="results/ai4privacy_fixed_theory_specialists_r40/base_eval_matched.json"
AGENTS_JSON='["clone-0","clone-1","clone-2","clone-3","clone-4","clone-5"]'

write_eval_config() {
  local out_path="$1"
  local output_dir="$2"
  local run_dir="$3"
  local agents_json="$4"
  local round_idx="$5"
  cat > "${out_path}" <<JSON
{
  "task": "run_eval",
  "seed": 0,
  "output_dir": "${output_dir}",
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "run_dir": "${run_dir}",
  "agents": ${agents_json},
  "rounds": [${round_idx}],
  "benchmarks": [
    {
      "kind": "beavertails",
      "path": "data/beavertails/30k_train.jsonl",
      "max_records": 5000
    },
    {
      "kind": "halueval",
      "path": "data/halueval",
      "tasks": ["qa", "dialogue"],
      "max_records": 5000
    },
    {
      "kind": "toxicchat",
      "path": "data/toxicchat",
      "score_mode": "jailbreaking",
      "human_annotated_only": false,
      "max_records": 5000
    },
    {
      "kind": "ai4privacy",
      "path": "data/ai4privacy",
      "score_mode": "density",
      "english_only": true,
      "max_records": 5000
    }
  ],
  "eval": {
    "max_seq_length": 1024,
    "forward_batch_size": 8,
    "max_eval_records": 256
  }
}
JSON
}

make_mean_position_overlay() {
  local spec_root="$1"
  local out_path="$2"
  "${PY}" - "$spec_root" "$out_path" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
out_path = Path(sys.argv[2])
by_agent: dict[str, list[np.ndarray]] = {}
for history_path in sorted(root.glob("seed*/history.json")):
    records = json.loads(history_path.read_text(encoding="utf-8"))
    if not records:
        continue
    for name, pos in records[-1]["positions"].items():
        by_agent.setdefault(name, []).append(np.asarray(pos, dtype=float))

positions = {
    name: np.stack(vals, axis=0).mean(axis=0).tolist()
    for name, vals in sorted(by_agent.items())
    if vals
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({"positions": positions}, indent=2), encoding="utf-8")
print(f"wrote {out_path}")
PY
}

run_round_sweep() {
  local rounds="$1"
  local round_idx=$((rounds - 1))
  local spec_root="results/ai4privacy_n6_sft_r${rounds}_sigma05_stretch_h25_p25"
  local gen_root="${spec_root}_generalist"
  local fig_dir="${FIG_ROOT}/r${rounds}"
  mkdir -p "${spec_root}/logs" "${gen_root}/logs" "${fig_dir}"

  echo "[r${rounds}] start $(date)" | tee -a "${spec_root}/logs/run_all.log"

  for seed in 0 1 2 3 4 5 6 7 8 9; do
    local run="${spec_root}/seed${seed}"
    local gen="${gen_root}/seed${seed}"
    mkdir -p "${run}" "${gen}"

    if [[ ! -f "${run}/history.json" ]]; then
      echo "[r${rounds}] seed${seed}: train specialists $(date)" | tee -a "${spec_root}/logs/run_all.log"
      "${PY}" -m infl_ens.training \
        --config "${CONFIG}" \
        "seed=${seed}" \
        "output_dir=${run}" \
        "sigma_fraction=0.5" \
        "closed_loop.n_rounds=${rounds}" \
        "closed_loop.batch_size=512" \
        "closed_loop.blend=0.5" \
        "closed_loop.init_noise=0.0" \
        "closed_loop.init_mode=theory_gradient_paired" \
        "closed_loop.centroid_mode=expected_pool" \
        "closed_loop.routing_weight=G" \
        "closed_loop.loss_reweight=position_only" \
        "closed_loop.save_per_round=true" \
        "closed_loop.sft.base_model=Qwen/Qwen2.5-1.5B-Instruct" \
        "closed_loop.sft.output_dir=${run}/agents" \
        "closed_loop.sft.max_seq_length=1024" \
        "closed_loop.sft.per_device_batch_size=8" \
        "closed_loop.sft.gradient_accumulation_steps=2" \
        "closed_loop.sft.num_train_epochs=1.0" \
        "closed_loop.sft.learning_rate=2.0e-4" \
        "closed_loop.sft.lora_r=16" \
        "closed_loop.sft.lora_alpha=32" \
        "closed_loop.sft.lora_dropout=0.05" \
        "closed_loop.sft.bf16=true" \
        "closed_loop.sft.gradient_checkpointing=true" \
        "closed_loop.sft.logging_steps=1" \
        "closed_loop.sft.cumulative_lora=true" \
        "closed_loop.sft.seed=${seed}" \
        > "${spec_root}/logs/seed${seed}_train.log" 2>&1
    else
      echo "[r${rounds}] seed${seed}: specialist history exists, skip" | tee -a "${spec_root}/logs/run_all.log"
    fi

    if [[ ! -f "${gen}/replay_summary.json" ]]; then
      echo "[r${rounds}] seed${seed}: train generalist replay $(date)" | tee -a "${gen_root}/logs/run_all.log"
      "${PY}" -m infl_ens.training \
        --config "${GEN_CONFIG}" \
        "history_path=${run}/history.json" \
        "output_dir=${gen}" \
        "sft.output_dir=${gen}/agents" \
        "seed=${seed}" \
        "sft.seed=${seed}" \
        "sft.num_train_epochs=1.0" \
        > "${gen_root}/logs/seed${seed}_replay.log" 2>&1
    else
      echo "[r${rounds}] seed${seed}: generalist replay exists, skip" | tee -a "${gen_root}/logs/run_all.log"
    fi

    if [[ ! -f "${run}/eval_final_round/eval_results.json" ]]; then
      echo "[r${rounds}] seed${seed}: eval specialists $(date)" | tee -a "${spec_root}/logs/run_all.log"
      write_eval_config \
        "${spec_root}/eval_seed${seed}.json" \
        "${run}/eval_final_round" \
        "${run}" \
        "${AGENTS_JSON}" \
        "${round_idx}"
      "${PY}" -m infl_ens.evaluation \
        --config "${spec_root}/eval_seed${seed}.json" \
        > "${spec_root}/logs/seed${seed}_eval.log" 2>&1
    else
      echo "[r${rounds}] seed${seed}: specialist eval exists, skip" | tee -a "${spec_root}/logs/run_all.log"
    fi

    if [[ ! -f "${gen}/eval_final_round/eval_results.json" ]]; then
      echo "[r${rounds}] seed${seed}: eval generalist $(date)" | tee -a "${gen_root}/logs/run_all.log"
      write_eval_config \
        "${gen_root}/eval_seed${seed}.json" \
        "${gen}/eval_final_round" \
        "${gen}" \
        '["generalist"]' \
        "${round_idx}"
      "${PY}" -m infl_ens.evaluation \
        --config "${gen_root}/eval_seed${seed}.json" \
        > "${gen_root}/logs/seed${seed}_eval.log" 2>&1
    else
      echo "[r${rounds}] seed${seed}: generalist eval exists, skip" | tee -a "${gen_root}/logs/run_all.log"
    fi
  done

  cp "${BASE_EVAL}" "${spec_root}/base_eval_matched.json"

  echo "[r${rounds}] plot benchmark comparison $(date)" | tee -a "${spec_root}/logs/run_all.log"
  "${PY}" scripts/plot_ai4privacy_fixed_vs_base_figure.py \
    --sweep-root "${spec_root}" \
    --generalist-root "${gen_root}" \
    --agents clone-0,clone-1,clone-2,clone-3,clone-4,clone-5 \
    --round-label "round ${round_idx}" \
    --output-stem "${fig_dir}/n6_sft_r${rounds}_sigma05_vs_generalist" \
    > "${spec_root}/logs/plot_compare.log" 2>&1

  local pos_json="${spec_root}/mean_final_positions.json"
  make_mean_position_overlay "${spec_root}" "${pos_json}"

  echo "[r${rounds}] plot mean positions on heatmap $(date)" | tee -a "${spec_root}/logs/run_all.log"
  "${PY}" scripts/plot_benchmark_space_heatmaps.py \
    --config "${CONFIG}" \
    --positions-json "${pos_json}" \
    --output-stem "${fig_dir}/n6_sft_r${rounds}_sigma05_mean_positions_on_resource" \
    --max-scatter 3000 \
    --title "AI4Privacy N=6 SFT r${rounds}, mean final positions over 10 seeds" \
    > "${spec_root}/logs/plot_positions_heatmap.log" 2>&1

  echo "[r${rounds}] done $(date)" | tee -a "${spec_root}/logs/run_all.log"
}

run_round_sweep 10
run_round_sweep 40

echo "all requested sweeps complete $(date)"
