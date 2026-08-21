#!/usr/bin/env bash
# Launch (or recreate) a tmux session with one window per seed + GPU monitor.
#
# Usage (on doob, from repo root):
#   bash scripts/tmux_monitor_seeds.sh
#   SEEDS="0 1 2" RUN_ROOT=results/my_sweep bash scripts/tmux_monitor_seeds.sh
#   GENERALIST_ROOT=results/my_sweep_generalist bash scripts/tmux_monitor_seeds.sh
#
# Attach:
#   tmux attach -t infl-seeds
#
# Windows: gpu, seed0, seed1, ...  (Ctrl-b n / Ctrl-b p to move)

set -euo pipefail

SESSION="${SESSION:-infl-seeds}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
RUN_ROOT="${RUN_ROOT:-results/ai4privacy_n6_sft_r10_sigma05_stretch_h25_p25}"
GENERALIST_ROOT="${GENERALIST_ROOT:-results/ai4privacy_n6_sft_r10_sigma05_stretch_h25_p25_generalist}"
REPO="${REPO:-$HOME/infl_ens}"

cd "${REPO}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session ${SESSION} already exists — attach with: tmux attach -t ${SESSION}"
    exit 0
fi

tmux new-session -d -s "${SESSION}" -n gpu \
    "watch -n 2 nvidia-smi"

for seed in ${SEEDS}; do
    train_log="${RUN_ROOT}/logs/seed${seed}_train.log"
    replay_log="${GENERALIST_ROOT}/logs/seed${seed}_replay.log"
    eval_log="${RUN_ROOT}/logs/seed${seed}_eval.log"
    eval_retry="${RUN_ROOT}/logs/seed${seed}_eval_retry.log"
  tmux new-window -t "${SESSION}" -n "seed${seed}" \
    "cd ${REPO} && echo '=== seed${seed} (tail -F; waiting if not started) ===' && \
     tail -F ${train_log} ${replay_log} ${eval_log} ${eval_retry} 2>/dev/null || \
     tail -f /dev/null"
done

echo "Created tmux session: ${SESSION}"
echo "  gpu window  — watch nvidia-smi every 2s"
echo "  seedN windows — train + replay + eval logs for ${RUN_ROOT}"
echo ""
echo "Attach:  tmux attach -t ${SESSION}"
echo "Detach:  Ctrl-b d"
echo "Switch:  Ctrl-b n / Ctrl-b p   or   Ctrl-b w (picker)"
