#!/usr/bin/env bash
# scripts/run_trait_representation_on_doob.sh
#
# Sync the repo to doob and build trait-space data-representation figures
# there, contrasting the legacy clipped calibration against the always-on
# quantile normalization.
#
# Per AGENTS.md §4 rule 9 — work that needs the GPU and the datasets goes
# through mlovett@doob.dartmouth.edu after a code sync. The datasets and
# the fingerprinted trait-space cache live only on the remote.
#
# The trait-space cache version bumped 2 -> 3, so the first run performs a
# FULL re-encode of the corpus with Qwen3-Embedding-8B. That takes hours,
# so the remote work is launched detached under nohup and polled, rather
# than held open on a single SSH connection.
#
# Usage:
#   bash scripts/run_trait_representation_on_doob.sh            # sync + launch
#   MODE=status bash scripts/run_trait_representation_on_doob.sh
#   MODE=pull   bash scripts/run_trait_representation_on_doob.sh
#
#   SMOKE=1 bash scripts/run_trait_representation_on_doob.sh    # 2-axis gate,
#       # runs in the foreground on the cheap safety_truth config first
#
# Environment variables:
#   CONFIG       Router YAML (default: seven-axis pair-merge split).
#   REMOTE       SSH target (default: mlovett@doob.dartmouth.edu).
#   REMOTE_REPO  Path on the remote (default: infl_ens).
#   MODE         launch (default) | status | pull
#   SMOKE=1      Run the cheap 2-axis validation instead of the full job.
#   MAX_PROMPTS  Prompts sampled for the figures (default: 8000).
#   SKIP_SYNC=1  Skip the scp code push.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/benchmark/router/seven_axis_pair_merge_split.yaml}"
REMOTE="${REMOTE:-mlovett@doob.dartmouth.edu}"
REMOTE_REPO="${REMOTE_REPO:-infl_ens}"
MODE="${MODE:-launch}"
MAX_PROMPTS="${MAX_PROMPTS:-8000}"
TMUX_SESSION="${TMUX_SESSION:-trait-repr}"
FIG_SUBDIR="scripts/figures/trait_repr"
LOG="results/trait_repr/run.log"

if [[ "${SMOKE:-0}" == "1" ]]; then
    CONFIG="configs/benchmark/router/safety_truth.yaml"
    MAX_PROMPTS="200"
fi

# ----------------------------------------------------------------------
# status / pull modes: no sync, no launch.
# ----------------------------------------------------------------------
if [[ "${MODE}" == "status" ]]; then
    echo "[status] tmux sessions on ${REMOTE}"
    ssh "${REMOTE}" "tmux ls 2>/dev/null || echo '(no tmux sessions - job finished or not started)'"
    echo
    echo "[status] tail of ${LOG}"
    ssh "${REMOTE}" "cd ${REMOTE_REPO} && tail -n 30 ${LOG} 2>/dev/null || echo 'no log yet'"
    echo
    echo "[status] GPU"
    ssh "${REMOTE}" "nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader"
    echo
    echo "[status] figures produced so far"
    ssh "${REMOTE}" "cd ${REMOTE_REPO} && ls -la ${FIG_SUBDIR}/ 2>/dev/null || echo 'none yet'"
    echo
    echo "  WATCH LIVE: ssh -t ${REMOTE} 'tmux attach -t ${TMUX_SESSION}'"
    exit 0
fi

if [[ "${MODE}" == "pull" ]]; then
    echo "[pull] figures + summary -> local ${FIG_SUBDIR}/"
    mkdir -p "${FIG_SUBDIR}" results/trait_repr
    # Figures and the JSON summary only; never the LoRA adapters or the
    # trait-space cache (those stay where they were computed).
    scp "${REMOTE}:${REMOTE_REPO}/${FIG_SUBDIR}/*.pdf" "${FIG_SUBDIR}/" 2>/dev/null || true
    scp "${REMOTE}:${REMOTE_REPO}/${FIG_SUBDIR}/*.png" "${FIG_SUBDIR}/" 2>/dev/null || true
    scp "${REMOTE}:${REMOTE_REPO}/${FIG_SUBDIR}/trait_repr_summary.json" \
        "${FIG_SUBDIR}/" 2>/dev/null || true
    scp "${REMOTE}:${REMOTE_REPO}/${LOG}" results/trait_repr/ 2>/dev/null || true
    echo "[pull] local contents:"
    ls -la "${FIG_SUBDIR}/" || true
    exit 0
fi

# ----------------------------------------------------------------------
# 1. Sync code via scp -r (no rsync locally).
#
# scp -r has no --exclude, so purge local __pycache__ first; otherwise
# Windows .pyc trees get shipped and clutter the remote.
# ----------------------------------------------------------------------
if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
    echo "[sync] purge local __pycache__"
    find src scripts tests -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

    echo "[sync] ensure remote tree exists"
    ssh "${REMOTE}" "mkdir -p ${REMOTE_REPO} && cd ${REMOTE_REPO} && \
        mkdir -p src scripts configs tests results scripts/figures ${FIG_SUBDIR} results/trait_repr"

    echo "[sync] purge remote __pycache__ so stale .pyc cannot shadow new code"
    ssh "${REMOTE}" "find ${REMOTE_REPO}/src ${REMOTE_REPO}/scripts -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null; true"

    echo "[sync] scp -r src scripts configs tests -> ${REMOTE}:${REMOTE_REPO}"
    scp -q -r src     "${REMOTE}:${REMOTE_REPO}/"
    scp -q -r scripts "${REMOTE}:${REMOTE_REPO}/"
    scp -q -r configs "${REMOTE}:${REMOTE_REPO}/"
    scp -q -r tests   "${REMOTE}:${REMOTE_REPO}/"
    scp -q    AGENTS.md structure.md pyproject.toml "${REMOTE}:${REMOTE_REPO}/"
    echo "[sync] done"
fi

# ----------------------------------------------------------------------
# 2. Remote work.
#
# SMOKE runs in the foreground (cheap, 2 axes, cache disabled) as a gate.
# The full job is launched detached because the cache rebuild is hours.
# ----------------------------------------------------------------------
if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "[remote] SMOKE gate: 2-axis, ${MAX_PROMPTS} prompts, foreground"
    ssh "${REMOTE}" bash -s <<REMOTE_EOF
set -euo pipefail
cd "${REMOTE_REPO}"
export PYTHONPATH=src
PY="\${PY:-.venv/bin/python}"

echo "[remote] python: \$(\${PY} --version)"
echo "[remote] pytest (offline suite)"
\${PY} -m pytest tests/test_trait_normalize.py tests/test_safety_trait_space.py \
    tests/test_trait_space_cache.py -q

echo "[remote] trait representation on ${CONFIG}"
\${PY} scripts/plot_trait_representation.py \
    --config "${CONFIG}" \
    --max-prompts ${MAX_PROMPTS} \
    --output-dir ${FIG_SUBDIR}_smoke
REMOTE_EOF
    echo "[smoke] complete. Re-run without SMOKE=1 to launch the full job."
    exit 0
fi

echo "[remote] launching full job in tmux session '${TMUX_SESSION}' (log: ${LOG})"
ssh "${REMOTE}" bash -s <<REMOTE_EOF
set -euo pipefail
cd "${REMOTE_REPO}"
mkdir -p results/trait_repr ${FIG_SUBDIR}

if tmux has-session -t ${TMUX_SESSION} 2>/dev/null; then
    echo "[remote] tmux session '${TMUX_SESSION}' ALREADY EXISTS; not starting another."
    echo "[remote] attach with: tmux attach -t ${TMUX_SESSION}"
    exit 0
fi

cat > results/trait_repr/_job.sh <<'JOB'
set -euo pipefail
export PYTHONPATH=src
PY="\${PY:-.venv/bin/python}"

echo "=== \$(date -Is) starting ==="
echo "[job] python: \$(\${PY} --version)"

echo "=== \$(date -Is) pytest ==="
\${PY} -m pytest tests/test_trait_normalize.py tests/test_safety_trait_space.py \
    tests/test_trait_space_cache.py -q

echo "=== \$(date -Is) trait representation (builds cache; slow) ==="
\${PY} scripts/plot_trait_representation.py \
    --config "__CONFIG__" \
    --max-prompts __MAX_PROMPTS__ \
    --output-dir __FIGDIR__

echo "=== \$(date -Is) resource-density slices ==="
SPLIT_ARG=""
if [ -f data/splits/seven_axis_seed0.json ]; then
    SPLIT_ARG="--split-manifest data/splits/seven_axis_seed0.json"
fi
\${PY} scripts/plot_benchmark_space_heatmaps.py \
    --config "__CONFIG__" \
    \${SPLIT_ARG} \
    --density-mode empirical \
    --pairwise-bins 64 \
    --smooth-sigma 1.2 \
    --mass-norm per_panel_power \
    --vmax-percentile 98 \
    --max-scatter 2500 \
    --scatter-alpha 0.08 \
    --dpi 220 \
    --title "Seven-axis trait space under quantile normalization" \
    --output-stem __FIGDIR__/seven_axis_resource_density

echo "=== \$(date -Is) DONE ==="
JOB

sed -i "s|__CONFIG__|${CONFIG}|g; s|__MAX_PROMPTS__|${MAX_PROMPTS}|g; s|__FIGDIR__|${FIG_SUBDIR}|g" \
    results/trait_repr/_job.sh

# Window 0 runs the job under 'tee' so the log persists even if the
# session is killed; window 1 watches the GPU. Mirrors the pattern in
# scripts/tmux_monitor_seeds.sh.
tmux new-session -d -s ${TMUX_SESSION} -n job \
    "bash results/trait_repr/_job.sh 2>&1 | tee ${LOG}; echo; echo '[job finished - press any key]'; read -n 1"
tmux new-window -t ${TMUX_SESSION} -n gpu "watch -n 5 nvidia-smi"
tmux select-window -t ${TMUX_SESSION}:job

echo "[remote] tmux session '${TMUX_SESSION}' started"
tmux ls
sleep 5
echo "--- first lines of log ---"
head -n 20 ${LOG} 2>/dev/null || echo "(log still starting)"
REMOTE_EOF

echo
echo "Launched in tmux. The cache rebuild takes hours (full Qwen3-8B encode)."
echo
echo "  WATCH LIVE :  ssh -t ${REMOTE} 'tmux attach -t ${TMUX_SESSION}'"
echo "                (window 0 = job, window 1 = nvidia-smi;"
echo "                 Ctrl-b n switches window, Ctrl-b d detaches)"
echo
echo "  monitor    :  MODE=status bash scripts/run_trait_representation_on_doob.sh"
echo "  collect    :  MODE=pull   bash scripts/run_trait_representation_on_doob.sh"
