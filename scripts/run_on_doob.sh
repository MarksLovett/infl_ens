#!/usr/bin/env bash
# scripts/run_on_doob.sh -- sync the repo to the GPU host and drive
# `python -m infl_ens.pipeline` there under tmux.
#
# This is the only shell script in the repository. Everything it launches
# is the Python pipeline; the experiment (arms, stages, evaluation window,
# figures) is described by the experiment YAML, not here.
#
# Usage:
#   bash scripts/run_on_doob.sh                          # sync + queue every stage
#   STAGES=routing,figures bash scripts/run_on_doob.sh   # re-run the analysis stages
#   MODE=smoke  bash scripts/run_on_doob.sh              # cheap gate, foreground
#   MODE=status bash scripts/run_on_doob.sh              # tmux, log tail, stage status, GPU
#   MODE=pull   bash scripts/run_on_doob.sh              # copy results + figures back
#
# Environment variables:
#   REMOTE       SSH target (default: mslovett@doob.dartmouth.edu).
#   REMOTE_REPO  Path on the remote (default: infl_ens).
#   EXPERIMENT   Experiment YAML (default: configs/experiments/seven_axis_3arm.yaml).
#   GPU          CUDA device to pin (default: 0).
#   STAGES       Comma-separated stage subset (default: the experiment's list).
#   ONLY_ARM     Restrict per-arm stages to one arm name.
#   FORCE=1      Re-run stages whose outputs exist.
#   MODE         launch (default) | smoke | status | pull
#   SKIP_SYNC=1  Skip the scp code push.
#   FORCE_GPU=1  Launch even if the target GPU already has a compute process.
#   PY           Remote python (default: .venv/bin/python).

set -euo pipefail
cd "$(dirname "$0")/.."

# Verified 2026-08-26: `mslovett@` is rejected by the host's key auth
# ("Permission denied (publickey,password)"); `mlovett@` is the account that
# owns ~/infl_ens, the venv and the datasets.
REMOTE="${REMOTE:-mlovett@doob.dartmouth.edu}"
REMOTE_REPO="${REMOTE_REPO:-infl_ens}"
EXPERIMENT="${EXPERIMENT:-configs/experiments/seven_axis_3arm.yaml}"
GPU="${GPU:-0}"
MODE="${MODE:-launch}"
PY="${PY:-.venv/bin/python}"

# Read the experiment name and directories from the YAML so the script and
# the pipeline can never disagree.
read -r EXP_NAME RESULT_DIR FIG_DIR <<<"$(python - "${EXPERIMENT}" <<'PYEOF'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
name = cfg["name"]
print(name, cfg.get("results_dir", f"results/{name}"), cfg.get("figures_dir", f"figures/{name}"))
PYEOF
)"
TMUX_SESSION="${TMUX_SESSION:-${EXP_NAME}}"
LOG="${RESULT_DIR}/run.log"

PIPELINE_ARGS=(-m infl_ens.pipeline --config "${EXPERIMENT}")
if [[ -n "${STAGES:-}" ]]; then PIPELINE_ARGS+=(--stages "${STAGES}"); fi
if [[ -n "${ONLY_ARM:-}" ]]; then PIPELINE_ARGS+=(--only-arm "${ONLY_ARM}"); fi
if [[ "${FORCE:-0}" == "1" ]]; then PIPELINE_ARGS+=(--force); fi

# ----------------------------------------------------------------------
# status / pull: no sync, no launch.
# ----------------------------------------------------------------------
if [[ "${MODE}" == "status" ]]; then
    echo "[status] tmux on ${REMOTE}"
    ssh "${REMOTE}" "tmux ls 2>/dev/null || echo '(no tmux sessions)'"
    echo
    echo "[status] tail of ${LOG}"
    ssh "${REMOTE}" "cd ${REMOTE_REPO} && tail -n 40 ${LOG} 2>/dev/null || echo 'no log yet'"
    echo
    echo "[status] stage status"
    ssh "${REMOTE}" "cd ${REMOTE_REPO} && cat ${RESULT_DIR}/stage_status.json 2>/dev/null || echo 'no status yet'"
    echo
    echo "[status] GPU"
    ssh "${REMOTE}" "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader"
    echo
    echo "  WATCH LIVE: ssh -t ${REMOTE} 'tmux attach -t ${TMUX_SESSION}'"
    exit 0
fi

if [[ "${MODE}" == "pull" ]]; then
    echo "[pull] run artifacts, tables and figures -> local"
    ARM_DIRS="$(python - "${EXPERIMENT}" <<'PYEOF'
import sys
from pathlib import Path

sys.path.insert(0, "src")
from infl_ens.experiment import load_experiment  # noqa: E402

exp = load_experiment(sys.argv[1])
print(" ".join(str(a.output_dir).replace("\\", "/") for a in exp.arms))
PYEOF
)"
    for run in ${ARM_DIRS}; do
        mkdir -p "${run}"
        for f in history.json resolved_config.yaml data_split.json \
                 routing_ensemble_diagnostics.json replay_summary.json; do
            scp "${REMOTE}:${REMOTE_REPO}/${run}/${f}" "${run}/" 2>/dev/null || true
        done
        for d in tables eval_train eval_test eval_val; do
            scp -r "${REMOTE}:${REMOTE_REPO}/${run}/${d}" "${run}/" 2>/dev/null || true
        done
    done
    mkdir -p "${FIG_DIR}" "${RESULT_DIR}"
    scp -r "${REMOTE}:${REMOTE_REPO}/${FIG_DIR}/"* "${FIG_DIR}/" 2>/dev/null || true
    scp "${REMOTE}:${REMOTE_REPO}/${LOG}" "${REMOTE}:${REMOTE_REPO}/${RESULT_DIR}/stage_status.json" \
        "${RESULT_DIR}/" 2>/dev/null || true
    echo "[pull] figures:"
    ls -la "${FIG_DIR}/" 2>/dev/null || true
    exit 0
fi

# ----------------------------------------------------------------------
# 1. Sync code (scp -r; there is no rsync on the local box).
# ----------------------------------------------------------------------
if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
    echo "[sync] purge local __pycache__"
    find src tests -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
    echo "[sync] ensure remote tree exists"
    ssh "${REMOTE}" "mkdir -p ${REMOTE_REPO} && cd ${REMOTE_REPO} && mkdir -p src configs tests scripts results figures ${RESULT_DIR}"
    echo "[sync] purge remote __pycache__"
    ssh "${REMOTE}" "find ${REMOTE_REPO}/src -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null; true"
    echo "[sync] scp -r src configs tests -> ${REMOTE}:${REMOTE_REPO}"
    scp -q -r src     "${REMOTE}:${REMOTE_REPO}/"
    scp -q -r configs "${REMOTE}:${REMOTE_REPO}/"
    scp -q -r tests   "${REMOTE}:${REMOTE_REPO}/"
    scp -q    pyproject.toml AGENTS.md structure.md scripts/run_on_doob.sh "${REMOTE}:${REMOTE_REPO}/"
    ssh "${REMOTE}" "cd ${REMOTE_REPO} && mv -f run_on_doob.sh scripts/run_on_doob.sh"
    echo "[sync] done"
fi

# ----------------------------------------------------------------------
# 2. Smoke gate: pytest subset + tiny closed loops, foreground.
# ----------------------------------------------------------------------
if [[ "${MODE}" == "smoke" ]]; then
    echo "[remote] smoke gate on GPU ${GPU}"
    ssh "${REMOTE}" "cd ${REMOTE_REPO} && PYTHONPATH=src CUDA_VISIBLE_DEVICES=${GPU} ${PY} -m infl_ens.pipeline --config ${EXPERIMENT} --smoke"
    echo "[smoke] complete. Re-run with MODE=launch to queue the full pipeline."
    exit 0
fi

# ----------------------------------------------------------------------
# 3. Full pipeline, detached under tmux (many hours).
# ----------------------------------------------------------------------
echo "[remote] queueing ${EXP_NAME} in tmux '${TMUX_SESSION}' (${PIPELINE_ARGS[*]})"
ssh "${REMOTE}" bash -s <<REMOTE_EOF
set -euo pipefail
cd "${REMOTE_REPO}"
mkdir -p ${RESULT_DIR}

if tmux has-session -t ${TMUX_SESSION} 2>/dev/null; then
    echo "[remote] tmux session '${TMUX_SESSION}' ALREADY EXISTS; not starting another."
    echo "[remote] attach with: tmux attach -t ${TMUX_SESSION}"
    exit 0
fi

BUSY="\$(nvidia-smi --id=${GPU} --query-compute-apps=pid --format=csv,noheader | wc -l)"
if [ "\${BUSY}" != "0" ] && [ "${FORCE_GPU:-0}" != "1" ]; then
    echo "[remote] GPU ${GPU} already has \${BUSY} compute process(es); refusing (FORCE_GPU=1 to override)."
    nvidia-smi --id=${GPU} --query-compute-apps=pid,process_name,used_memory --format=csv
    exit 1
fi

tmux new-session -d -s ${TMUX_SESSION} -n job \
    "PYTHONPATH=src CUDA_VISIBLE_DEVICES=${GPU} ${PY} ${PIPELINE_ARGS[*]} 2>&1 | tee ${LOG}; echo; echo '[pipeline finished - press any key]'; read -n 1"
tmux new-window -t ${TMUX_SESSION} -n gpu "watch -n 5 nvidia-smi"
tmux select-window -t ${TMUX_SESSION}:job
echo "[remote] tmux session '${TMUX_SESSION}' started"
tmux ls
sleep 10
echo "--- first lines of log ---"
head -n 25 ${LOG} 2>/dev/null || echo "(log still starting)"
REMOTE_EOF

echo
echo "Queued on GPU ${GPU}: ${PIPELINE_ARGS[*]}"
echo
echo "  WATCH LIVE :  ssh -t ${REMOTE} 'tmux attach -t ${TMUX_SESSION}'"
echo "  monitor    :  MODE=status bash scripts/run_on_doob.sh"
echo "  collect    :  MODE=pull   bash scripts/run_on_doob.sh"
