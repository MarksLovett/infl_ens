#!/usr/bin/env bash
# scripts/run_soft_pairs_on_doob.sh
#
# Sync the repo to doob and run the seven-axis soft-routing-over-pairs closed
# loop there: 2L clones initialised at L co-located grid-Nash positions, one
# cumulative LoRA per pair, each query routed softly over the pairs.
#
# Per AGENTS.md §4 rule 9 — work that needs the GPU and the datasets goes
# through mlovett@doob.dartmouth.edu after a code sync. The datasets, the
# fingerprinted trait-space cache and the adapters live only on the remote.
#
# The trait-space cache fingerprint is unchanged from
# seven_axis_pair_merge_split.yaml (3b42c68a8dd334c5), so the Qwen3-8B encode
# is reused; the projector still encodes the train+val pool once at startup.
#
# Usage:
#   bash scripts/run_soft_pairs_on_doob.sh                     # sync + launch
#   SMOKE=1 bash scripts/run_soft_pairs_on_doob.sh             # cheap gate
#   MODE=status bash scripts/run_soft_pairs_on_doob.sh
#   MODE=pull   bash scripts/run_soft_pairs_on_doob.sh
#
# Environment variables:
#   CONFIG       Router YAML (default: seven_axis_soft_pairs.yaml).
#   ARM          Arm name; sets results/<ARM> and the tmux session.
#   GPU          CUDA device index to pin (default: 0, the A100).
#   SEED_DIR     Run subdirectory under results/<ARM> (default: seed0).
#   REMOTE       SSH target (default: mlovett@doob.dartmouth.edu).
#   REMOTE_REPO  Path on the remote (default: infl_ens).
#   MODE         launch (default) | status | pull
#   SMOKE=1      Two tiny rounds off-manifest, in the foreground.
#   SKIP_SYNC=1  Skip the scp code push.
#   FORCE_GPU=1  Launch even if the target GPU already has a compute process.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/benchmark/router/seven_axis_soft_pairs.yaml}"
REMOTE="${REMOTE:-mlovett@doob.dartmouth.edu}"
REMOTE_REPO="${REMOTE_REPO:-infl_ens}"
MODE="${MODE:-launch}"
ARM="${ARM:-seven_axis_soft_pairs}"
GPU="${GPU:-0}"
SEED_DIR="${SEED_DIR:-seed0}"
TMUX_SESSION="${TMUX_SESSION:-${ARM//_/-}}"
RESULT_DIR="results/${ARM}"
RUN_DIR="${RESULT_DIR}/${SEED_DIR}"
LOG="${RESULT_DIR}/run.log"

# The split manifest path is declared in the config; read it rather than
# duplicating it here, so the builder and the trainer can never disagree.
MANIFEST="$(python - "${CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
print((cfg.get("data_split") or {}).get("manifest", ""))
PY
)"
if [[ -z "${MANIFEST}" ]]; then
    echo "error: ${CONFIG} has no data_split.manifest" >&2
    exit 1
fi

# ----------------------------------------------------------------------
# status / pull modes: no sync, no launch.
# ----------------------------------------------------------------------
if [[ "${MODE}" == "status" ]]; then
    echo "[status] tmux sessions on ${REMOTE}"
    ssh "${REMOTE}" "tmux ls 2>/dev/null || echo '(no tmux sessions - job finished or not started)'"
    echo
    echo "[status] tail of ${LOG}"
    ssh "${REMOTE}" "cd ${REMOTE_REPO} && tail -n 40 ${LOG} 2>/dev/null || echo 'no log yet'"
    echo
    echo "[status] GPU"
    ssh "${REMOTE}" "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader"
    echo
    echo "[status] adapters written so far"
    ssh "${REMOTE}" "cd ${REMOTE_REPO} && ls ${RUN_DIR}/agents 2>/dev/null || echo 'none yet'"
    echo
    echo "[status] per-round pair geometry"
    ssh "${REMOTE}" "cd ${REMOTE_REPO} && PYTHONPATH=src .venv/bin/python scripts/summarize_soft_pairs_history.py ${RUN_DIR}/history.json --tail 8"
    echo
    echo "  WATCH LIVE: ssh -t ${REMOTE} 'tmux attach -t ${TMUX_SESSION}'"
    exit 0
fi

if [[ "${MODE}" == "pull" ]]; then
    echo "[pull] history + resolved config + val evals -> local ${RUN_DIR}/"
    mkdir -p "${RUN_DIR}" "${RESULT_DIR}"
    # Never the adapters or the trait-space cache: those stay where they were
    # computed (tens of GB, and only meaningful next to the remote datasets).
    for f in history.json resolved_config.yaml data_split.json; do
        scp "${REMOTE}:${REMOTE_REPO}/${RUN_DIR}/${f}" "${RUN_DIR}/" 2>/dev/null || true
    done
    scp -r "${REMOTE}:${REMOTE_REPO}/${RUN_DIR}/eval_val" "${RUN_DIR}/" 2>/dev/null || true
    scp "${REMOTE}:${REMOTE_REPO}/${LOG}" "${RESULT_DIR}/" 2>/dev/null || true
    echo "[pull] local contents:"
    ls -la "${RUN_DIR}/" || true
    exit 0
fi

# ----------------------------------------------------------------------
# 1. Sync code via scp -r (no rsync locally).
#
# scp -r has no --exclude, so purge local __pycache__ first; otherwise
# Windows .pyc trees get shipped and stale bytecode can shadow new code.
# ----------------------------------------------------------------------
if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
    echo "[sync] purge local __pycache__"
    find src scripts tests -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

    echo "[sync] ensure remote tree exists"
    ssh "${REMOTE}" "mkdir -p ${REMOTE_REPO} && cd ${REMOTE_REPO} && \
        mkdir -p src scripts configs tests results ${RESULT_DIR} ${RUN_DIR}"

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
# 2. SMOKE gate: two tiny rounds in the foreground, off the split manifest.
#
# data_split must be disabled: with a manifest every round covers the whole
# train partition regardless of batch_size, so there is no cheap round.
# ----------------------------------------------------------------------
if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "[remote] SMOKE gate on GPU ${GPU} (2 rounds x 64 prompts, foreground)"
    ssh "${REMOTE}" bash -s <<REMOTE_EOF
set -euo pipefail
cd "${REMOTE_REPO}"
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${GPU}"
PY="\${PY:-.venv/bin/python}"

echo "[remote] python: \$(\${PY} --version)"
echo "[remote] pytest (offline suite)"
\${PY} -m pytest tests/test_soft_pairs.py tests/test_merge_training.py \
    tests/test_agent_init.py tests/test_weighted_sft_loss.py \
    tests/test_routing_eval.py -q

echo "[remote] closed loop, smoke config"
\${PY} -m infl_ens.training --config "${CONFIG}" -- \
    output_dir=${RESULT_DIR}_smoke/${SEED_DIR} \
    closed_loop.sft.output_dir=${RESULT_DIR}_smoke/${SEED_DIR}/agents \
    data_split=null \
    closed_loop.n_rounds=2 \
    closed_loop.batch_size=64 \
    closed_loop.val_eval.every_n_rounds=0 \
    closed_loop.theory_gradient.n_steps=500
REMOTE_EOF
    echo "[smoke] complete. Re-run without SMOKE=1 to launch the full job."
    exit 0
fi

# ----------------------------------------------------------------------
# 3. Full job, detached under tmux (hours).
# ----------------------------------------------------------------------
echo "[remote] launching full job in tmux session '${TMUX_SESSION}' (log: ${LOG})"
ssh "${REMOTE}" bash -s <<REMOTE_EOF
set -euo pipefail
cd "${REMOTE_REPO}"
mkdir -p ${RESULT_DIR} ${RUN_DIR}

if tmux has-session -t ${TMUX_SESSION} 2>/dev/null; then
    echo "[remote] tmux session '${TMUX_SESSION}' ALREADY EXISTS; not starting another."
    echo "[remote] attach with: tmux attach -t ${TMUX_SESSION}"
    exit 0
fi

# Refuse to share a card: this arm wants the whole GPU, and a co-tenant would
# both slow it down and risk OOM at 7 adapters per round.
BUSY="\$(nvidia-smi --id=${GPU} --query-compute-apps=pid --format=csv,noheader | wc -l)"
if [ "\${BUSY}" != "0" ] && [ "${FORCE_GPU:-0}" != "1" ]; then
    echo "[remote] GPU ${GPU} already has \${BUSY} compute process(es); refusing."
    echo "[remote] free it, pick another GPU=<n>, or re-run with FORCE_GPU=1."
    nvidia-smi --id=${GPU} --query-compute-apps=pid,process_name,used_memory --format=csv
    exit 1
fi

cat > ${RESULT_DIR}/_job.sh <<'JOB'
set -euo pipefail
export PYTHONPATH=src
# Pin one card. HuggingFaceEncoder defaults to device_map="auto", which
# otherwise splits the ~16GB (bf16-decompressed) encoder across every visible
# card and pays a PCIe hop per layer. device_map is not part of the config, so
# this does not affect the trait-space cache fingerprint.
export CUDA_VISIBLE_DEVICES="__GPU__"
PY="\${PY:-.venv/bin/python}"

echo "=== \$(date -Is) starting ==="
echo "[job] python: \$(\${PY} --version)"
echo "[job] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}"

echo "=== \$(date -Is) pytest (offline gate) ==="
\${PY} -m pytest tests/test_soft_pairs.py tests/test_merge_training.py \
    tests/test_agent_init.py tests/test_weighted_sft_loss.py \
    tests/test_routing_eval.py -q

echo "=== \$(date -Is) build split manifest ==="
if [ -f "__MANIFEST__" ]; then
    echo "[job] manifest already present: __MANIFEST__"
else
    \${PY} scripts/build_seven_axis_split.py \
        --config "__CONFIG__" \
        --output "__MANIFEST__"
fi

echo "=== \$(date -Is) closed loop (soft routing over pairs) ==="
\${PY} -m infl_ens.training --config "__CONFIG__"

echo "=== \$(date -Is) DONE ==="
JOB

sed -i "s|__CONFIG__|${CONFIG}|g; s|__MANIFEST__|${MANIFEST}|g; s|__GPU__|${GPU}|g" \
    ${RESULT_DIR}/_job.sh

# Window 0 runs the job under 'tee' so the log persists even if the session is
# killed; window 1 watches the GPU. Mirrors scripts/tmux_monitor_seeds.sh.
tmux new-session -d -s ${TMUX_SESSION} -n job \
    "bash ${RESULT_DIR}/_job.sh 2>&1 | tee ${LOG}; echo; echo '[job finished - press any key]'; read -n 1"
tmux new-window -t ${TMUX_SESSION} -n gpu "watch -n 5 nvidia-smi"
tmux select-window -t ${TMUX_SESSION}:job

echo "[remote] tmux session '${TMUX_SESSION}' started"
tmux ls
sleep 10
echo "--- first lines of log ---"
head -n 20 ${LOG} 2>/dev/null || echo "(log still starting)"
REMOTE_EOF

echo
echo "Launched in tmux on GPU ${GPU}."
echo
echo "  WATCH LIVE :  ssh -t ${REMOTE} 'tmux attach -t ${TMUX_SESSION}'"
echo "                (window 0 = job, window 1 = nvidia-smi;"
echo "                 Ctrl-b n switches window, Ctrl-b d detaches)"
echo
echo "  monitor    :  ARM=${ARM} MODE=status bash scripts/run_soft_pairs_on_doob.sh"
echo "  collect    :  ARM=${ARM} MODE=pull   bash scripts/run_soft_pairs_on_doob.sh"
