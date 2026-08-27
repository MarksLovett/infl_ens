#!/usr/bin/env bash
# scripts/run_seven_axis_3arm.sh
#
# Queue the full seven-axis routing comparison on the GPU host and produce
# every figure and table from it, in one resumable tmux pipeline.
#
# Three specialist arms sweep the top-k axis from fully distributed to a
# single winner, plus one shared generalist:
#
#   ARM 1a soft routing over co-located theory pairs, top_k = 7 (FULLY
#          distributed: every pair learns from every prompt)
#   ARM 1b soft routing over the same pairs, top_k = 3 (partial)
#   ARM 2  hard (SFT) routing over the same pairs, one sampled winner
#   ARM 3  pooled generalist replayed from ARM 1b's batches
#
# All arms share the manifest, seed and theory initialization, so they draw
# identical round batches and ONE generalist is data-matched to all three.
#
# Both specialist arms use `position_update: theory_matched`, which is required
# because they start from a theory initialization: the trait-space dynamics must
# stay the game's gradient flow.
#
# Deliverables written under results/<arm>/seed0/ and scripts/figures/three_arm/:
#   * three LaTeX bar figures (oracle vs generalist vs specialists, per arm and
#     cross-arm), compiled to PDF when latexmk is available,
#   * per-round pair NLL tables (round 4 -> last) as csv/md/tex/json,
#   * final pair-position and within-pair-separation figures per arm,
#   * a cross-arm analysis report.
#
# Usage:
#   bash scripts/run_seven_axis_3arm.sh                      # sync + queue all
#   SMOKE=1 bash scripts/run_seven_axis_3arm.sh              # cheap gate first
#   STAGES="routing figures" bash scripts/run_seven_axis_3arm.sh   # re-run analysis
#   MODE=status bash scripts/run_seven_axis_3arm.sh
#   MODE=pull   bash scripts/run_seven_axis_3arm.sh
#
# Environment variables:
#   REMOTE       SSH target. NOTE the other scripts in this repo default to
#                mlovett@doob.dartmouth.edu; this one defaults to the account
#                given for this experiment.
#   REMOTE_REPO  Path on the remote (default: infl_ens).
#   GPU          CUDA device to pin (default: 0).
#   STAGES       Space-separated subset of:
#                manifest softfull soft hard generalist perround routing figures
#                (default: all, in that order).
#   FIRST_ROUND  Early round in the NLL comparison table (default: 4). The
#                table reports exactly two rounds -- FIRST_ROUND and the final
#                round -- so the early/late change is a single column.
#   MAX_EVAL     Cap per benchmark for eval/oracle scoring (default: 1000).
#   MODE         launch (default) | status | pull
#   SMOKE=1      Two tiny rounds of each arm, foreground, off-manifest.
#   SKIP_SYNC=1  Skip the scp code push.
#   FORCE_GPU=1  Launch even if the target GPU already has a compute process.

set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="${REMOTE:-mslovett@doob.dartmouth.edu}"
REMOTE_REPO="${REMOTE_REPO:-infl_ens}"
MODE="${MODE:-launch}"
GPU="${GPU:-0}"
STAGES="${STAGES:-manifest softfull soft hard generalist perround routing figures}"
FIRST_ROUND="${FIRST_ROUND:-4}"
MAX_EVAL="${MAX_EVAL:-1000}"
TMUX_SESSION="${TMUX_SESSION:-seven-axis-3arm}"

CFG_FULL="configs/benchmark/router/seven_axis_soft_full_pairs.yaml"
CFG_SOFT="configs/benchmark/router/seven_axis_soft_topk3_pairs.yaml"
CFG_HARD="configs/benchmark/router/seven_axis_hard_pairs_matched.yaml"
CFG_GEN="configs/benchmark/router/seven_axis_3arm_generalist_replay.yaml"

RUN_FULL="results/seven_axis_soft_full_pairs/seed0"
RUN_SOFT="results/seven_axis_soft_topk3_pairs/seed0"
RUN_HARD="results/seven_axis_hard_pairs_matched/seed0"
RUN_GEN="results/seven_axis_3arm_generalist/seed0"

FIG_DIR="scripts/figures/three_arm"
RESULT_ROOT="results/seven_axis_3arm"
LOG="${RESULT_ROOT}/run.log"

AXIS_LABELS="harm hallucination jailbreak privacy overrefusal injection policy"

# The split manifest is declared in the config; read it rather than duplicating
# it, so the builder and every arm can never disagree.
MANIFEST="$(python - "${CFG_SOFT}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
print((cfg.get("data_split") or {}).get("manifest", ""))
PY
)"
if [[ -z "${MANIFEST}" ]]; then
    echo "error: ${CFG_SOFT} has no data_split.manifest" >&2
    exit 1
fi

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
    echo "[status] GPU"
    ssh "${REMOTE}" "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader"
    echo
    for run in "${RUN_FULL}" "${RUN_SOFT}" "${RUN_HARD}" "${RUN_GEN}"; do
        echo "[status] ${run}"
        ssh "${REMOTE}" "cd ${REMOTE_REPO} && \
            (PYTHONPATH=src .venv/bin/python -c \"
import json,sys
try:
    h=json.load(open('${run}/history.json'))
    print('  rounds done:', len(h), 'last round:', h[-1].get('round'))
except Exception as e:
    print('  no history yet')
\") ; ls ${run}/agents 2>/dev/null | head -n 8 || echo '  no adapters yet'"
    done
    echo
    echo "  WATCH LIVE: ssh -t ${REMOTE} 'tmux attach -t ${TMUX_SESSION}'"
    exit 0
fi

if [[ "${MODE}" == "pull" ]]; then
    echo "[pull] histories, tables, figures and reports -> local"
    for run in "${RUN_FULL}" "${RUN_SOFT}" "${RUN_HARD}" "${RUN_GEN}"; do
        mkdir -p "${run}"
        for f in history.json resolved_config.yaml data_split.json \
                 routing_ensemble_diagnostics.json replay_summary.json; do
            scp "${REMOTE}:${REMOTE_REPO}/${run}/${f}" "${run}/" 2>/dev/null || true
        done
        for d in tables eval_train eval_test eval_val; do
            scp -r "${REMOTE}:${REMOTE_REPO}/${run}/${d}" "${run}/" 2>/dev/null || true
        done
    done
    mkdir -p "${FIG_DIR}" "${RESULT_ROOT}"
    scp -r "${REMOTE}:${REMOTE_REPO}/${FIG_DIR}/"* "${FIG_DIR}/" 2>/dev/null || true
    scp "${REMOTE}:${REMOTE_REPO}/${LOG}" "${RESULT_ROOT}/" 2>/dev/null || true
    echo "[pull] figures:"
    ls -la "${FIG_DIR}/" 2>/dev/null || true
    exit 0
fi

# ----------------------------------------------------------------------
# 1. Sync code (scp -r; there is no rsync on the local box).
# ----------------------------------------------------------------------
if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
    echo "[sync] purge local __pycache__"
    find src scripts tests -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

    echo "[sync] ensure remote tree exists"
    ssh "${REMOTE}" "mkdir -p ${REMOTE_REPO} && cd ${REMOTE_REPO} && \
        mkdir -p src scripts configs tests results ${RESULT_ROOT} ${FIG_DIR} \
                 ${RUN_FULL} ${RUN_SOFT} ${RUN_HARD} ${RUN_GEN}"

    echo "[sync] purge remote __pycache__"
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
# 2. SMOKE gate: two tiny rounds of each arm, foreground, off-manifest.
# ----------------------------------------------------------------------
if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "[remote] SMOKE gate on GPU ${GPU}"
    ssh "${REMOTE}" bash -s <<REMOTE_EOF
set -euo pipefail
cd "${REMOTE_REPO}"
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${GPU}"
PY="\${PY:-.venv/bin/python}"

echo "[remote] python: \$(\${PY} --version)"
\${PY} -m pytest tests/test_soft_pairs.py tests/test_topk_matched.py \
    tests/test_unified_eval.py tests/test_merge_training.py \
    tests/test_weighted_sft_loss.py tests/test_routing_eval.py -q

for cfg in "${CFG_FULL}" "${CFG_SOFT}" "${CFG_HARD}"; do
    echo "[remote] smoke closed loop: \${cfg}"
    name="\$(basename \${cfg} .yaml)"
    \${PY} -m infl_ens.training --config "\${cfg}" -- \
        output_dir=results/_smoke_\${name}/seed0 \
        closed_loop.sft.output_dir=results/_smoke_\${name}/seed0/agents \
        data_split=null \
        eval=null \
        closed_loop.n_rounds=2 \
        closed_loop.batch_size=64 \
        closed_loop.theory_gradient.n_steps=500
done
echo "[remote] smoke OK"
REMOTE_EOF
    echo "[smoke] complete. Re-run without SMOKE=1 to queue the full pipeline."
    exit 0
fi

# ----------------------------------------------------------------------
# 3. Full pipeline, sequential, detached under tmux (many hours).
#
# Sequential by design: ARM 3 replays ARM 1's logged batches, and every
# analysis stage needs all three runs. Stages are individually re-runnable
# via STAGES=... so a failure late in the pipeline costs no GPU time.
# ----------------------------------------------------------------------
echo "[remote] queueing pipeline in tmux '${TMUX_SESSION}' (stages: ${STAGES})"
ssh "${REMOTE}" bash -s <<REMOTE_EOF
set -euo pipefail
cd "${REMOTE_REPO}"
mkdir -p ${RESULT_ROOT} ${FIG_DIR}

if tmux has-session -t ${TMUX_SESSION} 2>/dev/null; then
    echo "[remote] tmux session '${TMUX_SESSION}' ALREADY EXISTS; not starting another."
    echo "[remote] attach with: tmux attach -t ${TMUX_SESSION}"
    exit 0
fi

BUSY="\$(nvidia-smi --id=${GPU} --query-compute-apps=pid --format=csv,noheader | wc -l)"
if [ "\${BUSY}" != "0" ] && [ "${FORCE_GPU:-0}" != "1" ]; then
    echo "[remote] GPU ${GPU} already has \${BUSY} compute process(es); refusing."
    nvidia-smi --id=${GPU} --query-compute-apps=pid,process_name,used_memory --format=csv
    exit 1
fi

cat > ${RESULT_ROOT}/_job.sh <<'JOB'
set -euo pipefail
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="__GPU__"
PY="\${PY:-.venv/bin/python}"
STAGES="__STAGES__"
FIRST_ROUND=__FIRST_ROUND__
MAX_EVAL=__MAX_EVAL__

has_stage () { case " \${STAGES} " in *" \$1 "*) return 0;; *) return 1;; esac; }
banner () { echo; echo "=== \$(date -Is) \$* ==="; }

banner "pipeline start (stages: \${STAGES})"
\${PY} --version

if has_stage manifest; then
    banner "split manifest"
    if [ -f "__MANIFEST__" ]; then
        echo "[job] manifest already present: __MANIFEST__"
    else
        \${PY} scripts/build_seven_axis_split.py \
            --config "__CFG_SOFT__" --output "__MANIFEST__"
    fi
fi

if has_stage softfull; then
    banner "ARM 1a: FULLY distributed soft routing over pairs, top_k=7"
    \${PY} -m infl_ens.training --config "__CFG_FULL__"
fi

if has_stage soft; then
    banner "ARM 1b: soft routing over pairs, top_k=3"
    \${PY} -m infl_ens.training --config "__CFG_SOFT__"
fi

if has_stage hard; then
    banner "ARM 2: hard (SFT) routing over pairs"
    \${PY} -m infl_ens.training --config "__CFG_HARD__"
fi

if has_stage generalist; then
    banner "ARM 3: pooled generalist replay (from ARM 1 batches)"
    \${PY} -m infl_ens.training --config "__CFG_GEN__"
fi

# Final trained round, read from ARM 1's history.
FINAL_ROUND="\$(\${PY} -c "
import json
h = json.load(open('__RUN_SOFT__/history.json'))
print(int(h[-1]['round']))
")"
# Exactly two rounds: the early checkpoint and the final one, so the table is
# an early-vs-late comparison rather than a full sweep.
ROUNDS="[\${FIRST_ROUND},\${FINAL_ROUND}]"
echo "[job] final round \${FINAL_ROUND}; NLL comparison rounds \${ROUNDS}"

if has_stage perround; then
    banner "validation NLL at rounds \${FIRST_ROUND} and \${FINAL_ROUND}"
    # Skip an arm whose eval_val already covers both requested rounds: a
    # partial earlier run may have banked them, and re-scoring is expensive.
    for pair in "__CFG_FULL__|__RUN_FULL__" "__CFG_SOFT__|__RUN_SOFT__" "__CFG_HARD__|__RUN_HARD__"; do
        cfg="\${pair%%|*}"; run="\${pair##*|}"
        if \${PY} -c "
import json, sys
need = {\${FIRST_ROUND}, \${FINAL_ROUND}}
try:
    res = json.load(open(sys.argv[1] + '/eval_val/eval_results.json'))['results']
except Exception:
    sys.exit(1)
sys.exit(0 if need <= {r.get('round') for r in res} else 1)
" "\${run}"; then
            echo "[job] \${run}: eval_val already covers rounds \${ROUNDS}; skipping"
        else
            \${PY} -m infl_ens.evaluation --config "\${cfg}" -- \
                eval.partitions='["val"]' \
                eval.rounds="\${ROUNDS}" \
                eval.max_eval_records=\${MAX_EVAL}
        fi
    done
    \${PY} scripts/build_per_round_pair_nll_table.py \
        --eval-dir __RUN_FULL__/eval_val \
        --output-stem __RUN_FULL__/tables/pair_nll_by_round \
        --label "Soft fully distributed, k=7 (val)" \
        --first-round \${FIRST_ROUND} --rounds "\${FIRST_ROUND},\${FINAL_ROUND}"
    \${PY} scripts/build_per_round_pair_nll_table.py \
        --eval-dir __RUN_SOFT__/eval_val \
        --output-stem __RUN_SOFT__/tables/pair_nll_by_round \
        --label "Soft top-3 pairs (val)" \
        --first-round \${FIRST_ROUND} --rounds "\${FIRST_ROUND},\${FINAL_ROUND}"
    \${PY} scripts/build_per_round_pair_nll_table.py \
        --eval-dir __RUN_HARD__/eval_val \
        --output-stem __RUN_HARD__/tables/pair_nll_by_round \
        --label "Hard SFT routing pairs (val)" \
        --first-round \${FIRST_ROUND} --rounds "\${FIRST_ROUND},\${FINAL_ROUND}"
fi

if has_stage routing; then
    banner "oracle / route-then-score diagnostics (test partition)"
    \${PY} scripts/routing_ensemble_diagnostics.py \
        --router-config "__CFG_FULL__" \
        --history __RUN_FULL__/history.json \
        --merge-run-dir __RUN_FULL__ \
        --baseline-run-dir __RUN_GEN__ \
        --partition test --max-eval-records \${MAX_EVAL} \
        --round \${FINAL_ROUND} \
        --output-json __RUN_FULL__/routing_ensemble_diagnostics.json
    \${PY} scripts/routing_ensemble_diagnostics.py \
        --router-config "__CFG_SOFT__" \
        --history __RUN_SOFT__/history.json \
        --merge-run-dir __RUN_SOFT__ \
        --baseline-run-dir __RUN_GEN__ \
        --partition test --max-eval-records \${MAX_EVAL} \
        --round \${FINAL_ROUND} \
        --output-json __RUN_SOFT__/routing_ensemble_diagnostics.json
    \${PY} scripts/routing_ensemble_diagnostics.py \
        --router-config "__CFG_HARD__" \
        --history __RUN_HARD__/history.json \
        --merge-run-dir __RUN_HARD__ \
        --baseline-run-dir __RUN_GEN__ \
        --partition test --max-eval-records \${MAX_EVAL} \
        --round \${FINAL_ROUND} \
        --output-json __RUN_HARD__/routing_ensemble_diagnostics.json
fi

if has_stage figures; then
    banner "figures and tables"
    # Three bar figures: one oracle-vs-generalist-vs-specialist per arm.
    \${PY} scripts/write_oracle_routing_tex.py \
        --input __RUN_FULL__/routing_ensemble_diagnostics.json \
        --output __FIG_DIR__/soft_full_vs_oracle.tex \
        --experiment-label "Seven-axis soft pairs, fully distributed (k=7)"
    \${PY} scripts/write_oracle_routing_tex.py \
        --input __RUN_SOFT__/routing_ensemble_diagnostics.json \
        --output __FIG_DIR__/soft_topk3_vs_oracle.tex \
        --experiment-label "Seven-axis soft pairs (top-3)"
    \${PY} scripts/write_oracle_routing_tex.py \
        --input __RUN_HARD__/routing_ensemble_diagnostics.json \
        --output __FIG_DIR__/hard_sft_vs_oracle.tex \
        --experiment-label "Seven-axis hard (SFT) pairs"
    # Plus the cross-arm overlay.
    \${PY} scripts/write_arm_comparison_tex.py \
        --report "Soft k=7=__RUN_FULL__/routing_ensemble_diagnostics.json" \
        --report "Soft top-3=__RUN_SOFT__/routing_ensemble_diagnostics.json" \
        --report "Hard (SFT)=__RUN_HARD__/routing_ensemble_diagnostics.json" \
        --output __FIG_DIR__/arm_comparison.tex

    # Final pair positions + within-pair separation, per arm.
    \${PY} scripts/plot_pair_final_positions.py \
        --run-dir __RUN_FULL__ --output-stem __FIG_DIR__/soft_full \
        --axis-labels __AXIS_LABELS__ --title "Soft fully distributed (k=7)"
    \${PY} scripts/plot_pair_final_positions.py \
        --run-dir __RUN_SOFT__ --output-stem __FIG_DIR__/soft_topk3 \
        --axis-labels __AXIS_LABELS__ --title "Soft top-3 pairs"
    \${PY} scripts/plot_pair_final_positions.py \
        --run-dir __RUN_HARD__ --output-stem __FIG_DIR__/hard_sft \
        --axis-labels __AXIS_LABELS__ --title "Hard (SFT) pairs"

    # Cross-arm analysis (also verifies the data-matching precondition).
    \${PY} scripts/cross_analyze_three_arms.py \
        --arm "Soft k=7=__RUN_FULL__" \
        --arm "Soft top-3=__RUN_SOFT__" \
        --arm "Hard (SFT)=__RUN_HARD__" \
        --generalist-run-dir __RUN_GEN__ \
        --output-dir __FIG_DIR__

    if command -v latexmk >/dev/null 2>&1; then
        banner "compiling figures to PDF"
        for tex in __FIG_DIR__/*.tex; do
            (cd __FIG_DIR__ && latexmk -pdf -interaction=nonstopmode \
                -halt-on-error "\$(basename \${tex})" >/dev/null 2>&1) \
                && echo "[job] compiled \${tex}" \
                || echo "[job] WARNING could not compile \${tex}"
        done
        (cd __FIG_DIR__ && latexmk -c >/dev/null 2>&1) || true
    else
        echo "[job] latexmk not found; .tex written but not compiled"
    fi
fi

banner "DONE"
JOB

sed -i "s|__CFG_FULL__|${CFG_FULL}|g; s|__RUN_FULL__|${RUN_FULL}|g; \
        s|__CFG_SOFT__|${CFG_SOFT}|g; s|__CFG_HARD__|${CFG_HARD}|g; \
        s|__CFG_GEN__|${CFG_GEN}|g; s|__RUN_SOFT__|${RUN_SOFT}|g; \
        s|__RUN_HARD__|${RUN_HARD}|g; s|__RUN_GEN__|${RUN_GEN}|g; \
        s|__FIG_DIR__|${FIG_DIR}|g; s|__MANIFEST__|${MANIFEST}|g; \
        s|__GPU__|${GPU}|g; s|__STAGES__|${STAGES}|g; \
        s|__FIRST_ROUND__|${FIRST_ROUND}|g; s|__MAX_EVAL__|${MAX_EVAL}|g; \
        s|__AXIS_LABELS__|${AXIS_LABELS}|g" \
    ${RESULT_ROOT}/_job.sh

tmux new-session -d -s ${TMUX_SESSION} -n job \
    "bash ${RESULT_ROOT}/_job.sh 2>&1 | tee ${LOG}; echo; echo '[pipeline finished - press any key]'; read -n 1"
tmux new-window -t ${TMUX_SESSION} -n gpu "watch -n 5 nvidia-smi"
tmux select-window -t ${TMUX_SESSION}:job

echo "[remote] tmux session '${TMUX_SESSION}' started"
tmux ls
sleep 10
echo "--- first lines of log ---"
head -n 25 ${LOG} 2>/dev/null || echo "(log still starting)"
REMOTE_EOF

echo
echo "Queued on GPU ${GPU}: ${STAGES}"
echo
echo "  WATCH LIVE :  ssh -t ${REMOTE} 'tmux attach -t ${TMUX_SESSION}'"
echo "  monitor    :  MODE=status bash scripts/run_seven_axis_3arm.sh"
echo "  collect    :  MODE=pull   bash scripts/run_seven_axis_3arm.sh"
