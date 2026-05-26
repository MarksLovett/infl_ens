#!/usr/bin/env bash
# scripts/run_loss_reweight_on_doob.sh
#
# One-shot wrapper for the canonical-routing + (1-G) loss-reweight
# closed-loop run on doob.dartmouth.edu (a Linux host), with the full
# trajectory + theory-vs-SFT + capability-probe analysis stack run
# remotely and figures + JSON summaries pulled back to the local
# scripts/figures/ and results/ trees.
#
# Per AGENTS.md §4 rule 9 — training runs go through the SSH host
# mlovett@doob.dartmouth.edu after a code sync.
#
# Sync transport is scp (no rsync available locally is the assumption);
# code is pushed once at the start of each invocation, results are
# pulled back at the end. For tight inner loops over many seeds, set
# SKIP_SYNC=1 after the first call.
#
# Usage:
#   bash scripts/run_loss_reweight_on_doob.sh                 # default cfg, seed 0
#   SEED=1 bash scripts/run_loss_reweight_on_doob.sh          # different seed
#   CONFIG=configs/benchmark/router/safety_truth_n4_r10_loss_reweight.yaml \
#       bash scripts/run_loss_reweight_on_doob.sh             # explicit config
#
# Environment variables:
#   CONFIG        Path to the YAML (default: the loss_reweight config shipped here).
#   SEED          Override `seed` in the YAML (default: 0).
#   RUN_NAME      Slug used for figure stems and the local results dir
#                 (default: <config basename> + "_seed<SEED>").
#   REMOTE        SSH target (default: mlovett@doob.dartmouth.edu).
#   REMOTE_REPO   Path on the remote (default: ~/infl_ens).
#   SKIP_SYNC=1   Skip the scp code push.
#   SKIP_PULL=1   Skip scp'ing figures + JSON back at the end.

set -euo pipefail

CONFIG="${CONFIG:-configs/benchmark/router/safety_truth_n4_r10_loss_reweight.yaml}"
SEED="${SEED:-0}"
REMOTE="${REMOTE:-mlovett@doob.dartmouth.edu}"
REMOTE_REPO="${REMOTE_REPO:-infl_ens}"

# Slug used for figure stems and the local results directory.
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG}" .yaml)_seed${SEED}}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "CONFIG not found locally: ${CONFIG}" >&2
    exit 1
fi

echo "================================================================"
echo "  loss_reweight run on doob"
echo "  config       : ${CONFIG}"
echo "  seed         : ${SEED}"
echo "  remote       : ${REMOTE}:${REMOTE_REPO}"
echo "  run name     : ${RUN_NAME}"
echo "================================================================"

# ----------------------------------------------------------------------
# 1. Sync the code to doob via scp -r.
#
# scp -r copies whole directories but lacks rsync's --delete and
# --exclude; we push only the directories code lives in. Datasets and
# prior results stay on the remote.
# ----------------------------------------------------------------------
if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
    echo "[sync] ensure remote repo exists, then scp -r code -> ${REMOTE}:${REMOTE_REPO}"
    ssh "${REMOTE}" "mkdir -p ${REMOTE_REPO} && cd ${REMOTE_REPO} && mkdir -p src scripts configs tests results scripts/figures"

    # Clear stale __pycache__ on the remote so old .pyc files don't
    # shadow the new code (scp -r doesn't do --delete).
    ssh "${REMOTE}" "find ${REMOTE_REPO}/src ${REMOTE_REPO}/scripts -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null; true"

    # Push the package source and top-level metadata. AGENTS.md and
    # structure.md are required by the AGENTS.md §4 rule 10 / structure
    # cross-checks; pyproject + requirements let the remote install if
    # needed.
    scp -r src                          "${REMOTE}:${REMOTE_REPO}/"
    scp -r scripts                      "${REMOTE}:${REMOTE_REPO}/"
    scp -r configs                      "${REMOTE}:${REMOTE_REPO}/"
    scp -r tests                        "${REMOTE}:${REMOTE_REPO}/"
    scp    AGENTS.md structure.md pyproject.toml requirements.txt \
                                        "${REMOTE}:${REMOTE_REPO}/" 2>/dev/null || true
fi

# ----------------------------------------------------------------------
# 2. Run training + the three plotting/analysis steps on doob.
#
# Output dir is pinned by the YAML; we override `seed` and
# `closed_loop.sft.seed` via CLI overrides so one YAML can be reused
# across seeds without editing.
# ----------------------------------------------------------------------
echo "[remote] training + analysis on ${REMOTE}"
ssh "${REMOTE}" bash -s <<REMOTE_EOF
set -euo pipefail
cd "${REMOTE_REPO}"

CONFIG_R="${CONFIG}"
SEED_R="${SEED}"
RUN_NAME_R="${RUN_NAME}"

# Read output_dir from the YAML so figures and JSON paths line up
# with what the trainer actually writes.
OUT_DIR=\$(python - <<'PY'
import os, yaml
with open(os.environ["CONFIG_R"]) as fh:
    cfg = yaml.safe_load(fh)
print(cfg.get("output_dir", f"results/{os.environ['RUN_NAME_R']}"))
PY
)
export CONFIG_R SEED_R RUN_NAME_R OUT_DIR

echo "[remote] output_dir = \${OUT_DIR}"
mkdir -p "\${OUT_DIR}" scripts/figures

# 2a. Closed-loop training.
echo "[remote] training ..."
python -m infl_ens.training \
    --config "\${CONFIG_R}" \
    "seed=\${SEED_R}" \
    "closed_loop.sft.seed=\${SEED_R}" \
    2>&1 | tee "\${OUT_DIR}/training.log"

# 2b. Trait-space trajectories + utility tracking.
echo "[remote] plot_closed_loop_history ..."
PYTHONPATH=src python scripts/plot_closed_loop_history.py \
    --history "\${OUT_DIR}/history.json" \
    --axis-labels harm hallucination \
    --title "\${RUN_NAME_R}  (canonical + 1-G loss reweight)" \
    --output-stem "scripts/figures/\${RUN_NAME_R}"

# 2c. Theory vs SFT comparison — rebuilds the trait space from the
#     run's config and overlays the strategic-Nash endpoints on the
#     observed SFT trajectory.
echo "[remote] compare_theory_vs_sft ..."
PYTHONPATH=src python scripts/compare_theory_vs_sft.py \
    --config "\${CONFIG_R}" \
    --history "\${OUT_DIR}/history.json" \
    --axis-labels harm hallucination \
    --title "\${RUN_NAME_R}  theory vs SFT" \
    --output-stem "scripts/figures/\${RUN_NAME_R}_theory_vs_sft" \
    --summary-json "\${OUT_DIR}/theory_vs_sft.json"

# 2d. Capability probe — Tier 1 SFT loss curves + Tier 3 cross-
#     perplexity / specialisation margin. Requires save_per_round: true
#     in the config (this YAML sets it).
echo "[remote] probe_sft_capability ..."
PYTHONPATH=src python scripts/probe_sft_capability.py \
    --run-dir "\${OUT_DIR}" \
    --output-stem "scripts/figures/\${RUN_NAME_R}_probe"

echo "[remote] done."
REMOTE_EOF

# ----------------------------------------------------------------------
# 3. Pull figures + per-run JSON summaries back via scp.
#
# We deliberately do NOT pull the LoRA adapters (results/.../agents/)
# because they're ~25 MB each and stay on the remote where they were
# trained. Only the lightweight diagnostic outputs come back.
# ----------------------------------------------------------------------
if [[ "${SKIP_PULL:-0}" != "1" ]]; then
    echo "[pull] scp figures + JSON summaries back from ${REMOTE}"
    mkdir -p scripts/figures "results/${RUN_NAME}"

    # Figures: <run_name>.{pdf,png}, <run_name>_theory_vs_sft.{pdf,png},
    # <run_name>_probe*.{pdf,png}.
    scp "${REMOTE}:${REMOTE_REPO}/scripts/figures/${RUN_NAME}.pdf" \
        scripts/figures/ 2>/dev/null || true
    scp "${REMOTE}:${REMOTE_REPO}/scripts/figures/${RUN_NAME}.png" \
        scripts/figures/ 2>/dev/null || true
    scp "${REMOTE}:${REMOTE_REPO}/scripts/figures/${RUN_NAME}_theory_vs_sft.pdf" \
        scripts/figures/ 2>/dev/null || true
    scp "${REMOTE}:${REMOTE_REPO}/scripts/figures/${RUN_NAME}_theory_vs_sft.png" \
        scripts/figures/ 2>/dev/null || true
    # probe_sft_capability writes multiple files (loss curves + cross-
    # perplexity heatmap + summary CSV); pull anything matching the
    # _probe stem.
    scp "${REMOTE}:${REMOTE_REPO}/scripts/figures/${RUN_NAME}_probe*" \
        scripts/figures/ 2>/dev/null || true

    # Per-run lightweight artefacts.
    REMOTE_OUT_DIR=$(ssh "${REMOTE}" "cd ${REMOTE_REPO} && python - <<'PY'
import yaml
with open('${CONFIG}') as fh:
    cfg = yaml.safe_load(fh)
print(cfg.get('output_dir', 'results/${RUN_NAME}'))
PY")
    scp "${REMOTE}:${REMOTE_REPO}/${REMOTE_OUT_DIR}/history.json" \
        "results/${RUN_NAME}/" 2>/dev/null || true
    scp "${REMOTE}:${REMOTE_REPO}/${REMOTE_OUT_DIR}/theory_vs_sft.json" \
        "results/${RUN_NAME}/" 2>/dev/null || true
    scp "${REMOTE}:${REMOTE_REPO}/${REMOTE_OUT_DIR}/training.log" \
        "results/${RUN_NAME}/" 2>/dev/null || true
fi

echo
echo "loss_reweight run complete."
echo "  remote results : ${REMOTE}:${REMOTE_REPO}/results/${RUN_NAME}/"
echo "  local figures  : scripts/figures/${RUN_NAME}*.{pdf,png}"
echo "  local summary  : results/${RUN_NAME}/{history,theory_vs_sft}.json"
