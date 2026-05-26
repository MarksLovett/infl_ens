#!/usr/bin/env bash
# scripts/run_sigma_sweep_r20.sh
#
# One-shot sigma sweep for the cumulative-LoRA framework at 20 rounds.
#
# Steps performed, in order:
#
#   1. Pre-creates ``scripts/figures/sigma_sweep_r20_cum/sigma_{val}/``
#      subfolders, one per sigma value, so each experiment writes figures
#      into a unique directory.
#   2. Launches the sweep via ``run_sweep.sh sigma ...`` with
#      ``BASE_CONFIG=safety_truth_n4_r20_strategic_long_cum.yaml``. Sigma
#      values default to ``0.25 0.5 0.75 1.0 1.5`` but can be overridden
#      via ``SIGMAS`` env var. ``POST_PLOT`` and ``POST_THEORY`` are
#      disabled at the sweep level — we do them explicitly below into the
#      unique per-sigma figure subfolders.
#   3. After the sweep completes, runs three post-processing scripts per
#      sigma into the unique figure subfolder:
#        - ``plot_closed_loop_history.py``  → trajectory + utility tracking
#        - ``compare_theory_vs_sft.py``     → theory vs SFT comparison
#        - ``probe_sft_capability.py``      → cross-perplexity probe
#   4. Aggregates the whole sweep with ``plot_sweep.py --mode sigma``.
#   5. Prints a cross-sigma table of final-round specialisation margin.
#
# Environment variables (all optional):
#
#   SIGMAS         space-separated sigma_fraction values
#                  (default: "0.25 0.5 0.75 1.0 1.5")
#   SEED           seed override (default: 0)
#   BASE_CONFIG    base YAML config (default:
#                  configs/benchmark/router/safety_truth_n4_r20_strategic_long_cum.yaml)
#   RESULTS_ROOT   sweep results dir
#                  (default: results/sweep_sigma_r20_strategic_long_cum)
#   FIG_ROOT       figures root for this sweep
#                  (default: scripts/figures/sigma_sweep_r20_cum)
#   MAX_PROMPTS    --max-prompts forwarded to the probe (default: 128)
#   SKIP_EXISTING  set to 0 to force re-train sigmas with existing
#                  history.json (default: 1)
#   SKIP_TRAIN     set to 1 to skip the sweep launch and only re-run
#                  post-processing (default: 0)
#
# Usage examples:
#
#   bash scripts/run_sigma_sweep_r20.sh
#       Default cumulative-LoRA sweep at sigma in {0.25, 0.5, 0.75, 1.0, 1.5}.
#
#   SIGMAS="0.3 0.5 0.7" bash scripts/run_sigma_sweep_r20.sh
#       Custom sigma list.
#
#   SKIP_TRAIN=1 bash scripts/run_sigma_sweep_r20.sh
#       Post-process and aggregate without re-training.
#
#   BASE_CONFIG=configs/benchmark/router/safety_truth_n4_r20_strategic_long.yaml \
#   RESULTS_ROOT=results/sweep_sigma_r20_strategic_long \
#   FIG_ROOT=scripts/figures/sigma_sweep_r20_indep \
#       bash scripts/run_sigma_sweep_r20.sh
#       Same sweep against the independent-LoRA framework for comparison.

set -euo pipefail

# ---------- env defaults ----------
SIGMAS="${SIGMAS:-0.25 0.5 0.75 1.0 1.5}"
SEED="${SEED:-0}"
BASE_CONFIG="${BASE_CONFIG:-configs/benchmark/router/safety_truth_n4_r20_strategic_long_cum.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-results/sweep_sigma_r20_strategic_long_cum}"
FIG_ROOT="${FIG_ROOT:-scripts/figures/sigma_sweep_r20_cum}"
MAX_PROMPTS="${MAX_PROMPTS:-128}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# Resolve relative to repo root so the script works from any CWD.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
cd "${REPO_ROOT}"

echo "==========================================================="
echo " sigma sweep r20 — cumulative LoRA"
echo "==========================================================="
echo "  BASE_CONFIG    ${BASE_CONFIG}"
echo "  SIGMAS         ${SIGMAS}"
echo "  SEED           ${SEED}"
echo "  RESULTS_ROOT   ${RESULTS_ROOT}"
echo "  FIG_ROOT       ${FIG_ROOT}"
echo "  MAX_PROMPTS    ${MAX_PROMPTS}"
echo "  SKIP_EXISTING  ${SKIP_EXISTING}"
echo "  SKIP_TRAIN     ${SKIP_TRAIN}"
echo "==========================================================="

# ---------- prep ----------
mkdir -p "${FIG_ROOT}/aggregate"
for s in ${SIGMAS}; do
    mkdir -p "${FIG_ROOT}/sigma_${s}"
done

# ---------- 1. launch the sweep ----------
if [[ "${SKIP_TRAIN}" != "1" ]]; then
    echo ""
    echo "--- launching sweep ---"
    BASE_CONFIG="${BASE_CONFIG}" \
    RESULTS_ROOT="${RESULTS_ROOT}" \
    SEED="${SEED}" \
    POST_PLOT=0 \
    POST_THEORY=0 \
    SKIP_EXISTING="${SKIP_EXISTING}" \
    bash scripts/run_sweep.sh sigma ${SIGMAS}
else
    echo "SKIP_TRAIN=1 — skipping sweep launch"
fi

# ---------- 2. per-sigma post-processing ----------
echo ""
echo "--- per-sigma post-processing ---"
for s in ${SIGMAS}; do
    rundir="${RESULTS_ROOT}/sigma_${s}"
    figdir="${FIG_ROOT}/sigma_${s}"
    echo ""
    echo "=== sigma=${s} ==="
    if [[ ! -f "${rundir}/history.json" ]]; then
        echo "  no history.json in ${rundir} — skipping"
        continue
    fi

    echo "  [trajectory]"
    python scripts/plot_closed_loop_history.py \
        --history "${rundir}/history.json" \
        --axis-labels harm hallucination \
        --title "sigma_fraction=${s}, 20 rounds, cumulative LoRA" \
        --output-stem "${figdir}/trajectory"

    echo "  [theory_vs_sft]"
    python scripts/compare_theory_vs_sft.py \
        --config  "${BASE_CONFIG}" \
        --history "${rundir}/history.json" \
        --axis-labels harm hallucination \
        --title "Theory vs SFT (sigma_fraction=${s}, cumulative)" \
        --output-stem "${figdir}/theory_vs_sft" \
        --summary-json "${rundir}/theory_vs_sft.json"

    echo "  [capability probe]"
    python scripts/probe_sft_capability.py \
        --run-dir "${rundir}" \
        --base-sft-dir "${rundir}/agents" \
        --max-prompts "${MAX_PROMPTS}" \
        --output-stem "${figdir}/probe" \
        --title "Capability probe (sigma=${s}, cumulative, 20 rounds)"
done

# ---------- 3. aggregate ----------
echo ""
echo "--- aggregating sweep ---"
python scripts/plot_sweep.py \
    --root "${RESULTS_ROOT}" \
    --mode sigma \
    --title "Sigma sweep — cumulative LoRA, 20 rounds, seed ${SEED}" \
    --with-theory \
    --output-stem "${FIG_ROOT}/aggregate/sigma_sweep_r20_cum"

# ---------- 4. cross-sigma margin table ----------
echo ""
echo "--- cross-sigma final-round specialisation margins ---"
printf "%-8s %-14s %-12s %-12s\n" "sigma" "final_margin" "diag_NLL" "off_NLL"
printf "%-8s %-14s %-12s %-12s\n" "-----" "------------" "--------" "-------"
for s in ${SIGMAS}; do
    csv="${FIG_ROOT}/sigma_${s}/probe.csv"
    if [[ ! -f "${csv}" ]]; then
        printf "%-8s %s\n" "${s}" "(no probe csv)"
        continue
    fi
    python -c "
import csv as _csv, numpy as np
rows = list(_csv.DictReader(open('${csv}')))
last = max(int(r['round']) for r in rows)
diag = [float(r['nll']) for r in rows if r['agent_i']==r['agent_j'] and int(r['round'])==last]
off  = [float(r['nll']) for r in rows if r['agent_i']!=r['agent_j'] and int(r['round'])==last]
print(f'${s:<8} {np.mean(off)-np.mean(diag):<14.4f} {np.mean(diag):<12.4f} {np.mean(off):<12.4f}')
"
done

echo ""
echo "--- DONE ---"
echo "Figures written under: ${FIG_ROOT}/"
echo "Aggregate figure:      ${FIG_ROOT}/aggregate/sigma_sweep_r20_cum.{pdf,png,csv}"
