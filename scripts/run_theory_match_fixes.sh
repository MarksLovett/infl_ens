#!/usr/bin/env bash
# Run fixes 2–4 evaluation (blend calibration + matched-pool theory + grad diagnostic).
#
# Usage:
#   bash scripts/run_theory_match_fixes.sh
#   DIAGNOSE_ONLY=1 bash scripts/run_theory_match_fixes.sh

set -euo pipefail

ROOT="${ROOT:-results/theory_match_fixes}"
PY="${PY:-.venv/bin/python}"
DIAGNOSE_ONLY="${DIAGNOSE_ONLY:-0}"

mkdir -p "${ROOT}"

if [[ "${DIAGNOSE_ONLY}" != "1" ]]; then
  ${PY} scripts/evaluate_theory_match_fixes.py \
    --root "${ROOT}" \
    --run-sweeps \
    --n-rounds 20 \
    --theory-steps 3000
else
  ${PY} scripts/evaluate_theory_match_fixes.py \
    --root "${ROOT}" \
    --n-rounds 20 \
    --theory-steps 3000
fi
