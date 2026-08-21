#!/usr/bin/env bash
# Poll until seed-0 isolation finishes, then print summary.
set -euo pipefail
LOG="${1:-results/seed0_isolation/experiment.log}"
while ! grep -q 'seed-0 isolation finished' "$LOG" 2>/dev/null; do
  echo "$(date -Is) still running..."
  tail -1 "$LOG" 2>/dev/null || true
  sleep 120
done
echo "DONE"
tail -30 "$LOG"
if [[ -f results/seed0_isolation/summary.json ]]; then
  cat results/seed0_isolation/summary.json
fi
