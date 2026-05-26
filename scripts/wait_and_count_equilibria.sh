#!/usr/bin/env bash
# Wait for all history.json under ROOT, then run count_equilibrium_types.py.
set -euo pipefail
ROOT="${1:-results/position_only_sigma_r40_20seeds}"
TARGET="${2:-100}"
PY="${PY:-.venv/bin/python}"
while true; do
  n=$(find "${ROOT}" -name history.json 2>/dev/null | wc -l)
  echo "$(date +%H:%M:%S) ${ROOT}: ${n}/${TARGET}"
  if [[ "${n}" -ge "${TARGET}" ]]; then
    break
  fi
  sleep 120
done
cd "${ROOT%/results/*}" 2>/dev/null || cd ~/infl_ens
${PY} scripts/count_equilibrium_types.py --root "${ROOT}"
