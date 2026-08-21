#!/usr/bin/env bash
# Wait for 6-round posttrain/eval, then launch r24 pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
while pgrep -f run_seven_axis_split_posttrain >/dev/null 2>&1; do
  echo "waiting for posttrain..."
  sleep 30
done
while pgrep -f "infl_ens.evaluation.*seven_axis_split" >/dev/null 2>&1; do
  echo "waiting for split eval..."
  sleep 30
done
echo "starting r24 pipeline $(date -Is)"
bash scripts/run_seven_axis_split_r24.sh
