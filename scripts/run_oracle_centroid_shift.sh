#!/usr/bin/env bash
# Oracle-centroid shift: colocated init at 1-component oracle centroids.
#
#   bash scripts/run_oracle_centroid_shift.sh
#   VERIFY_ONLY=1 bash scripts/run_oracle_centroid_shift.sh

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-results/oracle_centroid_shift/experiment.log}"
BASELINE_DIR="${BASELINE_DIR:-results/seven_axis_collapse_hypercube_ga_baseline/seed0}"
REF_CONFIG="${REF_CONFIG:-configs/benchmark/router/attribution_2x2/ga_theory_pre.yaml}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-configs/benchmark/router/oracle_centroid_shift/ga_theory_pre.yaml}"
RUN_DIR="${RUN_DIR:-results/oracle_centroid_shift/ga_theory_pre/seed0}"
FORCE="${FORCE:-0}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"

mkdir -p results/oracle_centroid_shift
exec > >(tee -a "$LOG") 2>&1

echo "=== oracle-centroid shift started $(date -Is) ==="

echo "--- build 1-component oracle centroids ---"
"${PY}" scripts/build_oracle_centroid_positions.py

echo "--- build experiment config ---"
"${PY}" scripts/build_oracle_centroid_shift_configs.py \
  > results/oracle_centroid_shift/config_manifest.json

echo "--- programmatic config diff (allowlist) ---"
"${PY}" scripts/diff_router_configs.py \
  --reference "$REF_CONFIG" \
  --candidate "$EXPERIMENT_CONFIG"

echo "--- verify colocated init on trait space ---"
"${PY}" scripts/verify_oracle_centroid_init.py --config "$EXPERIMENT_CONFIG"

echo "--- NOTE: after training, verify_oracle_centroid_persistence.py gates results ---"
echo "--- (init shift is cosmetic if dynamics pull centers back to ref GA) ---"

if [[ "$VERIFY_ONLY" == "1" ]]; then
  echo "VERIFY_ONLY=1; stopping before training."
  exit 0
fi

hist="${RUN_DIR}/history.json"
if [[ -f "$hist" && "$FORCE" != "1" ]]; then
  echo "--- skip training (history exists) ---"
else
  echo "--- closed-loop training ---"
  "${PY}" -m infl_ens.training --config "$EXPERIMENT_CONFIG"
fi

echo "--- verify effective centers persisted (not cosmetic revert) ---"
"${PY}" scripts/verify_oracle_centroid_persistence.py \
  --history "$hist" \
  --output-json "${RUN_DIR}/centroid_persistence.json"

echo "--- routing gate ---"
"${PY}" scripts/compare_routing_weights.py \
  --router-config "$EXPERIMENT_CONFIG" \
  --history "$hist" \
  --merge-run-dir "$RUN_DIR" \
  --baseline-run-dir "$BASELINE_DIR" \
  --partition test --max-eval-records 1000 \
  --merge-nll-cache results/attribution_2x2/ga_theory_pre/seed0/merge_nll_test.npy \
  --output-json "${RUN_DIR}/routing_weight_comparison.json"

echo "--- gap decomposition ---"
"${PY}" scripts/decompose_routing_gap.py \
  --router-config "$EXPERIMENT_CONFIG" \
  --history "$hist" \
  --merge-run-dir "$RUN_DIR" \
  --baseline-run-dir "$BASELINE_DIR" \
  --merge-nll-cache results/attribution_2x2/ga_theory_pre/seed0/merge_nll_test.npy \
  --output-json "${RUN_DIR}/routing_gap_decomposition.json"

echo "--- theory G vs oracle ---"
"${PY}" scripts/compare_theory_g_vs_oracle.py \
  --router-config "$EXPERIMENT_CONFIG" \
  --history "$hist" \
  --merge-run-dir "$RUN_DIR" \
  --merge-nll-cache results/attribution_2x2/ga_theory_pre/seed0/merge_nll_test.npy \
  --output-json "${RUN_DIR}/theory_g_vs_oracle.json"

echo "--- summary ---"
"${PY}" scripts/summarize_oracle_centroid_shift.py \
  --run-dir "$RUN_DIR" \
  --output-json results/oracle_centroid_shift/summary.json

echo "=== oracle-centroid shift finished $(date -Is) ==="
