#!/usr/bin/env bash
# Two-seed comparison at 40 rounds with theory_gradient init.
#
#   results/theory_grad_r40_compare/r40/seed{0,1}/
#   scripts/figures/theory_grad_r40_compare/per_run/r40/seed*/
#
# Usage:
#   bash scripts/run_theory_grad_r40_2seeds.sh
#   AGGREGATE_ONLY=1 bash scripts/run_theory_grad_r40_2seeds.sh

set -euo pipefail

export ROUND_SWEEP_NAME=theory_grad_r40_compare
export SEEDS="0 1"
export ROUND_VALUES="40"
export SKIP_SIGMA_SWEEP=1
export SKIP_ROUND_SWEEP=0
export INIT_MODE=theory_gradient
export INIT_NOISE=0
export AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"

exec bash scripts/run_pairs_near_eq_sweeps.sh
