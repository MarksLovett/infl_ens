#!/usr/bin/env bash
# Full SFT sigma sweep: 20 seeds × 5 σ values × 40 rounds.
#
#   results/sigma_r40_20seeds/sigma*/seed*/
#   scripts/figures/sigma_r40_20seeds/per_run/... + aggregate/
#
# Default init: theory_gradient (separated start → GA → SFT at theory NE).
# Override: INIT_MODE=pairs_near_theory INIT_NOISE=0.01 bash ...
#
# After completion, classify final layouts:
#   .venv/bin/python scripts/count_equilibrium_types.py --root results/sigma_r40_20seeds
#
# Usage:
#   nohup bash scripts/run_sigma_sweep_r40_20seeds.sh \
#       > results/sigma_r40_20seeds/launch.log 2>&1 &

set -euo pipefail

export SIGMA_SWEEP_NAME=sigma_r40_20seeds
export ROUND_SWEEP_NAME=sigma_r40_20seeds_round_unused
export SKIP_ROUND_SWEEP=1
export SKIP_SIGMA_SWEEP=0
export INIT_MODE="${INIT_MODE:-theory_gradient}"
export INIT_NOISE="${INIT_NOISE:-0}"
export AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"
export SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
export SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.5 0.75 1 1.5}"
export SIGMA_SWEEP_N_ROUNDS="${SIGMA_SWEEP_N_ROUNDS:-40}"

exec bash scripts/run_pairs_near_eq_sweeps.sh
