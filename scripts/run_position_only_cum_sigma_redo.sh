#!/usr/bin/env bash
# scripts/run_position_only_cum_sigma_redo.sh
#
# Re-run ONLY the position_only sigma sweep after adding init_noise.
# Deletes prior results/figures under position_only_cum_sigma_sweep,
# then trains sigma_fraction ∈ {0.25, 0.5, 0.75, 1, 1.5} at 20 rounds
# with theory plots at the matching sigma.
#
# Usage (on doob):
#   nohup bash scripts/run_position_only_cum_sigma_redo.sh \
#       > results/position_only_cum_sigma_redo.log 2>&1 &

set -euo pipefail

export REDO_SIGMA_SWEEP=1
export SKIP_ROUND_SWEEP=1
export ROUND_VALUES=""
export SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.5 0.75 1 1.5}"
export SIGMA_SWEEP_N_ROUNDS="${SIGMA_SWEEP_N_ROUNDS:-20}"

exec bash scripts/run_position_only_cum_sweeps.sh
