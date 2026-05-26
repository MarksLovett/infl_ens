#!/usr/bin/env bash
# Resume pairs_near_eq_sigma_sweep (skips cells with history.json).
#
# Original sweep: pairs_near_theory init, 10 seeds, 20 rounds, Ïƒ âˆˆ {0.25â€¦1.5}.
# At last stop: Ïƒ=0.25 and Ïƒ=0.5 complete; Ïƒ=0.75 partial; Ïƒ=1 and 1.5 pending.
#
# Usage:
#   nohup bash scripts/run_finish_pairs_near_eq_sigma_sweep.sh \
#       >> results/pairs_near_eq_sigma_sweep/launch.log 2>&1 &

set -euo pipefail

export SIGMA_SWEEP_NAME=pairs_near_eq_sigma_sweep
export ROUND_SWEEP_NAME=pairs_near_eq_sigma_sweep_round_unused
export SKIP_ROUND_SWEEP=1
export SKIP_SIGMA_SWEEP=0
export INIT_MODE=pairs_near_theory
export INIT_NOISE=0.01
export THEORY_REF_ROOT="${THEORY_REF_ROOT:-results/theory_match_fixes/baseline_blend05}"
export AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"
export SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
export SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.5 0.75 1 1.5}"
export SIGMA_SWEEP_N_ROUNDS="${SIGMA_SWEEP_N_ROUNDS:-20}"

exec bash scripts/run_pairs_near_eq_sweeps.sh
