#!/usr/bin/env bash
# Full SFT sigma sweep: pairs_near_theory init (pairwise jitter around (2,2) ref).
#
#   10 seeds × σ ∈ {0.25, 0.5, 0.75, 1, 1.5}  ×  20 rounds
#
#   Agents 0,1 start near the low-harm pair; 2,3 near the high-harm pair of the
#   per-σ (2,2) reference (baseline_blend05 / pool_and_noise, else GA fallback).
#
#   results/pairs_near_theory_sigma_sweep/sigma*/seed*/
#   scripts/figures/pairs_near_theory_sigma_sweep/per_run/... + aggregate/
#
# Resumable: skips cells with history.json. Re-run aggregate only:
#   AGGREGATE_ONLY=1 bash scripts/run_pairs_near_theory_sigma_sweep.sh
#
# Usage:
#   nohup bash scripts/run_pairs_near_theory_sigma_sweep.sh \
#       > results/pairs_near_theory_sigma_sweep/launch.log 2>&1 &

set -euo pipefail

export SIGMA_SWEEP_NAME=pairs_near_theory_sigma_sweep
export ROUND_SWEEP_NAME=pairs_near_theory_sigma_sweep_round_unused
export SKIP_ROUND_SWEEP=1
export SKIP_SIGMA_SWEEP=0
export INIT_MODE=pairs_near_theory
export INIT_NOISE="${INIT_NOISE:-0.01}"
export THEORY_REF_ROOT="${THEORY_REF_ROOT:-results/theory_match_fixes/baseline_blend05}"
export AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"
export SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
export SIGMA_VALUES="${SIGMA_VALUES:-0.25 0.5 0.75 1 1.5}"
export SIGMA_SWEEP_N_ROUNDS="${SIGMA_SWEEP_N_ROUNDS:-20}"

exec bash scripts/run_pairs_near_eq_sweeps.sh
