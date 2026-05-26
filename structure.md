# structure.md

Authoritative file-by-file map of the `infl_ens` repository. See
`AGENTS.md` §4 rule 10: any PR adding, moving, renaming, or deleting a
file under `src/`, `scripts/`, `configs/`, or `tests/` MUST update this
file in the same edit.

## On-disk tree

```
infl_ens/
├── src/
│   └── infl_ens/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── __main__.py                       single CLI for the data submodule
│       │   ├── encoders.py                       SentenceTransformer / HF embedding wrappers
│       │   ├── trait_space.py                    TraitSpace + build_trait_space + position_from_corpus
│       │   └── benchmarks/
│       │       ├── __init__.py
│       │       ├── base.py                       BenchmarkSplit container
│       │       ├── beavertails.py                BeaverTails loader (harm axis)
│       │       ├── halueval.py                   HaluEval loader (hallucination axis)
│       │       └── safety_trait_space.py         2-D BeaverTails + HaluEval trait-space builder
│       ├── inflgame/
│       │   ├── __init__.py
│       │   └── router/
│       │       ├── __init__.py
│       │       ├── agents.py                     RouterAgent
│       │       ├── allocation.py                 G_i, u_i, ∇u_i
│       │       └── core.py                       InfluencerRouter
│       ├── training/
│       │   ├── __init__.py
│       │   ├── __main__.py                       single CLI; dispatches by `task` field
│       │   ├── router_training.py                gradient-ascent on positions
│       │   └── sft_training.py                   LoRA SFT (Qwen2.5-1.5B-Instruct by default)
│       └── utils/
│           ├── __init__.py
│           ├── agent_init.py                     mean_noise, pairs_near_theory, theory_gradient inits
│           ├── position_step.py                  adaptive EMA blend / expected_pool centroid
│           └── resource.py                       weighted_mean, weighted_covariance, σ₀*
├── scripts/
│   ├── download_beavertails.py                   one-off download (HF datasets)
│   ├── download_halueval.py                      one-off download (RUCAIBox/HaluEval JSON files)
│   ├── build_safety_trait_space.py               wrapper around `python -m infl_ens.data`
│   ├── compare_utility_estimators.py             u_grid vs u_pool vs share diagnostic
│   ├── diagnose_trait_support.py                 KDE vs empirical resource density; explains SFT-vs-theory gaps
│   ├── compare_theory_vs_sft.py                  strategic Nash vs SFT closed-loop endpoints
│   ├── plot_closed_loop_history.py               trajectories + utility tracking from history.json
│   ├── run_sweep.sh                              bash sweep launcher (seeds / sigma / kde)
│   ├── run_sigma_sweep_r20.sh                    end-to-end cumulative-LoRA sigma sweep at 20 rounds
│   ├── plot_sweep.py                             aggregate sweep results, classify equilibria
│   ├── probe_sft_capability.py                   cross-perplexity probe over saved per-round adapters
│   ├── closed_loop_demo.py                       toy closed-loop simulation
│   └── smoke_test.py                             pipeline sanity check
├── configs/
│   ├── model/
│   │   └── qwen2_5_1_5b.yaml                     base-model + LoRA hyperparams
│   ├── data/
│   │   ├── beavertails.yaml
│   │   └── halueval.yaml
│   └── benchmark/
│       └── router/
│           ├── example.yaml                      original synthetic three-anchor example
│           ├── safety_truth.yaml                 2-D BeaverTails + HaluEval (closed loop)
│           ├── safety_truth_n4_r10_strategic.yaml  G(1-G) strategic-gradient routing variant
│           ├── safety_truth_n4_r10_strategic_long.yaml  same routing, ~6× more SFT per round
│           ├── safety_truth_n4_r20_strategic_long.yaml  same as r10_strategic_long but 20 rounds
│           ├── safety_truth_n4_r40_strategic_long.yaml  same as r10_strategic_long but 40 rounds
│           ├── safety_truth_n4_r10_strategic_long_cum.yaml  cumulative-LoRA variant (10 rounds)
│           ├── safety_truth_n4_r20_strategic_long_cum.yaml  cumulative-LoRA variant (20 rounds)
│           ├── safety_truth_n4_r40_strategic_long_cum.yaml  cumulative-LoRA variant (40 rounds)
│           ├── beavertails_only.yaml             1-D harm-axis ablation
│           └── halueval_only.yaml                1-D hallucination-axis ablation
├── tests/
│   ├── test_benchmark_loaders.py                 offline tests with synthetic JSON fixtures
│   └── test_safety_trait_space.py                offline tests using a toy encoder
├── data/                                         gitignored
└── results/                                      gitignored
```

## File-by-file tables

### `src/infl_ens/data/`

| File | Role | Key public symbols |
|---|---|---|
| `__init__.py` | Re-exports | `TraitSpace`, `build_trait_space`, `position_from_corpus`, `SentenceTransformerEncoder`, `HuggingFaceEncoder`, `benchmarks` |
| `__main__.py` | Single CLI: `python -m infl_ens.data {preview,build-safety-trait-space}` | `main` |
| `encoders.py` | Sentence-embedding callables for trait-space construction | `SentenceTransformerEncoder`, `HuggingFaceEncoder` |
| `trait_space.py` | Trait space :math:`\mathbb{B}` and resource distribution :math:`B(b)` | `TraitSpace`, `build_trait_space`, `position_from_corpus` |

### `src/infl_ens/data/benchmarks/`

| File | Role | Key public symbols |
|---|---|---|
| `__init__.py` | Re-exports | `BenchmarkSplit`, `load_beavertails`, `load_halueval`, `build_safety_trait_space`, `LearnedAxis`, `BEAVERTAILS_CATEGORIES`, `HALUEVAL_TASKS` |
| `base.py` | Uniform benchmark record container | `BenchmarkSplit` |
| `beavertails.py` | BeaverTails loader and harm-score scoring | `load_beavertails`, `BEAVERTAILS_CATEGORIES` |
| `halueval.py` | HaluEval loader and hallucination-score scoring | `load_halueval`, `HALUEVAL_TASKS` |
| `safety_trait_space.py` | 2-D learned-anchor trait space from labelled benchmarks | `build_safety_trait_space`, `LearnedAxis` |

### `src/infl_ens/inflgame/router/`

| File | Role | Key public symbols |
|---|---|---|
| `__init__.py` | Re-exports | `RouterAgent`, `InfluencerRouter`, `allocation_weights`, `expected_utilities`, `empirical_utility`, `strategic_routing_weights`, `utility_gradient` |
| `agents.py` | Router-agent dataclass and calibration-based init | `RouterAgent`, `RouterAgent.from_calibration` |
| `allocation.py` | Allocation math :math:`G_i, u_i, \hat u_i, \nabla_{x_i} u_i, p_i^{strat}` | `allocation_weights`, `expected_utilities`, `empirical_utility`, `strategic_routing_weights`, `utility_gradient` |
| `core.py` | Public router class | `InfluencerRouter` |

### `src/infl_ens/training/`

| File | Role | Key public symbols |
|---|---|---|
| `__init__.py` | Eager re-export of router training; lazy proxy for SFT | `RouterTrainingConfig`, `train_router_positions`, (lazy) `SFTTrainingConfig`, `sft_train_agent` |
| `__main__.py` | Single CLI: `python -m infl_ens.training --config <path>` dispatches on the config's `task` field. Closed-loop task honours `closed_loop.routing_weight` (`G` / `G_times_1mG`), `closed_loop.loss_reweight`, `closed_loop.init_noise` (Gaussian symmetry-breaking at clone start), and `closed_loop.save_per_round`; always logs `agent_prompts` / `agent_responses` / `agent_sft_logs` per round in `history.json`. | `main` |
| `router_training.py` | Gradient-ascent loop on agent positions | `RouterTrainingConfig`, `train_router_positions` |
| `sft_training.py` | LoRA SFT for a single :class:`RouterAgent`; accepts `out_dir_override` for per-round adapter archiving; accepts `cfg.cumulative_lora=True` to load and continue training the prior adapter rather than starting fresh; returns `log_history` and `loaded_prior_lora` from the SFT trainer's state | `SFTTrainingConfig`, `sft_train_agent` |

### `src/infl_ens/utils/`

| File | Role | Key public symbols |
|---|---|---|
| `__init__.py` | Re-exports | `weighted_mean`, `weighted_covariance`, `gaussian_stability_threshold` |
| `resource.py` | Pure helpers on (grid, weights) pairs | `weighted_mean`, `weighted_covariance`, `gaussian_stability_threshold` |

### `scripts/`

| File | Role |
|---|---|
| `download_beavertails.py` | Downloads `PKU-Alignment/BeaverTails` to `data/beavertails/` via the `datasets` library |
| `download_halueval.py` | Downloads HaluEval task JSON files from `RUCAIBox/HaluEval` to `data/halueval/` |
| `build_safety_trait_space.py` | Convenience wrapper around `python -m infl_ens.data build-safety-trait-space` |
| `compare_utility_estimators.py` | Side-by-side comparison of grid :math:`u_i`, empirical-pool :math:`\hat u_i`, and finite-batch proportional share. `--mode {toy,safety}`. |
| `diagnose_trait_support.py` | Diagnoses the KDE-smoothed resource density :math:`B(b)` vs the actual prompt projections in trait space, recomputes theoretical Nash, and reports per-agent density / prompt-count metrics at theory NE and SFT end. Supports `--density-mode {kde,empirical,both}`: `empirical` rebuilds :math:`B(b)` as a 2-D histogram of projected prompts on the same `n_grid × n_grid` lattice (mass only where prompts actually project; tunable `--empirical-smoothing-cells` to avoid hard zeros), while `both` produces a 2-panel comparison figure plus both console tables side-by-side. `--config-override KEY=VAL` repeatable for bandwidth/sigma sweeps without YAML edits. |
| `compare_theory_vs_sft.py` | Rebuilds the trait space from a closed-loop run's config, initialises agents from `history.json` round 0, runs `train_router_positions`, and compares the strategic-Nash endpoints with the SFT trajectory in trait space. |
| `plot_closed_loop_history.py` | Reads `history.json` from a closed-loop run and renders trajectories + utility tracking to PDF/PNG under `scripts/figures/`. |
| `run_sweep.sh` | Bash launcher that sweeps one parameter (seeds, sigma_fraction, or kde_bandwidth) over the closed-loop trainer. Skips runs whose `history.json` already exists; optionally runs per-run plotting and theory comparison after each training. |
| `run_sigma_sweep_r20.sh` | End-to-end wrapper for the cumulative-LoRA sigma sweep at 20 rounds: pre-creates unique figure subfolders, launches `run_sweep.sh sigma`, runs trajectory + theory_vs_sft + capability probe per sigma into its own subfolder, aggregates with `plot_sweep.py`, prints a cross-sigma specialisation-margin table. Defaults to cumulative framework + `safety_truth_n4_r20_strategic_long_cum.yaml`; switchable to independent framework via env-var overrides. |
| `plot_sweep.py` | Aggregates a sweep root directory into one figure: per-run trajectory panels, equilibrium-type classification by single-linkage clustering, optional overlay of theoretical Nash endpoints, CSV summary. |
| `probe_sft_capability.py` | Capability probe: reads a closed-loop run with `save_per_round: true` and computes Tier 1 (per-agent SFT loss curves) + Tier 3 (cross-perplexity matrix and specialisation margin per round). Headline output is `μ(r) = NLL(others) − NLL(own)`; negligible μ means SFT isn't actually differentiating the agents and observed position dynamics are pure routing-centroid geometry. |
| `run_position_only_cum_r10.sh` | Single 10-round launcher for the matched `position_only` config (`batch_size=256`, cumulative LoRA): trains to `results/position_only_cum_round_sweep/r10/`, then runs trajectory + theory_vs_sft + capability probe figures. |
| `run_position_only_cum_sweeps.sh` | Two-pass sweep (rounds 10/20/40; sigma 0.25–1.5× threshold at 20 rounds) over the matched `position_only_cum` config. Mirrors `run_loss_reweight_cum_sweeps.sh`. Supports `REDO_SIGMA_SWEEP=1`, `SKIP_ROUND_SWEEP=1`. |
| `run_position_only_cum_sigma_redo.sh` | Sigma sweep only: wipes `position_only_cum_sigma_sweep`, re-trains with `init_noise` from config. |
| `run_position_only_seed_sigma_sweep.sh` | Resumable seed×sigma grid (default 5 seeds × 5 sigmas, 20 rounds) for `position_only_cum`; per-run figures under `scripts/figures/<SWEEP_NAME>/per_run/`; calls `aggregate_seed_sigma_sweep.py` for mean±std aggregates. Extend via `SEEDS` / `SIGMA_VALUES`. |
| `aggregate_seed_sigma_sweep.py` | Aggregates `results/<sweep>/sigma*/seed*/` or `r*/seed*/` into trajectory, spread, probe-margin, and overview figures plus `summary.csv` under `scripts/figures/<sweep>/aggregate/`. |
| `run_pairs_near_eq_sweeps.sh` | Full SFT sweeps with `pairs_near_theory` init: PASS 1 seed×rounds `{10,20,40}`; PASS 2 seed×σ `{0.25…1.5}` at 20 rounds; probe + aggregate per pass. |
| `run_pairs_near_theory_10seeds.sh` | Position-only sim: `pairs_near_theory` init, 10 seeds × 2 σ (fast, no SFT). |
| `summarize_pairs_near_theory.py` | (2,2) vs collapsed summary for position-only pair-init runs. |
| `simulate_position_only_loop.py` | Fast closed-loop with routing + `(1-G)` centroid updates only (no SFT). Centroid modes: `batch`, `full_pool` (~10k prompts), `expected_pool` (deterministic static limit). |
| `run_large_batch_static_analysis.sh` | Compares batch 256, batch 10k, full pool, and expected pool at σ=0.25/0.75 (no SFT). |
| `compare_batch_size_static.py` | Summarises `large_batch_static_analysis` vs optional prior batch-256 sweep. |
| `run_position_fix_comparison.sh` | A/B fixes at σ=0.25/0.75, batch 256: baseline vs `expected_pool` vs `init_noise=0.01` (sim, no SFT). |
| `run_pool_and_noise_10seeds.sh` | Both fixes (`expected_pool` + `init_noise=0.01`), 10 seeds; calls `aggregate_final_positions.py`. |
| `aggregate_final_positions.py` | Mean ± std of final clone positions and pairwise spread across seeds. |
| `run_position_step_stability_test.sh` | Pre-sweep grid over position-step policies at σ=0.25/0.75 via `simulate_position_only_loop.py` (seconds per cell, not full training). |
| `compare_position_step_modes.py` | Summarises stability-test runs: final spread, mean effective blend, collapse flags; overview figure under `scripts/figures/position_step_stability_test/`. |
| `verify_position_update.py` | Audits `history.json` position deltas against weighted vs unweighted centroid predictions (detects pre-fix `position_only` centroid bugs). |
| `compare_histories.py` | Compares per-round position trajectories between two closed-loop `history.json` files. |
| `closed_loop_demo.py` | Toy hash-bag closed-loop simulation (no external deps) |
| `smoke_test.py` | End-to-end pipeline sanity check |

### `configs/`

| File | Role |
|---|---|
| `model/qwen2_5_1_5b.yaml` | Base-model + LoRA hyperparameters for the SFT trainer |
| `data/beavertails.yaml` | Static BeaverTails loader settings |
| `data/halueval.yaml` | Static HaluEval loader settings |
| `benchmark/router/example.yaml` | Original synthetic three-anchor example |
| `benchmark/router/safety_truth.yaml` | 2-D BeaverTails + HaluEval closed-loop config |
| `benchmark/router/safety_truth_n4_r10_strategic.yaml` | Same as safety_truth but with `routing_weight: G_times_1mG` (strategic-gradient correspondence under MV-Gaussian kernels) |
| `benchmark/router/safety_truth_n4_r10_strategic_long.yaml` | Same as ..._strategic but pushes the SFT step harder (3 epochs, batch 512, per-device 16, logging_steps=1). Use the capability probe as the overfitting detector on this run. |
| `benchmark/router/safety_truth_n4_r20_strategic_long.yaml` | Same as ..._r10_strategic_long but 20 rounds — tests stability of the strategic (2,2) basin under additional SFT. |
| `benchmark/router/safety_truth_n4_r40_strategic_long.yaml` | Same as ..._r10_strategic_long but 40 rounds — long-horizon stability + overfitting check. |
| `benchmark/router/safety_truth_n4_r{10,20,40}_strategic_long_cum.yaml` | Cumulative-LoRA variants of the three strategic_long configs: each agent loads its prior adapter and continues training rather than restarting from the base model every round. Capability accumulates across rounds. Saves to its own `results/safety_truth_n4_r*_strategic_long_cum/` directories so the original (independent-round) framework is preserved for comparison. |
| `benchmark/router/beavertails_only.yaml` | 1-D harm-axis ablation |
| `benchmark/router/halueval_only.yaml` | 1-D hallucination-axis ablation |

### `tests/`

| File | Role |
|---|---|
| `test_benchmark_loaders.py` | Offline tests for BeaverTails and HaluEval loaders |
| `test_safety_trait_space.py` | Offline tests for `build_safety_trait_space` |

## `__init__.py` re-export summary

- `src/infl_ens/__init__.py`: minimal — does not import subpackages eagerly.
- `src/infl_ens/data/__init__.py`: `TraitSpace`, `build_trait_space`, `position_from_corpus`, `SentenceTransformerEncoder`, `HuggingFaceEncoder`, `benchmarks`.
- `src/infl_ens/data/benchmarks/__init__.py`: `BenchmarkSplit`, `load_beavertails`, `load_halueval`, `build_safety_trait_space`, `LearnedAxis`, `BEAVERTAILS_CATEGORIES`, `HALUEVAL_TASKS`.
- `src/infl_ens/inflgame/__init__.py`: re-exports the `router` subpackage.
- `src/infl_ens/inflgame/router/__init__.py`: `InfluencerRouter`, `RouterAgent`, `allocation_weights`, `expected_utilities`, `utility_gradient`.
- `src/infl_ens/training/__init__.py`: `RouterTrainingConfig`, `train_router_positions`; lazy `SFTTrainingConfig`, `sft_train_agent` (avoids importing torch/transformers at package import time).
- `src/infl_ens/utils/__init__.py`: `weighted_mean`, `weighted_covariance`, `gaussian_stability_threshold`.
