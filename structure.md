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
│       │   ├── trait_space_cache.py              fingerprinted on-disk cache + build_or_load helper
│       │   ├── trait_linear_transform.py           frozen unsupervised affine trait transforms
│       │   ├── trait_normalize.py                per-axis quantile (empirical-CDF) normalizer to [0,1]^L
│       │   ├── position_blend.py                 EMA blend toward corpus centroid (apply_position_update)
│       │   └── benchmarks/
│       │       ├── __init__.py
│       │       ├── base.py                       BenchmarkSplit container
│       │       ├── beavertails.py                BeaverTails loader (harm axis)
│       │       ├── halueval.py                   HaluEval loader (hallucination axis)
│       │       ├── jbb_behaviors.py              JBB-Behaviors loader (jailbreak axis)
│       │       ├── toxicchat.py                  ToxicChat loader (legacy jailbreak axis)
│       │       ├── ai4privacy.py                 AI4Privacy loader (privacy axis)
│       │       ├── orbench.py                    OR-Bench loader (over-refusal axis)
│       │       ├── prompt_injection.py           prompt-injection loader (injection axis)
│       │       ├── do_not_answer.py              Do-Not-Answer loader (policy-violation axis)
│       │       └── safety_trait_space.py         multi-axis learned benchmark trait-space builder
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
│       │   ├── sft_training.py                   LoRA SFT (Qwen2.5-1.5B-Instruct by default)
│       │   ├── baseline_replay.py                pooled baseline SFT from history.json batches
│       │   ├── merge_training.py                 pair-merge SFT helpers (4 routers, 2 LoRAs)
│       │   ├── sweep_aggregate.py                sweep summaries, CSV export, seed×σ aggregation
│       │   └── theory_vs_sft.py                  strategic ascent vs SFT trajectory helpers
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── __main__.py                       single CLI; adapter / run eval on benchmarks
│       │   ├── adapters.py                       resolve + load saved LoRA checkpoints
│       │   ├── benchmarks.py                     YAML-driven BeaverTails / HaluEval loading
│       │   ├── metrics.py                        mean NLL on chat-formatted splits
│       │   ├── evaluate.py                       evaluate_adapter_on_* + JSON report writer
│       │   ├── aggregate.py                      mean ± std of eval metrics across seeds
│       │   ├── base_eval.py                      base-model (no adapter) benchmark NLL
│       │   ├── compare.py                        baseline/specialist/merge adapter comparison
│       │   └── capability_probe.py               SFT cross-perplexity probe helpers
│       ├── vis/
│       │   ├── __init__.py
│       │   ├── benchmark_nll_bar.py              grouped bar chart: base vs adapter benchmark NLL
│       │   ├── benchmark_space.py                pairwise trait-space resource heatmaps
│       │   ├── capability_probe.py               SFT loss curves + cross-NLL probe figure
│       │   ├── closed_loop.py                    closed-loop trajectories + utility tracking
│       │   ├── save.py                           optional PDF/PNG export helper
│       │   ├── sweeps.py                         flat / seed×σ sweep comparison figures
│       │   └── theory_vs_sft.py                  theory gradient ascent vs SFT overlay figure
│       ├── latex/
│       │   ├── With canonical routing,.tex       derivation of position-only Gaussian update
│       │   ├── kernel_agnostic_gradient_step.tex kernel-agnostic gradient-matched update note
│       │   └── trait_axis_assignment.tex         how prompts get trait-axis coordinates
│       └── utils/
│           ├── __init__.py
│           ├── agent_init.py                     mean_noise, pairs_near_theory, merge_near_theory, theory_gradient inits
│           ├── checkpoints.py                    prune intermediate round-* LoRA dirs
│           ├── position_step.py                  blend schedule + expected_pool centroid (re-exports position_blend)
│           ├── resource.py                       weighted_mean, weighted_covariance, σ₀*
│           └── sweep_discovery.py                discover sigma×seed history.json trees
├── scripts/
│   ├── download_beavertails.py                   one-off download (HF datasets)
│   ├── download_halueval.py                      one-off download (RUCAIBox/HaluEval JSON files)
│   ├── download_jbb_behaviors.py                 one-off download (JailbreakBench JBB-Behaviors CSVs)
│   ├── download_toxicchat.py                     one-off download (LMSYS ToxicChat CSV files)
│   ├── download_ai4privacy.py                    one-off download (AI4Privacy JSONL files)
│   ├── download_orbench.py                       one-off download (OR-Bench CSV files)
│   ├── download_prompt_injection.py              one-off download (Protect AI injection JSONL)
│   ├── download_do_not_answer.py                 one-off download (Do-Not-Answer + Alpaca negatives)
│   ├── build_safety_trait_space.py               wrapper around `python -m infl_ens.data`
│   ├── plot_trait_representation.py              clipped-vs-quantile trait representation figures
│   ├── run_trait_representation_on_doob.sh       sync to doob, build representation figures, pull back
│   ├── compare_utility_estimators.py             u_grid vs u_pool vs share diagnostic
│   ├── diagnose_axis_separability.py             axis AUC / Cohen's d / saturation diagnostics
│   ├── diagnose_trait_support.py                 KDE vs empirical resource density; explains SFT-vs-theory gaps
│   ├── compare_theory_vs_sft.py                  strategic Nash vs SFT closed-loop endpoints
│   ├── compare_runs.py                           thin CLI → :mod:`infl_ens.vis.closed_loop.plot_trajectory_overlay`
│   ├── plot_closed_loop_history.py               thin CLI → :mod:`infl_ens.vis.closed_loop.plot_history`
│   ├── plot_benchmark_space_heatmaps.py          thin CLI → :mod:`infl_ens.vis.benchmark_space.plot_pairwise_heatmaps`
│   ├── plot_pairwise_position_updates.py         thin CLI → :mod:`infl_ens.vis.closed_loop.plot_pairwise_position_updates`
│   ├── run_sweep.sh                              bash sweep launcher (seeds / sigma / kde)
│   ├── run_sigma_sweep_r20.sh                    end-to-end cumulative-LoRA sigma sweep at 20 rounds
│   ├── plot_sweep.py                             thin CLI → :mod:`infl_ens.training.sweep_aggregate` + :mod:`infl_ens.vis.sweeps`
│   ├── prune_final_round_adapters.py             thin CLI → :mod:`infl_ens.utils.checkpoints.prune_intermediate_adapters`
│   ├── probe_sft_capability.py                   cross-perplexity probe over saved per-round adapters
│   ├── closed_loop_demo.py                       toy closed-loop simulation
│   ├── run_eval_pairs_near_r40_seed0.sh          final-round eval for one seed
│   ├── run_eval_pairs_near_r40_all_seeds.sh      final-round eval seeds 0–9
│   ├── run_baseline_replay_r40_all_seeds.sh      pooled baseline 40-round replay seeds 0–9
│   ├── compare_baseline_vs_specialists.py        baseline vs clone round-39 NLL comparison
│   ├── run_pair_merge_r40_all_seeds.sh           40-round fixed pair-merge closed loop seeds 0–9
│   ├── run_proximity_merge_r40_all_seeds.sh      40-round proximity merge (paired theory init)
│   ├── run_proximity_plus_specialists_r40.sh     proximity + per-clone SFT, theory_gradient init
│   ├── run_ai4privacy_fixed_theory_specialists_10seeds.sh  10-seed AI4Privacy specialist sweep from fixed theory positions
│   ├── run_ai4privacy_fixed_theory_generalist_10seeds.sh  pooled generalist replay of specialist-routed AI4Privacy batches
│   ├── eval_ai4privacy_fixed_theory_generalist.sh  final-round benchmark eval for pooled generalist replay
│   ├── postprocess_ai4privacy_fixed_theory_specialists.sh  plots, evals, and specialization probes after the AI4Privacy sweep
│   ├── compare_matched_proximity_vs_specialists.py  same-run merge vs specialist benchmark table
│   ├── replay_corner_single_pool_ablation.py       one LoRA/corner on merge-sized batch (replay)
│   ├── run_corner_single_pool_ablation.sh            run replay + eval on proximity_plus_specialists
│   ├── compare_all_r40_models.py                 base + pooled + specialists + merge-low/high
│   ├── aggregate_compare_all_seeds.py            mean ± std over per-seed compare_all JSON
│   ├── aggregate_merge_by_corner.py              thin CLI: merge NLL by corner role across seeds
│   ├── run_compare_all_r40_all_seeds.sh          run compare_all + aggregate for all seeds
│   ├── aggregate_eval_across_seeds.py            mean ± std NLL over per-seed eval JSONs
│   ├── export_eval_matrix.py                     agent×benchmark CSV / MD / LaTeX / JSON
│   ├── write_oracle_routing_tex.py               oracle vs pooled vs learned specialists pgfplots figure
│   ├── plot_oracle_run_figures.py                full oracle-run suite: NLL, positions, split, support
│   ├── rebuild_oracle_resource_density.sh        detailed empirical resource density for oracle run
│   ├── eval_base_model.py                        thin CLI → :mod:`infl_ens.evaluation.base_eval.evaluate_base_model`
│   └── smoke_test.py                             pipeline sanity check
├── configs/
│   ├── model/
│   │   └── qwen2_5_1_5b.yaml                     base-model + LoRA hyperparams
│   ├── data/
│   │   ├── beavertails.yaml
│   │   ├── halueval.yaml
│   │   ├── toxicchat.yaml
│   │   ├── ai4privacy.yaml
│   │   ├── orbench.yaml
│   │   ├── prompt_injection.yaml
│   │   └── do_not_answer.yaml
│   ├── evaluation/
│   │   ├── adapter_on_benchmarks.yaml            score one adapter on BeaverTails + HaluEval
│   │   ├── run_on_benchmarks.yaml                score every adapter under a run directory
│   │   └── run_final_round.yaml                  final round only, all agents (cross-agent compare)
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
│           ├── halueval_only.yaml                1-D hallucination-axis ablation
│           ├── safety_truth_toxicchat_n4.yaml    3-D harm + hallucination + jailbreak router training
│           ├── three_new_axes.yaml               3-D over-refusal + injection + policy-violation preview
│           ├── six_axis_safety.yaml              6-D safety trait space for distribution slice figures
│           ├── six_axis_theory_n12.yaml           12-agent theory-only Nash on 6-D safety space
│           ├── six_axis_pair_merge_r40.yaml       12 routers + 6 pair-merged LoRAs, position_only, 40 rounds
│           ├── six_axis_baseline_replay_r40.yaml  pooled baseline replay after pair-merge closed loop
│           ├── safety_truth_ai4privacy_n6_theory_only_sigma04.yaml  6-agent paired-theory no-SFT AI4Privacy run at 0.4σ*
│           └── ai4privacy_fixed_theory_generalist_replay_r40.yaml  pooled replay generalist for the fixed-theory AI4Privacy sweep
├── tests/
│   ├── test_benchmark_loaders.py                 offline tests with synthetic JSON fixtures
│   ├── test_ai4privacy_loader.py                 offline tests with synthetic AI4Privacy JSONL fixture
│   ├── test_evaluation.py                        offline tests for adapter discovery + eval I/O
│   ├── test_toxicchat_loader.py                  offline tests with synthetic ToxicChat CSV fixture
│   ├── test_new_benchmark_loaders.py             offline tests for OR-Bench, injection, Do-Not-Answer
│   ├── test_encoders.py                          offline tests for Hugging Face embedding extraction
│   ├── test_trait_normalize.py                   offline tests for the quantile normalizer
│   └── test_safety_trait_space.py                offline tests using a toy encoder
├── data/                                         gitignored
└── results/                                      gitignored
```

## File-by-file tables

### `src/infl_ens/data/`

| File | Role | Key public symbols |
|---|---|---|
| `__init__.py` | Re-exports | `TraitSpace`, `build_trait_space`, `position_from_corpus`, `HuggingFaceEncoder`, `benchmarks` |
| `__main__.py` | Single CLI: `python -m infl_ens.data {preview,build-safety-trait-space}` | `main` |
| `encoders.py` | Direct Hugging Face embedding backend for trait-space construction | `HuggingFaceEncoder` |
| `trait_space.py` | Trait space :math:`\mathbb{B}` and resource distribution :math:`B(b)` | `TraitSpace`, `build_trait_space`, `position_from_corpus` |
| `trait_space_cache.py` | Persist/reload safety trait spaces; config fingerprint + `build_or_load_safety_trait_space` | `build_or_load_safety_trait_space`, `trait_space_fingerprint`, `save_safety_trait_space_cache`, `load_safety_trait_space_cache`, `coordinate_chain_from_cache` |
| `trait_linear_transform.py` | Frozen unsupervised affine trait transforms (optional pre-normalizer pipeline stage) | `FrozenLinearTransform`, `fit_standardize`, `fit_whiten`, `apply_trait_space` |
| `trait_normalize.py` | Always-on per-axis quantile (empirical-CDF) normalization to `[0,1]^L` | `QuantileNormalizer`, `AxisQuantileMap`, `fit_quantile_normalizer` |
| `position_blend.py` | EMA toward trait-space centroid after corpus projection | `apply_position_update`, `effective_blend`, `parse_position_step` |
| `splits.py` | Stratified train/val/test partitions per benchmark; exact train-coverage batch planning | `DataSplitManifest`, `build_split_manifest`, `load_split_manifest`, `apply_manifest_partition`, `choose_exact_train_coverage` |

### `src/infl_ens/data/benchmarks/`

| File | Role | Key public symbols |
|---|---|---|
| `__init__.py` | Re-exports | `BenchmarkSplit`, `load_beavertails`, `load_halueval`, `load_toxicchat`, `load_ai4privacy`, `load_orbench`, `load_prompt_injection`, `load_do_not_answer`, `build_safety_trait_space`, `LearnedAxis`, `BEAVERTAILS_CATEGORIES`, `HALUEVAL_TASKS`, `TOXICCHAT_SCORE_MODES`, `PII_SCORE_MODES`, `ORBENCH_CONFIGS` |
| `base.py` | Uniform benchmark record container | `BenchmarkSplit` |
| `beavertails.py` | BeaverTails loader and harm-score scoring | `load_beavertails`, `BEAVERTAILS_CATEGORIES` |
| `halueval.py` | HaluEval loader and hallucination-score scoring | `load_halueval`, `HALUEVAL_TASKS` |
| `jbb_behaviors.py` | JBB-Behaviors loader and jailbreak-axis scoring | `load_jbb_behaviors`, `HARMFUL_FILENAME`, `BENIGN_FILENAME` |
| `toxicchat.py` | ToxicChat loader and jailbreak-score scoring | `load_toxicchat`, `TOXICCHAT_SCORE_MODES` |
| `ai4privacy.py` | AI4Privacy loader and privacy-density scoring | `load_ai4privacy`, `PII_SCORE_MODES` |
| `orbench.py` | OR-Bench loader and over-refusal scoring | `load_orbench`, `ORBENCH_CONFIGS` |
| `prompt_injection.py` | Prompt-injection loader and injection scoring | `load_prompt_injection` |
| `do_not_answer.py` | Do-Not-Answer loader and policy-violation scoring | `load_do_not_answer` |
| `safety_trait_space.py` | Multi-axis learned-anchor trait space from labelled benchmarks | `build_safety_trait_space`, `build_safety_trait_space_bundle`, `LearnedAxis`, `SafetyTraitSpaceBundle` |

### `src/infl_ens/inflgame/router/`

| File | Role | Key public symbols |
|---|---|---|
| `__init__.py` | Re-exports | `RouterAgent`, `InfluencerRouter`, `allocation_weights`, `expected_utilities`, `empirical_utility`, `strategic_routing_weights`, `utility_gradient` |
| `agents.py` | Router-agent dataclass and calibration-based init | `RouterAgent`, `RouterAgent.from_calibration` |
| `allocation.py` | Allocation math :math:`G_i, u_i, \hat u_i, \nabla_{x_i} u_i, p_i^{strat}` | `allocation_weights`, `expected_utilities`, `empirical_utility`, `strategic_routing_weights`, `utility_gradient` |
| `verification.py` | Numerical drift-vs-gradient alignment for canonical/strategic routing rules | `run_reweighted_drift_report` |
| `core.py` | Public router class | `InfluencerRouter` |

### `src/infl_ens/evaluation/`

| File | Role | Key public symbols |
|---|---|---|
| `__init__.py` | Re-exports eval API; lazy metrics | `AdapterEvalConfig`, `BenchmarkEvalResult`, `load_benchmark_splits`, `evaluate_adapter_on_splits`, `evaluate_run_adapters`, `run_eval_job`, `discover_adapters`, (lazy) `mean_token_nll` |
| `__main__.py` | Single CLI: `python -m infl_ens.evaluation --config <path>` | `main` |
| `adapters.py` | Adapter path validation and HF+PEFT loading | `is_adapter_dir`, `resolve_adapter_dir`, `discover_adapters`, `AdapterRef`, `load_base_causal_lm`, `load_adapter_model` |
| `benchmarks.py` | YAML ``benchmarks`` block → :class:`BenchmarkSplit` list; optional manifest partition | `BENCHMARK_KINDS`, `load_benchmark_splits`, `load_benchmark_splits_with_partition`, `subsample_split` |
| `metrics.py` | Chat formatting + mean token NLL | `format_chat_example`, `split_to_texts`, `mean_token_nll` |
| `evaluate.py` | Orchestration and JSON reports | `AdapterEvalConfig`, `BenchmarkEvalResult`, `EvalJobConfig`, `evaluate_adapter_on_split`, `evaluate_adapter_on_splits`, `evaluate_run_adapters`, `run_eval_job`, `write_eval_report` |
| `aggregate.py` | Mean ± std over seeds; matrix export | `AggregatedEvalMetric`, `EvalMatrix`, `aggregate_eval_across_seeds`, `build_eval_matrix`, `write_eval_matrix_outputs` |
| `compare.py` | Baseline/specialist/merge adapter comparison; corner aggregation | `ModelScore`, `DEFAULT_SAFETY_BENCHMARKS`, `resolve_adapter_at`, `eval_adapter`, `compare_baseline_vs_specialists`, `compare_all_models`, `process_merge_seed`, `aggregate_merge_by_corner`, `aggregate_compare_reports`, `print_aggregate_compare_table` |
| `specialist_tables.py` | Train/test tables: flat route-then-score headline + diagnostic per-benchmark specialist vs ``pooled-baseline`` | `build_specialist_vs_pooled_table`, `write_specialist_comparison_tables`, `load_routing_headline`, `merge_eval_scores`, `AXIS_SPECIALIST` |
| `routing_eval.py` | Flat-pool route-then-score (expected/sampled/argmax proportional :math:`G`, oracle ceiling) | `run_flat_routing_eval`, `FlatRoutingReport`, `format_headline_markdown`, `report_to_dict` |
| `axis_niche.py` | Per-axis niche gates: variance (PCA), ICA, mid-mass (distinctness + :math:`G`) | `run_axis_niche_diagnostic`, `AxisNicheResult`, `format_niche_markdown` |
| `capability_probe.py` | SFT capability probe (cross-NLL matrix, margins) | `probe_run`, `cross_batch_margin`, `write_probe_csv` |

### `src/infl_ens/training/`

| File | Role | Key public symbols |
|---|---|---|
| `__init__.py` | Eager re-export of router training; lazy proxy for SFT | `RouterTrainingConfig`, `train_router_positions`, (lazy) `SFTTrainingConfig`, `sft_train_agent` |
| `__main__.py` | Single CLI: `python -m infl_ens.training --config <path>` dispatches on the config's `task` field. Closed-loop task honours `closed_loop.routing_weight` (`G` / `G_times_1mG`), `closed_loop.loss_reweight`, `closed_loop.init_noise` (Gaussian symmetry-breaking at clone start), `closed_loop.sft_merge_groups` (optional pair-merge SFT: four routers, two physical LoRAs), and `closed_loop.save_per_round`; always logs `agent_prompts` / `agent_responses` / `agent_sft_logs` per round in `history.json` (plus `merge_*` fields when pair-merge is enabled). Baseline replay also writes a centroid-tracking `history.json` for pooled generalist plots. | `main` |
| `router_training.py` | Gradient-ascent loop on agent positions | `RouterTrainingConfig`, `train_router_positions` |
| `sft_training.py` | LoRA SFT for a single :class:`RouterAgent`; accepts `out_dir_override` for per-round adapter archiving; accepts `cfg.cumulative_lora=True` to load and continue training the prior adapter rather than starting fresh; returns `log_history` and `loaded_prior_lora` from the SFT trainer's state | `SFTTrainingConfig`, `sft_train_agent` |
| `baseline_replay.py` | Replay pooled baseline/generalist SFT from closed-loop `history.json` routed batches; records pooled batch and cumulative centroids | `load_closed_loop_history`, `pooled_batch_from_round`, `replay_pooled_baseline_sft`, `make_pooled_baseline_agent` |
| `merge_training.py` | Pair-merge closed loop: fixed or proximity merge groups, routed-batch concat, router-only centroid updates | `resolve_dynamic_merge_groups`, `merge_train_name`, `parse_sft_merge_groups`, `merge_routed_batch`, `closed_loop_weight_args` |
| `sweep_aggregate.py` | Sweep summaries and seed×σ aggregation (orchestrates :mod:`infl_ens.vis.sweeps`) | `classify_equilibrium_clusters`, `summarise_flat_sweep_run`, `write_flat_sweep_csv`, `aggregate_group_seed_sweep`, `aggregate_final_positions`, `print_final_positions_report`, `summarize_pairs_near_theory_sweep` |
| `position_only.py` | Position-only closed-loop simulation and replay (no SFT) | `simulate_position_only_loop`, `replay_position_updates`, `main` |
| `history_audit.py` | Verify logged position updates against centroid predictions | `verify_history`, `build_trait_space_from_config`, `load_config` |
| `position_stability.py` | Summarize batch-size and position-step stability sweeps | `run_batch_size_static_comparison`, `run_position_step_modes_comparison` |
| `data_split.py` | Closed-loop train/val/test partition resolution and exact-coverage batches | `resolve_closed_loop_data_split`, `shuffled_train_batch_indices`, `partitioned_splits_for_eval` |
| `closed_loop_eval.py` | Mid-training validation NLL on merge adapters | `run_closed_loop_val_eval` |
| `theory_vs_sft.py` | Strategic gradient-ascent vs SFT trajectory comparison | `run_strategic_ascent`, `sft_trajectory_from_history`, `build_theory_trait_space`, `build_theory_summary`, `sigma_from_cfg` |

### `src/infl_ens/latex/`

| File | Role |
|---|---|
| `With canonical routing,.tex` | Derivation note for canonical routing and position-only update under MV-Gaussian kernels |
| `kernel_agnostic_gradient_step.tex` | Derivation note for gradient-matched position updates for any differentiable positive influence kernel |
| `trait_axis_assignment.tex` | Methodology note: how prompts are mapped to safety trait-axis coordinates |

### `src/infl_ens/utils/`

| File | Role | Key public symbols |
|---|---|---|
| `__init__.py` | Re-exports | `weighted_mean`, `weighted_covariance`, `gaussian_stability_threshold` |
| `resource.py` | Pure helpers on (grid, weights) pairs | `weighted_mean`, `weighted_covariance`, `gaussian_stability_threshold` |
| `sweep_discovery.py` | Discover flat and nested sweep directories | `RunCell`, `load_history`, `agent_order`, `position_tensor`, `discover_group_seed_runs`, `discover_flat_sweep_runs`, `iter_sigma_seed_histories`, `discover_sigma_seed_history_paths`, `final_positions`, `collect_final_layout_labels` |
| `init_noise_calibration.py` | Calibrate ``init_noise`` for target mean pairwise spread | `solve_init_noise`, `mean_pairwise_spread`, `expected_pairwise_two_agent` |

### `scripts/`

| File | Role |
|---|---|
| `download_beavertails.py` | Downloads `PKU-Alignment/BeaverTails` to `data/beavertails/` via the `datasets` library |
| `download_halueval.py` | Downloads HaluEval task JSON files from `RUCAIBox/HaluEval` to `data/halueval/` |
| `download_jbb_behaviors.py` | Downloads `JailbreakBench/JBB-Behaviors` CSV files to `data/jbb_behaviors/` |
| `download_toxicchat.py` | Downloads `lmsys/toxic-chat` CSV files to `data/toxicchat/` |
| `download_ai4privacy.py` | Downloads `ai4privacy/pii-masking-200k` JSONL files to `data/ai4privacy/` |
| `download_orbench.py` | Downloads `orbench-llm/or-bench` CSV configs to `data/orbench/` |
| `download_prompt_injection.py` | Downloads `neuralchemy/prompt-injection-Threat-Matrix` (binary, cap 5k) to `data/prompt_injection/`; legacy `--source deepset` |
| `download_do_not_answer.py` | Downloads `LibrAI/do-not-answer` plus Alpaca benign negatives to `data/do_not_answer/` |
| `build_safety_trait_space.py` | Convenience wrapper around `python -m infl_ens.data build-safety-trait-space` |
| `plot_trait_representation.py` | Data-representation figures contrasting the legacy clipped calibration with the always-on quantile normalization. Derives both coordinate sets from **one** encode pass (a second pass would need a different `trait_space` block and thus a full re-encode). Writes `trait_marginals_old_vs_new`, `trait_pairs_old_vs_new`, `dataset_composition`, and `trait_repr_summary.json`. |
| `run_trait_representation_on_doob.sh` | Sync (`scp -r`) to `mlovett@doob.dartmouth.edu`, run the representation + resource-density figures there under `nohup`, and pull figures back. `MODE=launch\|status\|pull`; `SMOKE=1` runs a cheap 2-axis foreground gate first. |
| `compare_utility_estimators.py` | Side-by-side comparison of grid :math:`u_i`, empirical-pool :math:`\hat u_i`, and finite-batch proportional share. `--mode {toy,safety}`. |
| `diagnose_axis_separability.py` | Compares mean-difference vs shrinkage-Fisher trait-axis estimators, percentile calibration, saturation, and optional Gram-Schmidt decorrelation before changing the production trait-space builder. |
| `diagnose_axis_confusability.py` | Diagnoses benchmark-origin confusion in learned trait coordinates, including field-map A/Bs, subspace residuals, nonlinear probes, score→coordinate leak matrices, and benchmark coordinate-overlap matrices. |
| `diagnose_trait_support.py` | Diagnoses the KDE-smoothed resource density :math:`B(b)` vs the actual prompt projections in trait space, recomputes theoretical Nash, and reports per-agent density / prompt-count metrics at theory NE and SFT end. Supports `--density-mode {kde,empirical,both}`: `empirical` rebuilds :math:`B(b)` as a 2-D histogram of projected prompts on the same `n_grid × n_grid` lattice (mass only where prompts actually project; tunable `--empirical-smoothing-cells` to avoid hard zeros), while `both` produces a 2-panel comparison figure plus both console tables side-by-side. `--config-override KEY=VAL` repeatable for bandwidth/sigma sweeps without YAML edits. |
| `compare_theory_vs_sft.py` | Thin CLI: calls :mod:`infl_ens.training.theory_vs_sft` and :func:`infl_ens.vis.theory_vs_sft.plot_theory_vs_sft_comparison` to compare strategic-Nash endpoints with the SFT trajectory. |
| `compare_runs.py` | Thin CLI: rebuilds trait space from YAML, extracts two `history.json` trajectories and theory NE, calls :func:`infl_ens.vis.closed_loop.plot_trajectory_overlay`. |
| `plot_closed_loop_history.py` | Thin CLI: loads `history.json`, calls :func:`infl_ens.vis.closed_loop.plot_history`, saves PDF/PNG under `scripts/figures/`. |
| `plot_benchmark_space_heatmaps.py` | Thin CLI: rebuilds trait space from YAML, calls :func:`infl_ens.vis.benchmark_space.plot_pairwise_heatmaps`, saves PDF/PNG. |
| `plot_pairwise_position_updates.py` | Thin CLI: loads `history.json`, calls :func:`infl_ens.vis.closed_loop.plot_pairwise_position_updates` for multi-axis runs. |
| `run_sweep.sh` | Bash launcher that sweeps one parameter (seeds, sigma_fraction, or kde_bandwidth) over the closed-loop trainer. Skips runs whose `history.json` already exists; optionally runs per-run plotting and theory comparison after each training. |
| `run_sigma_sweep_r20.sh` | End-to-end wrapper for the cumulative-LoRA sigma sweep at 20 rounds: pre-creates unique figure subfolders, launches `run_sweep.sh sigma`, runs trajectory + theory_vs_sft + capability probe per sigma into its own subfolder, aggregates with `plot_sweep.py`, prints a cross-sigma specialisation-margin table. Defaults to cumulative framework + `safety_truth_n4_r20_strategic_long_cum.yaml`; switchable to independent framework via env-var overrides. |
| `plot_sweep.py` | Thin CLI: discovers flat sweep runs, calls :func:`infl_ens.training.sweep_aggregate.summarise_flat_sweep_run`, :func:`infl_ens.vis.sweeps.plot_sweep_grid`, and writes CSV via :func:`infl_ens.training.sweep_aggregate.write_flat_sweep_csv`. |
| `prune_final_round_adapters.py` | Walks ``results/**/agents/<agent>/round-NN/`` and deletes every round directory except the highest index per agent; flat ``agents/<agent>/`` checkpoints are untouched. |
| `probe_sft_capability.py` | Thin CLI: calls :mod:`infl_ens.evaluation.capability_probe` and :func:`infl_ens.vis.capability_probe.plot_probe` for Tier 1 (SFT loss) + Tier 3 (cross-NLL margin) diagnostics. |
| `run_position_only_cum_r10.sh` | Single 10-round launcher for the matched `position_only` config (`batch_size=256`, cumulative LoRA): trains to `results/position_only_cum_round_sweep/r10/`, then runs trajectory + theory_vs_sft + capability probe figures. |
| `run_position_only_cum_sweeps.sh` | Two-pass sweep (rounds 10/20/40; sigma 0.25–1.5× threshold at 20 rounds) over the matched `position_only_cum` config. Mirrors `run_loss_reweight_cum_sweeps.sh`. Supports `REDO_SIGMA_SWEEP=1`, `SKIP_ROUND_SWEEP=1`. |
| `run_position_only_cum_sigma_redo.sh` | Sigma sweep only: wipes `position_only_cum_sigma_sweep`, re-trains with `init_noise` from config. |
| `run_position_only_seed_sigma_sweep.sh` | Resumable seed×sigma grid (default 5 seeds × 5 sigmas, 20 rounds) for `position_only_cum`; per-run figures under `scripts/figures/<SWEEP_NAME>/per_run/`; calls `aggregate_seed_sigma_sweep.py` for mean±std aggregates. Extend via `SEEDS` / `SIGMA_VALUES`. |
| `aggregate_seed_sigma_sweep.py` | Thin CLI: discovers ``sigma*/seed*`` or ``r*/seed*`` cells via :func:`infl_ens.utils.sweep_discovery.discover_group_seed_runs`, aggregates via :func:`infl_ens.training.sweep_aggregate.aggregate_group_seed_sweep`. |
| `run_pairs_near_eq_sweeps.sh` | Full SFT sweeps with `pairs_near_theory` init: PASS 1 seed×rounds `{10,20,40}`; PASS 2 seed×σ `{0.25…1.5}` at 20 rounds; probe + aggregate per pass. |
| `run_pairs_near_theory_10seeds.sh` | Position-only sim: `pairs_near_theory` init, 10 seeds × 2 σ (fast, no SFT). |
| `summarize_pairs_near_theory.py` | Thin CLI: calls :func:`infl_ens.training.sweep_aggregate.summarize_pairs_near_theory_sweep`. |
| `simulate_position_only_loop.py` | Thin CLI: calls :mod:`infl_ens.training.position_only` for fast routing + `(1-G)` centroid updates (no SFT). |
| `run_seven_axis_posttrain.sh` | After pair-merge closed loop: pooled baseline replay, merge eval (round 39), baseline eval |
| `routing_ensemble_diagnostics.py` | Flat test-pool route-then-score: expected/sampled/argmax proportional :math:`G` + oracle ceiling |
| `diagnose_seven_axis_niche.py` | Seven-axis variance / ICA / mid-mass niche gates |
| `compare_routing_weights.py` | Flat-pool naive-G vs G(1−G) expected routing comparison (adapter-free) |
| `build_five_axis_split.py` | Build 70/10/20 manifest for five-axis collapse config |
| `build_five_axis_collapse_init.py` | Extract 10-clone / 5-D init positions from six-axis theory n12 |
| `analyze_pair_occupancy.py` | Pair occupancy + recommended agent count from routing JSON |
| `run_collapse_experiment.sh` | Full collapse pipeline: split → train → baseline → routing gate → occupancy |
| `run_hypercube_collapse_experiment.sh` | Hypercube GA init → theory pre + SFT-only merge → baseline → routing gate |
| `build_attribution_2x2_configs.py` | Regenerate four 2×2 attribution cell YAMLs from `_base.yaml` |
| `build_attribution_spread_rerun_configs.py` | Generate spread-calibrated random + GA reproducibility YAMLs |
| `build_seed0_isolation_configs.py` | GA no-pre on fixed `five_axis_seed0.json`; vary training seed 1–3 |
| `run_attribution_2x2.sh` | Seed-0 init×theory_pre sweep; routing gate only; geometry summary |
| `run_attribution_spread_rerun.sh` | Spread re-run: random (2 levels × 3 seeds) + GA repro; verify spread first |
| `run_seed0_isolation.sh` | Seed-0 split isolation: GA no-pre, training seeds 1–3, fixed manifest |
| `summarize_attribution_2x2.py` | Tabulate NLL + post-pre within-merge L2 across attribution cells |
| `summarize_attribution_spread_rerun.py` | Tabulate NLL + geometry across spread re-run cells |
| `summarize_seed0_isolation.py` | Pooled/learned NLL for seed-0 isolation vs train0 reference |
| `decompose_routing_gap.py` | Per-prompt argmax-oracle agreement and NLL gap by benchmark axis |
| `compare_theory_g_vs_oracle.py` | Theory G merge ranking vs oracle NLL; soft vs miscalibrated verdict |
| `inspect_overrefusal_dilution.py` | Orbench weight diffuseness + sharpening counterfactuals for +0.060 gap |
| `build_oracle_centroid_positions.py` | 1-component oracle-prompt centroids → colocated fixed_positions |
| `build_oracle_centroid_shift_configs.py` | Oracle-centroid shift YAML from ga_theory_pre reference |
| `diff_router_configs.py` | Programmatic YAML diff with allowlist (pre-launch gate) |
| `verify_oracle_centroid_init.py` | Verify colocated init matches oracle centroids on trait space |
| `verify_oracle_centroid_persistence.py` | Per-round center trace; fail if shift reverts during training |
| `run_oracle_centroid_shift.sh` | Build → diff → verify → train → routing eval → summary |
| `summarize_oracle_centroid_shift.py` | Per-axis agreement/gap vs ga_theory_pre reference |
| `fit_whitening_transform.py` | Fit standardize/whiten on seed-1 trait vectors (unsupervised) |
| `build_trait_whitening_configs.py` | Baseline / standardize / whiten arm YAMLs |
| `verify_whitening_transform.py` | VERIFY_ONLY seed-0 variance/decorrelation check |
| `evaluate_whitening.py` | Per-arm gap, agreement, theory↔oracle alignment L2 |
| `run_trait_whitening.sh` | Fit → diff → verify → three-arm train + eval |
| `diagnose_tail_separability.py` | Expensive-tail collision vs separability; irreducible vs routing-fixable |
| `analyze_merge_oracle_geometry.py` | Oracle-winning prompt trait geometry per merge (spread design) |
| `build_oracle_spread_positions.py` | k=2 oracle centers → aligned / misaligned fixed_positions JSON |
| `build_within_merge_spread_configs.py` | Within-merge spread experiment YAMLs (seed-0 split) |
| `run_within_merge_spread.sh` | Launch aligned + misaligned spread arms + routing eval |
| `summarize_within_merge_spread.py` | Oracle−learned gap + per-round within_merge summary |
| `verify_init_spread.py` | Log realized init mean_pairwise for spread-calibrated random configs |
| `run_merge_near_gradient_ascent.py` | Pair init (hypercube edges or merge-near) + grid Nash; writes `results/*/fixed_positions.json` |
| `build_six_axis_split.py` | Build persisted 70/10/20 stratified split manifest + batch/round plan (6 benchmarks) |
| `run_six_axis_pipeline.sh` | Theory n12 → slice figure → conditional r24 split + pooled baseline + tables |
| `run_six_axis_split_r24.sh` | 24-round six-axis split closed loop + baseline replay + eval + tables |
| `build_six_axis_split_tables.py` | Write train/test specialist-vs-pooled markdown tables from eval JSON |
| `patch_six_axis_configs.py` | One-off converter from seven-axis/JBB configs to six-axis (12 agents) |
| `compare_reweighted_drift.py` | Thin CLI: calls :func:`infl_ens.inflgame.router.verification.run_reweighted_drift_report`. |
| `run_large_batch_static_analysis.sh` | Compares batch 256, batch 10k, full pool, and expected pool at σ=0.25/0.75 (no SFT). |
| `compare_batch_size_static.py` | Thin CLI: calls :func:`infl_ens.training.position_stability.run_batch_size_static_comparison`. |
| `run_position_fix_comparison.sh` | A/B fixes at σ=0.25/0.75, batch 256: baseline vs `expected_pool` vs `init_noise=0.01` (sim, no SFT). |
| `run_pool_and_noise_10seeds.sh` | Both fixes (`expected_pool` + `init_noise=0.01`), 10 seeds; calls `aggregate_final_positions.py`. |
| `aggregate_final_positions.py` | Thin CLI: calls :func:`infl_ens.training.sweep_aggregate.aggregate_final_positions`. |
| `run_position_step_stability_test.sh` | Pre-sweep grid over position-step policies at σ=0.25/0.75 via `simulate_position_only_loop.py` (seconds per cell, not full training). |
| `compare_position_step_modes.py` | Thin CLI: calls :func:`infl_ens.training.position_stability.run_position_step_modes_comparison`. |
| `verify_position_update.py` | Thin CLI: calls :func:`infl_ens.training.history_audit.verify_history`. |
| `closed_loop_demo.py` | Toy hash-bag closed-loop simulation (no external deps) |
| `run_eval_pairs_near_r40_seed0.sh` | Final-round eval (`round-39`) for clone-0…3 on one seed |
| `run_eval_pairs_near_r40_all_seeds.sh` | Same eval for seeds 0–9 (skips existing `eval_results.json`) |
| `run_baseline_replay_r40_all_seeds.sh` | Pooled baseline replay (40 rounds) for seeds 0–9 from `pairs_near_eq` histories |
| `compare_baseline_vs_specialists.py` | Thin CLI: calls :func:`infl_ens.evaluation.compare.compare_baseline_vs_specialists` for pooled-baseline vs clone round-N NLL. |
| `run_pair_merge_r40_all_seeds.sh` | Train pair-merge closed loop (40 rounds, seeds 0–9) → `results/pair_merge_round_sweep/r40/seed*` |
| `run_ai4privacy_fixed_theory_specialists_10seeds.sh` | Runs the 40-round four-specialist AI4Privacy sweep for seeds 0–9 from fixed theory positions, preserving clone pair assignments across seeds. |
| `run_ai4privacy_fixed_theory_generalist_10seeds.sh` | Replays each AI4Privacy specialist seed into one cumulative generalist LoRA that sees the exact union of routed specialist batches and plots its cumulative centroid. |
| `eval_ai4privacy_fixed_theory_generalist.sh` | Final-round benchmark eval for the pooled generalist seeds across BeaverTails, HaluEval, ToxicChat, and AI4Privacy. |
| `postprocess_ai4privacy_fixed_theory_specialists.sh` | Waits for the AI4Privacy 10-seed sweep, then renders position/update plots, final-round per-benchmark evals, and specialization probe figures. |
| `compare_ai4privacy_fixed_vs_base.py` | Aggregates round-39 adapter evals (10 seeds) vs a matched base-model NLL run on all four benchmarks; writes `base_eval_matched.json` and `compare_vs_base.json` under the sweep root. |
| `plot_ai4privacy_fixed_vs_base_figure.py` | Bar chart (PDF/PNG) + standalone pgfplots `.tex` for the AI4Privacy fixed-theory specialist vs base vs generalist NLL table. |
| `plot_seven_axis_eval_figure.py` | Seven-axis merge-specialist vs pooled-baseline NLL bar chart (PDF/PNG) + pgfplots `.tex`. |
| `write_seven_axis_eval_tex.py` | Standalone pgfplots `.tex` writer for seven-axis eval scores (no matplotlib). |
| `write_oracle_routing_tex.py` | Standalone pgfplots `.tex` for flat-pool + per-benchmark oracle vs pooled generalist vs learned specialists. |
| `plot_oracle_run_figures.py` | Full oracle-run figure suite (specialist-pair NLL, positions, data split, oracle support, merge mass). |
| `rebuild_oracle_resource_density.sh` | Empirical 64-bin resource density + round-5 positions for the oracle run (doob). |
| `compare_all_r40_models.py` | Thin CLI: calls :func:`infl_ens.evaluation.compare.compare_all_models` (base + pooled + specialists + merge). |
| `aggregate_merge_by_corner.py` | Thin CLI: calls :func:`infl_ens.evaluation.compare.process_merge_seed` and :func:`aggregate_merge_by_corner`. |
| `aggregate_compare_all_seeds.py` | Thin CLI: calls :func:`infl_ens.evaluation.compare.aggregate_compare_reports`. |
| `run_compare_all_r40_all_seeds.sh` | Per-seed `compare_all_r40_models.py` then aggregate |
| `aggregate_eval_across_seeds.py` | Averages per-seed `eval_results.json` → `eval_aggregate.json` |
| `export_eval_matrix.py` | Writes `eval_matrix.{csv,md,tex,json}` from `eval_aggregate.json` |
| `smoke_test.py` | End-to-end pipeline sanity check |

### `configs/`

| File | Role |
|---|---|
| `model/qwen2_5_1_5b.yaml` | Base-model + LoRA hyperparameters for the SFT trainer |
| `data/beavertails.yaml` | Static BeaverTails loader settings |
| `data/halueval.yaml` | Static HaluEval loader settings |
| `data/toxicchat.yaml` | Static ToxicChat loader settings |
| `data/ai4privacy.yaml` | Static AI4Privacy loader settings |
| `data/orbench.yaml` | Static OR-Bench loader settings |
| `data/prompt_injection.yaml` | Static prompt-injection loader settings |
| `data/do_not_answer.yaml` | Static Do-Not-Answer loader settings |
| `benchmark/router/example.yaml` | Original synthetic three-anchor example |
| `benchmark/router/safety_truth.yaml` | 2-D BeaverTails + HaluEval closed-loop config |
| `benchmark/router/safety_truth_n4_r10_strategic.yaml` | Same as safety_truth but with `routing_weight: G_times_1mG` (strategic-gradient correspondence under MV-Gaussian kernels) |
| `benchmark/router/safety_truth_n4_r10_strategic_long.yaml` | Same as ..._strategic but pushes the SFT step harder (3 epochs, batch 512, per-device 16, logging_steps=1). Use the capability probe as the overfitting detector on this run. |
| `benchmark/router/safety_truth_n4_r20_strategic_long.yaml` | Same as ..._r10_strategic_long but 20 rounds — tests stability of the strategic (2,2) basin under additional SFT. |
| `benchmark/router/safety_truth_n4_r40_strategic_long.yaml` | Same as ..._r10_strategic_long but 40 rounds — long-horizon stability + overfitting check. |
| `benchmark/router/safety_truth_n4_r{10,20,40}_strategic_long_cum.yaml` | Cumulative-LoRA variants of the three strategic_long configs: each agent loads its prior adapter and continues training rather than restarting from the base model every round. Capability accumulates across rounds. Saves to its own `results/safety_truth_n4_r*_strategic_long_cum/` directories so the original (independent-round) framework is preserved for comparison. |
| `benchmark/router/beavertails_only.yaml` | 1-D harm-axis ablation |
| `benchmark/router/halueval_only.yaml` | 1-D hallucination-axis ablation |
| `benchmark/router/safety_truth_toxicchat_n4.yaml` | 3-D router-position training config over BeaverTails harm, HaluEval hallucination, and ToxicChat jailbreak axes |
| `benchmark/router/three_new_axes.yaml` | 3-D over-refusal + injection + policy-violation preview config |
| `benchmark/router/six_axis_safety.yaml` | 6-D safety trait space for pairwise distribution slice figures |
| `benchmark/router/six_axis_theory_n12.yaml` | 12-agent theory-only Nash on 6-D safety space (`theory_gradient_paired`) |
| `benchmark/router/six_axis_pair_merge_r40.yaml` | 12 routers at fixed theory positions, 6 pair-merged cumulative LoRAs, `position_only`, 40 rounds |
| `benchmark/router/six_axis_pair_merge_split.yaml` | Same as pair-merge but 70/10/20 stratified split, exact train coverage, val eval every 5 rounds |
| `benchmark/router/six_axis_pair_merge_split_r12.yaml` | Same split; 12 rounds × smaller batch (`target_n_rounds: 12`) |
| `benchmark/router/six_axis_pair_merge_split_r24.yaml` | Same split; 24 rounds × ~840 batch (`target_n_rounds: 24`) |
| `benchmark/router/six_axis_baseline_replay_r40.yaml` | Pooled baseline replay from pair-merge `history.json` |
| `benchmark/router/six_axis_baseline_replay_split.yaml` | Pooled baseline replay from split closed-loop `history.json` |
| `benchmark/router/six_axis_baseline_replay_split_r12.yaml` | Pooled baseline replay for 12-round split run |
| `benchmark/router/six_axis_baseline_replay_split_r24.yaml` | Pooled baseline replay for 24-round split run |
| `evaluation/six_axis_run_eval.yaml` | Final-round eval on all 6 safety benchmarks for merge adapters |
| `evaluation/six_axis_split_eval_train.yaml` | Final-round eval on train partition (cap 1000/benchmark) |
| `benchmark/router/seven_axis_pair_merge_split.yaml` | 14 routers, 7 pair merges, 70/10/20 seven-axis split |
| `benchmark/router/seven_axis_pair_merge_split_nostretch.yaml` | Same as above but `coordinate_stretch_gamma(s)` dropped, so the router sees the quantile normalizer's near-uniform marginals directly. Stretch-free comparison arm. |
| `benchmark/router/seven_axis_collapse_dead_axes.yaml` | 5-axis collapse track (drops jailbreak + injection); 10 clones / 5 merges |
| `benchmark/router/seven_axis_collapse_near_theory.yaml` | Collapse track with merge-pair-near init + theory pre-warmup |
| `benchmark/router/seven_axis_collapse_hypercube_ga.yaml` | Collapse from hypercube GA positions; SFT-only collapsed-pair merge |
| `benchmark/router/seven_axis_collapse_hypercube_ga_baseline_replay.yaml` | Pooled baseline for hypercube GA collapse run |
| `benchmark/router/attribution_2x2/_base.yaml` | Shared template for 2×2 init×theory_pre attribution cells |
| `benchmark/router/attribution_2x2/ga_theory_pre.yaml` | GA init + theory pre (cell 1,1) |
| `benchmark/router/attribution_2x2/ga_no_theory_pre.yaml` | GA init, no theory pre (cell 1,0) |
| `benchmark/router/attribution_2x2/random_theory_pre.yaml` | Random init + theory pre (cell 0,1) |
| `benchmark/router/attribution_2x2/random_no_theory_pre.yaml` | Random init, no theory pre (cell 0,0) |
| `benchmark/router/attribution_spread_rerun/manifest.json` | Calibrated `init_noise` + config list for spread re-run |
| `benchmark/router/attribution_spread_rerun/ga_no_theory_pre_seed{1-4}.yaml` | GA reproducibility (no theory pre) |
| `benchmark/router/attribution_spread_rerun/ga_theory_pre_seed1.yaml` | GA theory-pre spot-check at seed 1 |
| `benchmark/router/attribution_spread_rerun/random_s09_*_seed{0-2}.yaml` | Matched-spread (~0.9) random arms |
| `benchmark/router/attribution_spread_rerun/random_s045_theory_pre_seed{0-2}.yaml` | Moderate-spread (~0.45) random + theory pre |
| `benchmark/router/seed0_isolation/manifest.json` | Fixed split isolation: training seeds 1–3 on `five_axis_seed0.json` |
| `benchmark/router/seed0_isolation/ga_no_theory_pre_train{1-3}.yaml` | GA no-pre isolation arms |
| `benchmark/router/within_merge_spread/manifest.json` | Oracle k=2 spread vs misaligned control |
| `benchmark/router/within_merge_spread/oracle_k2_{aligned,misaligned}.yaml` | Within-merge spread cells |
| `benchmark/router/seven_axis_collapse_baseline_replay.yaml` | Pooled baseline replay for collapse closed loop |
| `benchmark/router/seven_axis_router_improve_split.yaml` | Seven-axis router-improve track (`G_times_1mG`, `sigma_fraction: 0.1`) |
| `evaluation/seven_axis_split_eval_test.yaml` | Final-round eval on withheld seven-axis test partition |
| `benchmark/router/safety_truth_ai4privacy_n6_theory_only_sigma04.yaml` | 6-agent paired-theory no-SFT AI4Privacy position-only run at `sigma_fraction: 0.4` |
| `benchmark/router/baseline_replay_r40.yaml` | Pooled baseline 40-round replay from `history.json` (`task: baseline_replay`) |
| `benchmark/router/ai4privacy_fixed_theory_generalist_replay_r40.yaml` | Pooled generalist replay config for the fixed-theory AI4Privacy specialist sweep |
| `benchmark/router/safety_truth_n4_r40_pair_merge_cum.yaml` | 40-round closed loop: four routers + fixed `sft_merge_groups` |
| `benchmark/router/safety_truth_n4_r40_proximity_merge_cum.yaml` | 40-round: `theory_gradient_paired` + `sft_merge_mode: proximity` |
| `benchmark/router/safety_truth_n4_r40_proximity_plus_specialists_cum.yaml` | Same init as pairs_near_eq; proximity merge + `sft_also_train_individual` |
| `evaluation/adapter_on_benchmarks.yaml` | Single-adapter eval on BeaverTails + HaluEval (`task: adapter_eval`) |
| `evaluation/run_on_benchmarks.yaml` | Discover all adapters under a run and eval on both benchmarks (`task: run_eval`) |
| `evaluation/run_final_round.yaml` | `run_eval` with `rounds: [39]` and all four clone agents |

### `tests/`

| File | Role |
|---|---|
| `test_benchmark_loaders.py` | Offline tests for BeaverTails and HaluEval loaders |
| `test_ai4privacy_loader.py` | Offline tests for AI4Privacy JSONL parsing and privacy-density scoring |
| `test_evaluation.py` | Offline tests for adapter discovery, benchmark config loading, eval JSON reports |
| `test_routing_eval.py` | Offline tests for merge-level :math:`G` aggregation in route-then-score eval |
| `test_merge_training.py` | Offline tests for `sft_merge_groups` parsing and batch merge |
| `test_safety_trait_space.py` | Offline tests for `build_safety_trait_space` |
| `test_trait_space_cache.py` | Embedding dedupe + on-disk trait-space cache roundtrip |
| `test_trait_normalize.py` | Quantile-normalizer monotonicity, ties, out-of-range clamping, JSON roundtrip |
| `test_encoders.py` | Offline tests for direct Hugging Face embedding extraction and configuration |
| `test_jbb_behaviors_loader.py` | Offline tests for JBB-Behaviors CSV parsing |
| `test_toxicchat_loader.py` | Offline tests for ToxicChat CSV parsing and score modes |
| `test_new_benchmark_loaders.py` | Offline tests for OR-Bench, prompt-injection, and Do-Not-Answer loaders |
| `test_theory_ref_resolve.py` | Slow smoke test for `resolve_theory_22_reference` at high sigma |

## `__init__.py` re-export summary

- `src/infl_ens/__init__.py`: minimal — does not import subpackages eagerly.
- `src/infl_ens/data/__init__.py`: `TraitSpace`, `build_trait_space`, `position_from_corpus`, `HuggingFaceEncoder`, `FrozenLinearTransform`, `QuantileNormalizer`, `benchmarks`.
- `src/infl_ens/data/benchmarks/__init__.py`: `BenchmarkSplit`, `load_beavertails`, `load_halueval`, `load_jbb_behaviors`, `load_toxicchat`, `load_ai4privacy`, `load_orbench`, `load_prompt_injection`, `load_do_not_answer`, `build_safety_trait_space`, `LearnedAxis`, `BEAVERTAILS_CATEGORIES`, `HALUEVAL_TASKS`, `TOXICCHAT_SCORE_MODES`, `PII_SCORE_MODES`, `ORBENCH_CONFIGS`.
- `src/infl_ens/inflgame/__init__.py`: re-exports the `router` subpackage.
- `src/infl_ens/inflgame/router/__init__.py`: `InfluencerRouter`, `RouterAgent`, `allocation_weights`, `expected_utilities`, `utility_gradient`.
- `src/infl_ens/training/__init__.py`: `RouterTrainingConfig`, `train_router_positions`; lazy `SFTTrainingConfig`, `sft_train_agent` (avoids importing torch/transformers at package import time).
- `src/infl_ens/evaluation/__init__.py`: `AdapterEvalConfig`, `BenchmarkEvalResult`, `EvalJobConfig`, `load_benchmark_splits`, `evaluate_adapter_on_splits`, `evaluate_run_adapters`, `run_eval_job`, `discover_adapters`, `resolve_adapter_dir`; lazy `mean_token_nll`, `format_chat_example`, `evaluate_base_model`, `write_base_eval_report`, compare helpers (`ModelScore`, `compare_baseline_vs_specialists`, …), capability probe (`probe_run`, `cross_batch_margin`).
- `src/infl_ens/vis/__init__.py`: `plot_benchmark_nll_comparison`, `plot_pairwise_heatmaps`, `plot_history`, `plot_pairwise_position_updates`, `plot_trajectory_overlay`, `plot_probe`, `plot_theory_vs_sft_comparison`, `plot_sweep_grid`, `plot_trajectory_mean_std`, `plot_series_mean_std`, `plot_overview`, `plot_spread_by_mode_sigma`, `save_figure`.
- `src/infl_ens/utils/__init__.py`: `weighted_mean`, `weighted_covariance`, `gaussian_stability_threshold`.
