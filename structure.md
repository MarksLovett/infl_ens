# structure.md

Authoritative file-by-file map of the `infl_ens` repository. See
`AGENTS.md` rule 10: any change adding, moving, renaming, or deleting a
file under `src/`, `configs/`, `scripts/`, or `tests/` MUST update this
file in the same edit.

## On-disk tree

```
infl_ens/
├── src/infl_ens/
│   ├── __init__.py
│   ├── config.py                         layered YAML: includes, dotted overrides, key validation, resolve_sft_block
│   ├── experiment.py                     experiment files: ArmSpec / ExperimentConfig / load_experiment
│   ├── data/
│   │   ├── __init__.py
│   │   ├── encoders.py                   HuggingFaceEncoder (generic AutoModel + pooling) + make_encoder(cfg)
│   │   ├── trait_space.py                TraitSpace + build_trait_space + position_from_corpus
│   │   ├── trait_space_cache.py          fingerprinted on-disk cache + build_or_load_safety_trait_space
│   │   ├── trait_normalize.py            per-axis quantile (empirical-CDF) normalizer to [0,1]^L
│   │   ├── position_blend.py             EMA blend toward corpus centroid (apply_position_update)
│   │   ├── splits.py                     DataSplitManifest, stratified splits, exact train coverage, build_manifest_from_config
│   │   ├── download.py                   one downloader per benchmark kind + DOWNLOADERS registry
│   │   └── benchmarks/
│   │       ├── __init__.py
│   │       ├── base.py                   BenchmarkSplit container
│   │       ├── loading.py                load_benchmark_splits from a config `benchmarks` list (+ partition variant)
│   │       ├── beavertails.py            BeaverTails loader (harm axis)
│   │       ├── halueval.py               HaluEval loader (hallucination axis)
│   │       ├── jbb_behaviors.py          JBB-Behaviors loader (jailbreak axis)
│   │       ├── ai4privacy.py             AI4Privacy loader (privacy axis)
│   │       ├── orbench.py                OR-Bench loader (over-refusal axis)
│   │       ├── prompt_injection.py       prompt-injection loader (injection axis)
│   │       ├── do_not_answer.py          Do-Not-Answer loader (policy-violation axis)
│   │       └── safety_trait_space.py     multi-axis learned benchmark trait-space builder
│   ├── inflgame/
│   │   ├── __init__.py
│   │   └── router/
│   │       ├── __init__.py
│   │       ├── agents.py                 RouterAgent
│   │       ├── allocation.py             G_i, u_i, ∇u_i, top-k / sampled top-k / per-group soft weights
│   │       ├── core.py                   InfluencerRouter
│   │       └── verification.py           numerical gradient-alignment checks of the position updates
│   ├── training/
│   │   ├── __init__.py
│   │   ├── __main__.py                   thin CLI: load_config -> TASKS[task]
│   │   ├── tasks.py                      TASKS registry; run_baseline_replay
│   │   ├── closed_loop.py                run_closed_loop (route -> SFT -> position update), knob validation, agent init
│   │   ├── setup.py                      load_splits, make_trait_space, sigma_from_config, init_agents, history/resolved-config writers
│   │   ├── agent_init.py                 theory_gradient / theory_gradient_paired inits, pairing rules, resolve_agent_entries
│   │   ├── position_step.py              blend schedule + expected_pool centroid (re-exports position_blend)
│   │   ├── router_training.py            gradient ascent on positions (the theory solve)
│   │   ├── sft_training.py               LoRA SFT trainer (weighted loss, cumulative adapters)
│   │   ├── merge_training.py             pair-merge SFT helpers; soft routing over pairs
│   │   ├── baseline_replay.py            pooled generalist replayed from history.json batches
│   │   ├── data_split.py                 resolve train/val/test partitions + batch plan for a run
│   │   ├── closed_loop_eval.py           periodic validation NLL during training
│   │   └── pool_dynamics.py              grid-Nash gradient ascent, layout classification, pair geometry
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── __main__.py                   single CLI; unified (training-YAML) or standalone eval jobs
│   │   ├── evaluate.py                   evaluate_adapter_on_*, run_unified_eval, JSON reports
│   │   ├── routing_eval.py               flat-pool route-then-score: pooled / learned / oracle
│   │   ├── adapters.py                   resolve + load saved LoRA checkpoints
│   │   ├── metrics.py                    mean NLL on chat-formatted splits
│   │   └── benchmarks.py                 re-export shim of data.benchmarks.loading
│   ├── figures/
│   │   ├── __init__.py
│   │   ├── __main__.py                   python -m infl_ens.figures --config <experiment> [--only ...] [--list]
│   │   ├── render.py                     FIGURES registry; the only module reading run artifacts
│   │   ├── style.py                      shared rcParams, benchmark order / labels
│   │   ├── save.py                       save_figure (pdf + png)
│   │   ├── closed_loop.py                trajectories + utility tracking, pairwise position updates, overlays
│   │   ├── pair_positions.py             final pair positions over every axis pair; within-pair separation
│   │   ├── benchmark_space.py            pairwise trait-space resource heatmaps
│   │   ├── benchmark_nll_bar.py          grouped bar chart: base vs adapter benchmark NLL
│   │   ├── scale_family.py               family x scale held-out NLL grid + table (pure builders)
│   │   ├── trait_representation.py       clipped-vs-quantile trait marginals / pair densities / stats
│   │   ├── pgf_tex.py                    oracle_routing_tex, arm_comparison_tex, compile_tex
│   │   ├── per_round_tables.py           held-out NLL by pair at selected rounds (csv/md/tex/json)
│   │   └── cross_arm_report.py           data matching, routing headline, pair stability, NLL movement
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── __main__.py                   python -m infl_ens.pipeline --config <experiment> [--stages] [--smoke] [--dry-run]
│   │   └── stages.py                     download / manifest / train / perround / routing / figures / prune
│   ├── latex/                            derivation notes (TeX + compiled PDFs; not touched by tooling)
│   │   ├── With canonical routing,.tex
│   │   ├── kernel_agnostic_gradient_step.tex
│   │   ├── position_update_comparison.tex
│   │   └── trait_axis_assignment.tex
│   └── utils/
│       ├── __init__.py
│       ├── resource.py                   weighted_mean, weighted_covariance, σ₀*
│       └── checkpoints.py                prune intermediate round-* LoRA dirs
├── configs/
│   ├── encoders/
│   │   ├── qwen3_embedding_8b_awq.yaml   default trait-space encoder (cache identity + HF kwargs)
│   │   └── bge_large_en_v1_5.yaml        template for another Hugging Face encoder
│   ├── trait_space/
│   │   └── seven_axis.yaml               geometry knobs; includes the encoder preset
│   ├── data/
│   │   └── seven_axis_safety.yaml        the seven benchmarks + the 70/10/20 split
│   ├── models/
│   │   └── qwen2_5_1_5b_instruct.yaml    base LM + LoRA hyperparameters (sft block)
│   ├── arms/
│   │   ├── _closed_loop_base.yaml        everything the specialist arms share
│   │   ├── soft_full_pairs.yaml          soft, k = 7, share-weighted
│   │   ├── soft_topk3_pairs.yaml         soft, k = 3, share-weighted
│   │   ├── topk3_unit_pairs.yaml         soft, k = 3, unit weight
│   │   ├── hard_topk3_pairs.yaml         sampled top-3 without replacement, unit weight
│   │   ├── hard_pairs_matched.yaml       hard (one sampled winner), unit weight
│   │   ├── generalist_replay.yaml        pooled generalist replayed from the k = 3 arm
│   │   └── scale_family/                 model scale-family sweep cells (share the pinned fingerprint)
│   │       ├── _specialist_base.yaml     soft k = 3 base; cells override only output_dir + sft.base_model
│   │       ├── _generalist_base.yaml     pooled-baseline replay base; cells add base_model + history_path
│   │       ├── {qwen,llama,gemma}_{1b,3b,8b}.yaml        9 specialist cells (3 family x 3 scale)
│   │       └── {qwen,llama,gemma}_{1b,3b,8b}_gen.yaml    9 per-cell pooled generalists
│   └── experiments/
│       ├── seven_axis_3arm.yaml          the canonical experiment: arms, stages, eval window, figures, smoke
│       └── scale_family_sweep.yaml       18-arm sweep (9 specialist + 9 generalist); family x scale NLL figure
├── scripts/
│   ├── run_on_doob.sh                    the only shell script: sync + tmux launch + status + pull
│   └── figures/seven_axis_safety_resource_separated.png   included by docs/project_overview/project_overview.tex
├── tests/                                pytest (see table below)
├── docs/                                 Sphinx site; docs/project_overview/*.tex is the TeX overview
├── data/                                 raw datasets, splits, trait-space cache (gitignored)
├── results/                              run outputs (gitignored)
└── figures/                              rendered figures per experiment (gitignored)
```

## Entry points

| Command | Module | Purpose |
|---|---|---|
| `python -m infl_ens.pipeline --config configs/experiments/<name>.yaml` | `pipeline/__main__.py` | run an experiment end to end (`--stages`, `--only-arm`, `--force`, `--smoke`, `--dry-run`) |
| `python -m infl_ens.training --config configs/arms/<arm>.yaml [-- k=v]` | `training/__main__.py` | one arm: `closed_loop` or `baseline_replay` |
| `python -m infl_ens.evaluation --config <run>/resolved_config.yaml [-- k=v]` | `evaluation/__main__.py` | score archived adapters on the held-out partitions |
| `python -m infl_ens.figures --config <experiment> [--only a,b] [--list]` | `figures/__main__.py` | render figures and tables into `figures/<experiment>/` |
| `bash scripts/run_on_doob.sh` | — | drive the pipeline on the GPU host under tmux |

## `src/infl_ens/` — top level

| File | Role |
|---|---|
| `config.py` | `load_config` (includes → overrides → validation), key tables (`TOP_LEVEL_KEYS`, `CLOSED_LOOP_KEYS`, ...), `resolve_sft_block`, `ConfigError` |
| `experiment.py` | `load_experiment` → `ExperimentConfig` (`arms`, `stages`, `eval`, `figures`, `smoke`), `ArmSpec` (with optional `family`/`scale` + `cell`); `generalists`/`generalist_for` pair each specialist with its same `(family, scale)` generalist |

## `src/infl_ens/data/`

| File | Role |
|---|---|
| `encoders.py` | `HuggingFaceEncoder` (AutoModel + mean/cls/last_token pooling, L2 norm); `make_encoder(cfg)` / `encoder_kwargs_from_config` read `trait_space.encoder` + the `encoder` block |
| `trait_space.py` | `TraitSpace` (grid, weights, project), `build_trait_space` (anchor/PCA), `position_from_corpus` |
| `trait_space_cache.py` | `trait_space_fingerprint` (hashes `benchmarks` + `trait_space` minus throughput keys), save/load cache, `build_or_load_safety_trait_space`, `load_cache_artifacts`, `coordinate_chain_from_cache` |
| `trait_normalize.py` | `QuantileNormalizer`, `fit_quantile_normalizer` |
| `position_blend.py` | `apply_position_update`, `parse_position_step`, `effective_blend` |
| `splits.py` | `DataSplitManifest`, `build_split_manifest`, `choose_exact_train_coverage`, `apply_manifest_partition`, `build_manifest_from_config` |
| `download.py` | `download_<kind>` functions, `DOWNLOADERS`, `download_for_entry`, `entry_is_present` |
| `benchmarks/loading.py` | `BENCHMARK_KINDS`, `load_benchmark_split(s)`, `load_benchmark_splits_with_partition`, `subsample_split` |
| `benchmarks/<kind>.py` | one offline loader per benchmark returning a `BenchmarkSplit` |
| `benchmarks/safety_trait_space.py` | `build_safety_trait_space_bundle` (learned Fisher axes, residualisation, quantile normalisation, KDE grid), `LearnedAxis` |

## `src/infl_ens/inflgame/router/`

| File | Role |
|---|---|
| `agents.py` | `RouterAgent` (name, position, `from_calibration`, `update_position_from_corpus`) |
| `allocation.py` | `allocation_weights`, `expected_utilities`, `empirical_utility`, `utility_gradient`, `strategic_routing_weights`, `top_k_allocation_weights`, `sampled_top_k_mask`, `matched_centroid_mass`, `group_allocation_weights` |
| `core.py` | `InfluencerRouter` (`route`, `route_batch`, `expected_utilities`) |
| `verification.py` | expected-drift derivations and Monte-Carlo checks of every routing / position-update rule (used by tests) |

## `src/infl_ens/training/`

| File | Role |
|---|---|
| `__main__.py` | argparse → `load_config` → `TASKS[cfg["task"]]`; exit 2 on config errors |
| `tasks.py` | `TASKS = {closed_loop, baseline_replay}`, `run_baseline_replay` |
| `closed_loop.py` | `run_closed_loop`, `validate_routing_and_loss_modes`, `init_agents_closed_loop`; module docstring lists every `closed_loop.*` knob |
| `setup.py` | `load_splits`, `make_trait_space`, `sigma_from_config`, `init_agents`, `coords_for_prompts`, `write_history`, `write_resolved_config` |
| `agent_init.py` | `resolve_agent_entries`, `init_agents_theory_gradient(_paired)`, `co_locate_theory_pairs`, pairing rules, separated random starts |
| `position_step.py` | `blend_for_round`, `expected_pool_centroid` (+ re-exports of `position_blend`) |
| `router_training.py` | `RouterTrainingConfig`, `train_router_positions` |
| `sft_training.py` | `SFTTrainingConfig`, `sft_train_agent`, weighted causal-LM loss, `make_chat_formatter` (base-model chat template with Qwen fallback) |
| `merge_training.py` | `parse_sft_merge_groups`, `merge_groups_from_theory_pairs`, `snap_configured_merge_pairs`, `soft_pair_assignments`, `soft_pair_position_target`, `closed_loop_weight_args` |
| `baseline_replay.py` | `pooled_batch_from_round`, `replay_pooled_baseline_sft`, `make_pooled_baseline_agent` |
| `data_split.py` | `resolve_closed_loop_data_split`, `shuffled_train_batch_indices`, `partitioned_splits_for_eval` |
| `closed_loop_eval.py` | `run_closed_loop_val_eval`, `append_val_eval_summary` |
| `pool_dynamics.py` | `run_gradient_ascent_theory`, `classify_layout`, `pairwise_spread`, `agent_pairwise_geometry` |

## `src/infl_ens/evaluation/`

| File | Role |
|---|---|
| `__main__.py` | argparse → `load_config` → `run_unified_eval` (training YAML with `eval`) or `run_eval_job` |
| `evaluate.py` | `AdapterEvalConfig`, `EvalJobConfig` (+ `from_unified`), `evaluate_adapter_on_split(s)`, `evaluate_run_adapters`, `run_unified_eval`, `final_round_from_history`, `write_eval_report` |
| `routing_eval.py` | `run_flat_routing_eval` (pooled / expected / sampled / argmax G / oracle), `report_to_dict`, `format_headline_markdown` |
| `adapters.py` | `AdapterRef`, `discover_adapters`, `resolve_adapter_dir`, `load_adapter_model` |
| `metrics.py` | `format_chat_example`, `build_chat_formatter` (chat formatter from a base-model id), `mean_token_nll`, `split_to_texts` |
| `benchmarks.py` | re-exports `data.benchmarks.loading` |

## `src/infl_ens/figures/`

| File | Role |
|---|---|
| `render.py` | `FigureSpec`, `FIGURES` (`oracle_routing`, `arm_comparison`, `pair_positions`, `within_pair`, `closed_loop_history`, `per_round_tables`, `cross_arm_report`, `family_scale_nll`, gpu: `trait_representation`, `benchmark_space`), `render_all` |
| `__main__.py` | CLI over `render_all` |
| `style.py` | `apply_paper_style`, `BENCHMARK_ORDER`, `BENCHMARK_LABELS`, `PGF_BENCHMARK_ORDER` |
| `save.py` | `save_figure` |
| `closed_loop.py` | `plot_history` (projects L > 2 onto the first two axes), `plot_pairwise_position_updates`, `plot_trajectory_overlay` |
| `pair_positions.py` | `plot_final_positions`, `plot_within_pair`, `merge_groups_from_config/history`, `within_pair_series` |
| `benchmark_space.py` | `plot_pairwise_heatmaps` |
| `benchmark_nll_bar.py` | `plot_benchmark_nll_comparison` |
| `scale_family.py` | `CellNLL`, `plot_family_scale_nll` (family x scale NLL heatmap), `write_family_scale_table` (csv/md/tex/json) |
| `trait_representation.py` | `legacy_coordinates`, `representation_stats`, `plot_marginals`, `plot_pair_comparison`, `plot_dataset_composition`, `stratified_sample` |
| `pgf_tex.py` | `oracle_routing_tex`, `arm_comparison_tex`, `compile_tex`, `tex_escape` |
| `per_round_tables.py` | `load_eval_rows`, `eval_rows_cover`, `pivot_per_round`, `write_per_round_outputs`, `build_per_round_tables` |
| `cross_arm_report.py` | `data_matching`, `round_prompt_sets`, `within_pair_distances`, `build_cross_arm_report`, `write_cross_arm_report` |

## `src/infl_ens/pipeline/`

| File | Role |
|---|---|
| `stages.py` | `PipelineContext`, `STAGES`, `run_pipeline`, `run_smoke`, `run_is_complete`, `smoke_config`, `resolved_run_config` |
| `__main__.py` | argparse, `--dry-run` planner (`describe`), logging to `<results_dir>/pipeline.log` |

## `src/infl_ens/utils/`

| File | Role |
|---|---|
| `resource.py` | `weighted_mean`, `weighted_covariance`, `gaussian_stability_threshold` |
| `checkpoints.py` | `prune_intermediate_adapters` |

## `configs/`

| File | Role |
|---|---|
| `encoders/qwen3_embedding_8b_awq.yaml` | `trait_space.encoder` = `drawais/Qwen3-Embedding-8B-AWQ-INT4` (fingerprinted) + `encoder` kwargs (left padding, last-token pooling, device_map auto) |
| `encoders/bge_large_en_v1_5.yaml` | worked template for a different Hugging Face encoder (cls pooling, right padding) |
| `trait_space/seven_axis.yaml` | includes the encoder preset; cache dir, `n_grid: 3`, `kde_bandwidth: 0.08`, residualisation, mode-alignment weights, stretch off |
| `data/seven_axis_safety.yaml` | the seven `benchmarks` entries + `data_split` (70/10/20, exact train coverage, 12 rounds) |
| `models/qwen2_5_1_5b_instruct.yaml` | top-level `sft` block: base model, LoRA r/alpha/dropout, batch, epochs, bf16, cumulative LoRA |
| `arms/_closed_loop_base.yaml` | includes data + trait_space + model; theory-paired init, `sft_merge_groups: from_init`, `position_update: theory_matched`, final-round `eval` |
| `arms/*.yaml` | one arm each: only `output_dir` and the routing knobs differ (see the on-disk tree) |
| `arms/scale_family/_specialist_base.yaml` | includes `_closed_loop_base.yaml` + soft k = 3 routing; the shared base for every sweep cell |
| `arms/scale_family/_generalist_base.yaml` | pooled-baseline replay base (data + trait_space + model), no hardcoded run paths |
| `arms/scale_family/{qwen,llama,gemma}_{1b,3b,8b}.yaml` | 9 specialist cells; each overrides only `output_dir` + `sft.base_model` |
| `arms/scale_family/{qwen,llama,gemma}_{1b,3b,8b}_gen.yaml` | 9 per-cell generalists; each sets `sft.base_model`, `history_path`, `output_dir` |
| `experiments/seven_axis_3arm.yaml` | five specialist arms + generalist, stages, `perround_rounds: [4, final]`, figure list, smoke gate |
| `experiments/scale_family_sweep.yaml` | 3 family x 3 scale sweep: 9 specialist + 9 generalist arms (with `family`/`scale`), routing per cell, `family_scale_nll` figure |

Every arm (including all scale-family cells) resolves to byte-identical `benchmarks` + `trait_space` blocks (cache fingerprint `3b42c68a8dd334c5`), enforced by `tests/test_config_fingerprint.py` and `tests/test_scale_family.py`.

## `tests/`

| File | Covers |
|---|---|
| `test_config.py` | includes, overrides, key validation of `infl_ens.config` |
| `test_config_fingerprint.py` | every arm keeps the cache fingerprint; arms differ only in routing knobs |
| `test_scale_family.py` | the 9 sweep cells keep the fingerprint and differ only in `sft.base_model`; experiment pairs each cell's generalist; `make_chat_formatter` template + fallback; `family_scale_nll` figure/table |
| `test_encoder_config.py` | encoder presets and `make_encoder` resolution (no torch) |
| `test_encoders.py` | `HuggingFaceEncoder` pooling / placement (mocked transformers; needs torch) |
| `test_training_cli.py` | `python -m infl_ens.training` dispatch and error exits |
| `test_pipeline.py` | experiment loading, `--dry-run`, stage skip logic, smoke overrides, status file |
| `test_figures_smoke.py` | every plot/table builder on synthetic inputs; `render_all` over a fake experiment |
| `test_checkpoints.py` | `prune_intermediate_adapters` |
| `test_soft_pairs.py`, `test_topk_matched.py`, `test_sampled_topk.py`, `test_weighted_sft_loss.py` | soft / top-k / sampled routing, theory-matched updates, weighted SFT loss, stubbed closed loops |
| `test_merge_training.py`, `test_agent_init.py`, `test_position_step.py`, `test_data_splits.py` | merge groups, theory init, position steps, split manifests |
| `test_evaluation.py`, `test_unified_eval.py`, `test_routing_eval.py` | adapter scoring, unified eval from a training YAML, route-then-score |
| `test_benchmark_loaders.py`, `test_new_benchmark_loaders.py`, `test_ai4privacy_loader.py`, `test_jbb_behaviors_loader.py` | offline benchmark loaders |
| `test_safety_trait_space.py`, `test_trait_normalize.py`, `test_trait_space_cache.py` | trait-space construction, quantile normaliser, cache round-trip and fingerprint |

## Re-exports

- `infl_ens.data`: `TraitSpace`, `build_trait_space`, `position_from_corpus`, `HuggingFaceEncoder`, `QuantileNormalizer`, `benchmarks`
- `infl_ens.data.benchmarks`: `BenchmarkSplit`, `LearnedAxis`, `build_safety_trait_space`, the seven `load_*` loaders and their constants
- `infl_ens.inflgame.router`: `InfluencerRouter`, `RouterAgent`, `allocation_weights`, `expected_utilities`, `empirical_utility`, `utility_gradient`, `strategic_routing_weights`, `top_k_allocation_weights`, `sampled_top_k_mask`, `matched_centroid_mass`, `group_allocation_weights`
- `infl_ens.training`: `RouterTrainingConfig`, `train_router_positions` (eager); `SFTTrainingConfig`, `sft_train_agent`, `make_chat_formatter` (lazy)
- `infl_ens.evaluation`: `AdapterEvalConfig`, `BenchmarkEvalResult`, `EvalJobConfig`, `evaluate_adapter_on_split(s)`, `evaluate_run_adapters`, `run_eval_job`, `run_unified_eval`, `final_round_from_history`, `write_eval_report`, `AdapterRef`, `discover_adapters`, `is_adapter_dir`, `resolve_adapter_dir`, `BENCHMARK_KINDS`, `load_benchmark_splits`, `subsample_split`; lazy `build_chat_formatter`, `format_chat_example`, `mean_token_nll`, `split_to_texts`
- `infl_ens.figures`: the pure plot functions (incl. `plot_family_scale_nll`, `write_family_scale_table`, `CellNLL`), `oracle_routing_tex`, `arm_comparison_tex`, `save_figure`, `apply_paper_style`, `BENCHMARK_ORDER`, `BENCHMARK_LABELS`
- `infl_ens.pipeline`: `STAGES`, `PipelineContext`, `run_pipeline`, `run_smoke`
- `infl_ens.utils`: `weighted_mean`, `weighted_covariance`, `gaussian_stability_threshold`

## Deferred cleanups

- `closed_loop.position_step` adaptive modes (`cap_linf`, `cap_l2`, `trust_box`) and `blend_schedule` / `blend_start` are still accepted and tested (`tests/test_position_step.py`) but unused by the canonical arms; removing them means threading a kwarg out of ~12 call sites in `training/closed_loop.py`.
- `infl_ens.data.trait_space.build_trait_space` (anchor / PCA trait spaces) is only used by tests; the pipeline builds trait spaces from labelled benchmarks.
