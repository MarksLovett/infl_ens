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
│           └── resource.py                       weighted_mean, weighted_covariance, σ₀*
├── scripts/
│   ├── download_beavertails.py                   one-off download (HF datasets)
│   ├── download_halueval.py                      one-off download (RUCAIBox/HaluEval JSON files)
│   ├── build_safety_trait_space.py               wrapper around `python -m infl_ens.data`
│   ├── compare_utility_estimators.py             u_grid vs u_pool vs share diagnostic
│   ├── compare_theory_vs_sft.py                  strategic Nash vs SFT closed-loop endpoints
│   ├── plot_closed_loop_history.py               trajectories + utility tracking from history.json
│   ├── run_sweep.sh                              bash sweep launcher (seeds / sigma / kde)
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
├── docs/
│   ├── conf.py                                   Sphinx config (autodoc, autosummary, MyST, Furo)
│   ├── index.rst                                 main TOC: user guide / API / tooling
│   ├── getting_started.rst                       wraps README.md via myst-parser
│   ├── structure.rst                             wraps this file via myst-parser
│   ├── scripts.rst                               script table + invocation notes
│   ├── configs.rst                               YAML config tree reference
│   ├── requirements.txt                          docs build deps (sphinx, furo, myst, copybutton)
│   ├── Makefile                                  ``make -C docs html`` for local builds
│   ├── make.bat                                  Windows equivalent
│   ├── _static/.gitkeep                          tracked placeholder for static assets
│   ├── _templates/.gitkeep                       tracked placeholder for template overrides
│   └── api/
│       ├── data.rst                              recursive autosummary for infl_ens.data
│       ├── inflgame.rst                          recursive autosummary for infl_ens.inflgame
│       ├── training.rst                          recursive autosummary for infl_ens.training
│       └── utils.rst                             recursive autosummary for infl_ens.utils
├── .github/
│   └── workflows/
│       └── docs.yml                              builds Sphinx + deploys to GitHub Pages
├── pyproject.toml                                hatchling build backend, src/ layout, optional extras
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
| `__main__.py` | Single CLI: `python -m infl_ens.training --config <path>` dispatches on the config's `task` field. Closed-loop task honours `closed_loop.routing_weight` (`G` / `G_times_1mG`) and `closed_loop.save_per_round` (per-round adapter archiving); always logs `agent_prompts` / `agent_responses` / `agent_sft_logs` per round in `history.json`. | `main` |
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
| `compare_theory_vs_sft.py` | Rebuilds the trait space from a closed-loop run's config, initialises agents from `history.json` round 0, runs `train_router_positions`, and compares the strategic-Nash endpoints with the SFT trajectory in trait space. |
| `plot_closed_loop_history.py` | Reads `history.json` from a closed-loop run and renders trajectories + utility tracking to PDF/PNG under `scripts/figures/`. |
| `run_sweep.sh` | Bash launcher that sweeps one parameter (seeds, sigma_fraction, or kde_bandwidth) over the closed-loop trainer. Skips runs whose `history.json` already exists; optionally runs per-run plotting and theory comparison after each training. |
| `plot_sweep.py` | Aggregates a sweep root directory into one figure: per-run trajectory panels, equilibrium-type classification by single-linkage clustering, optional overlay of theoretical Nash endpoints, CSV summary. |
| `probe_sft_capability.py` | Capability probe: reads a closed-loop run with `save_per_round: true` and computes Tier 1 (per-agent SFT loss curves) + Tier 3 (cross-perplexity matrix and specialisation margin per round). Headline output is `μ(r) = NLL(others) − NLL(own)`; negligible μ means SFT isn't actually differentiating the agents and observed position dynamics are pure routing-centroid geometry. |
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

### `docs/`

Sphinx project that builds the public documentation site. The HTML
output is published to GitHub Pages by `.github/workflows/docs.yml` on
every push to `main` that touches `src/`, `docs/`, `README.md`, or
`structure.md`. Heavy ML dependencies (torch, transformers, datasets,
peft, ...) are *not* required to build the docs — they are mocked via
`autodoc_mock_imports` in `docs/conf.py`.

| File | Role |
|---|---|
| `conf.py` | Sphinx configuration: autodoc + autosummary (`:recursive:`), MyST, Furo theme, mathjax, copybutton, intersphinx; adds `../src` to `sys.path`; mocks heavy deps |
| `index.rst` | Main page; defines the three-section TOC (User guide / API reference / Tooling) |
| `getting_started.rst` | Includes `../README.md` verbatim via MyST so the quick-start stays in one place |
| `structure.rst` | Includes `../structure.md` verbatim via MyST |
| `scripts.rst` | Table of every `scripts/*.py` with role + invocation snippet |
| `configs.rst` | Table of every `configs/**/*.yaml` |
| `api/data.rst` | Recursive autosummary entry point for `infl_ens.data` |
| `api/inflgame.rst` | Recursive autosummary entry point for `infl_ens.inflgame` |
| `api/training.rst` | Recursive autosummary entry point for `infl_ens.training` |
| `api/utils.rst` | Recursive autosummary entry point for `infl_ens.utils` |
| `requirements.txt` | Docs-only build deps: `sphinx`, `furo`, `myst-parser`, `sphinx-copybutton` |
| `Makefile` / `make.bat` | Local build (`make -C docs html`) |
| `_static/`, `_templates/` | Tracked placeholders for static assets and template overrides |

### `.github/workflows/`

| File | Role |
|---|---|
| `docs.yml` | Builds `docs/` with Sphinx and publishes to GitHub Pages via `actions/upload-pages-artifact` + `actions/deploy-pages`. Writes `.nojekyll` into the artifact so directories like `_static/` are served. One-time setup: **Settings → Pages → Source: GitHub Actions**. |

### `pyproject.toml`

Single source of build and tooling configuration. Uses **hatchling** as
the build backend, declares the package at `src/infl_ens`, and pins core
runtime deps to the minimum (`numpy>=1.24`). Heavy ML deps are opt-in
via extras so `pip install infl_ens` stays slim. The cross-extra `dev`
includes everything plus `ruff` and `mypy`.

| Section | Role |
|---|---|
| `[build-system]` | Hatchling backend |
| `[project]` | Metadata + core deps (just `numpy`) |
| `[project.optional-dependencies]` | Extras: `ml`, `data`, `vis`, `configs`, `docs`, `test`, `dev` |
| `[project.scripts]` | Console-script aliases `infl-ens-train`, `infl-ens-data` |
| `[tool.hatch.build.targets.{wheel,sdist}]` | Ships `src/infl_ens` in the wheel; sdist also bundles `configs/`, `scripts/`, `docs/`, `tests/` |
| `[tool.pytest.ini_options]` | `testpaths = ["tests"]`, `pythonpath = ["src"]` |
| `[tool.ruff]` / `[tool.ruff.lint]` | Lint config; `F401`/`F403` ignored in `__init__.py` |
| `[tool.mypy]` | Targets `src/infl_ens`, ignores missing imports for heavy deps |
| `[tool.coverage.*]` | Branch coverage of `src/infl_ens` |

Install workflows:

```bash
# Minimal install (analytical pieces only)
pip install .

# Full ML stack (LoRA SFT, sentence-transformer encoders)
pip install ".[ml,data,configs]"

# Everything (tests + docs + linters)
pip install -e ".[dev]"
```

## `__init__.py` re-export summary

- `src/infl_ens/__init__.py`: minimal — does not import subpackages eagerly.
- `src/infl_ens/data/__init__.py`: `TraitSpace`, `build_trait_space`, `position_from_corpus`, `SentenceTransformerEncoder`, `HuggingFaceEncoder`, `benchmarks`.
- `src/infl_ens/data/benchmarks/__init__.py`: `BenchmarkSplit`, `load_beavertails`, `load_halueval`, `build_safety_trait_space`, `LearnedAxis`, `BEAVERTAILS_CATEGORIES`, `HALUEVAL_TASKS`.
- `src/infl_ens/inflgame/__init__.py`: re-exports the `router` subpackage.
- `src/infl_ens/inflgame/router/__init__.py`: `InfluencerRouter`, `RouterAgent`, `allocation_weights`, `expected_utilities`, `utility_gradient`.
- `src/infl_ens/training/__init__.py`: `RouterTrainingConfig`, `train_router_positions`; lazy `SFTTrainingConfig`, `sft_train_agent` (avoids importing torch/transformers at package import time).
- `src/infl_ens/utils/__init__.py`: `weighted_mean`, `weighted_covariance`, `gaussian_stability_threshold`.
