# infl_ens

Influencer-game routing for small language models. `infl_ens` implements
the **influencer's game** (Lovett & Fu, 2024) and uses it to train an
ensemble of LoRA specialists: a router places each prompt in a learned
safety *trait space*, allocates it to agents with the proportional rule

$$
G_i(\mathbf{x}, b) = \frac{f_i(x_i, b)}{\sum_{j=1}^{N} f_j(x_j, b)}
$$

(a Gaussian influence kernel around each agent's position), fine-tunes
the agents on what they were routed, and moves every agent toward the
gradient-matched centroid of its routed mass — round after round.

## Install

```bash
pip install -e ".[configs,figures,test]"   # numpy + pyyaml + matplotlib + pytest
pip install -e ".[ml]"                     # torch, transformers, peft, trl: training + eval
```

The GPU host uses a `.venv/` with the `ml` extra; see
`scripts/run_on_doob.sh`.

## Run an experiment

Everything is driven by one experiment file (`configs/experiments/`):

```bash
python -m infl_ens.pipeline --config configs/experiments/seven_axis_3arm.yaml --dry-run   # validate + plan
python -m infl_ens.pipeline --config configs/experiments/seven_axis_3arm.yaml --smoke     # cheap gate
python -m infl_ens.pipeline --config configs/experiments/seven_axis_3arm.yaml             # the real thing
python -m infl_ens.pipeline --config configs/experiments/seven_axis_3arm.yaml --stages routing,figures
```

Stages: `download` (opt-in) → `manifest` → `train` → `perround` →
`routing` → `figures` → `prune` (opt-in). Each stage skips work whose
outputs already exist. Results land under `results/<arm>/seed0/` and
figures under `figures/<experiment>/`; both directories are gitignored.

On the GPU host:

```bash
MODE=smoke bash scripts/run_on_doob.sh    # sync + gate
bash scripts/run_on_doob.sh               # sync + queue under tmux
MODE=status bash scripts/run_on_doob.sh
MODE=pull   bash scripts/run_on_doob.sh
```

The per-stage CLIs (`python -m infl_ens.training`, `infl_ens.evaluation`,
`infl_ens.figures`) take the same YAML files with optional
`-- KEY=VALUE` overrides.

## Configure an experiment

Configs are small YAML fragments composed with `includes:`:

```text
configs/encoders/    which Hugging Face encoder builds the trait space
configs/trait_space/ trait-space geometry (grid, KDE bandwidth, residualisation)
configs/data/        the seven safety benchmarks + the train/val/test split
configs/models/      base LM + LoRA hyperparameters
configs/arms/        one file per arm; only the routing knobs differ
configs/experiments/ which arms, which stages, evaluation window, figures
```

```yaml
# configs/arms/soft_topk3_pairs.yaml
includes: [_closed_loop_base.yaml]
output_dir: results/seven_axis_soft_topk3_pairs/seed0
closed_loop: {routing_mode: soft, soft_top_k: 3, soft_loss: weighted}
```

Unknown keys are rejected with the file that introduced them. To try a
different encoder, copy `configs/encoders/bge_large_en_v1_5.yaml`, set the
model id and pooling, and include it from a `configs/trait_space/*.yaml`.

**Cache contract.** The resolved `benchmarks` + `trait_space` blocks are
hashed into the trait-space cache fingerprint; the GPU host holds the
encode for `3b42c68a8dd334c5`. Every arm resolves to exactly those blocks
(`tests/test_config_fingerprint.py` guards it), so a launch loads the
cache instead of spending hours re-encoding. Changing the encoder model
id or any geometry key starts a fresh encode.

## Design rules

- **Positions reflect observed capability, not strategic choice.** They
  are re-estimated after every SFT round from the routed corpus.
- **Fixed σ.** The allocation sharpens naturally as positions spread;
  `sigma_fraction` picks σ relative to the closed-form stability threshold
  σ₀\* (below 1 the symmetric equilibrium can break and specialists emerge).
- **Theory-matched dynamics.** Under `position_update: theory_matched` the
  expected trait-space drift is parallel to the strategic gradient
  ∇uᵢ ∝ Σ B(b) Gᵢ(1−Gᵢ)(b − xᵢ) in every routing mode.
- **The generalist is data-matched.** The pooled baseline replays the
  batches the specialist arms logged, so the comparison isolates routing.

## Repository map

```text
src/infl_ens/
  config.py, experiment.py    layered YAML + experiment files
  data/                       encoders, trait space + cache, benchmark loaders/downloaders, splits
  inflgame/router/            RouterAgent, InfluencerRouter, allocation math (G, u, ∇u, top-k)
  training/                   closed loop, pooled replay, LoRA SFT, theory init, task registry
  evaluation/                 adapter NLL scoring, route-then-score diagnostics
  figures/                    pure plot/table builders + the artifact-reading render layer
  pipeline/                   stages + the end-to-end CLI
  utils/                      resource moments, σ₀*, checkpoint pruning
configs/                      encoders, trait_space, data, models, arms, experiments
scripts/run_on_doob.sh        the one shell script (sync + tmux + status + pull)
tests/                        pytest (offline; torch-dependent tests skip without it)
docs/                         Sphinx site; docs/project_overview/ holds the TeX overview
```

`structure.md` is the file-by-file map and must be updated with any file
added, moved or removed (AGENTS.md rule 10).

## References

Lovett, M. & Fu, X. (2024). *Learning Dynamics of the Influencer's Game
in Resource Landscapes.*
