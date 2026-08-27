# AGENTS.md — Coding Instructions for `infl_ens`

You are contributing to a Python research package that implements the **influencer's game** (Lovett & Fu, 2024) and extends it to align small language models (SLMs, e.g. Qwen) as the learning agents. Follow these rules without exception.

---

## 1. Repository layout

```
infl_ens/
├── src/infl_ens/            # the importable package
│   ├── config.py            # layered YAML loading shared by every CLI
│   ├── experiment.py        # experiment files: arms, stages, analysis settings
│   ├── data/                # encoders, trait space + cache, benchmark loaders/downloaders, splits (CODE ONLY)
│   ├── inflgame/            # game env: router, allocation math
│   ├── training/            # the closed loop, pooled replay, LoRA SFT, theory init; one CLI
│   ├── evaluation/          # adapter scoring, route-then-score; one CLI
│   ├── figures/             # pure plot/table builders + render.py (the only artifact reader); one CLI
│   ├── pipeline/            # stages + the end-to-end CLI
│   ├── utils/               # resource moments, σ₀*, checkpoint pruning (no sibling imports)
│   └── latex/               # derivation notes (TeX; never moved by tooling)
├── configs/                 # YAML fragments: encoders, trait_space, data, models, arms, experiments
├── scripts/                 # exactly one file: run_on_doob.sh (+ the PNG docs/project_overview includes)
├── tests/                   # pytest
├── docs/                    # Sphinx site + docs/project_overview/*.tex
├── data/                    # raw datasets, splits, trait-space cache (gitignored)
├── results/                 # run outputs: history.json, adapters, eval reports (gitignored)
└── figures/                 # rendered figures per experiment (gitignored)
```

**Never put data files inside `src/`. Never put checkpoints inside `src/`.** Code lives in the package; artifacts live at the repo root (`data/`, `results/`, `figures/`).

The tree above is intentionally schematic. The **authoritative, file-by-file map of the current repo** is in `structure.md` at the repo root. Read it before adding code, and update it whenever you add, move, rename, or delete a file (see §4 rule 10).

---

## 2. Documentation style — Sphinx, always

Every public function, class, and module uses **Sphinx-style docstrings**. Example:

```python
def stability_threshold(n: int, sigma_b: float) -> float:
    """Compute the first-bifurcation threshold for a Gaussian influencer's game.

    Implements Corollary 8 (Lovett & Fu, 2024):
    :math:`\\sigma_0^* = \\sqrt{(N-2)/(N-1)}\\,\\sigma_B`.

    :param n: Number of players. Must satisfy ``n >= 2``.
    :type n: int
    :param sigma_b: Standard deviation of the resource distribution.
    :type sigma_b: float
    :returns: Critical competitive reach below which the symmetric Nash
              equilibrium destabilizes.
    :rtype: float
    :raises ValueError: If ``n < 2`` or ``sigma_b <= 0``.
    """
```

Use `:param:`, `:type:`, `:returns:`, `:rtype:`, `:raises:`, `:math:` (LaTeX), and `:cite:` where relevant. No NumPy-style, no Google-style.

---

## 3. Module placement rules

Before adding a file, decide where it goes using this checklist:

- [ ] Does it define the game's reward, kernel, dynamics, or equilibrium math? → `inflgame/`
- [ ] Does it load, download, score, split or embed a benchmark dataset? → `data/`
- [ ] Does it train a model or run a training loop? → `training/`
- [ ] Does it score adapters or a routed ensemble? → `evaluation/`
- [ ] Does it produce a `matplotlib.figure.Figure`, a TeX figure or a table? → `figures/` (pure builder) and, if it must read run artifacts, a renderer in `figures/render.py`
- [ ] Does it orchestrate stages of an experiment? → `pipeline/stages.py`
- [ ] Is it a helper used by ≥2 subpackages with no sibling imports? → `utils/`
- [ ] Is it a key of a run config? → a table in `config.py` (and a fragment under `configs/`), never a new script

If you can't pick exactly one, the design is wrong — stop and clarify before writing code.

---

## 4. Hard rules

1. **One pipeline entry point, no scripts.** Experiments run through `python -m infl_ens.pipeline --config configs/experiments/<name>.yaml`; the per-stage CLIs are `python -m infl_ens.training`, `infl_ens.evaluation` and `infl_ens.figures`, all driven by the same YAML files with `-- KEY=VAL` overrides. Do not add `run_*.sh`, `plot_*.py`, `build_*_configs.py` or any other one-off: a new experiment is a new file under `configs/arms/` or `configs/experiments/`; a new analysis is a renderer in `figures/render.py` or a stage in `pipeline/stages.py`. `scripts/run_on_doob.sh` is the only shell script and it only invokes the pipeline.
2. **The environment owns the reward.** Trainers consume `inflgame`. They do not reimplement payoff math. If you find yourself computing :math:`G_i(\mathbf{x}, b)` outside `inflgame/`, you are wrong.
3. **`figures/` builders are pure.** Plot and table functions accept records/arrays and return Figures, TeX strings or tables. No disk reads, no training calls, no `plt.show()`. `figures/render.py` is the only place that reads run artifacts; outputs go to `figures/<experiment>/`.
4. **`utils/` does not import siblings.** If a util needs `data`, `inflgame` or `training`, it is not a util — move it.
5. **Outputs go to `results/<run>/`.** Every run writes `history.json` and `resolved_config.yaml` under its config's `output_dir`; downstream stages read the resolved config, never the arm YAML. Never hardcode paths.
6. **Seed everything.** Every training task seeds NumPy and the SFT trainer from the config's `seed` before any RNG use.
7. **Type hints required** on every public function signature. Use `from __future__ import annotations` at the top of every module.
8. **No global mutable state** (no module-level lists/dicts being mutated). Configs are passed; they are not read from globals.
9. **Train on the GPU host** (`mslovett@doob.dartmouth.edu`, override with `REMOTE=`) via `scripts/run_on_doob.sh`; run `MODE=smoke` before queueing a full pipeline.
10. **Keep `structure.md` in sync.** Any change that adds, moves, renames, or deletes a file under `src/`, `configs/`, `scripts/`, or `tests/` MUST update `structure.md` in the same edit: the on-disk tree at the top, the matching row in the file-by-file table, a new section for a new subpackage, and the re-export list when an `__init__.py` changes. A PR that introduces a file without touching `structure.md` is incomplete.
11. **Update the subpackage `__init__.py`** when you add a public module to it, and add a key to the tables in `config.py` (with a test) when you add a config knob.
12. **Protect the trait-space cache.** The resolved `benchmarks` + `trait_space` blocks are hashed into the cache fingerprint (`3b42c68a8dd334c5` on the GPU host). Never add or edit keys under `trait_space` or the shared data fragment casually; `tests/test_config_fingerprint.py` must keep passing unless a re-encode is intended.
13. **Do not move the TeX documentation.** `docs/project_overview/*.tex` and `src/infl_ens/latex/*.tex` stay where they are; `scripts/figures/seven_axis_safety_resource_separated.png` stays because the overview includes it by relative path.

---
