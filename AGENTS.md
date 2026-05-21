# AGENTS.md — Coding Instructions for `inflai`

You are contributing to a Python research package that implements the **influencer's game** (Lovett & Fu, 2024) and extends it to align small language models (SLMs, e.g. Qwen) as the learning agents. Follow these rules without exception.

---

## 1. Repository layout

```
infl_ens/
├── src/
    ├──infl_ens     # the importable package
       ├── inflgame/            # game env, kernels, dynamics, equilibrium
       ├── data/                # benchmark loaders + preprocessing (CODE ONLY)
       ├── training/            # one CLI, many trainers
       ├── vis/                 # plotting (returns Figures, saves as pngs and pdfs in /scripts/figures)
       └── utils/               # seeding, config, runs, io, metrics, geometry
├── scripts/                 # one-off downloads, dataset builds, ad-hoc analysis
├── configs/                 # Hydra/YAML configs (benchmark, algo, model)
├── tests/                   # pytest
├── data/                    # raw + processed datasets (gitignored)
└── results/                 # checkpoints, metrics, rollouts (gitignored)
```

**Never put data files inside `src/`. Never put checkpoints inside `src/`.** Code lives in the package (`src/`, importable as `inflai`); artifacts live at the repo root (`data/`, `results/`).

The tree above is intentionally schematic. The **authoritative, file-by-file
map of the current repo** is in `structure.md` at the repo root. Read it before
adding code, and update it whenever you add, move, rename, or delete a script
(see §4 rule 10).

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
- [ ] Does it load, score, or preprocess a benchmark dataset? → `data/`
- [ ] Does it train a model or run a training loop? → `training/`
- [ ] Does it produce a `matplotlib.figure.Figure`? → `vis/`
- [ ] Is it a helper used by ≥2 subpackages? → `utils/`
- [ ] Is it a one-off (download, manual sweep, paper-figure regeneration)? → `scripts/`, not the package

If you can't pick exactly one, the design is wrong — stop and clarify before writing code.

---

## 4. Hard rules

1. **One training entry point.** `python -m inflai.training` with Hydra config overrides. Do not add `train_linkedin.py`, `train_synthetic.py`, etc. Add a config file under `configs/benchmark/` instead.
2. **The environment owns the reward.** Trainers consume `inflgame.env`. They do not reimplement payoff math. If you find yourself computing :math:`G_i(\mathbf{x}, b)` outside `inflgame/`, you are wrong.
3. **`vis/` is pure.** Plotting functions accept arrays/frames and return Figures. No disk reads, no training calls, no `plt.show()`, figures are saved to \scripts\figures as pdfs and pngs.
4. **`utils/` does not import siblings.** If a util needs `inflgame` or `training`, it is not a util — move it.
5. **Outputs go to `results/<run_id>/`.** Use `inflai.utils.runs.new_run()`. Never hardcode paths.
6. **Seed everything.** Every training script calls `inflai.utils.seeding.seed_all(seed)` before any RNG use.
7. **Type hints required** on every public function signature. Use `from __future__ import annotations` at the top of every module.
8. **No global mutable state** (no module-level lists/dicts being mutated). Configs are passed; they are not read from globals.
9. **SSH to mlovett@doob.dartmouth.edu** for training after you sync the code to the remote server whenever you fine tune\train a model
10. **Keep `structure.md` in sync.** Any change that adds, moves, renames, or deletes a file under `src/`, `scripts/`, `configs/`, or `tests/` MUST update `structure.md` in the same edit.
11. ** Update the submodule `__init__.py`** when you add a new script to a submodule. 
 Specifically:
    - Update the on-disk tree at the top of `structure.md`.
    - Add or update the matching row in the relevant file-by-file table.
    - If you added a new subpackage, add a new section for it.
    - If you re-exported new symbols from an `__init__.py`, list them.
    Do not split this into a follow-up commit — a PR that introduces a new script without also touching `structure.md` is incomplete.

---
