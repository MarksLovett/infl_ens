# Influencer-Game Router (`inflai_router`)

Drop-in router package for the `infl_ens` project: routes queries to candidate
models via the proportional-allocation rule

```math
G_i(\mathbf{x}, b) = \frac{f_i(x_i, b)}{\sum_{j=1}^{N} f_j(x_j, b)}
```

from Lovett & Fu (2024), with a multivariate-Gaussian influence kernel on a
trait space constructed automatically from a calibration corpus.

## Contents

```
inflai_router/
├── README.md                                       (this file)
├── structure_md_addition.md                        patch for project structure.md
├── smoke_test.py                                   end-to-end sanity check
├── closed_loop_demo.py                             clones-to-specialists demo
├── configs/benchmark/router/example.yaml           Hydra config (AGENTS.md §4 rule 1)
└── src/infl_ens/
    ├── data/trait_space.py                         TraitSpace + build_trait_space + position_from_corpus
    ├── inflgame/router/
    │   ├── agents.py                               RouterAgent (+ from_calibration, update_position_from_corpus)
    │   ├── allocation.py                           allocation_weights, expected_utilities, utility_gradient
    │   └── core.py                                 InfluencerRouter
    ├── training/router_training.py                 RouterTrainingConfig, train_router_positions
    └── utils/resource.py                           weighted_mean, weighted_covariance, gaussian_stability_threshold
```

## Install / use

The layout mirrors `src/infl_ens/...` exactly, so you can either

- **Merge into your existing repo**: copy each `src/infl_ens/...` file to the
  matching path in your tree, then patch `structure.md` per
  `structure_md_addition.md`.
- **Run standalone**: add `<this dir>/src` to `PYTHONPATH` and the smoke
  test / closed-loop demo work as-is. The demos use a toy hash-bag encoder
  so no embedding model is required to verify the pipeline.

```bash
cd inflai_router
PYTHONPATH=src python smoke_test.py
PYTHONPATH=src python closed_loop_demo.py
```

## Quick start

```python
import numpy as np
from infl_ens.data.trait_space import build_trait_space
from infl_ens.inflgame.router import InfluencerRouter, RouterAgent
from infl_ens.utils.resource import gaussian_stability_threshold

# 1. Build trait space from a calibration corpus.
space = build_trait_space(
    queries=my_calibration_queries,
    encoder=my_sentence_encoder,                   # e.g. sentence-transformers
    anchors=["math reasoning", "code", "creative writing"],
)

# 2. Initialise N clones of a base model at the calibration centroid.
x0 = space.mean
agents = [RouterAgent(name=f"clone-{i}", position=x0.copy()) for i in range(3)]

# 3. Pick sigma relative to the closed-form stability threshold.
sigma_star = gaussian_stability_threshold(len(agents), space.grid, space.weights)
sigma = 0.5 * sigma_star          # below threshold → symmetry can break
                                  # above threshold → clones stay clones

# 4. Construct the router.
router = InfluencerRouter(space, agents, sigma=sigma, policy="argmax")

# 5a. Passive use: route queries (positions stay fixed).
chosen = router.route("solve the integral of x squared")

# 5b. Closed-loop use: route -> fine-tune each agent -> re-estimate positions.
for round_idx in range(n_rounds):
    queries = sample_from_resource(batch_size)
    choices = router.route_batch(queries)
    for agent in agents:
        my_finetune(agent, [q for q, c in zip(queries, choices) if c is agent])
        agent.update_position_from_corpus(my_eval_queries[agent.name], space.project)
```

## Design notes

See the two delivered demos for the full picture, but the headline rules are:

- **Positions reflect observed capability**, not strategic choice. Re-estimate
  them after each training round via `RouterAgent.update_position_from_corpus`
  or `position_from_corpus`.
- **Use fixed σ.** The softmax in $G_i$ naturally sharpens as positions
  spread. No σ schedule needed.
- **Pick σ relative to σ₀\***:
  - σ > σ₀\* → symmetric Nash is locally stable; clones stay clones.
  - σ < σ₀\* → bifurcation regime; closed-loop training drives symmetry-breaking.
- **Random tiebreak in argmax** so the identical-positions case routes
  uniformly instead of always picking agent 0.

## What lives where

| Concern | Module |
|---|---|
| Trait-space construction & resource distribution | `data/trait_space.py` |
| Position estimation from a query corpus | `data/trait_space.position_from_corpus` |
| Allocation math ($G_i$, $u_i$, $\nabla u_i$) | `inflgame/router/allocation.py` |
| `RouterAgent` (data + position-update method) | `inflgame/router/agents.py` |
| `InfluencerRouter` (public routing class) | `inflgame/router/core.py` |
| Strategic-agent gradient ascent on positions | `training/router_training.py` |
| Stability threshold + resource summaries | `utils/resource.py` |
| Example Hydra config | `configs/benchmark/router/example.yaml` |

All modules use Sphinx-style docstrings, type hints, and
`from __future__ import annotations` (AGENTS.md §2, §4 rule 7).
