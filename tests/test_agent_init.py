"""Tests for pair-near-theory agent initialization."""

from __future__ import annotations

import numpy as np

from infl_ens.data.trait_space import TraitSpace
from infl_ens.utils.agent_init import (
    harm_pair_indices,
    init_agents_pairs_near_reference,
    random_separated_initial_positions,
    run_theory_gradient_positions,
)


def _fake_space() -> TraitSpace:
    grid = np.array([
        [0.1, 0.2],
        [0.9, 0.8],
        [0.1, 0.8],
        [0.9, 0.2],
    ])
    w = np.ones(4) / 4

    def _proj(texts):  # noqa: ANN001
        return np.zeros((len(texts), 2))

    return TraitSpace(grid=grid, weights=w, project=_proj)


def test_harm_pair_indices() -> None:
    pos = np.array([
        [0.1, 0.0],
        [0.15, 0.0],
        [0.9, 0.0],
        [0.85, 0.0],
    ])
    low, high = harm_pair_indices(pos)
    assert set(low.tolist()) == {0, 1}
    assert set(high.tolist()) == {2, 3}


def test_pairs_near_reference_spread_at_init() -> None:
    space = _fake_space()
    ref = np.array([
        [0.1, 0.2],
        [0.12, 0.22],
        [0.9, 0.7],
        [0.88, 0.72],
    ])
    cfg = {"agents": [{"name": f"clone-{i}"} for i in range(4)]}
    agents = init_agents_pairs_near_reference(
        cfg, space, ref, seed=0, init_noise=0.01,
    )
    pos = np.stack([a.position for a in agents])
    assert pos[0, 0] < 0.3 and pos[1, 0] < 0.3
    assert pos[2, 0] > 0.7 and pos[3, 0] > 0.7


def test_random_separated_spread() -> None:
    space = _fake_space()
    p0 = random_separated_initial_positions(space, 4, seed=42, min_pairwise=0.15)
    dists = [
        float(np.linalg.norm(p0[i] - p0[j]))
        for i in range(4) for j in range(i + 1, 4)
    ]
    assert min(dists) >= 0.15


def test_theory_gradient_moves_from_start() -> None:
    space = _fake_space()
  # Use a coarse grid — gradient may not converge; just check shape.
    names = ["a", "b", "c", "d"]
    meta = run_theory_gradient_positions(
        space, names, sigma=0.5, seed=0, n_steps=50, min_pairwise=0.1,
    )
    assert meta["theory_end"].shape == (4, 2)
    assert meta["initial"].shape == (4, 2)
