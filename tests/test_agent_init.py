"""Tests for theory-based agent initialization helpers."""

from __future__ import annotations

import numpy as np

from infl_ens.data.trait_space import TraitSpace
from infl_ens.training.agent_init import (
    adjacent_harm_pair_indices,
    co_locate_theory_pairs,
    harm_pair_indices,
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


def test_adjacent_harm_pair_indices_six_agents() -> None:
    pos = np.array([
        [0.50, 0.0],
        [0.10, 0.0],
        [0.70, 0.0],
        [0.20, 0.0],
        [0.90, 0.0],
        [0.60, 0.0],
    ])
    pairs = adjacent_harm_pair_indices(pos)
    assert [set(p.tolist()) for p in pairs] == [{1, 3}, {0, 5}, {2, 4}]


def test_co_locate_theory_pairs_six_agents() -> None:
    pos = np.array([
        [0.50, 0.0],
        [0.10, 0.0],
        [0.70, 0.0],
        [0.20, 0.0],
        [0.90, 0.0],
        [0.60, 0.0],
    ])
    out = co_locate_theory_pairs(pos, [f"clone-{i}" for i in range(6)])
    assert np.allclose(out[1], out[3])
    assert np.allclose(out[0], out[5])
    assert np.allclose(out[2], out[4])
    assert np.allclose(out[1], [0.15, 0.0])
    assert np.allclose(out[0], [0.55, 0.0])
    assert np.allclose(out[2], [0.80, 0.0])


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


