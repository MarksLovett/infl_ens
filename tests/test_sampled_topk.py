"""Offline tests for sampled (without-replacement) top-k routing selection.

:func:`infl_ens.inflgame.router.allocation.sampled_top_k_mask` draws ``k``
distinct agents per query from the allocation shares. It is the stochastic
counterpart of the deterministic top-``k`` gate, and the generalisation of
hard routing past a single winner, so the closed loop can vary *selection*
(argmax vs sampling) independently of ``k`` and of the loss weighting.

These tests pin the distribution it implements — Gumbel-top-k must equal
successive sampling without replacement — rather than merely checking
shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from infl_ens.inflgame.router.allocation import (
    sampled_top_k_mask,
    top_k_allocation_weights,
)


def _reference_successive_sample(
    probs: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Draw ``k`` distinct indices by explicit renormalise-and-redraw."""
    remaining = probs.astype(float).copy()
    picked: list[int] = []
    for _ in range(k):
        total = remaining.sum()
        idx = int(rng.choice(len(remaining), p=remaining / total))
        picked.append(idx)
        remaining[idx] = 0.0
    return tuple(sorted(picked))


def test_shape_and_cardinality() -> None:
    """Exactly k true entries per query column."""
    rng = np.random.default_rng(0)
    G = rng.random((5, 20))
    G /= G.sum(axis=0, keepdims=True)
    for k in (1, 2, 4):
        mask = sampled_top_k_mask(G, k, np.random.default_rng(1))
        assert mask.shape == G.shape
        assert np.all(mask.sum(axis=0) == k)
    # k >= N selects everyone.
    assert np.all(sampled_top_k_mask(G, 5, np.random.default_rng(1)))
    assert np.all(sampled_top_k_mask(G, 9, np.random.default_rng(1)))


def test_k1_matches_categorical_distribution() -> None:
    """At k=1 the draw is the plain categorical G — i.e. hard routing."""
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    G = np.tile(probs[:, None], (1, 4000))
    mask = sampled_top_k_mask(G, 1, np.random.default_rng(7))
    freq = mask.sum(axis=1) / mask.shape[1]
    assert np.allclose(freq, probs, atol=0.02), freq


def test_matches_successive_sampling_without_replacement() -> None:
    """Gumbel-top-k reproduces renormalise-and-redraw, over the subset law."""
    probs = np.array([0.45, 0.30, 0.15, 0.10])
    n_draws, k = 30_000, 2

    G = np.tile(probs[:, None], (1, n_draws))
    mask = sampled_top_k_mask(G, k, np.random.default_rng(3))
    got: dict[tuple[int, ...], int] = {}
    for col in range(n_draws):
        key = tuple(sorted(np.flatnonzero(mask[:, col]).tolist()))
        got[key] = got.get(key, 0) + 1

    ref_rng = np.random.default_rng(11)
    want: dict[tuple[int, ...], int] = {}
    for _ in range(n_draws):
        key = _reference_successive_sample(probs, k, ref_rng)
        want[key] = want.get(key, 0) + 1

    keys = set(got) | set(want)
    assert len(keys) == 6  # all C(4,2) subsets appear
    for key in keys:
        p_got = got.get(key, 0) / n_draws
        p_want = want.get(key, 0) / n_draws
        assert abs(p_got - p_want) < 0.02, (key, p_got, p_want)


def test_zero_share_agents_are_not_selected() -> None:
    """An agent with zero share is never drawn while others remain."""
    probs = np.array([0.6, 0.4, 0.0, 0.0])
    G = np.tile(probs[:, None], (1, 500))
    mask = sampled_top_k_mask(G, 2, np.random.default_rng(5))
    assert not mask[2].any()
    assert not mask[3].any()
    assert np.all(mask[:2])


def test_sampling_differs_from_argmax_gate() -> None:
    """Sampling explores past the k largest shares; argmax never does."""
    rng = np.random.default_rng(0)
    G = rng.random((6, 300))
    G /= G.sum(axis=0, keepdims=True)
    k = 2
    argmax_mask = top_k_allocation_weights(G, k) > 0.0
    sampled = sampled_top_k_mask(G, k, np.random.default_rng(2))
    # Same budget per query ...
    assert np.all(sampled.sum(axis=0) == argmax_mask.sum(axis=0))
    # ... but a substantial share of queries pick a different set.
    differing = np.mean(np.any(sampled != argmax_mask, axis=0))
    assert differing > 0.2, differing


def test_validates_arguments() -> None:
    G = np.ones((3, 4)) / 3.0
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sampled_top_k_mask(G, 0, rng)
    with pytest.raises(ValueError):
        sampled_top_k_mask(np.ones(3), 1, rng)


def test_soft_pair_assignments_sample_mode() -> None:
    """The merge-group wrapper honours select='sample' and needs an rng."""
    from infl_ens.training.merge_training import soft_pair_assignments

    rng_setup = np.random.default_rng(0)
    pair_pos = rng_setup.random((4, 2))
    clones = np.repeat(pair_pos, 2, axis=0)
    coords = rng_setup.random((40, 2))
    from infl_ens.inflgame.router.allocation import allocation_weights

    g_clone = allocation_weights(clones, coords, 0.2 ** 2 * np.eye(2))
    group_index = np.repeat(np.arange(4), 2)

    W, idx, weights = soft_pair_assignments(
        g_clone, group_index, 4, 2,
        select="sample", rng=np.random.default_rng(1),
    )
    assert np.all((W > 0).sum(axis=0) == 2)
    assert np.allclose(W.sum(axis=0), 1.0)
    for p in range(4):
        assert weights[p].shape == idx[p].shape

    with pytest.raises(ValueError, match="rng"):
        soft_pair_assignments(g_clone, group_index, 4, 2, select="sample")
    with pytest.raises(ValueError, match="select"):
        soft_pair_assignments(g_clone, group_index, 4, 2, select="bogus")


def test_validation_matrix_soft_select() -> None:
    """soft_select is a soft-routing knob and must be a known value."""
    from infl_ens.training.closed_loop import validate_routing_and_loss_modes as check

    check("G", None, routing_mode="soft", soft_top_k=3, n_agents=14,
          has_merge_groups=True, n_groups=7,
          soft_select="sample", soft_loss="unit")
    with pytest.raises(ValueError, match="soft_select"):
        check("G", None, routing_mode="hard", soft_select="sample")
    with pytest.raises(ValueError, match="soft_select"):
        check("G", None, routing_mode="soft", soft_top_k=2, n_agents=4,
              soft_select="bogus")
