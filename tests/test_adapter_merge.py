"""Tests for merging reconstructed LoRA deltas."""

from __future__ import annotations

import numpy as np

from infl_ens.training.adapter_merge import (
    dare_ties_merge,
    knots_merge,
    merge_delta_family,
    reconstruct_delta,
    svd_refactor,
    ties_merge,
)


def test_reconstruction_and_svd_refactor() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 4))
    b = rng.normal(size=(3, 2))
    delta = reconstruct_delta(a, b, scaling=2.0)
    out_a, out_b = svd_refactor(delta, rank=2)
    np.testing.assert_allclose(out_b @ out_a, delta, atol=1e-5)


def test_ties_resolves_sign_conflicts() -> None:
    first = np.array([[3.0, -2.0]])
    second = np.array([[1.0, 4.0]])
    merged = ties_merge([first, second], density=1.0)
    assert merged[0, 0] > 0 and merged[0, 1] > 0


def test_dare_is_seeded_and_family_merge_never_averages_factors() -> None:
    deltas = [np.arange(9, dtype=float).reshape(3, 3), np.eye(3)]
    left = dare_ties_merge(deltas, density=0.5, drop_rate=0.2, seed=7)
    right = dare_ties_merge(deltas, density=0.5, drop_rate=0.2, seed=7)
    np.testing.assert_array_equal(left, right)
    family = merge_delta_family(
        [{"m": deltas[0]}, {"m": deltas[1]}], method="linear",
    )
    np.testing.assert_allclose(family["m"], (deltas[0] + deltas[1]) / 2)


def test_knots_linear_reconstructs_mean_in_shared_svd_basis() -> None:
    rng = np.random.default_rng(4)
    deltas = [rng.normal(size=(5, 4)), rng.normal(size=(5, 4))]
    merged = knots_merge(deltas, merge_method="linear", rank=5)
    np.testing.assert_allclose(merged, np.mean(deltas, axis=0), atol=1e-5)
