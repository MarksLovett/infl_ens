"""Tests for common expert-routing policies and pure routing scores."""

from __future__ import annotations

import numpy as np
import pytest

from infl_ens.evaluation.routers import (
    build_validation_router_suite,
    gaussian_router,
    hindsight_oracle_router,
    low_support_mask,
    metadata_router,
    nearest_centroid_router,
    normalize_routing_weights,
    permute_router,
    uniform_router,
    validation_selected_permutation,
)
from infl_ens.evaluation.routing_eval import score_routing_weights
from infl_ens.inflgame.router.allocation import (
    balanced_assignment_mask,
    expert_choice_mask,
)


def test_basic_routers_are_row_stochastic() -> None:
    coordinates = np.array([[0.0], [0.9]])
    centers = np.array([[0.0], [1.0]])
    for weights in (
        gaussian_router(centers, coordinates, sigma=0.2),
        nearest_centroid_router(centers, coordinates),
        uniform_router(2, 2),
        metadata_router(["a", "b"], ["domain-a", "domain-b"]),
    ):
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        assert np.all(weights >= 0)


def test_normalization_and_permutation_validate_inputs() -> None:
    np.testing.assert_allclose(
        normalize_routing_weights(np.array([[2.0, 2.0]])), [[0.5, 0.5]],
    )
    with pytest.raises(ValueError):
        normalize_routing_weights(np.zeros((1, 2)))
    weights = np.array([[0.8, 0.2]])
    np.testing.assert_allclose(permute_router(weights, [1, 0]), [[0.2, 0.8]])


def test_pure_scorer_obeys_hindsight_floor() -> None:
    nll = np.array([[1.0, 2.0], [3.0, 0.5]])
    weights = np.array([[0.8, 0.2], [0.7, 0.3]])
    scores = score_routing_weights(weights, nll, np.array([2.0, 3.0]), ["a", "b"])
    oracle = score_routing_weights(
        hindsight_oracle_router(nll), nll, np.array([2.0, 3.0]), ["a", "b"],
    )
    assert oracle.expected_nll <= scores.expected_nll
    assert scores.effective_experts > 1.0


def test_validation_permutation_enumerates_every_mapping() -> None:
    base = np.eye(3)
    nll = np.array([[3.0, 2.0, 1.0], [2.0, 1.0, 3.0], [1.0, 3.0, 2.0]])
    permutation, summary = validation_selected_permutation(base, nll)
    assert summary["n_permutations"] == 6
    assert len(permutation) == 3


def test_validation_suite_fits_without_test_nll() -> None:
    coordinates = np.linspace(0, 1, 20)[:, None]
    labels = ["a"] * 10 + ["b"] * 10
    nll = np.column_stack([coordinates[:, 0], 1.0 - coordinates[:, 0]])
    native = gaussian_router(np.array([[0.0], [1.0]]), coordinates, sigma=0.3)
    suite, metadata = build_validation_router_suite(
        coordinates,
        nll,
        np.full(20, 3.0),
        labels,
        native,
        coordinates,
        labels,
        native,
        ["domain-a", "domain-b"],
        max_steps=20,
    )
    assert {"fitted_classification", "fitted_soft", "fitted_regression"} <= set(suite)
    assert "metadata" in suite and "prompt_to_label" in suite
    assert metadata["validation_selected_permutation"]["n_permutations"] == 2


def test_low_support_is_stratified() -> None:
    coordinates = np.arange(20, dtype=float)[:, None]
    labels = ["a"] * 10 + ["b"] * 10
    mask = low_support_mask(
        coordinates,
        labels,
        {"a": np.array([4.5]), "b": np.array([14.5])},
        np.array([9.5]),
        np.array([6.0]),
        fraction=0.2,
    )
    assert mask[:10].sum() == 2 and mask[10:].sum() == 2


def test_balanced_and_expert_choice_allocation_semantics() -> None:
    pytest.importorskip("scipy")
    affinity = np.array([
        [0.9, 0.8, 0.1, 0.2, 0.3],
        [0.1, 0.2, 0.9, 0.8, 0.7],
    ])
    balanced = balanced_assignment_mask(affinity)
    assert np.all(balanced.sum(axis=0) == 1)
    assert max(balanced.sum(axis=1)) - min(balanced.sum(axis=1)) <= 1
    choice = expert_choice_mask(affinity, capacity_factor=1.2)
    assert choice.sum() == round(1.2 * affinity.shape[1])
    assert np.any(choice.sum(axis=0) != 1)
