"""Tests for exact fixed-partition replay planning."""

from __future__ import annotations

import numpy as np
import pytest

from infl_ens.training.fixed_partition import (
    audit_reference_batches,
    build_partition_plan,
    hard_kmeans_partition,
)


def test_benchmark_partition_names_experts_directly() -> None:
    coordinates = np.arange(12, dtype=float).reshape(6, 2)
    labels = ["a", "a", "b", "b", "c", "c"]
    plan = build_partition_plan("benchmark", coordinates, labels, n_experts=3)
    assert plan.expert_names == ("domain-a", "domain-b", "domain-c")
    assert plan.assignment.tolist() == [0, 0, 1, 1, 2, 2]


def test_random_partition_is_balanced_and_deterministic() -> None:
    coordinates = np.arange(22, dtype=float).reshape(11, 2)
    labels = ["x"] * 11
    left = build_partition_plan("random", coordinates, labels, n_experts=3, seed=4)
    right = build_partition_plan("random", coordinates, labels, n_experts=3, seed=4)
    np.testing.assert_array_equal(left.assignment, right.assignment)
    counts = np.bincount(left.assignment)
    assert counts.max() - counts.min() <= 1


def test_hard_kmeans_is_deterministic_and_separates_blobs() -> None:
    coordinates = np.array([[0.0], [0.1], [0.2], [9.8], [9.9], [10.0]])
    left, centers = hard_kmeans_partition(coordinates, 2, seed=0, n_init=4)
    right, _ = hard_kmeans_partition(coordinates, 2, seed=0, n_init=4)
    np.testing.assert_array_equal(left, right)
    assert len(set(left[:3])) == 1 and len(set(left[3:])) == 1
    assert centers.shape == (2, 1)


def test_reference_audit_uses_record_ids_not_duplicate_text() -> None:
    prompts = ["same", "same"]
    responses = ["r", "r"]
    ids = ["a:0", "b:0"]
    batches = [np.array([1, 0])]
    history = [{"round": 0, "batch_record_ids": ["b:0", "a:0"]}]
    audit_reference_batches(history, batches, prompts, responses, ids)
    history[0]["batch_record_ids"] = ["a:0", "b:0"]
    with pytest.raises(ValueError, match="record IDs"):
        audit_reference_batches(history, batches, prompts, responses, ids)
