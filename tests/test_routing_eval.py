"""Tests for flat-pool routing evaluation helpers."""

from __future__ import annotations

import numpy as np

from infl_ens.evaluation.routing_eval import aggregate_clone_g_to_merge


def test_aggregate_clone_g_to_merge_sums_pairs() -> None:
    """Clone G weights within a merge pair should sum."""
    g_clone = np.array([
        [0.4, 0.1],
        [0.1, 0.5],
        [0.3, 0.2],
        [0.2, 0.2],
    ])
    agent_names = ["clone-0", "clone-1", "clone-2", "clone-3"]
    clone_to_merge = {
        "clone-0": "merge-a",
        "clone-1": "merge-a",
        "clone-2": "merge-b",
        "clone-3": "merge-b",
    }
    merge_names = ["merge-a", "merge-b"]
    name_map = {"merge-a": "merge-a", "merge-b": "merge-b"}
    g_merge = aggregate_clone_g_to_merge(
        g_clone, agent_names, clone_to_merge, merge_names, name_map,
    )
    assert g_merge.shape == (2, 2)
    np.testing.assert_allclose(g_merge[0], [0.5, 0.6])
    np.testing.assert_allclose(g_merge[1], [0.5, 0.4])
    np.testing.assert_allclose(g_merge.sum(axis=0), [1.0, 1.0])
