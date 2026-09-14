"""Tests for flat-pool routing evaluation helpers."""

from __future__ import annotations

import numpy as np

from infl_ens.evaluation.routing_eval import (
    FlatRoutingReport,
    FlatRoutingScores,
    aggregate_clone_g_to_merge,
    report_to_dict,
)


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


def test_schema_v2_preserves_legacy_headline_aliases() -> None:
    flat = FlatRoutingScores(
        pooled_nll=1.2,
        learned_argmax_nll=1.1,
        learned_expected_nll=1.05,
        learned_sampled_nll=1.08,
        strategic_expected_nll=1.04,
        strategic_argmax_nll=1.02,
        oracle_nll=0.9,
        n_prompts=2,
        round_idx=11,
    )
    payload = report_to_dict(
        FlatRoutingReport(
            flat=flat,
            merge_names=["a", "b"],
            merge_name_map={"a": "a", "b": "b"},
            routers={"native": {"expected_nll": 1.05}},
            references={"generalist_r16": {"mean_nll": 1.2}},
            oracle={"mean_nll": 0.9},
            slices={"full": {}},
            nll_artifact_digest="digest",
            partition="test",
        )
    )
    assert payload["schema_version"] == 2
    assert payload["evaluation"]["partition"] == "test"
    assert payload["flat"]["pooled_nll"] == 1.2
    assert payload["flat"]["oracle_routing_nll"] == 0.9
    assert payload["flat"]["learned_routing_expected_nll"] == 1.05
