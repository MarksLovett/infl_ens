"""Tests for pair-merge closed-loop helpers."""

from __future__ import annotations

import pytest


from infl_ens.training.merge_training import (
    closed_loop_weight_args,
    merge_routed_batch,
    merge_train_name,
    parse_sft_merge_groups,
)


def test_parse_sft_merge_groups() -> None:
    cl = {
        "sft_merge_groups": [
            {"train_as": "merge-low", "names": ["clone-0", "clone-1"]},
            {"train_as": "merge-high", "names": ["clone-2", "clone-3"]},
        ],
    }
    groups = parse_sft_merge_groups(cl, ["clone-0", "clone-1", "clone-2", "clone-3"])
    assert groups == [
        ("merge-low", ["clone-0", "clone-1"]),
        ("merge-high", ["clone-2", "clone-3"]),
    ]


def test_parse_sft_merge_groups_rejects_overlap() -> None:
    cl = {"sft_merge_groups": [["clone-0", "clone-1"], ["clone-1", "clone-2"]]}
    with pytest.raises(ValueError, match="partition"):
        parse_sft_merge_groups(cl, ["clone-0", "clone-1", "clone-2", "clone-3"])


def test_merge_routed_batch_concatenates() -> None:
    prompts = {"a": ["p1"], "b": ["p2", "p3"]}
    responses = {"a": ["r1"], "b": ["r2", "r3"]}
    mp, mr, w = merge_routed_batch(prompts, responses, ["a", "b"])
    assert mp == ["p1", "p2", "p3"]
    assert mr == ["r1", "r2", "r3"]
    assert w is None


def test_merge_train_name_stable() -> None:
    assert merge_train_name(["clone-1", "clone-0"]) == "merge-clone-0-clone-1"


def test_closed_loop_weight_args_position_only() -> None:
    sw, ew, skip = closed_loop_weight_args(
        "position_only", "expected_pool", [0.5, 0.5],
    )
    assert sw is None and ew is None and skip is True
    sw, ew, skip = closed_loop_weight_args(
        "position_only", "batch", [0.5, 0.5],
    )
    assert sw is None and ew == [0.5, 0.5] and skip is False


def test_closed_loop_weight_args_position_update_knob() -> None:
    """The centroid weights follow position_update; the loss follows loss_reweight."""
    w = [0.5, 0.5]
    # Default: unit loss, theory-matched (1-G) centroid == the old position_only.
    assert closed_loop_weight_args(None, "batch", w) == (None, w, False)
    # Naive centroid: nothing weighted.
    assert closed_loop_weight_args(
        None, "batch", w, position_update="naive",
    ) == (None, None, False)
    # one_minus_G weights the loss; the centroid still follows the knob.
    assert closed_loop_weight_args("one_minus_G", "batch", w) == (w, w, False)
    assert closed_loop_weight_args(
        "one_minus_G", "batch", w, position_update="naive",
    ) == (w, None, False)
    # Strategic routing passes no weights: the centroid stays uniform.
    assert closed_loop_weight_args(None, "batch", None) == (None, None, False)
    # The alias overrides an explicit naive request (validation rejects it
    # upstream; here it must at least stay gradient-matched).
    assert closed_loop_weight_args(
        "position_only", "batch", w, position_update="naive",
    ) == (None, w, False)
