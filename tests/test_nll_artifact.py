"""Tests for versioned, provenance-checked NLL matrix artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from infl_ens.evaluation.nll_artifact import (
    NLL_ARTIFACT_SCHEMA_VERSION,
    NllMatrixArtifact,
    checkpoint_sha256,
    ordered_records_sha256,
)


def _artifact() -> NllMatrixArtifact:
    return NllMatrixArtifact(
        expert_nll=np.array([[1.0, 2.0], [2.0, 1.0]]),
        reference_nll=np.array([[1.5], [1.4]]),
        token_counts=np.array([3.0, 4.0]),
        expert_names=("a", "b"),
        reference_names=("pooled",),
        bench_labels=("x", "y"),
        record_ids=("x:0", "y:0"),
        provenance={"schema_version": NLL_ARTIFACT_SCHEMA_VERSION, "round": 1},
    )


def test_artifact_round_trip_is_pickle_free(tmp_path: Path) -> None:
    artifact = _artifact()
    path = artifact.save(tmp_path / "nll.npz")
    loaded = NllMatrixArtifact.load(path, expected_provenance=artifact.provenance)
    np.testing.assert_array_equal(loaded.expert_nll, artifact.expert_nll)
    assert loaded.digest == artifact.digest


def test_artifact_rejects_wrong_provenance(tmp_path: Path) -> None:
    path = _artifact().save(tmp_path / "nll.npz")
    with pytest.raises(ValueError, match="provenance"):
        NllMatrixArtifact.load(
            path,
            expected_provenance={"schema_version": NLL_ARTIFACT_SCHEMA_VERSION, "round": 2},
        )


def test_artifact_rejects_duplicate_record_ids() -> None:
    artifact = _artifact()
    bad = NllMatrixArtifact(
        **{**artifact.__dict__, "record_ids": ("same", "same")},
    )
    with pytest.raises(ValueError, match="record_ids"):
        bad.validate()


def test_record_and_checkpoint_hashes_are_sensitive(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    file_path = checkpoint / "adapter_config.json"
    file_path.write_text("one", encoding="utf-8")
    first = checkpoint_sha256(checkpoint)
    file_path.write_text("two", encoding="utf-8")
    assert checkpoint_sha256(checkpoint) != first
    left = ordered_records_sha256(["x:0"], ["p"], ["r"])
    right = ordered_records_sha256(["x:0"], ["p"], ["different"])
    assert left != right
    ordered = ordered_records_sha256(["x:0", "x:1"], ["a", "b"], ["c", "d"])
    reordered = ordered_records_sha256(["x:1", "x:0"], ["b", "a"], ["d", "c"])
    assert ordered != reordered
