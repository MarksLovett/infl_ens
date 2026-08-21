"""Offline tests for the AI4Privacy PII loader.

These tests do not touch the network: they write tiny synthetic JSONL
fixtures matching the AI4Privacy ``pii-masking-200k`` schema and verify
that the loader parses them, computes the privacy-density axis correctly
(including overlap merging and key-name aliases), and produces a
well-formed :class:`BenchmarkSplit`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from infl_ens.data.benchmarks import (
    BenchmarkSplit,
    PII_SCORE_MODES,
    load_ai4privacy,
)


def _write_fixture(tmp_path: Path, name: str = "english_pii_test.jsonl") -> Path:
    """Write a tiny AI4Privacy-format JSONL fixture.

    :param tmp_path: Pytest tmp dir.
    :type tmp_path: pathlib.Path
    :param name: Filename (defaults to an english-marked shard name).
    :type name: str
    :returns: Path to the written fixture.
    :rtype: pathlib.Path
    """
    f = tmp_path / name
    # source_text length 20; PII span [0,10) covers 10 chars -> density 0.5
    rows = [
        {
            "source_text": "John Smith lives here",  # len 21
            "target_text": "[NAME] lives here",
            "privacy_mask": [
                {"value": "John Smith", "start": 0, "end": 10, "label": "NAME"}
            ],
        },
        # Two overlapping spans must merge: [0,5) and [3,8) -> covered 8 chars.
        {
            "source_text": "abcdefghij",  # len 10
            "target_text": "[X]",
            "privacy_mask": [
                {"value": "abcde", "start": 0, "end": 5, "label": "A"},
                {"value": "defgh", "start": 3, "end": 8, "label": "B"},
            ],
        },
        # privacy_mask provided as a JSON-encoded string (Isotonic mirror).
        {
            "unmasked_text": "0123456789",  # len 10, alias key
            "masked_text": "[Y]",
            "privacy_mask": json.dumps(
                [{"value": "01234", "start": 0, "end": 5, "label": "C"}]
            ),
        },
    ]
    with f.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return f


def test_load_ai4privacy_density_axis(tmp_path: Path) -> None:
    """Default density score is character coverage / source length."""
    f = _write_fixture(tmp_path)
    split = load_ai4privacy(f)
    assert isinstance(split, BenchmarkSplit)
    assert split.axis_name == "privacy"
    assert split.name == "ai4privacy"
    assert split.n == 3
    assert split.metadata["score_target"] == "prompt"
    # Row 0: 10 / 21, Row 1 (merged): 8 / 10, Row 2: 5 / 10.
    assert np.isclose(split.scores[0], 10 / 21)
    assert np.isclose(split.scores[1], 0.8)
    assert np.isclose(split.scores[2], 0.5)


def test_load_ai4privacy_overlap_merged(tmp_path: Path) -> None:
    """Overlapping PII spans are merged, not double-counted."""
    f = _write_fixture(tmp_path)
    split = load_ai4privacy(f)
    # If overlaps were summed, row 1 would be (5+5)/10 = 1.0; merged = 0.8.
    assert np.isclose(split.scores[1], 0.8)


def test_load_ai4privacy_binary_mode(tmp_path: Path) -> None:
    """Binary mode flags any-PII presence as 1.0."""
    f = _write_fixture(tmp_path)
    split = load_ai4privacy(f, score_mode="binary")
    assert np.allclose(split.scores, 1.0)


def test_load_ai4privacy_alias_keys(tmp_path: Path) -> None:
    """The Isotonic-mirror unmasked_text/masked_text aliases are accepted."""
    f = _write_fixture(tmp_path)
    split = load_ai4privacy(f)
    # Row 2 used alias keys; its prompt must still be present.
    assert split.prompts[2] == "0123456789"


def test_load_ai4privacy_max_records(tmp_path: Path) -> None:
    """``max_records`` honours the cap."""
    f = _write_fixture(tmp_path)
    split = load_ai4privacy(f, max_records=2)
    assert split.n == 2


def test_load_ai4privacy_english_only_filter(tmp_path: Path) -> None:
    """Directory mode with english_only skips non-english shards."""
    _write_fixture(tmp_path, name="english_pii_test.jsonl")
    _write_fixture(tmp_path, name="french_pii_test.jsonl")
    split = load_ai4privacy(tmp_path, english_only=True)
    assert split.metadata["n_files"] == 1  # only the english shard
    split_all = load_ai4privacy(tmp_path, english_only=False)
    assert split_all.metadata["n_files"] == 2


def test_load_ai4privacy_bad_score_mode(tmp_path: Path) -> None:
    """An unknown score_mode raises ValueError; advertised modes accepted."""
    f = _write_fixture(tmp_path)
    with pytest.raises(ValueError):
        load_ai4privacy(f, score_mode="nope")
    for mode in PII_SCORE_MODES:
        load_ai4privacy(f, score_mode=mode)


def test_load_ai4privacy_missing_path(tmp_path: Path) -> None:
    """A missing path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_ai4privacy(tmp_path / "does_not_exist.jsonl")
