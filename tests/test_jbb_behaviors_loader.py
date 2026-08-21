"""Tests for :mod:`infl_ens.data.benchmarks.jbb_behaviors`."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from infl_ens.data.benchmarks.jbb_behaviors import (
    BENIGN_FILENAME,
    HARMFUL_FILENAME,
    load_jbb_behaviors,
)


def _write_jbb_fixture(tmp_path: Path) -> Path:
    """Write minimal harmful and benign JBB CSV fixtures."""
    harmful = tmp_path / HARMFUL_FILENAME
    benign = tmp_path / BENIGN_FILENAME
    with harmful.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Goal", "Target"])
        writer.writeheader()
        writer.writerow({"Goal": "hack the system", "Target": "Sure"})
        writer.writerow({"Goal": "bypass safety", "Target": "OK"})
    with benign.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Goal", "Target"])
        writer.writeheader()
        writer.writerow({"Goal": "write a poem", "Target": "Here"})
    return tmp_path


def test_load_jbb_behaviors_balanced_axis(tmp_path: Path) -> None:
    root = _write_jbb_fixture(tmp_path)
    split = load_jbb_behaviors(root)
    assert split.name == "jbb_behaviors"
    assert split.axis_name == "jailbreak"
    assert len(split.prompts) == 3
    assert np.allclose(split.scores, [1.0, 1.0, 0.0])


def test_load_jbb_behaviors_harmful_only(tmp_path: Path) -> None:
    root = _write_jbb_fixture(tmp_path)
    split = load_jbb_behaviors(root, include_benign=False)
    assert len(split.prompts) == 2
    assert np.all(split.scores == 1.0)


def test_load_jbb_behaviors_max_records(tmp_path: Path) -> None:
    root = _write_jbb_fixture(tmp_path)
    split = load_jbb_behaviors(root, max_records=2)
    assert len(split.prompts) == 2


def test_load_jbb_behaviors_missing_harmful(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_jbb_behaviors(tmp_path)
