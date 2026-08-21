"""JailbreakBench JBB-Behaviors loader and jailbreak-axis scoring.

JBB-Behaviors (Chao et al., 2024) is a curated jailbreak benchmark with
100 harmful and 100 benign behaviors. Each record supplies a ``Goal``
(user instruction) and ``Target`` (intended model completion). Harmful
goals are deliberate jailbreak attempts; benign goals are superficially
similar but non-adversarial.

The loader scores the **prompt** (``Goal``) on the ``jailbreak`` axis:
harmful behaviors receive ``1.0``, benign behaviors ``0.0``. This gives
a balanced 50/50 class mix unlike ToxicChat's sparse jailbreak labels.

On-disk layout (written by ``scripts/download_jbb_behaviors.py``)::

    data/jbb_behaviors/
        harmful_behaviors.csv
        benign_behaviors.csv

This module is offline: it reads from disk and performs no network IO.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from infl_ens.data.benchmarks.base import BenchmarkSplit


PathLike = Union[str, Path]

#: Default CSV filenames written by the download script.
HARMFUL_FILENAME: str = "harmful_behaviors.csv"
BENIGN_FILENAME: str = "benign_behaviors.csv"


def _iter_csv(path: Path) -> Iterable[dict[str, Any]]:
    """Yield rows from a CSV file.

    :param path: Path to a ``.csv`` file.
    :type path: pathlib.Path
    :returns: Iterator of row dictionaries.
    :rtype: Iterable[dict]
    """
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def _extract_goal(row: dict[str, Any]) -> Optional[str]:
    """Extract the jailbreak prompt from a JBB record.

    :param row: Raw CSV row.
    :type row: dict
    :returns: Goal text if present, else ``None``.
    :rtype: str | None
    """
    for key in ("Goal", "goal", "prompt", "user_input"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return None


def _resolve_csv_paths(path: Path) -> tuple[Path, Optional[Path]]:
    """Resolve harmful and optional benign CSV paths under ``path``.

    :param path: Directory or single CSV file.
    :type path: pathlib.Path
    :returns: ``(harmful_path, benign_path_or_none)``.
    :rtype: tuple[pathlib.Path, pathlib.Path | None]
    :raises FileNotFoundError: If no harmful behaviors file is found.
    """
    if path.is_file():
        return path, None
    harmful = path / HARMFUL_FILENAME
    benign = path / BENIGN_FILENAME
    if not harmful.is_file():
        alt = path / "harmful-behaviors.csv"
        if alt.is_file():
            harmful = alt
        else:
            raise FileNotFoundError(
                f"JBB harmful behaviors not found under {path}; "
                f"expected {HARMFUL_FILENAME}"
            )
    if not benign.is_file():
        alt = path / "benign-behaviors.csv"
        benign = alt if alt.is_file() else None
    return harmful, benign


def load_jbb_behaviors(
    path: PathLike,
    *,
    include_benign: bool = True,
    max_records: Optional[int] = None,
) -> BenchmarkSplit:
    """Load JBB-Behaviors for the jailbreak trait-space axis.

    :param path: Directory containing JBB CSV files, or a single harmful
        behaviors CSV.
    :type path: str | pathlib.Path
    :param include_benign: Whether to append benign behaviors as
        negatives (``score=0.0``).
    :type include_benign: bool
    :param max_records: Cap on total records after concatenation.
        ``None`` keeps all rows.
    :type max_records: int | None
    :returns: Benchmark split with ``axis_name='jailbreak'``.
    :rtype: BenchmarkSplit
    :raises FileNotFoundError: If required CSV files are missing.
    :raises ValueError: If no valid prompts are found.
    """
    root = Path(path)
    harmful_path, benign_path = _resolve_csv_paths(root)

    records: list[tuple[str, float]] = []
    for row in _iter_csv(harmful_path):
        goal = _extract_goal(row)
        if goal:
            records.append((goal, 1.0))

    if include_benign and benign_path is not None:
        for row in _iter_csv(benign_path):
            goal = _extract_goal(row)
            if goal:
                records.append((goal, 0.0))

    if not records:
        raise ValueError(f"no JBB prompts found under {path}")

    if max_records is not None and len(records) > max_records:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(records), size=int(max_records), replace=False)
        records = [records[int(i)] for i in sorted(idx)]

    prompts = [p for p, _ in records]
    scores = np.asarray([s for _, s in records], dtype=float)

    return BenchmarkSplit(
        name="jbb_behaviors",
        prompts=prompts,
        scores=scores,
        axis_name="jailbreak",
        metadata={
            "source": "JailbreakBench/JBB-Behaviors",
            "license": "mit",
            "score_target": "prompt",
            "include_benign": include_benign,
            "n_harmful_file": str(harmful_path),
            "n_benign_file": str(benign_path) if benign_path else None,
            "n_records": len(prompts),
        },
    )
