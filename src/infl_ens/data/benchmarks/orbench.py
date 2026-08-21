"""OR-Bench loader and over-refusal axis scoring.

OR-Bench (Cui et al., 2024) is a large-scale benchmark for measuring
*over-refusal* in aligned language models: prompts that appear harmful but
are benign. It is released on HuggingFace as ``orbench-llm/or-bench``
(``cc-by-4.0``) and ships three CSV configs:

- ``or-bench-80k``: seemingly toxic but benign prompts (score ``1.0``).
- ``or-bench-hard-1k``: harder over-refusal cases (score ``1.0``).
- ``or-bench-toxic``: genuinely toxic prompts used as negatives (score
  ``0.0``).

This module is offline: populate ``data/orbench/`` with
``scripts/download_orbench.py``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from infl_ens.data.benchmarks.base import BenchmarkSplit


PathLike = Union[str, Path]

#: Supported OR-Bench CSV configs written by the download script.
ORBENCH_CONFIGS: tuple[str, ...] = (
    "or-bench-80k",
    "or-bench-hard-1k",
    "or-bench-toxic",
)

#: Positive over-refusal configs (benign prompts that look harmful).
_ORBENCH_POSITIVE_CONFIGS: frozenset[str] = frozenset(
    {"or-bench-80k", "or-bench-hard-1k"},
)

#: Negative configs (genuinely toxic prompts).
_ORBENCH_NEGATIVE_CONFIGS: frozenset[str] = frozenset({"or-bench-toxic"})


def _iter_csv(path: Path) -> Iterable[dict[str, str]]:
    """Yield rows from an OR-Bench CSV file.

    :param path: Path to a ``.csv`` file.
    :type path: pathlib.Path
    :returns: Iterator of row dictionaries keyed by column name.
    :rtype: Iterable[dict]
    """
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def _resolve_files(
    path: Path,
    configs: Optional[tuple[str, ...]],
) -> list[tuple[Path, float]]:
    """Resolve CSV files and their axis scores under ``path``.

    :param path: Directory containing OR-Bench CSV files or a single CSV.
    :type path: pathlib.Path
    :param configs: Optional subset of :data:`ORBENCH_CONFIGS` to load.
    :type configs: tuple[str, ...] | None
    :returns: List of ``(file_path, score)`` pairs.
    :rtype: list[tuple[pathlib.Path, float]]
    :raises FileNotFoundError: If no matching CSV files are found.
    """
    if path.is_file():
        name = path.stem
        if name in _ORBENCH_POSITIVE_CONFIGS:
            return [(path, 1.0)]
        if name in _ORBENCH_NEGATIVE_CONFIGS:
            return [(path, 0.0)]
        raise ValueError(
            f"unknown OR-Bench file {path.name!r}; expected one of "
            f"{ORBENCH_CONFIGS}"
        )

    selected = configs or ORBENCH_CONFIGS
    unknown = set(selected) - set(ORBENCH_CONFIGS)
    if unknown:
        raise ValueError(
            f"unknown OR-Bench configs {sorted(unknown)!r}; "
            f"expected subset of {ORBENCH_CONFIGS}"
        )

    files: list[tuple[Path, float]] = []
    for cfg in selected:
        candidate = path / f"{cfg}.csv"
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        score = 1.0 if cfg in _ORBENCH_POSITIVE_CONFIGS else 0.0
        files.append((candidate, score))
    return files


def load_orbench(
    path: PathLike,
    *,
    configs: Optional[tuple[str, ...]] = None,
    max_records: Optional[int] = None,
) -> BenchmarkSplit:
    """Load OR-Bench from CSV files into an over-refusal axis split.

    :param path: Directory containing ``or-bench-*.csv`` files or a single
        CSV file.
    :type path: str | pathlib.Path
    :param configs: Optional subset of :data:`ORBENCH_CONFIGS`. Defaults
        to all three configs (80k positives + toxic negatives).
    :type configs: tuple[str, ...] | None
    :param max_records: Optional cap on total records returned.
    :type max_records: int | None
    :returns: A :class:`BenchmarkSplit` with ``axis_name='overrefusal'``.
    :rtype: BenchmarkSplit
    :raises FileNotFoundError: If ``path`` does not exist.
    :raises ValueError: If no parsable records are found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    prompts: list[str] = []
    scores: list[float] = []
    loaded_configs: list[str] = []

    file_entries = _resolve_files(path, configs)
    per_file_cap: Optional[int] = None
    if max_records is not None:
        per_file_cap = max(1, (max_records + len(file_entries) - 1) // len(file_entries))

    for file_path, score in file_entries:
        loaded_configs.append(file_path.stem)
        n_this_file = 0
        for row in _iter_csv(file_path):
            prompt = row.get("prompt") or row.get("text")
            if not prompt:
                continue
            prompts.append(str(prompt))
            scores.append(score)
            n_this_file += 1
            if per_file_cap is not None and n_this_file >= per_file_cap:
                break
        if max_records is not None and len(prompts) >= max_records:
            break

    if not prompts:
        raise ValueError(f"no parsable records found under {path}")

    return BenchmarkSplit(
        name="orbench",
        prompts=prompts,
        scores=np.asarray(scores, dtype=float),
        axis_name="overrefusal",
        metadata={
            "source": "orbench-llm/or-bench",
            "license": "cc-by-4.0",
            "score_target": "prompt",
            "configs": tuple(loaded_configs),
        },
    )
