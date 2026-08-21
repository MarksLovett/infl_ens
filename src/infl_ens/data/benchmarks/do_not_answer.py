"""Do-Not-Answer loader and policy-violation axis scoring.

Do-Not-Answer (Wang et al., 2024) is an open-source dataset of 939
instructions that responsible language models should refuse. It is released
on HuggingFace as ``LibrAI/do-not-answer`` (``apache-2.0``) and organises
prompts across five risk areas and twelve harm types.

Because every Do-Not-Answer prompt is refusal-worthy, the loader mixes in
benign negatives (written by ``scripts/download_do_not_answer.py`` as
``benign_negatives.jsonl``) so the Fisher axis has both classes.

This module is offline: populate ``data/do_not_answer/`` with
``scripts/download_do_not_answer.py``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from infl_ens.data.benchmarks.base import BenchmarkSplit


PathLike = Union[str, Path]

#: Default benign-negative filename written by the download script.
DEFAULT_BENIGN_FILE: str = "benign_negatives.jsonl"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield rows from a JSONL file.

    :param path: Path to a ``.jsonl`` file.
    :type path: pathlib.Path
    :returns: Iterator of parsed JSON dicts.
    :rtype: Iterable[dict]
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


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


def _extract_prompt(row: dict[str, Any]) -> Optional[str]:
    """Extract a prompt string from a Do-Not-Answer or benign record.

    :param row: Raw record dictionary.
    :type row: dict
    :returns: Prompt text if present, else ``None``.
    :rtype: str | None
    """
    for key in ("question", "instruction", "prompt", "text"):
        value = row.get(key)
        if value:
            return str(value)
    return None


def _load_positive_records(path: Path) -> list[tuple[str, float]]:
    """Load Do-Not-Answer refusal prompts scored ``1.0``.

    :param path: Directory or file containing Do-Not-Answer records.
    :type path: pathlib.Path
    :returns: List of ``(prompt, score)`` pairs.
    :rtype: list[tuple[str, float]]
    """
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            list(path.glob("do_not_answer*.jsonl"))
            + list(path.glob("do_not_answer*.csv"))
            + list(path.glob("*.jsonl"))
        )
        files = [
            f for f in files
            if f.name != DEFAULT_BENIGN_FILE and "benign" not in f.stem
        ]
        if not files:
            raise FileNotFoundError(f"no Do-Not-Answer files in {path}")

    out: list[tuple[str, float]] = []
    for file_path in files:
        if file_path.name == DEFAULT_BENIGN_FILE:
            continue
        iterator: Iterable[dict[str, Any]]
        if file_path.suffix.lower() == ".jsonl":
            iterator = _iter_jsonl(file_path)
        else:
            iterator = _iter_csv(file_path)
        for row in iterator:
            prompt = _extract_prompt(row)
            if prompt is None:
                continue
            out.append((prompt, 1.0))
    return out


def _load_benign_records(path: Path) -> list[tuple[str, float]]:
    """Load benign negative prompts scored ``0.0``.

    :param path: File or directory containing ``benign_negatives.jsonl``.
    :type path: pathlib.Path
    :returns: List of ``(prompt, score)`` pairs.
    :rtype: list[tuple[str, float]]
    :raises FileNotFoundError: If no benign-negative file is found.
    """
    if path.is_file():
        benign_path = path
    else:
        benign_path = path / DEFAULT_BENIGN_FILE
    if not benign_path.exists():
        raise FileNotFoundError(
            f"benign negatives not found at {benign_path}; run "
            "scripts/download_do_not_answer.py first"
        )

    out: list[tuple[str, float]] = []
    iterator: Iterable[dict[str, Any]]
    if benign_path.suffix.lower() == ".jsonl":
        iterator = _iter_jsonl(benign_path)
    else:
        iterator = _iter_csv(benign_path)
    for row in iterator:
        prompt = _extract_prompt(row)
        if prompt is None:
            continue
        out.append((prompt, 0.0))
    return out


def load_do_not_answer(
    path: PathLike,
    *,
    benign_path: Optional[PathLike] = None,
    include_benign: bool = True,
    max_records: Optional[int] = None,
) -> BenchmarkSplit:
    """Load Do-Not-Answer plus optional benign negatives.

    :param path: Directory or file containing Do-Not-Answer records.
    :type path: str | pathlib.Path
    :param benign_path: Optional explicit path to benign negatives. When
        ``None`` and ``include_benign`` is ``True``, looks for
        ``benign_negatives.jsonl`` beside the Do-Not-Answer files.
    :type benign_path: str | pathlib.Path | None
    :param include_benign: If ``True``, append benign negatives so the axis
        has both classes.
    :type include_benign: bool
    :param max_records: Optional cap on total records returned.
    :type max_records: int | None
    :returns: A :class:`BenchmarkSplit` with
        ``axis_name='policy_violation'``.
    :rtype: BenchmarkSplit
    :raises FileNotFoundError: If required files are missing.
    :raises ValueError: If no parsable records are found.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)

    records = _load_positive_records(root)
    if include_benign:
        benign_root = Path(benign_path) if benign_path is not None else root
        records.extend(_load_benign_records(benign_root))

    if not records:
        raise ValueError(f"no parsable records found under {root}")

    if max_records is not None and len(records) > max_records:
        records = records[:max_records]

    prompts = [p for p, _ in records]
    scores = [s for _, s in records]

    return BenchmarkSplit(
        name="do_not_answer",
        prompts=prompts,
        scores=np.asarray(scores, dtype=float),
        axis_name="policy_violation",
        metadata={
            "source": "LibrAI/do-not-answer",
            "license": "apache-2.0",
            "score_target": "prompt",
            "include_benign": include_benign,
            "n_positive": int(sum(s >= 0.5 for s in scores)),
            "n_negative": int(sum(s < 0.5 for s in scores)),
        },
    )
