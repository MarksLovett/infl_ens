"""BeaverTails loader and harm-axis scoring.

BeaverTails (Ji et al., 2023) is a large-scale safety preference dataset
released by PKU-Alignment under the name ``PKU-Alignment/BeaverTails``. Each
record contains a prompt, a response, and a 14-way harm-category label set
together with an aggregate ``is_safe`` flag. We collapse that to a single
``harm`` score in ``[0, 1]``: ``1.0`` when the response is unsafe along
*any* category, ``0.0`` otherwise.

This module is offline: it reads from disk and does no network IO. Use
``scripts/download_beavertails.py`` to populate ``data/beavertails/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from infl_ens.data.benchmarks.base import BenchmarkSplit


PathLike = Union[str, Path]


#: Categorical harm labels in BeaverTails. Listed here so callers can pass
#: ``categories=...`` to focus on a subset (e.g. only physical-harm types).
BEAVERTAILS_CATEGORIES: tuple[str, ...] = (
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "discrimination,stereotype,injustice",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "hate_speech,offensive_language",
    "misinformation_regarding_ethics,laws_and_safety",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "sexually_explicit,adult_content",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield rows from a JSON-Lines file.

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


def _coerce_record(
    row: dict[str, Any],
    categories: tuple[str, ...],
) -> Optional[tuple[str, str, float]]:
    """Project one BeaverTails record to ``(prompt, response, harm_score)``.

    BeaverTails has two on-disk schemas (the released JSONL files and the
    HuggingFace datasets version) that disagree slightly on key names. We
    accept both.

    :param row: Raw record dictionary.
    :type row: dict
    :param categories: Categories that count toward the harm score.
    :type categories: tuple[str, ...]
    :returns: ``(prompt, response, harm)`` if the row parses, else ``None``.
    :rtype: tuple[str, str, float] | None
    """
    prompt = row.get("prompt") or row.get("question")
    response = row.get("response") or row.get("answer")
    if not prompt or not response:
        return None
    cats = row.get("category")
    if isinstance(cats, dict):
        harm = float(any(bool(cats.get(c)) for c in categories))
    elif "is_safe" in row:
        harm = 0.0 if bool(row["is_safe"]) else 1.0
    else:
        return None
    return str(prompt), str(response), harm


def load_beavertails(
    path: PathLike,
    *,
    categories: Optional[Iterable[str]] = None,
    max_records: Optional[int] = None,
) -> BenchmarkSplit:
    """Load BeaverTails from a JSONL file or HuggingFace-cached directory.

    :param path: Path to a ``.jsonl`` file or to a directory containing
        ``train.jsonl`` / ``test.jsonl``. A directory with several JSONL
        files is concatenated in lexicographic order.
    :type path: str | pathlib.Path
    :param categories: Optional iterable of category names that should
        count toward the harm score. Defaults to all 14 categories.
    :type categories: Iterable[str] | None
    :param max_records: Optional cap on the number of records to return
        (post-filtering). Useful for fast smoke tests.
    :type max_records: int | None
    :returns: A :class:`BenchmarkSplit` with ``axis_name='harm'``.
    :rtype: BenchmarkSplit
    :raises FileNotFoundError: If ``path`` does not exist.
    :raises ValueError: If no records can be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        files = sorted(p for p in path.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"no .jsonl files in {path}")
    else:
        files = [path]

    cats = tuple(categories) if categories is not None else BEAVERTAILS_CATEGORIES

    prompts: list[str] = []
    responses: list[str] = []
    scores: list[float] = []
    for f in files:
        for row in _iter_jsonl(f):
            rec = _coerce_record(row, cats)
            if rec is None:
                continue
            p, r, h = rec
            prompts.append(p)
            responses.append(r)
            scores.append(h)
            if max_records is not None and len(prompts) >= max_records:
                break
        if max_records is not None and len(prompts) >= max_records:
            break

    if not prompts:
        raise ValueError(f"no parsable records found under {path}")

    return BenchmarkSplit(
        name="beavertails",
        prompts=prompts,
        responses=responses,
        scores=np.asarray(scores, dtype=float),
        axis_name="harm",
        metadata={
            "source": "PKU-Alignment/BeaverTails",
            "n_files": len(files),
            "categories": list(cats),
        },
    )
