"""Prompt-injection loader and injection-axis scoring.

The Protect AI validation corpus ``protectai/prompt-injection-validation``
aggregates several public prompt-injection benchmarks into a single binary
labelled set: ``0`` for benign requests and ``1`` for prompt injections or
jailbreak-style attacks embedded in user text.

This axis is *prompt-intrinsic* and complements ToxicChat jailbreak intent:
injections hide adversarial instructions inside otherwise benign-looking
content, whereas jailbreaks are direct user attempts to bypass safeguards.

This module is offline: populate ``data/prompt_injection/`` with
``scripts/download_prompt_injection.py`` (default:
``neuralchemy/prompt-injection-Threat-Matrix`` binary config).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from infl_ens.data.benchmarks.base import BenchmarkSplit


PathLike = Union[str, Path]


def _truthy_label(value: Any) -> float:
    """Coerce a dataset label to ``0.0`` or ``1.0``.

    :param value: Raw label cell.
    :type value: Any
    :returns: ``1.0`` for positive injection labels, else ``0.0``.
    :rtype: float
    """
    if isinstance(value, (int, float)):
        return 1.0 if int(value) == 1 else 0.0
    s = str(value).strip().lower()
    return 1.0 if s in ("1", "true", "yes", "injection") else 0.0


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


def _coerce_record(row: dict[str, Any]) -> Optional[tuple[str, float]]:
    """Project one record to ``(prompt, injection_score)``.

    :param row: Raw record dictionary.
    :type row: dict
    :returns: ``(prompt, score)`` if the row parses, else ``None``.
    :rtype: tuple[str, float] | None
    """
    prompt = row.get("text") or row.get("prompt")
    if not prompt:
        return None
    label = row.get("label")
    if label is None:
        label = row.get("binary_label")
    if label is None:
        label = row.get("injection")
    if label is None:
        return None
    if isinstance(label, str):
        positive = label.strip().lower() in (
            "1", "true", "yes", "injection", "malicious", "jailbreak",
        )
        return str(prompt), 1.0 if positive else 0.0
    return str(prompt), _truthy_label(label)


def load_prompt_injection(
    path: PathLike,
    *,
    max_records: Optional[int] = None,
) -> BenchmarkSplit:
    """Load prompt-injection records from CSV or JSONL files.

    :param path: Path to a ``.csv`` / ``.jsonl`` file or a directory
        containing one or more such files (concatenated lexicographically).
    :type path: str | pathlib.Path
    :param max_records: Optional cap on the number of records returned.
    :type max_records: int | None
    :returns: A :class:`BenchmarkSplit` with ``axis_name='injection'``.
    :rtype: BenchmarkSplit
    :raises FileNotFoundError: If ``path`` does not exist or no data files
        are found under a directory.
    :raises ValueError: If no parsable records are found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        files = sorted(
            list(path.glob("*.csv")) + list(path.glob("*.jsonl")),
        )
        if not files:
            raise FileNotFoundError(f"no .csv or .jsonl files in {path}")
    else:
        files = [path]

    prompts: list[str] = []
    scores: list[float] = []
    for file_path in files:
        iterator: Iterable[dict[str, Any]]
        if file_path.suffix.lower() == ".jsonl":
            iterator = _iter_jsonl(file_path)
        else:
            iterator = _iter_csv(file_path)
        for row in iterator:
            rec = _coerce_record(row)
            if rec is None:
                continue
            prompt, score = rec
            prompts.append(prompt)
            scores.append(score)
            if max_records is not None and len(prompts) >= max_records:
                break
        if max_records is not None and len(prompts) >= max_records:
            break

    if not prompts:
        raise ValueError(f"no parsable records found under {path}")

    manifest_path = path / "manifest.json" if path.is_dir() else path.parent / "manifest.json"
    source = "prompt_injection"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source = str(manifest.get("source", source))
        except json.JSONDecodeError:
            pass

    return BenchmarkSplit(
        name="prompt_injection",
        prompts=prompts,
        scores=np.asarray(scores, dtype=float),
        axis_name="injection",
        metadata={
            "source": source,
            "score_target": "prompt",
            "n_files": len(files),
        },
    )
