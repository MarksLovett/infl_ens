"""AI4Privacy PII-masking loader and privacy-density axis scoring.

AI4Privacy ``pii-masking-200k`` (Ai4Privacy, 2023) is a large open dataset
of natural-language texts annotated with personally identifiable
information (PII) spans. It is released on HuggingFace as
``ai4privacy/pii-masking-200k`` under ``apache-2.0`` and was built to train
PII-detection / redaction models, so every record ships:

- the natural ``source_text`` (a.k.a. ``unmasked_text``),
- a PII-free ``target_text`` (a.k.a. ``masked_text``) with placeholders,
- a ``privacy_mask``: a list of ``{value, start, end, label}`` spans
  marking exactly where the private information sits in ``source_text``.

This module exposes a third trait-space axis that is *prompt-intrinsic* and
chosen to be embedding-separable from the existing axes. The prompts read
like emails, forms, records, and business correspondence — structurally
unlike BeaverTails' harmful requests or HaluEval's factual QA — which is
the property that makes the axis a candidate for genuine independence in
the MiniLM embedding space (verify with the independence diagnostic before
promoting it to a Nash axis).

The axis is **PII density**: the fraction of the prompt occupied by private
information, computed directly from the ``privacy_mask`` span lengths. No
model judgement is required — the score is read off the labels, the same
"easy to measure" property HaluEval and BeaverTails enjoy:

.. math::

    \\mathrm{density}(x) = \\min\\!\\left(1,\\;
        \\frac{\\sum_{s \\in \\text{mask}} (\\text{end}_s - \\text{start}_s)}
             {\\max(1, |\\text{source\\_text}|)}\\right)

Two scoring modes are provided (see :data:`PII_SCORE_MODES`):

- ``density`` (default): continuous character-coverage fraction in
  ``[0, 1]`` — a graded axis with mass across the interval.
- ``binary``: ``1.0`` if the record contains *any* PII span, else ``0.0``.
  Because every ``pii-masking-200k`` record contains PII, the binary mode is
  only meaningful when mixed with PII-free negatives from another source.

This module is offline: it reads from disk and does no network IO. Use
``scripts/download_ai4privacy.py`` to populate ``data/ai4privacy/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from infl_ens.data.benchmarks.base import BenchmarkSplit


PathLike = Union[str, Path]


#: Supported scoring modes for the privacy axis. ``density`` is the graded
#: character-coverage fraction; ``binary`` flags any-PII presence.
PII_SCORE_MODES: tuple[str, ...] = ("density", "binary")

#: Substring used to select English-language files when ``path`` is a
#: directory holding the per-language JSONL shards
#: (``english_pii_43k.jsonl``, ``french_pii_62k.jsonl``, ...).
_ENGLISH_FILE_MARKER: str = "english"


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


def _parse_privacy_mask(raw: Any) -> list[dict[str, Any]]:
    """Normalise a ``privacy_mask`` field to a list of span dicts.

    The field is sometimes a JSON-encoded string (Isotonic mirror) and
    sometimes a native list (HuggingFace ``datasets`` export). This accepts
    both and returns ``[]`` for anything unparseable.

    :param raw: Raw ``privacy_mask`` value.
    :type raw: Any
    :returns: List of span dicts with at least ``start`` / ``end`` keys.
    :rtype: list[dict]
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, dict)]
    return []


def _pii_char_coverage(
    text: str,
    spans: list[dict[str, Any]],
) -> int:
    """Total number of characters in ``text`` covered by PII spans.

    Overlapping spans are merged so coverage never exceeds ``len(text)``.

    :param text: The ``source_text`` whose characters may be covered.
    :type text: str
    :param spans: Parsed ``privacy_mask`` spans with ``start`` / ``end``.
    :type spans: list[dict]
    :returns: Number of covered characters (after merging overlaps).
    :rtype: int
    """
    intervals: list[tuple[int, int]] = []
    n = len(text)
    for s in spans:
        try:
            start = int(s["start"])
            end = int(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        start = max(0, min(start, n))
        end = max(0, min(end, n))
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return 0
    intervals.sort()
    covered = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            covered += cur_end - cur_start
            cur_start, cur_end = start, end
    covered += cur_end - cur_start
    return covered


def _coerce_record(
    row: dict[str, Any],
    score_mode: str,
) -> Optional[tuple[str, str, float]]:
    """Project one AI4Privacy record to ``(prompt, response, score)``.

    The unmasked natural text is the prompt; the masked text is used as the
    response so the split carries a (prompt, response) pair like the other
    loaders. Accepts both the native (``source_text`` / ``target_text``) and
    Isotonic-mirror (``unmasked_text`` / ``masked_text``) key names.

    :param row: Raw record dictionary.
    :type row: dict
    :param score_mode: One of :data:`PII_SCORE_MODES`.
    :type score_mode: str
    :returns: ``(prompt, response, score)`` if parseable, else ``None``.
    :rtype: tuple[str, str, float] | None
    """
    prompt = row.get("source_text") or row.get("unmasked_text")
    response = row.get("target_text") or row.get("masked_text") or ""
    if not prompt:
        return None
    spans = _parse_privacy_mask(row.get("privacy_mask"))
    if score_mode == "binary":
        score = 1.0 if spans else 0.0
    elif score_mode == "density":
        covered = _pii_char_coverage(str(prompt), spans)
        score = min(1.0, covered / max(1, len(str(prompt))))
    else:  # pragma: no cover - guarded by caller
        raise ValueError(
            f"unknown score_mode {score_mode!r}; "
            f"expected one of {PII_SCORE_MODES}"
        )
    return str(prompt), str(response), float(score)


def load_ai4privacy(
    path: PathLike,
    *,
    score_mode: str = "density",
    english_only: bool = True,
    max_records: Optional[int] = None,
) -> BenchmarkSplit:
    """Load AI4Privacy PII-masking data from a JSONL file or directory.

    :param path: Path to a ``.jsonl`` file, or to a directory containing the
        per-language shards (``english_pii_43k.jsonl`` etc.). A directory is
        concatenated in lexicographic order.
    :type path: str | pathlib.Path
    :param score_mode: Axis scoring mode. One of :data:`PII_SCORE_MODES`.
        ``density`` (default) is the graded character-coverage fraction.
    :type score_mode: str
    :param english_only: When ``path`` is a directory, restrict to files
        whose name contains ``"english"``. Ignored for single-file paths.
    :type english_only: bool
    :param max_records: Optional cap on the number of records to return.
    :type max_records: int | None
    :returns: A :class:`BenchmarkSplit` with ``axis_name='privacy'``.
    :rtype: BenchmarkSplit
    :raises FileNotFoundError: If ``path`` does not exist or no matching
        JSONL files are found under a directory.
    :raises ValueError: If ``score_mode`` is unknown or no records parse.
    """
    if score_mode not in PII_SCORE_MODES:
        raise ValueError(
            f"unknown score_mode {score_mode!r}; "
            f"expected one of {PII_SCORE_MODES}"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        files = sorted(p for p in path.glob("*.jsonl"))
        if english_only:
            files = [
                f for f in files if _ENGLISH_FILE_MARKER in f.name.lower()
            ]
        if not files:
            raise FileNotFoundError(
                f"no matching .jsonl files in {path} "
                f"(english_only={english_only})"
            )
    else:
        files = [path]

    prompts: list[str] = []
    responses: list[str] = []
    scores: list[float] = []
    for f in files:
        for row in _iter_jsonl(f):
            rec = _coerce_record(row, score_mode)
            if rec is None:
                continue
            p, r, s = rec
            prompts.append(p)
            responses.append(r)
            scores.append(s)
            if max_records is not None and len(prompts) >= max_records:
                break
        if max_records is not None and len(prompts) >= max_records:
            break

    if not prompts:
        raise ValueError(f"no parsable records found under {path}")

    return BenchmarkSplit(
        name="ai4privacy",
        prompts=prompts,
        responses=responses,
        scores=np.asarray(scores, dtype=float),
        axis_name="privacy",
        metadata={
            "source": "ai4privacy/pii-masking-200k",
            "license": "apache-2.0",
            "score_mode": score_mode,
            "score_target": "prompt",
            "english_only": english_only,
            "n_files": len(files),
        },
    )
