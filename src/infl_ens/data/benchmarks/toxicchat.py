"""ToxicChat loader and jailbreak-axis scoring.

ToxicChat (Lin et al., 2023) is a real-world content-moderation benchmark
released by LMSYS at ``lmsys/toxic-chat``. It collects anonymised user
queries from the Vicuna online demo and ships per-prompt **integer** labels
for two intrinsic properties:

- ``toxicity`` (0/1): whether the user input is toxic.
- ``jailbreaking`` (0/1): whether the user input is a deliberate attempt to
  trick the model into producing disallowed content while looking benign.

This module exposes a third trait-space axis that is *prompt-intrinsic* and
deliberately independent of the existing axes:

- BeaverTails scores whether a **response** is harmful (``harm``).
- HaluEval scores whether a **response** hallucinates (``hallucination``).
- ToxicChat scores whether the **prompt itself** is adversarial
  (``jailbreak``), regardless of any response.

Because the label is supplied by the dataset (no model judgement is
required), the axis is "easy to measure" in the same sense as HaluEval:
the score is read off the record, not computed by inference.

The on-disk format is CSV (version ``toxicchat0124``) with columns
``conv_id, user_input, model_output, human_annotation, toxicity,
jailbreaking, openai_moderation``. This module is offline: it reads from
disk and does no network IO. Use ``scripts/download_toxicchat.py`` to
populate ``data/toxicchat/``.

.. note::

   ToxicChat is released under ``cc-by-nc-4.0`` (non-commercial). This is
   recorded in :attr:`BenchmarkSplit.metadata` under ``license`` so
   downstream reporting can surface it.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from infl_ens.data.benchmarks.base import BenchmarkSplit


PathLike = Union[str, Path]


#: Scoring modes for the jailbreak axis. ``jailbreaking`` uses the dataset's
#: ``jailbreaking`` label; ``toxicity`` uses the ``toxicity`` label; ``either``
#: scores ``1.0`` when *either* label is positive (a broader "adversarial
#: prompt" axis). The default, ``jailbreaking``, is the most independent of
#: the harm/hallucination axes since it captures *intent of the prompt*
#: rather than *harmfulness of content*.
TOXICCHAT_SCORE_MODES: tuple[str, ...] = ("jailbreaking", "toxicity", "either")


def _truthy_int(value: Any) -> int:
    """Coerce a CSV cell to a 0/1 integer label.

    ToxicChat stores labels as integers, but CSV round-trips them as
    strings (``"0"`` / ``"1"``). Some mirrors export booleans. This accepts
    all three and returns ``0`` or ``1``.

    :param value: Raw cell value.
    :type value: Any
    :returns: ``1`` if the value denotes a positive label, else ``0``.
    :rtype: int
    """
    s = str(value).strip().lower()
    return 1 if s in ("1", "true", "yes") else 0


def _iter_csv(path: Path) -> Iterable[dict[str, Any]]:
    """Yield rows from a ToxicChat CSV file as dicts.

    :param path: Path to a ``.csv`` file.
    :type path: pathlib.Path
    :returns: Iterator of row dictionaries keyed by column name.
    :rtype: Iterable[dict]
    """
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def _coerce_record(
    row: dict[str, Any],
    score_mode: str,
) -> Optional[tuple[str, str, float]]:
    """Project one ToxicChat record to ``(prompt, response, jailbreak_score)``.

    :param row: Raw record dictionary (one CSV row).
    :type row: dict
    :param score_mode: One of :data:`TOXICCHAT_SCORE_MODES`.
    :type score_mode: str
    :returns: ``(prompt, response, score)`` if the row parses, else ``None``.
    :rtype: tuple[str, str, float] | None
    """
    prompt = row.get("user_input")
    if not prompt:
        return None
    response = row.get("model_output") or ""
    tox = _truthy_int(row.get("toxicity", 0))
    jail = _truthy_int(row.get("jailbreaking", 0))
    if score_mode == "jailbreaking":
        score = float(jail)
    elif score_mode == "toxicity":
        score = float(tox)
    elif score_mode == "either":
        score = float(tox or jail)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(
            f"unknown score_mode {score_mode!r}; "
            f"expected one of {TOXICCHAT_SCORE_MODES}"
        )
    return str(prompt), str(response), score


def load_toxicchat(
    path: PathLike,
    *,
    score_mode: str = "jailbreaking",
    human_annotated_only: bool = False,
    max_records: Optional[int] = None,
) -> BenchmarkSplit:
    """Load ToxicChat from a CSV file or a directory of CSV files.

    :param path: Path to a ``.csv`` file or to a directory containing one
        or more ToxicChat CSV files (e.g. ``toxic-chat_annotation_train.csv``
        / ``toxic-chat_annotation_test.csv``). A directory with several CSV
        files is concatenated in lexicographic order.
    :type path: str | pathlib.Path
    :param score_mode: Which intrinsic label defines the axis. One of
        :data:`TOXICCHAT_SCORE_MODES`. Defaults to ``'jailbreaking'``, the
        axis most independent of the harm/hallucination axes.
    :type score_mode: str
    :param human_annotated_only: If ``True``, keep only rows whose
        ``human_annotation`` column is truthy (drops the auto-filtered
        non-toxic majority and yields a denser, higher-confidence split).
    :type human_annotated_only: bool
    :param max_records: Optional cap on the number of records to return
        (post-filtering). Useful for fast smoke tests.
    :type max_records: int | None
    :returns: A :class:`BenchmarkSplit` with ``axis_name='jailbreak'``.
    :rtype: BenchmarkSplit
    :raises FileNotFoundError: If ``path`` does not exist or no CSV files
        are found under a directory.
    :raises ValueError: If ``score_mode`` is unknown or no records parse.
    """
    if score_mode not in TOXICCHAT_SCORE_MODES:
        raise ValueError(
            f"unknown score_mode {score_mode!r}; "
            f"expected one of {TOXICCHAT_SCORE_MODES}"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        files = sorted(p for p in path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no .csv files in {path}")
    else:
        files = [path]

    prompts: list[str] = []
    responses: list[str] = []
    scores: list[float] = []
    for f in files:
        for row in _iter_csv(f):
            if human_annotated_only and not _truthy_int(
                row.get("human_annotation", 0)
            ):
                continue
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
        name="toxicchat",
        prompts=prompts,
        responses=responses,
        scores=np.asarray(scores, dtype=float),
        axis_name="jailbreak",
        metadata={
            "source": "lmsys/toxic-chat",
            "version": "toxicchat0124",
            "license": "cc-by-nc-4.0",
            "score_mode": score_mode,
            "score_target": "prompt",
            "human_annotated_only": human_annotated_only,
            "n_files": len(files),
        },
    )
