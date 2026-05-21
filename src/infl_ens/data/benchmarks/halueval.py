"""HaluEval loader and hallucination-axis scoring.

HaluEval (Li et al., 2023) is a large hallucination evaluation benchmark
covering QA, dialogue, and summarization. It is released by RUCAIBox at
``https://github.com/RUCAIBox/HaluEval``. Each task ships pairs of
``right_answer`` / ``hallucinated_answer`` responses to the same prompt.

We expose each pair as two records along the ``hallucination`` axis:

- ``(prompt, right_answer)`` with score ``0.0``,
- ``(prompt, hallucinated_answer)`` with score ``1.0``.

This doubles the effective dataset size and gives a clean signed axis for
the trait space. As with the BeaverTails loader, this module is offline:
populate ``data/halueval/`` with ``scripts/download_halueval.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from infl_ens.data.benchmarks.base import BenchmarkSplit


PathLike = Union[str, Path]


#: HaluEval task identifiers and the corresponding JSON filename used by
#: the upstream release.
HALUEVAL_TASKS: dict[str, str] = {
    "qa": "qa_data.json",
    "dialogue": "dialogue_data.json",
    "summarization": "summarization_data.json",
    "general": "general_data.json",
}


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    """Yield rows from either ``.json`` (list of dicts) or ``.jsonl`` files.

    HaluEval ships some splits as JSON-Lines and others as a single JSON
    list, so we sniff the first non-whitespace character.

    :param path: Path to a ``.json`` or ``.jsonl`` file.
    :type path: pathlib.Path
    :returns: Iterator of parsed JSON dicts.
    :rtype: Iterable[dict]
    """
    with path.open("r", encoding="utf-8") as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":
            data = json.load(fh)
            for row in data:
                yield row
        else:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _extract_pair(
    row: dict[str, Any],
    task: str,
) -> Optional[tuple[str, str, str]]:
    """Extract ``(prompt, right_answer, hallucinated_answer)`` from one row.

    The key names vary across HaluEval tasks; this function consolidates
    them and returns ``None`` for rows missing required fields.

    :param row: Raw record.
    :type row: dict
    :param task: Task identifier (``'qa'``, ``'dialogue'``, etc.).
    :type task: str
    :returns: Tuple of strings, or ``None`` if the row is unusable.
    :rtype: tuple[str, str, str] | None
    """
    if task == "qa":
        prompt = row.get("question")
        right = row.get("right_answer")
        wrong = row.get("hallucinated_answer")
    elif task == "dialogue":
        prompt = row.get("dialogue_history") or row.get("knowledge")
        right = row.get("right_response")
        wrong = row.get("hallucinated_response")
    elif task == "summarization":
        prompt = row.get("document")
        right = row.get("right_summary")
        wrong = row.get("hallucinated_summary")
    elif task == "general":
        prompt = row.get("user_query") or row.get("question")
        resp = row.get("chatgpt_response") or row.get("response")
        label = row.get("hallucination")
        if not prompt or resp is None or label is None:
            return None
        # General split has a single response with a binary label.
        return (
            str(prompt),
            str(resp) if str(label).lower() in ("no", "0", "false") else "",
            str(resp) if str(label).lower() in ("yes", "1", "true") else "",
        )
    else:  # pragma: no cover - guarded by HALUEVAL_TASKS
        raise ValueError(f"unknown HaluEval task {task!r}")

    if not prompt or not right or not wrong:
        return None
    return str(prompt), str(right), str(wrong)


def load_halueval(
    path: PathLike,
    *,
    tasks: Optional[Iterable[str]] = None,
    max_records: Optional[int] = None,
) -> BenchmarkSplit:
    """Load HaluEval from a single file or a directory of task files.

    :param path: Path to a single HaluEval task JSON/JSONL file, or to a
        directory containing some subset of
        ``{qa,dialogue,summarization,general}_data.json``.
    :type path: str | pathlib.Path
    :param tasks: Subset of task identifiers to include. Defaults to all
        tasks whose files are present at ``path``.
    :type tasks: Iterable[str] | None
    :param max_records: Optional cap on the number of records (counting
        both right and hallucinated entries).
    :type max_records: int | None
    :returns: A :class:`BenchmarkSplit` with
        ``axis_name='hallucination'``.
    :rtype: BenchmarkSplit
    :raises FileNotFoundError: If ``path`` does not exist.
    :raises ValueError: If no records parse.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        present: list[tuple[str, Path]] = []
        for task, fname in HALUEVAL_TASKS.items():
            f = path / fname
            if f.exists():
                present.append((task, f))
        if not present:
            raise FileNotFoundError(f"no HaluEval task files in {path}")
        if tasks is not None:
            want = set(tasks)
            present = [(t, p) for t, p in present if t in want]
    else:
        # Single-file mode: caller must specify which task it is.
        task = next(iter(tasks)) if tasks else "qa"
        present = [(task, path)]

    prompts: list[str] = []
    responses: list[str] = []
    scores: list[float] = []
    for task, f in present:
        for row in _iter_records(f):
            triple = _extract_pair(row, task)
            if triple is None:
                continue
            prompt, right, wrong = triple
            if right:
                prompts.append(prompt)
                responses.append(right)
                scores.append(0.0)
            if wrong:
                prompts.append(prompt)
                responses.append(wrong)
                scores.append(1.0)
            if max_records is not None and len(prompts) >= max_records:
                break
        if max_records is not None and len(prompts) >= max_records:
            break

    if not prompts:
        raise ValueError(f"no parsable records found under {path}")

    return BenchmarkSplit(
        name="halueval",
        prompts=prompts,
        responses=responses,
        scores=np.asarray(scores, dtype=float),
        axis_name="hallucination",
        metadata={
            "source": "RUCAIBox/HaluEval",
            "tasks": [t for t, _ in present],
        },
    )
