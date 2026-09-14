"""Offline loader for TruthfulQA generation records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from infl_ens.data.behavioral.base import BehavioralCase, BehavioralSuite, stable_case_id
from infl_ens.data.behavioral.common import (
    coerce_messages,
    first_text,
    read_rows,
    source_files,
    split_values,
)


def load_truthfulqa(entry: Mapping[str, Any]) -> BehavioralSuite:
    """Load TruthfulQA generation questions and answer references.

    :param entry: Suite config with ``path`` and optional ``version``.
    :type entry: Mapping[str, Any]
    :returns: Pinned TruthfulQA suite.
    :rtype: BehavioralSuite
    """
    rows = read_rows(Path(str(entry["path"])))
    cases: list[BehavioralCase] = []
    for index, row in enumerate(rows):
        messages = coerce_messages(row, prompt_keys=("Question", "question", "prompt"))
        if not messages:
            continue
        raw_id = first_text(row, ("id", "question_id"))
        case_id = f"truthfulqa:{raw_id}" if raw_id else stable_case_id("truthfulqa", messages)
        best = first_text(row, ("Best Answer", "best_answer"))
        correct = split_values(row.get("Correct Answers", row.get("correct_answers")))
        incorrect = split_values(row.get("Incorrect Answers", row.get("incorrect_answers")))
        cases.append(BehavioralCase(
            suite="truthfulqa",
            case_id=case_id,
            messages=messages,
            category=first_text(row, ("Category", "category")) or "all",
            metadata={"row": index, "source_file": row.get("_source_file")},
            scoring_payload={
                "best_answer": best,
                "correct_answers": correct,
                "incorrect_answers": incorrect,
            },
        ))
    return BehavioralSuite(
        name="truthfulqa",
        version=str(entry.get("version", "pinned-local")),
        source="sylinrl/TruthfulQA",
        cases=tuple(cases),
        metadata={"source_files": source_files(rows)},
    ).take(entry.get("max_cases"))


__all__ = ["load_truthfulqa"]
