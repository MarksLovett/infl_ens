"""Offline loader for XSTest safe and unsafe contrast prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from infl_ens.data.behavioral.base import BehavioralCase, BehavioralSuite, stable_case_id
from infl_ens.data.behavioral.common import coerce_messages, first_text, read_rows, source_files


def _is_safe(row: Mapping[str, Any]) -> bool:
    label = first_text(row, ("label", "safety_label", "gold_label"))
    if label is not None:
        return label.casefold() in {"safe", "benign", "0", "false"}
    prompt_type = first_text(row, ("type",)) or ""
    return not prompt_type.casefold().startswith("contrast_")


def load_xstest(entry: Mapping[str, Any]) -> BehavioralSuite:
    """Load XSTest prompts and their safe/unsafe contrast labels.

    :param entry: Suite config with ``path`` and optional ``version``.
    :type entry: Mapping[str, Any]
    :returns: Pinned XSTest suite.
    :rtype: BehavioralSuite
    """
    rows = read_rows(Path(str(entry["path"])))
    cases: list[BehavioralCase] = []
    for index, row in enumerate(rows):
        messages = coerce_messages(row, prompt_keys=("prompt", "instruction", "text"))
        if not messages:
            continue
        safe = _is_safe(row)
        raw_id = first_text(row, ("id", "prompt_id"))
        case_id = f"xstest:{raw_id}" if raw_id else stable_case_id("xstest", messages)
        cases.append(BehavioralCase(
            suite="xstest",
            case_id=case_id,
            messages=messages,
            category=first_text(row, ("type", "category", "focus")) or "all",
            metadata={"row": index, "source_file": row.get("_source_file")},
            scoring_payload={"safe": safe, "expected_behavior": "answer" if safe else "refuse"},
        ))
    return BehavioralSuite(
        name="xstest",
        version=str(entry.get("version", "pinned-local")),
        source="paul-rottger/exaggerated-safety",
        cases=tuple(cases),
        metadata={"source_files": source_files(rows)},
    ).take(entry.get("max_cases"))


__all__ = ["load_xstest"]
