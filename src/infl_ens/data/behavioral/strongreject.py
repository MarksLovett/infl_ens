"""Offline loader for StrongREJECT forbidden-prompt records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from infl_ens.data.behavioral.base import BehavioralCase, BehavioralSuite, stable_case_id
from infl_ens.data.behavioral.common import coerce_messages, first_text, read_rows, source_files


def load_strongreject(entry: Mapping[str, Any]) -> BehavioralSuite:
    """Load StrongREJECT data exported as JSON, JSONL or CSV.

    :param entry: Suite config with ``path`` and optional ``version``.
    :type entry: Mapping[str, Any]
    :returns: Pinned StrongREJECT suite.
    :rtype: BehavioralSuite
    """
    rows = read_rows(Path(str(entry["path"])))
    cases: list[BehavioralCase] = []
    for index, row in enumerate(rows):
        messages = coerce_messages(
            row, prompt_keys=("forbidden_prompt", "prompt", "goal", "instruction"),
        )
        if not messages:
            continue
        raw_id = first_text(row, ("id", "prompt_id", "source_id"))
        case_id = f"strongreject:{raw_id}" if raw_id else stable_case_id("strongreject", messages)
        cases.append(BehavioralCase(
            suite="strongreject",
            case_id=case_id,
            messages=messages,
            category=first_text(row, ("category", "harm_category")) or "all",
            metadata={"row": index, "source_file": row.get("_source_file")},
            scoring_payload={"forbidden_prompt": messages[-1].content},
        ))
    return BehavioralSuite(
        name="strongreject",
        version=str(entry.get("version", "pinned-local")),
        source="dsbowen/strong_reject",
        cases=tuple(cases),
        metadata={"source_files": source_files(rows)},
    ).take(entry.get("max_cases"))


__all__ = ["load_strongreject"]
