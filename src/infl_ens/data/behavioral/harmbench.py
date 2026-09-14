"""Offline loader for HarmBench standard-behavior records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from infl_ens.data.behavioral.base import BehavioralCase, BehavioralSuite, stable_case_id
from infl_ens.data.behavioral.common import coerce_messages, first_text, read_rows, source_files


def load_harmbench(entry: Mapping[str, Any]) -> BehavioralSuite:
    """Load HarmBench behavior CSV/JSON records.

    :param entry: Suite config with ``path`` and optional ``version``.
    :type entry: Mapping[str, Any]
    :returns: Pinned HarmBench suite.
    :rtype: BehavioralSuite
    """
    rows = read_rows(Path(str(entry["path"])))
    wanted_functional = {
        str(value).casefold() for value in entry.get("functional_categories", [])
    }
    cases: list[BehavioralCase] = []
    for index, row in enumerate(rows):
        functional = first_text(row, ("FunctionalCategory", "functional_category"))
        if wanted_functional and (
            functional is None or functional.casefold() not in wanted_functional
        ):
            continue
        messages = coerce_messages(row, prompt_keys=("Behavior", "behavior", "prompt", "goal"))
        if not messages:
            continue
        raw_id = first_text(row, ("BehaviorID", "behavior_id", "id"))
        case_id = f"harmbench:{raw_id}" if raw_id else stable_case_id("harmbench", messages)
        category = first_text(
            row, ("SemanticCategory", "semantic_category", "category", "FunctionalCategory"),
        ) or "all"
        cases.append(BehavioralCase(
            suite="harmbench",
            case_id=case_id,
            messages=messages,
            category=category,
            metadata={
                "row": index,
                "functional_category": functional,
                "source_file": row.get("_source_file"),
            },
            scoring_payload={
                "expected_behavior": "refuse",
                "forbidden_prompt": messages[-1].content,
            },
        ))
    return BehavioralSuite(
        name="harmbench",
        version=str(entry.get("version", "pinned-local")),
        source="centerforaisafety/HarmBench",
        cases=tuple(cases),
        metadata={"source_files": source_files(rows)},
    ).take(entry.get("max_cases"))


__all__ = ["load_harmbench"]
