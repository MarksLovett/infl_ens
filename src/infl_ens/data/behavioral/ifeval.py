"""Offline loader for Google's Instruction-Following Eval records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from infl_ens.data.behavioral.base import BehavioralCase, BehavioralSuite, stable_case_id
from infl_ens.data.behavioral.common import coerce_messages, first_text, read_rows, source_files


def load_ifeval(entry: Mapping[str, Any]) -> BehavioralSuite:
    """Load IFEval prompts and deterministic checker arguments.

    :param entry: Suite config with ``path`` and optional ``version``.
    :type entry: Mapping[str, Any]
    :returns: Pinned IFEval suite.
    :rtype: BehavioralSuite
    """
    rows = read_rows(Path(str(entry["path"])))
    cases: list[BehavioralCase] = []
    for index, row in enumerate(rows):
        messages = coerce_messages(row, prompt_keys=("prompt", "instruction"))
        if not messages:
            continue
        raw_id = first_text(row, ("key", "id", "prompt_id"))
        case_id = f"ifeval:{raw_id}" if raw_id else stable_case_id("ifeval", messages)
        instruction_ids = list(row.get("instruction_id_list") or [])
        cases.append(BehavioralCase(
            suite="ifeval",
            case_id=case_id,
            messages=messages,
            category=str(instruction_ids[0]) if instruction_ids else "all",
            metadata={"row": index, "source_file": row.get("_source_file")},
            scoring_payload={
                "prompt": messages[-1].content,
                "instruction_id_list": instruction_ids,
                "kwargs": list(row.get("kwargs") or []),
            },
        ))
    return BehavioralSuite(
        name="ifeval",
        version=str(entry.get("version", "pinned-local")),
        source="google-research/instruction_following_eval",
        cases=tuple(cases),
        metadata={"source_files": source_files(rows)},
    ).take(entry.get("max_cases"))


__all__ = ["load_ifeval"]
