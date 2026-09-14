"""Offline loader for prepared BIPIA indirect-injection cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from infl_ens.data.behavioral.base import (
    BehavioralCase,
    BehavioralSuite,
    ChatMessage,
    stable_case_id,
)
from infl_ens.data.behavioral.common import (
    coerce_bool,
    coerce_messages,
    first_text,
    read_rows,
    source_files,
)


def _messages(row: Mapping[str, Any]) -> tuple[ChatMessage, ...] | None:
    if isinstance(row.get("messages"), list):
        messages = coerce_messages(row, prompt_keys=())
        if messages:
            return messages
    query = first_text(row, ("user_query", "query", "task"))
    context = first_text(row, ("context", "external_content", "injected_context"))
    if query:
        visible = query if not context else f"{query}\n\nExternal content:\n{context}"
        return (ChatMessage("user", visible),)
    return coerce_messages(row, prompt_keys=("prompt", "input", "instruction"))


def load_bipia(entry: Mapping[str, Any]) -> BehavioralSuite:
    """Load BIPIA cases prepared by the official builder.

    The upstream BIPIA package constructs attacked examples from clean task
    and attack files. This loader expects that constructed JSON/JSONL/CSV
    output and never invents attacks itself.

    :param entry: Suite config with ``path`` and optional ``version``.
    :type entry: Mapping[str, Any]
    :returns: Pinned BIPIA suite.
    :rtype: BehavioralSuite
    """
    rows = read_rows(Path(str(entry["path"])))
    cases: list[BehavioralCase] = []
    for index, row in enumerate(rows):
        messages = _messages(row)
        if not messages:
            continue
        raw_id = first_text(row, ("id", "case_id", "sample_id"))
        case_id = f"bipia:{raw_id}" if raw_id else stable_case_id("bipia", messages)
        cases.append(BehavioralCase(
            suite="bipia",
            case_id=case_id,
            messages=messages,
            category=first_text(row, ("task_name", "task", "dataset_name")) or "all",
            metadata={"row": index, "source_file": row.get("_source_file")},
            scoring_payload={
                "attack_target": row.get("attack_target", row.get("target")),
                "reference": row.get("reference", row.get("answer")),
                "attacked": coerce_bool(
                    row.get("attacked", row.get("attack")), default=True,
                ),
            },
        ))
    return BehavioralSuite(
        name="bipia",
        version=str(entry.get("version", "pinned-local")),
        source="microsoft/BIPIA",
        cases=tuple(cases),
        metadata={"source_files": source_files(rows)},
    ).take(entry.get("max_cases"))


__all__ = ["load_bipia"]
