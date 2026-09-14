"""Offline loader for ConfAIde contextual-privacy cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from infl_ens.data.behavioral.base import BehavioralCase, BehavioralSuite, stable_case_id
from infl_ens.data.behavioral.common import (
    coerce_bool,
    coerce_messages,
    first_text,
    read_rows,
    source_files,
)


def _upstream_rows(root: Path, tiers: set[str]) -> list[dict[str, Any]]:
    unsupported = tiers - {"1", "2a", "2b"}
    if unsupported:
        raise ValueError(
            "raw ConfAIde loading supports tiers 1, 2a and 2b; tiers 3/4 require "
            "prepared records because the upstream protocol expands each scenario into "
            "multiple interactive tasks",
        )
    rows: list[dict[str, Any]] = []
    for tier in ("1", "2a", "2b"):
        if tier not in tiers:
            continue
        prompts_path = root / f"tier_{tier}.txt"
        labels_path = root / f"tier_{tier[0]}_labels.txt"
        if not prompts_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(
                f"ConfAIde tier {tier} requires {prompts_path.name} and {labels_path.name}",
            )
        prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines()]
        labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
        if len(prompts) != len(labels):
            raise ValueError(f"ConfAIde tier {tier} prompt/label counts disagree")
        for index, (prompt, label) in enumerate(zip(prompts, labels)):
            rows.append({
                "id": f"{tier}:{index}",
                "prompt": prompt,
                "tier": tier,
                "expected_rating": float(label),
                "_source_file": prompts_path.as_posix(),
            })
    return rows


def load_confaide(entry: Mapping[str, Any]) -> BehavioralSuite:
    """Load normalized or upstream-like ConfAIde records.

    :param entry: Suite config with ``path``, optional ``tiers`` and ``version``.
    :type entry: Mapping[str, Any]
    :returns: Pinned ConfAIde suite.
    :rtype: BehavioralSuite
    """
    root = Path(str(entry["path"]))
    wanted = {str(value).lower() for value in entry.get("tiers", [])}
    rows = (
        _upstream_rows(root, wanted or {"1", "2a", "2b"})
        if root.is_dir() and (root / "tier_1.txt").is_file()
        else read_rows(root)
    )
    cases: list[BehavioralCase] = []
    for index, row in enumerate(rows):
        tier = first_text(row, ("tier", "data_tier", "level")) or "unknown"
        source_file = str(row.get("_source_file", ""))
        if tier == "unknown":
            for candidate in ("1", "2a", "2b", "3", "4"):
                if f"tier{candidate}" in source_file.lower().replace("_", ""):
                    tier = candidate
                    break
        if wanted and tier.lower() not in wanted:
            continue
        messages = coerce_messages(
            row,
            prompt_keys=("prompt", "question", "scenario", "context", "input"),
        )
        if not messages:
            continue
        raw_id = first_text(row, ("id", "case_id", "scenario_id"))
        case_id = f"confaide:{raw_id}" if raw_id else stable_case_id("confaide", messages)
        if row.get("expected_rating") is not None:
            expected_disclosure: Any = None
        elif row.get("should_disclose") is not None:
            expected_disclosure: Any = row["should_disclose"]
        elif row.get("private") is not None:
            expected_disclosure = not coerce_bool(row["private"], default=True)
        else:
            expected_disclosure = row.get("label")
        cases.append(BehavioralCase(
            suite="confaide",
            case_id=case_id,
            messages=messages,
            category=f"tier-{tier}",
            metadata={"row": index, "tier": tier, "source_file": source_file},
            scoring_payload={
                "expected_disclosure": expected_disclosure,
                "expected_rating": row.get("expected_rating"),
                "rating_values": (
                    [1, 2, 3, 4]
                    if tier == "1" else [-100, -50, 0, 50, 100]
                ),
                "reference": row.get("answer", row.get("reference")),
            },
        ))
    return BehavioralSuite(
        name="confaide",
        version=str(entry.get("version", "pinned-local")),
        source="skywalker023/confaide",
        cases=tuple(cases),
        metadata={"source_files": source_files(rows)},
    ).take(entry.get("max_cases"))


__all__ = ["load_confaide"]
