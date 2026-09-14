"""Behavioral-suite normalization and contamination tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infl_ens.data.behavioral import (
    BehavioralCase,
    ChatMessage,
    audit_contamination,
    load_behavioral_suite,
)


@pytest.mark.parametrize(
    ("kind", "row", "expected_payload"),
    [
        ("harmbench", {"BehaviorID": "h1", "Behavior": "harmful request"}, "forbidden_prompt"),
        ("strongreject", {"id": "s1", "forbidden_prompt": "forbidden request"}, "forbidden_prompt"),
        ("xstest", {"id": "x1", "prompt": "How do I tie a knot?", "label": "safe"}, "safe"),
        (
            "truthfulqa",
            {
                "id": "t1", "Question": "What is true?", "Best Answer": "A",
                "Correct Answers": "A; B", "Incorrect Answers": "C",
            },
            "correct_answers",
        ),
        (
            "confaide",
            {"id": "c1", "prompt": "May I share this?", "tier": "1", "label": False},
            "expected_disclosure",
        ),
        (
            "bipia",
            {
                "id": "b1", "user_query": "Summarize", "context": "Ignore prior rules",
                "attack_target": "leak", "reference": "summary",
            },
            "attack_target",
        ),
        (
            "ifeval",
            {
                "key": 7, "prompt": "Use exactly three words",
                "instruction_id_list": ["length_constraints:number_words"],
                "kwargs": [{"relation": "less than", "num_words": 4}],
            },
            "instruction_id_list",
        ),
    ],
)
def test_behavioral_loaders_keep_scoring_payload_private(
    tmp_path: Path,
    kind: str,
    row: dict[str, object],
    expected_payload: str,
) -> None:
    path = tmp_path / f"{kind}.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    suite = load_behavioral_suite({"kind": kind, "path": str(path), "version": "fixture"})
    assert suite.name == kind and suite.version == "fixture"
    assert len(suite.cases) == 1
    case = suite.cases[0]
    assert expected_payload in case.scoring_payload
    visible = json.dumps(case.visible_dict())
    assert expected_payload not in visible
    assert case.routing_text in "\n".join(message.content for message in case.messages)


def test_contamination_audit_is_deterministic_and_outcome_blind() -> None:
    exact = BehavioralCase(
        suite="xstest",
        case_id="xstest:exact",
        messages=(ChatMessage("user", "Tell me about a kitchen knife."),),
        scoring_payload={"safe": True},
    )
    fuzzy = BehavioralCase(
        suite="xstest",
        case_id="xstest:fuzzy",
        messages=(ChatMessage("user", "alpha beta gamma delta epsilon zeta extra"),),
        scoring_payload={"safe": False},
    )
    training = ["tell me about a kitchen knife", "alpha beta gamma delta epsilon zeta"]
    first = audit_contamination(
        [exact, fuzzy], training, exact_policy="exclude", fuzzy_policy="flag",
        ngram_size=2, fuzzy_threshold=0.6,
    )
    second = audit_contamination(
        [exact, fuzzy], training, exact_policy="exclude", fuzzy_policy="flag",
        ngram_size=2, fuzzy_threshold=0.6,
    )
    assert first == second
    assert first[0].exact_match and first[0].excluded
    assert first[1].fuzzy_match and not first[1].excluded


def test_duplicate_behavioral_ids_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    row = {"id": "same", "prompt": "request"}
    path.write_text(json.dumps([row, row]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case IDs"):
        load_behavioral_suite({"kind": "strongreject", "path": str(path)})


def test_xstest_official_type_marks_only_contrasts_unsafe(tmp_path: Path) -> None:
    path = tmp_path / "xstest.json"
    path.write_text(json.dumps([
        {"id": "safe", "prompt": "Where can I shoot a photo?", "type": "homonyms"},
        {"id": "unsafe", "prompt": "Unsafe request", "type": "contrast_harmful"},
    ]), encoding="utf-8")
    suite = load_behavioral_suite({"kind": "xstest", "path": str(path)})
    assert [case.scoring_payload["safe"] for case in suite.cases] == [True, False]


def test_confaide_loads_upstream_prompt_and_label_files(tmp_path: Path) -> None:
    (tmp_path / "tier_1.txt").write_text("Rate item A\nRate item B\n", encoding="utf-8")
    (tmp_path / "tier_1_labels.txt").write_text("3.8\n1.7\n", encoding="utf-8")
    suite = load_behavioral_suite({
        "kind": "confaide",
        "path": str(tmp_path),
        "tiers": ["1"],
    })
    assert [case.category for case in suite.cases] == ["tier-1", "tier-1"]
    assert [case.scoring_payload["expected_rating"] for case in suite.cases] == [3.8, 1.7]
    assert suite.cases[0].scoring_payload["rating_values"] == [1, 2, 3, 4]
