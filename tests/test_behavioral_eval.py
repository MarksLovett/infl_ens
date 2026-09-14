"""Behavioral artifacts, graders, mixture math and pipeline interfaces."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from infl_ens.data.behavioral import BehavioralCase, BehavioralSuite, ChatMessage
from infl_ens.evaluation.behavioral_artifact import (
    GenerationArtifact,
    GenerationRecord,
    generation_artifact_is_current,
    write_generation_artifact,
)
from infl_ens.evaluation.behavioral_graders import (
    aggregate_behavioral_scores,
    grade_generation_records,
    resolved_grader_identity,
)
from infl_ens.experiment import (
    BehavioralSettings,
    EvalSettings,
    ExperimentConfig,
    FigureSettings,
    SmokeSettings,
    load_experiment,
)
from infl_ens.figures.behavioral_report import (
    behavioral_csv,
    behavioral_markdown,
    behavioral_rows,
)
from infl_ens.pipeline.stages import PipelineContext, stage_behavioral

ROOT = Path(__file__).resolve().parents[1]
BASELINE_EXPERIMENT = ROOT / "configs" / "experiments" / "seven_axis_baselines.yaml"


def _suite(*, safe: bool = False) -> BehavioralSuite:
    return BehavioralSuite(
        name="xstest",
        version="fixture",
        source="fixture",
        cases=(BehavioralCase(
            suite="xstest",
            case_id="xstest:1",
            messages=(ChatMessage("user", "benign request"),),
            category="safe" if safe else "unsafe",
            scoring_payload={"safe": safe},
        ),),
    )


def _record(response: str) -> GenerationRecord:
    return GenerationRecord(
        case_id="xstest:1",
        suite="xstest",
        target="model",
        protocol="native",
        messages=({"role": "user", "content": "benign request"},),
        response=response,
        token_ids=(1, 2),
    )


def test_generation_artifact_round_trip_and_provenance_invalidation(tmp_path: Path) -> None:
    provenance = {"model": "fixture", "checkpoint_sha256": "abc", "case_ids": ["xstest:1"]}
    artifact = write_generation_artifact(tmp_path, [_record("I cannot help")], provenance)
    assert artifact == GenerationArtifact.load(tmp_path)
    assert generation_artifact_is_current(tmp_path, provenance)
    assert not generation_artifact_is_current(tmp_path, {**provenance, "model": "changed"})
    manifest_path = tmp_path / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["n_records"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not generation_artifact_is_current(tmp_path, provenance)


def test_current_generation_cache_does_not_load_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import infl_ens.evaluation.behavioral_eval as module
    import infl_ens.evaluation.adapters as adapters

    suite = _suite()
    experiment = ExperimentConfig(
        name="fixture",
        path=tmp_path / "experiment.yaml",
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
        arms=(),
        stages=("behavioral",),
        eval=EvalSettings(),
        figures=FigureSettings(),
        smoke=SmokeSettings(),
        behavioral_eval=BehavioralSettings(suites=({
            "kind": "xstest",
            "path": "unused",
        },)),
    )
    target = module.BehavioralTarget(
        name="base_model",
        kind="base",
        base_model="fixture/base",
    )
    coordinates = np.asarray([[0.25, 0.75]])
    provenance = module._target_provenance(
        target,
        suite,
        "native",
        experiment,
        None,
        coordinates,
        {},
        experiment.behavioral_eval.generation,
    )
    output = tmp_path / "results" / "behavioral" / "base_model" / "native" / "xstest"
    write_generation_artifact(output, [_record("cached")], provenance)

    def fail_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("a current generation artifact must bypass model loading")

    monkeypatch.setattr(adapters, "load_base_causal_lm", fail_load)
    artifacts = module._generate_target_batch(
        target,
        [(suite, coordinates)],
        "native",
        object(),
        experiment,
        force=False,
        checkpoint_hashes={},
    )
    assert artifacts[0][1].records[0].response == "cached"


def test_proxy_grader_reports_safe_and_unsafe_behavior_separately() -> None:
    unsafe_scores = grade_generation_records(
        _suite(safe=False), [_record("I cannot assist with that")], {"kind": "smoke_proxy"},
    )
    safe_scores = grade_generation_records(
        _suite(safe=True), [_record("Here is the benign answer")], {"kind": "smoke_proxy"},
    )
    assert unsafe_scores[0].metrics["unsafe_compliance"] == 0.0
    assert "safe_refusal" not in unsafe_scores[0].metrics
    assert safe_scores[0].metrics["safe_refusal"] == 0.0
    aggregate = aggregate_behavioral_scores([*unsafe_scores, *safe_scores])
    assert aggregate["overall"]["calibrated_behavior"] == 1.0
    assert aggregate["overall"]["n_errors"] == 0


def test_grader_identity_preserves_evaluator_configuration() -> None:
    identity = resolved_grader_identity(
        "harmbench",
        {
            "kind": "official",
            "evaluator": "harmbench",
            "kwargs": {"max_response_length": 1024},
        },
    )
    assert identity["kwargs"] == {"max_response_length": 1024}
    assert identity["adapter_implementation_version"] == 1


def test_strongreject_adapter_uses_single_pair_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def evaluate(
        forbidden_prompt: str,
        response: str,
        evaluators: list[str],
        **kwargs: object,
    ) -> list[dict[str, float]]:
        observed.update(
            forbidden_prompt=forbidden_prompt,
            response=response,
            evaluators=evaluators,
            kwargs=kwargs,
        )
        return [{"score": 0.25}]

    package = types.ModuleType("strong_reject")
    module = types.ModuleType("strong_reject.evaluate")
    module.evaluate = evaluate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "strong_reject", package)
    monkeypatch.setitem(sys.modules, "strong_reject.evaluate", module)
    case = BehavioralCase(
        suite="strongreject",
        case_id="strongreject:1",
        messages=(ChatMessage("user", "forbidden"),),
        scoring_payload={"forbidden_prompt": "forbidden"},
    )
    record = GenerationRecord(
        case_id=case.case_id,
        suite=case.suite,
        target="model",
        protocol="native",
        messages=tuple(message.to_dict() for message in case.messages),
        response="response",
    )
    scores = grade_generation_records(
        BehavioralSuite("strongreject", "fixture", "fixture", (case,)),
        [record],
        {
            "kind": "official",
            "evaluator": "strongreject_finetuned",
            "kwargs": {"max_response_length": 1024},
        },
    )
    assert scores[0].metrics["score"] == 0.25
    assert observed["evaluators"] == ["strongreject_finetuned"]
    assert observed["kwargs"] == {"max_response_length": 1024}


def test_confaide_official_rating_parser_and_correlation() -> None:
    cases = tuple(
        BehavioralCase(
            suite="confaide",
            case_id=f"confaide:{index}",
            messages=(ChatMessage("user", f"rate {index}"),),
            category="tier-1",
            scoring_payload={"expected_rating": expected, "rating_values": [1, 2, 3, 4]},
        )
        for index, expected in enumerate((1.2, 3.8))
    )
    suite = BehavioralSuite("confaide", "fixture", "fixture", cases)
    records = [
        GenerationRecord(
            case_id=case.case_id,
            suite="confaide",
            target="model",
            protocol="native",
            messages=tuple(message.to_dict() for message in case.messages),
            response=response,
        )
        for case, response in zip(cases, ("1) not sensitive", "Answer: 4"))
    ]
    scores = grade_generation_records(suite, records, {"kind": "official"})
    assert [score.metrics["model_rating"] for score in scores] == [1.0, 4.0]
    aggregate = aggregate_behavioral_scores(scores)
    assert aggregate["overall"]["pearson_correlation"] == pytest.approx(1.0)


def test_probability_mixture_matches_manual_distribution() -> None:
    torch = pytest.importorskip("torch")
    from infl_ens.evaluation.generation import mixture_log_probabilities

    logits = torch.tensor([
        [[2.0, 0.0, -1.0]],
        [[0.0, 1.0, 2.0]],
    ])
    weights = torch.tensor([[0.25, 0.75]])
    actual = mixture_log_probabilities(logits, weights).exp()
    expected = 0.25 * torch.softmax(logits[0], dim=-1) + 0.75 * torch.softmax(
        logits[1], dim=-1,
    )
    assert torch.allclose(actual, expected, atol=1e-7)
    with pytest.raises(ValueError, match="nonnegative"):
        mixture_log_probabilities(logits, torch.tensor([[1.1, -0.1]]))


def test_baseline_experiment_registers_behavioral_protocol() -> None:
    experiment = load_experiment(BASELINE_EXPERIMENT)
    settings = experiment.behavioral_eval
    assert settings is not None
    assert settings.checkpoints == "final"
    assert settings.protocols == ("native", "hard_argmax")
    assert {entry["kind"] for entry in settings.suites} == {
        "harmbench", "strongreject", "xstest", "truthfulqa", "confaide", "bipia", "ifeval",
    }
    assert "behavioral" in experiment.stages


def test_behavioral_stage_passes_arm_restriction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_experiment(BASELINE_EXPERIMENT)
    observed: dict[str, object] = {}

    def fake_run(
        exp: object,
        *,
        repo_root: Path,
        selected_arms: tuple[str, ...],
        force: bool,
    ) -> Path:
        observed.update(
            exp=exp,
            repo_root=repo_root,
            selected_arms=selected_arms,
            force=force,
        )
        return tmp_path / "summary.json"

    import infl_ens.evaluation.behavioral_eval as module

    monkeypatch.setattr(module, "run_behavioral_eval", fake_run)
    stage_behavioral(PipelineContext(
        exp=experiment,
        repo_root=tmp_path,
        only_arms=("game",),
    ))
    assert observed["repo_root"] == tmp_path
    assert observed["selected_arms"] == ("game",)
    assert observed["force"] is False


def test_behavioral_report_outputs_long_form_rows() -> None:
    summary = {
        "schema_version": 1,
        "rows": [{
            "target": "game",
            "target_kind": "routed",
            "protocol": "native",
            "suite": "xstest",
            "aggregate": {
                "overall": {
                    "n": 10,
                    "n_scored": 9,
                    "n_errors": 1,
                    "calibrated_behavior": 0.75,
                    "calibrated_behavior_n": 8,
                },
                "graders": ["lexical_refusal"],
                "fidelity": ["smoke_proxy"],
            },
        }],
    }
    rows = behavioral_rows(summary)
    assert rows == [{
        "target": "game", "target_kind": "routed", "protocol": "native",
        "suite": "xstest", "graders": "lexical_refusal", "fidelity": "smoke_proxy",
        "category": "all",
        "metric": "calibrated_behavior", "value": 0.75,
        "n": 10, "n_scored": 9, "n_errors": 1,
    }]
    assert (
        "game,routed,native,xstest,lexical_refusal,smoke_proxy,all,"
        "calibrated_behavior,0.75"
    ) in behavioral_csv(rows)
    markdown = behavioral_markdown(rows)
    assert (
        "| game | native | xstest | smoke_proxy | all | calibrated_behavior | 0.7500"
        in markdown
    )
