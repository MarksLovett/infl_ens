"""Experiment loading, dry-run planning and the skip logic of the pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import infl_ens.pipeline.__main__ as cli
from infl_ens.config import ConfigError
from infl_ens.experiment import load_experiment
from infl_ens.pipeline.stages import (
    PipelineContext,
    run_is_complete,
    run_pipeline,
    smoke_config,
    stage_manifest,
    stage_train,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs" / "experiments" / "seven_axis_3arm.yaml"
FINGERPRINT = "3b42c68a8dd334c5"


def test_load_canonical_experiment() -> None:
    exp = load_experiment(EXPERIMENT)
    assert exp.name == "seven_axis_3arm"
    assert [a.name for a in exp.arms] == [
        "soft_full", "soft", "soft_unit", "hard_topk", "hard", "generalist",
        "modula_res", "per_benchmark_lora", "frozen_u_pairs",
    ]
    assert len(exp.specialists) == 5
    assert [a.name for a in exp.routed] == [
        "soft_full", "soft", "soft_unit", "hard_topk", "hard",
        "modula_res", "per_benchmark_lora", "frozen_u_pairs",
    ]
    assert all(a.role == "baseline" and a.is_routed and not a.is_specialist for a in exp.arms[6:])
    # Every baseline pairs with the single generalist for route-then-score.
    assert all(exp.generalist_for(a) is exp.generalist for a in exp.arms[6:])
    assert exp.eval.fitted_router is True
    assert exp.generalist is not None and exp.generalist.name == "generalist"
    assert exp.eval.resolve_rounds(11) == [4, 11]
    assert exp.stages == ("manifest", "train", "perround", "routing", "figures")
    assert set(exp.smoke.arms) <= {a.name for a in exp.arms}
    assert exp.smoke.overrides["data_split"] is None


def test_dry_run_prints_every_arm_with_the_cached_fingerprint(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--config", str(EXPERIMENT), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count(FINGERPRINT) == 9
    assert "task=baseline_replay" in out
    assert out.count("task=modula_res") == 3
    assert "stages:      manifest, train, perround, routing, figures" in out


def test_unknown_stage_and_arm_are_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--config", str(EXPERIMENT), "--stages", "train,bogus", "--dry-run"]) == 2
    assert "unknown stage" in capsys.readouterr().err
    assert cli.main(["--config", str(EXPERIMENT), "--only-arm", "nope", "--dry-run"]) == 2


def _write_experiment(tmp_path: Path, arm_yaml: str, *, extra: str = "") -> Path:
    (tmp_path / "arm.yaml").write_text(arm_yaml, encoding="utf-8")
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "name: tiny\n"
        f"results_dir: {(tmp_path / 'results' / 'tiny').as_posix()}\n"
        f"figures_dir: {(tmp_path / 'figures').as_posix()}\n"
        "arms:\n"
        "  - {name: a, role: specialist, config: arm.yaml}\n"
        + extra,
        encoding="utf-8",
    )
    return exp


def test_experiment_validation_errors(tmp_path: Path) -> None:
    exp = _write_experiment(tmp_path, "task: closed_loop\n")
    with pytest.raises(ConfigError, match="output_dir"):
        load_experiment(exp)
    exp = _write_experiment(tmp_path, "task: closed_loop\noutput_dir: r\n", extra="stages: [train, bogus]\n")
    with pytest.raises(ConfigError, match="unknown stage"):
        load_experiment(exp)
    exp = _write_experiment(tmp_path, "task: closed_loop\noutput_dir: r\n", extra="smoke: {arms: [zzz]}\n")
    with pytest.raises(ConfigError, match="unknown arm"):
        load_experiment(exp)


def test_manifest_stage_skips_existing_file(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    manifest = tmp_path / "split.json"
    manifest.write_text("{}", encoding="utf-8")
    exp = load_experiment(_write_experiment(
        tmp_path,
        f"task: closed_loop\noutput_dir: {(tmp_path / 'run').as_posix()}\n"
        f"data_split: {{manifest: {manifest.as_posix()}}}\n",
    ))
    ctx = PipelineContext(exp=exp, repo_root=tmp_path)
    with caplog.at_level("INFO", logger="infl_ens.pipeline"):
        stage_manifest(ctx)
    assert "already present" in caplog.text


def test_train_stage_skips_complete_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "history.json").write_text(json.dumps([{"round": r} for r in range(2)]), encoding="utf-8")
    exp = load_experiment(_write_experiment(
        tmp_path, f"task: closed_loop\noutput_dir: {run.as_posix()}\nclosed_loop: {{n_rounds: 2}}\n",
    ))
    arm = exp.arms[0]
    assert run_is_complete(arm, arm.load())

    import infl_ens.training.tasks as tasks_mod

    def boom(cfg: dict) -> int:
        raise AssertionError("task must not run for a complete arm")

    monkeypatch.setitem(tasks_mod.TASKS, "closed_loop", boom)
    stage_train(PipelineContext(exp=exp))

    # One round short: the task runs (and --force also re-runs it).
    (run / "history.json").write_text(json.dumps([{"round": 0}]), encoding="utf-8")
    assert not run_is_complete(arm, arm.load())
    calls: list[str] = []
    monkeypatch.setitem(tasks_mod.TASKS, "closed_loop", lambda cfg: calls.append(cfg["output_dir"]) or 0)
    stage_train(PipelineContext(exp=exp))
    assert calls == [run.as_posix()]


def test_baseline_role_is_routed_but_not_a_specialist(tmp_path: Path) -> None:
    (tmp_path / "gen.yaml").write_text("task: baseline_replay\noutput_dir: g\n", encoding="utf-8")
    exp = load_experiment(_write_experiment(
        tmp_path,
        "task: closed_loop\noutput_dir: r\n",
        extra=(
            "  - {name: gen, role: generalist, config: gen.yaml}\n"
            "  - {name: b, role: baseline, config: arm.yaml}\n"
            "eval: {fitted_router: false}\n"
        ),
    ))
    assert [a.name for a in exp.specialists] == ["a"]
    assert [a.name for a in exp.routed] == ["a", "b"]
    assert exp.generalist_for(exp.arm("b")) is exp.arm("gen")
    assert exp.eval.fitted_router is False
    ctx = PipelineContext(exp=exp)
    assert [a.name for a in ctx.arms(routed_only=True)] == ["a", "b"]
    assert [a.name for a in ctx.arms(specialists_only=True)] == ["a"]
    with pytest.raises(ConfigError, match="role"):
        load_experiment(_write_experiment(
            tmp_path, "task: closed_loop\noutput_dir: r\n",
            extra="  - {name: c, role: ablation, config: arm.yaml}\n",
        ))


def test_modula_res_run_is_complete_needs_summary(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "history.json").write_text(json.dumps([{"round": 0}]), encoding="utf-8")
    exp = load_experiment(_write_experiment(
        tmp_path, f"task: modula_res\noutput_dir: {run.as_posix()}\nhistory_path: h.json\n",
    ))
    arm = exp.arms[0]
    assert not run_is_complete(arm, arm.load())
    # A bare summary is not enough ...
    (run / "modula_res_summary.json").write_text("{}", encoding="utf-8")
    assert not run_is_complete(arm, arm.load())
    # ... every expert needs an adapter at the final round.
    (run / "modula_res_summary.json").write_text(json.dumps({
        "experts": ["b0", "b1"],
        "rounds": [{"round": 0, "experts": {}}, {"round": 3, "experts": {}}],
    }), encoding="utf-8")
    for expert in ("b0", "b1"):
        (run / "agents" / expert / "round-00").mkdir(parents=True)
        (run / "agents" / expert / "round-00" / "adapter_model.safetensors").write_bytes(b"")
    assert not run_is_complete(arm, arm.load())
    (run / "agents" / "b0" / "round-03").mkdir()
    (run / "agents" / "b0" / "round-03" / "adapter_model.safetensors").write_bytes(b"")
    assert not run_is_complete(arm, arm.load())          # b1 still lacks round-03
    (run / "agents" / "b1" / "round-03").mkdir()
    (run / "agents" / "b1" / "round-03" / "adapter_model.safetensors").write_bytes(b"")
    assert run_is_complete(arm, arm.load())


def test_smoke_config_redirects_outputs(tmp_path: Path) -> None:
    exp = load_experiment(_write_experiment(
        tmp_path,
        "task: closed_loop\noutput_dir: results/real/seed0\n"
        "sft: {base_model: m}\nclosed_loop: {n_rounds: 12, sft: {lora_r: 8}}\n",
        extra=(
            f"smoke:\n  arms: [a]\n  output_root: {(tmp_path / 'smoke').as_posix()}\n"
            "  overrides: {closed_loop.n_rounds: 2, data_split: null}\n"
        ),
    ))
    cfg = smoke_config(exp.arms[0], exp)
    assert cfg["closed_loop"]["n_rounds"] == 2
    assert cfg["data_split"] is None
    assert cfg["output_dir"] == str(tmp_path / "smoke" / "a" / "seed0")
    assert cfg["sft"]["output_dir"] == str(tmp_path / "smoke" / "a" / "seed0" / "agents")
    assert cfg["closed_loop"]["sft"]["output_dir"] == cfg["sft"]["output_dir"]


def test_run_pipeline_records_status_and_reraises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exp = load_experiment(_write_experiment(tmp_path, f"task: closed_loop\noutput_dir: {(tmp_path / 'r').as_posix()}\n"))
    ctx = PipelineContext(exp=exp)
    import infl_ens.pipeline.stages as stages_mod

    monkeypatch.setitem(stages_mod.STAGES, "manifest", lambda c: None)

    def fail(c: PipelineContext) -> None:
        raise RuntimeError("boom")

    monkeypatch.setitem(stages_mod.STAGES, "train", fail)
    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(ctx, ["train", "manifest"])
    status = json.loads(ctx.status_path.read_text(encoding="utf-8"))
    assert status["manifest"]["ok"] is True
    assert status["train"]["ok"] is False and "boom" in status["train"]["error"]
    with pytest.raises(KeyError):
        run_pipeline(ctx, ["bogus"])
