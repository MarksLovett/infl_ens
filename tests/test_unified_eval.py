"""Offline tests for the unified training + evaluation config.

One closed-loop YAML carries the encoder (``trait_space``), the training
(``closed_loop``) and the evaluation (``eval``) blocks. Covers:

- :meth:`infl_ens.evaluation.evaluate.EvalJobConfig.from_unified` deriving
  the run directory, base model, benchmarks and split manifest from the
  training blocks,
- :func:`infl_ens.evaluation.evaluate.run_unified_eval` looping the
  configured partitions, resolving ``rounds: final`` from ``history.json``
  and scoring an optional baseline run,
- the closed-loop trainer running that evaluation automatically after the
  last round, and
- the evaluation CLI accepting the training YAML as-is.

The scoring itself is stubbed: no model is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

import infl_ens.evaluation.evaluate as evaluate_mod
from infl_ens.evaluation.evaluate import (
    EvalJobConfig,
    final_round_from_history,
    is_unified_config,
    run_unified_eval,
)


def _unified_cfg(run_dir: Path) -> dict[str, Any]:
    return {
        "task": "closed_loop",
        "seed": 3,
        "output_dir": str(run_dir),
        "benchmarks": [{"kind": "beavertails", "path": "x.jsonl"}],
        "trait_space": {"encoder": "enc", "n_grid": 3},
        "data_split": {"manifest": "data/splits/fake.json"},
        "closed_loop": {
            "sft": {"base_model": "org/model", "max_seq_length": 512},
        },
        "eval": {"max_eval_records": 10, "forward_batch_size": 4},
    }


def _install_recorder(monkeypatch: pytest.MonkeyPatch) -> list[EvalJobConfig]:
    jobs: list[EvalJobConfig] = []

    def fake_run_eval_job(job: EvalJobConfig) -> list:
        jobs.append(job)
        return []

    monkeypatch.setattr(evaluate_mod, "run_eval_job", fake_run_eval_job)
    return jobs


# ---------------------------------------------------------------------------
# EvalJobConfig.from_unified
# ---------------------------------------------------------------------------


def test_from_unified_derives_fields(tmp_path: Path) -> None:
    cfg = _unified_cfg(tmp_path / "run")
    job = EvalJobConfig.from_unified(cfg, partition="test", rounds=[3])
    assert job.task == "run_eval"
    assert job.seed == 3
    assert job.run_dir == str(tmp_path / "run")
    assert job.output_dir == str(tmp_path / "run" / "eval_test")
    assert job.base_model == "org/model"
    assert job.benchmarks == cfg["benchmarks"]
    assert job.rounds == [3]
    assert job.agents is None
    assert job.data_split_manifest == "data/splits/fake.json"
    assert job.data_split_partition == "test"
    adapter_cfg = job.to_adapter_eval_config()
    assert adapter_cfg.max_seq_length == 512      # inherited from closed_loop.sft
    assert adapter_cfg.forward_batch_size == 4
    assert adapter_cfg.max_eval_records == 10
    assert adapter_cfg.seed == 3

    # Eval-block overrides win over the training defaults.
    cfg["eval"]["max_seq_length"] = 256
    cfg["eval"]["base_model"] = "org/other"
    cfg["eval"]["agents"] = ["pair-0"]
    job = EvalJobConfig.from_unified(cfg, partition="train")
    assert job.to_adapter_eval_config().max_seq_length == 256
    assert job.base_model == "org/other"
    assert job.agents == ["pair-0"]
    assert job.rounds is None

    # A manifest the trainer wrote itself is accepted too.
    cfg["data_split"] = {"write_manifest": "splits/built.json"}
    assert EvalJobConfig.from_unified(cfg, partition="val").data_split_manifest == (
        "splits/built.json"
    )


def test_from_unified_rejects_bad_shapes(tmp_path: Path) -> None:
    cfg = _unified_cfg(tmp_path)
    with pytest.raises(ValueError, match="closed_loop"):
        EvalJobConfig.from_unified({"task": "run_eval"}, partition="test")
    cfg["data_split"] = {}
    with pytest.raises(ValueError, match="manifest"):
        EvalJobConfig.from_unified(cfg, partition="test")


def test_is_unified_config(tmp_path: Path) -> None:
    assert is_unified_config(_unified_cfg(tmp_path))
    assert not is_unified_config({"task": "run_eval", "eval": {}})
    legacy = _unified_cfg(tmp_path)
    legacy.pop("eval")
    assert not is_unified_config(legacy)


# ---------------------------------------------------------------------------
# run_unified_eval
# ---------------------------------------------------------------------------


def _write_history(run_dir: Path, n_rounds: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.json").write_text(
        json.dumps([{"round": r} for r in range(n_rounds)]), encoding="utf-8",
    )


def test_run_unified_eval_partitions_and_final_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_history(run_dir, 4)
    assert final_round_from_history(run_dir) == 3
    jobs = _install_recorder(monkeypatch)

    cfg = _unified_cfg(run_dir)
    reports = run_unified_eval(cfg)
    assert [j.data_split_partition for j in jobs] == ["train", "test"]
    assert all(j.rounds == [3] for j in jobs)
    assert all(j.run_dir == str(run_dir) for j in jobs)
    assert reports == [
        run_dir / "eval_train" / "eval_results.json",
        run_dir / "eval_test" / "eval_results.json",
    ]

    # The trainer passes the round it just finished; history is not needed.
    jobs.clear()
    (run_dir / "history.json").unlink()
    run_unified_eval(cfg, final_round=7)
    assert all(j.rounds == [7] for j in jobs)

    # Explicit partitions / rounds are honoured verbatim.
    jobs.clear()
    cfg["eval"]["partitions"] = ["val"]
    cfg["eval"]["rounds"] = [1, 2]
    run_unified_eval(cfg)
    assert [(j.data_split_partition, j.rounds) for j in jobs] == [("val", [1, 2])]

    # A baseline run is scored on the same partitions and rounds.
    jobs.clear()
    baseline = tmp_path / "pooled"
    cfg["eval"]["baseline_run_dir"] = str(baseline)
    cfg["eval"]["baseline_agents"] = ["pooled-generalist"]
    reports = run_unified_eval(cfg)
    assert [j.run_dir for j in jobs] == [str(run_dir), str(baseline)]
    assert jobs[1].output_dir == str(baseline / "eval_val")
    assert jobs[1].agents == ["pooled-generalist"]
    assert jobs[1].rounds == [1, 2]
    assert reports[1] == baseline / "eval_val" / "eval_results.json"


def test_final_round_from_history_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        final_round_from_history(tmp_path)
    (tmp_path / "history.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        final_round_from_history(tmp_path)


# ---------------------------------------------------------------------------
# closed-loop trainer hook
# ---------------------------------------------------------------------------


def _install_fake_training(monkeypatch: pytest.MonkeyPatch, prompts: list[str]) -> None:
    """Deterministic toy trait space plus a no-op SFT stub."""
    from infl_ens.data.benchmarks import BenchmarkSplit
    from infl_ens.data.trait_space import TraitSpace
    import infl_ens.training.closed_loop as driver
    import infl_ens.training.sft_training as sft_mod

    rng = np.random.default_rng(0)
    coords = {p: rng.random(2) for p in prompts}

    def project(queries: Sequence[str]) -> np.ndarray:
        return np.stack([coords[q] for q in queries], axis=0)

    space = TraitSpace(
        grid=np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
        weights=np.ones(4) / 4,
        project=project,
        axis_labels=("harm", "other"),
    )
    split = BenchmarkSplit(
        name="fake",
        prompts=list(prompts),
        scores=np.zeros(len(prompts)),
        axis_name="harm",
        responses=[f"r-{p}" for p in prompts],
    )
    monkeypatch.setattr(driver, "_load_splits", lambda cfg: [split])
    monkeypatch.setattr(driver, "_make_trait_space", lambda cfg, splits: space)

    def fake_sft_train_agent(agent, prompts, responses, cfg, **kwargs):  # noqa: ANN001
        return {
            "output_dir": str(kwargs.get("out_dir_override") or cfg.output_dir),
            "n_train": len(prompts),
            "log_history": [],
            "loaded_prior_lora": None,
        }

    monkeypatch.setattr(sft_mod, "sft_train_agent", fake_sft_train_agent)


def _trainer_cfg(tmp_path: Path, *, with_eval: bool = True) -> dict[str, Any]:
    out_dir = tmp_path / "run"
    cfg: dict[str, Any] = {
        "task": "closed_loop",
        "seed": 0,
        "repo_root": str(tmp_path),
        "output_dir": str(out_dir),
        "policy": "proportional",
        "benchmarks": [{"kind": "fake", "path": "unused"}],
        "agents": [{"name": "clone-0"}, {"name": "clone-1"}],
        "sigma_mode": "absolute",
        "sigma": 0.35,
        "data_split": {
            "seed": 0,
            "train_frac": 0.5,
            "val_frac": 0.25,
            "test_frac": 0.25,
            "write_manifest": "splits/fake.json",
            "cover_train_exactly": False,
        },
        "closed_loop": {
            "init_mode": "mean_noise",
            "init_noise": 0.05,
            "n_rounds": 2,
            "batch_size": 8,
            "sft": {"output_dir": str(out_dir / "agents"), "base_model": "org/m"},
        },
    }
    if with_eval:
        cfg["eval"] = {"partitions": ["train", "test"], "max_eval_records": 5}
    return cfg


def test_trainer_runs_unified_eval_after_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from infl_ens.training.closed_loop import run_closed_loop as _task_closed_loop

    _install_fake_training(monkeypatch, [f"q{i}" for i in range(40)])
    jobs = _install_recorder(monkeypatch)

    cfg = _trainer_cfg(tmp_path)
    assert _task_closed_loop(cfg) == 0
    assert [j.data_split_partition for j in jobs] == ["train", "test"]
    assert all(j.rounds == [1] for j in jobs)          # n_rounds - 1
    assert all(j.run_dir == str(tmp_path / "run") for j in jobs)
    assert all(j.base_model == "org/m" for j in jobs)
    assert all(j.data_split_manifest == "splits/fake.json" for j in jobs)
    assert jobs[0].output_dir == str(tmp_path / "run" / "eval_train")

    # after_training: false leaves the eval to the standalone CLI.
    jobs.clear()
    cfg = _trainer_cfg(tmp_path)
    cfg["eval"]["after_training"] = False
    assert _task_closed_loop(cfg) == 0
    assert jobs == []

    # No eval block: nothing runs.
    cfg = _trainer_cfg(tmp_path, with_eval=False)
    assert _task_closed_loop(cfg) == 0
    assert jobs == []


# ---------------------------------------------------------------------------
# evaluation CLI
# ---------------------------------------------------------------------------


def test_evaluation_cli_dispatches_unified_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    yaml = pytest.importorskip("yaml")
    import infl_ens.evaluation.__main__ as cli

    seen: list[dict[str, Any]] = []

    def fake_run_unified_eval(cfg: dict[str, Any]) -> list[Path]:
        seen.append(cfg)
        return [tmp_path / "eval_test" / "eval_results.json"]

    monkeypatch.setattr(cli, "run_unified_eval", fake_run_unified_eval)
    cfg_path = tmp_path / "unified.yaml"
    cfg_path.write_text(yaml.safe_dump(_unified_cfg(tmp_path / "run")), encoding="utf-8")

    assert cli.main(["--config", str(cfg_path), "eval.max_eval_records=7"]) == 0
    assert len(seen) == 1
    assert seen[0]["task"] == "closed_loop"           # task is not consulted
    assert seen[0]["eval"]["max_eval_records"] == 7    # overrides still apply
