"""Tests for the thin ``python -m infl_ens.training`` entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

import infl_ens.training.__main__ as cli
import infl_ens.training.tasks as tasks_mod


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_task_is_rejected_with_exit_code_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _write(tmp_path / "bad.yaml", "task: router_training\n")
    assert cli.main(["--config", str(cfg)]) == 2
    assert "unknown task" in capsys.readouterr().err


def test_unknown_key_is_rejected_before_dispatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _write(tmp_path / "bad.yaml", "task: closed_loop\nclosed_loop: {theory_pre: {}}\n")
    assert cli.main(["--config", str(cfg)]) == 2
    err = capsys.readouterr().err
    assert "theory_pre" in err
    assert "bad.yaml" in err


def test_dispatch_passes_overrides_to_the_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_task(cfg: dict) -> int:
        seen.update(cfg)
        return 0

    monkeypatch.setitem(tasks_mod.TASKS, "closed_loop", fake_task)
    monkeypatch.setattr(cli, "TASKS", tasks_mod.TASKS)
    cfg = _write(
        tmp_path / "run.yaml",
        "task: closed_loop\nseed: 3\nclosed_loop: {n_rounds: 12, batch_size: 64}\n",
    )
    rc = cli.main(["--config", str(cfg), "closed_loop.n_rounds=2", "data_split=null"])
    assert rc == 0
    assert seen["seed"] == 3
    assert seen["closed_loop"] == {"n_rounds": 2, "batch_size": 64}
    assert seen["data_split"] is None
