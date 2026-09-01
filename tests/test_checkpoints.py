"""Tests for :func:`infl_ens.utils.checkpoints.prune_intermediate_adapters`."""

from __future__ import annotations

from pathlib import Path

from infl_ens.utils.checkpoints import prune_intermediate_adapters


def _touch_adapter(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_model.safetensors").write_bytes(b"x")


def _make_run(root: Path) -> Path:
    run = root / "results" / "arm" / "seed0"
    for agent in ("pair-0", "pair-1"):
        for rnd in range(3):
            _touch_adapter(run / "agents" / agent / f"round-{rnd:02d}")
    # A flat adapter without round-* children must be left untouched.
    _touch_adapter(run / "agents" / "pooled-baseline")
    return run


def test_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    stats = prune_intermediate_adapters(run, dry_run=True)
    assert stats == {
        "agents_scanned": 2,
        "round_dirs_removed": 4,
        "adapters_before": 7,
        "adapters_after": 7,
    }
    assert (run / "agents" / "pair-0" / "round-00").is_dir()


def test_prune_keeps_only_the_final_round(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    stats = prune_intermediate_adapters(run)
    assert stats["round_dirs_removed"] == 4
    assert stats["adapters_after"] == 3
    for agent in ("pair-0", "pair-1"):
        kept = sorted(p.name for p in (run / "agents" / agent).iterdir())
        assert kept == ["round-02"]
    assert (run / "agents" / "pooled-baseline" / "adapter_model.safetensors").is_file()
