"""Offline tests for the evaluation subpackage (no torch required)."""

from __future__ import annotations

import json
from pathlib import Path

from infl_ens.evaluation.adapters import (
    AdapterRef,
    discover_adapters,
    is_adapter_dir,
    latest_round_dir,
    resolve_adapter_dir,
)
from infl_ens.evaluation.benchmarks import load_benchmark_splits, subsample_split
from infl_ens.evaluation.evaluate import (
    BenchmarkEvalResult,
    EvalJobConfig,
    write_eval_report,
)
from infl_ens.data.benchmarks import BenchmarkSplit
import numpy as np


def _write_beavertails_fixture(tmp_path: Path) -> Path:
    f = tmp_path / "tiny.jsonl"
    rows = [
        {"prompt": "p1", "response": "r1", "is_safe": True},
        {"prompt": "p2", "response": "r2", "is_safe": False,
         "category": {"violence,aiding_and_abetting,incitement": True}},
        {"prompt": "p3", "response": "r3", "is_safe": True},
    ]
    with f.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return f


def test_load_benchmark_splits_beavertails(tmp_path: Path) -> None:
    path = _write_beavertails_fixture(tmp_path)
    splits = load_benchmark_splits([
        {"kind": "beavertails", "path": str(path), "max_records": 2},
    ])
    assert len(splits) == 1
    assert splits[0].name == "beavertails"
    assert splits[0].n == 2


def test_subsample_split_unchanged_when_small() -> None:
    split = BenchmarkSplit(
        name="toy",
        prompts=["a", "b"],
        scores=np.array([0.0, 1.0]),
        axis_name="harm",
    )
    out = subsample_split(split, 10, seed=0)
    assert out.n == 2


def test_is_adapter_dir(tmp_path: Path) -> None:
    adapter = tmp_path / "lora"
    adapter.mkdir()
    assert not is_adapter_dir(adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    assert is_adapter_dir(adapter)
    resolve_adapter_dir(adapter)


def test_discover_adapters_per_round(tmp_path: Path) -> None:
    agents = tmp_path / "agents" / "clone-0" / "round-01"
    agents.mkdir(parents=True)
    (agents / "adapter_model.safetensors").write_bytes(b"")
    found = discover_adapters(tmp_path)
    assert found == [AdapterRef(agent="clone-0", round=1, path=agents)]


def _mk_adapter(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_model.safetensors").write_bytes(b"")
    return path


def test_latest_round_dir_falls_back_to_newest_earlier_round(tmp_path: Path) -> None:
    """Cumulative adapters: a round the agent skipped resolves to its last trained round."""
    agent = tmp_path / "agents" / "jbb"
    r0 = _mk_adapter(agent / "round-00")
    r3 = _mk_adapter(agent / "round-03")
    (agent / "round-05").mkdir()                       # empty dir: not an adapter
    assert latest_round_dir(agent, 3) == r3            # exact hit
    assert latest_round_dir(agent, 11) == r3           # skipped rounds -> newest earlier
    assert latest_round_dir(agent, 5) == r3            # empty round dir is ignored
    assert latest_round_dir(agent, 2) == r0
    assert latest_round_dir(agent, 0) == r0
    assert latest_round_dir(tmp_path / "agents" / "missing", 4) is None
    _mk_adapter(tmp_path / "agents" / "late" / "round-07")
    assert latest_round_dir(tmp_path / "agents" / "late", 4) is None   # nothing at or before 4


def test_discover_adapters_requested_rounds_keep_every_agent(tmp_path: Path) -> None:
    """With explicit rounds, an agent missing a round is reported with the adapter in effect."""
    a0 = _mk_adapter(tmp_path / "agents" / "a" / "round-00")
    a1 = _mk_adapter(tmp_path / "agents" / "a" / "round-01")
    b0 = _mk_adapter(tmp_path / "agents" / "b" / "round-00")   # b never trained again
    found = discover_adapters(tmp_path, rounds=[0, 1])
    assert found == [
        AdapterRef(agent="a", round=0, path=a0),
        AdapterRef(agent="a", round=1, path=a1),
        AdapterRef(agent="b", round=0, path=b0),
        AdapterRef(agent="b", round=1, path=b0),          # stands in for the missing round-01
    ]
    # Without a round filter, only what is on disk is listed.
    assert [(r.agent, r.round) for r in discover_adapters(tmp_path)] == [("a", 0), ("a", 1), ("b", 0)]


def test_resolve_merge_adapters_uses_adapter_in_effect(tmp_path: Path) -> None:
    from infl_ens.evaluation.routing_eval import resolve_merge_adapters

    _mk_adapter(tmp_path / "agents" / "jbb_behaviors" / "round-09")
    _mk_adapter(tmp_path / "agents" / "halueval" / "round-11")
    names, name_map = resolve_merge_adapters(
        tmp_path, 11, ["pair-0", "pair-1"],
        aliases={"pair-0": "jbb_behaviors", "pair-1": "halueval"},
    )
    assert names == ["jbb_behaviors", "halueval"]
    assert name_map == {"pair-0": "jbb_behaviors", "pair-1": "halueval"}
    import pytest

    with pytest.raises(FileNotFoundError, match="at or before round 5"):
        resolve_merge_adapters(tmp_path, 5, ["pair-0"], aliases={"pair-0": "jbb_behaviors"})


def test_write_eval_report(tmp_path: Path) -> None:
    results = [
        BenchmarkEvalResult(
            benchmark="beavertails",
            axis_name="harm",
            mean_nll=1.5,
            n_examples=10,
            n_tokens=100,
            adapter_dir=str(tmp_path / "a"),
        ),
    ]
    path = write_eval_report(results, tmp_path, meta={"task": "adapter_eval"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["meta"]["task"] == "adapter_eval"
    assert len(payload["results"]) == 1


def test_eval_job_config_from_mapping() -> None:
    job = EvalJobConfig.from_mapping({
        "task": "adapter_eval",
        "adapter_dir": "/tmp/a",
        "benchmarks": [{"kind": "beavertails", "path": "data/x"}],
        "eval": {"max_eval_records": 64},
    })
    cfg = job.to_adapter_eval_config()
    assert cfg.max_eval_records == 64
