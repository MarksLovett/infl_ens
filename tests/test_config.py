"""Tests for the layered YAML loader in :mod:`infl_ens.config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from infl_ens.config import (
    ConfigError,
    apply_overrides,
    deep_merge,
    load_config,
    resolve_includes,
    validate_config,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_deep_merge_recurses_mappings_and_replaces_lists() -> None:
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    override = {"a": {"y": [3], "z": 2}, "c": None}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": [3], "z": 2}, "b": 1, "c": None}
    # Inputs are untouched and the result does not alias them.
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    merged["a"]["y"].append(9)
    assert override["a"]["y"] == [3]


def test_includes_merge_in_order_and_own_keys_win(tmp_path: Path) -> None:
    _write(tmp_path / "frag" / "a.yaml", "x: 1\nnested: {p: a, q: a}\n")
    _write(tmp_path / "frag" / "b.yaml", "x: 2\nnested: {q: b}\n")
    top = _write(
        tmp_path / "top.yaml",
        "includes: [frag/a.yaml, frag/b.yaml]\nnested: {r: top}\n",
    )
    cfg = load_config(top, validate=False)
    assert cfg == {"x": 2, "nested": {"p": "a", "q": "b", "r": "top"}}
    assert "includes" not in cfg


def test_nested_include_is_relative_to_including_file(tmp_path: Path) -> None:
    _write(tmp_path / "enc" / "e.yaml", "encoder: {model_name: m}\n")
    _write(tmp_path / "ts" / "t.yaml", "includes: [../enc/e.yaml]\ntrait_space: {n_grid: 3}\n")
    top = _write(tmp_path / "arms" / "arm.yaml", "includes: [../ts/t.yaml]\nseed: 1\n")
    cfg = load_config(top)
    assert cfg == {"encoder": {"model_name": "m"}, "trait_space": {"n_grid": 3}, "seed": 1}


def test_include_cycle_raises(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "includes: [b.yaml]\n")
    _write(tmp_path / "b.yaml", "includes: [a.yaml]\n")
    with pytest.raises(ConfigError, match="include cycle"):
        load_config(tmp_path / "a.yaml", validate=False)


def test_missing_include_raises(tmp_path: Path) -> None:
    top = _write(tmp_path / "a.yaml", "includes: [nope.yaml]\n")
    with pytest.raises(ConfigError, match="not found"):
        load_config(top, validate=False)


def test_resolve_includes_rejects_non_list(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a list"):
        resolve_includes({"includes": "a.yaml"}, tmp_path)


def test_string_overrides_parse_json_and_create_paths() -> None:
    cfg = {"closed_loop": {"n_rounds": 12}, "data_split": {"seed": 0}}
    apply_overrides(
        cfg,
        [
            "closed_loop.n_rounds=2",
            "data_split=null",
            "closed_loop.theory_gradient.n_steps=500",
            'eval.partitions=["val"]',
            "output_dir=results/x",
        ],
    )
    assert cfg["closed_loop"]["n_rounds"] == 2
    assert cfg["data_split"] is None
    assert cfg["closed_loop"]["theory_gradient"] == {"n_steps": 500}
    assert cfg["eval"] == {"partitions": ["val"]}
    assert cfg["output_dir"] == "results/x"


def test_mapping_overrides_use_values_verbatim() -> None:
    cfg: dict = {}
    apply_overrides(cfg, {"closed_loop.batch_size": 64, "eval": None})
    assert cfg == {"closed_loop": {"batch_size": 64}, "eval": None}


def test_override_without_equals_raises() -> None:
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        apply_overrides({}, ["oops"])


def test_unknown_closed_loop_key_names_key_and_source() -> None:
    cfg = {"task": "closed_loop", "closed_loop": {"theory_pre": {}}}
    with pytest.raises(ConfigError) as info:
        validate_config(cfg, source="configs/arms/x.yaml")
    msg = str(info.value)
    assert "closed_loop" in msg
    assert "'theory_pre'" in msg
    assert "configs/arms/x.yaml" in msg


def test_unknown_top_level_key_and_task_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown key 'training'"):
        validate_config({"training": {}})
    with pytest.raises(ConfigError, match="unknown task"):
        validate_config({"task": "router_training"})


def test_benchmark_entries_are_checked_per_kind() -> None:
    validate_config({"benchmarks": [{"kind": "halueval", "path": "p", "tasks": ["qa"]}]})
    with pytest.raises(ConfigError, match="unknown benchmark kind"):
        validate_config({"benchmarks": [{"kind": "toxicchat", "path": "p"}]})
    with pytest.raises(ConfigError, match="unknown key 'tasks'"):
        validate_config({"benchmarks": [{"kind": "beavertails", "path": "p", "tasks": []}]})
    with pytest.raises(ConfigError, match="missing required key 'path'"):
        validate_config({"benchmarks": [{"kind": "beavertails"}]})


def test_agents_accept_list_or_pairs_mapping() -> None:
    validate_config({"agents": [{"name": "clone-0"}, {"name": "clone-1", "calibration": "harm"}]})
    validate_config({"agents": {"pairs_from_axes": True, "name_prefix": "clone"}})
    with pytest.raises(ConfigError, match="pairs_from_axes"):
        validate_config({"agents": {"name_prefix": "clone"}})
    with pytest.raises(ConfigError, match="missing required key 'name'"):
        validate_config({"agents": [{"calibration": "harm"}]})


def test_null_blocks_are_allowed() -> None:
    validate_config({"data_split": None, "eval": None, "closed_loop": {"n_rounds": 2}})


def test_validation_does_not_mutate_trait_space() -> None:
    ts = {"encoder": "some/model", "n_grid": 3, "mode_alignment_weights": {"a": 1.0}}
    cfg = {"trait_space": ts, "encoder": {"model_name": "some/model"}}
    snapshot = {"trait_space": dict(ts), "encoder": {"model_name": "some/model"}}
    validate_config(cfg)
    assert cfg == snapshot
    assert cfg["trait_space"] is ts
