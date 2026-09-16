"""Offline tests for the ``modula_res`` task (MoDULA-Res-style baseline arms).

Covers the config surface, the two domain partitions (benchmark labels vs.
game-logged pairs), universal-adapter resolution, the source router copy
and aliasing, the stubbed end-to-end task (history / summary / resolved
config), and the evaluation-side merge of the universal adapter.
No model is loaded.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import infl_ens.training.tasks as tasks_mod
from infl_ens.config import KNOWN_TASKS, MODULA_RES_KEYS, TOP_LEVEL_KEYS, load_config
from infl_ens.training.modula_res import (
    DomainBatch,
    SourceRouterBlocks,
    dominant_axis_by_merge_group,
    history_domain_batches,
    label_domain_batches,
    pair_to_benchmark_aliases,
    resolve_universal_adapter_dir,
    round_batch_indices,
    source_router_blocks,
    train_modula_res,
)

BENCHES = ["b0", "b1", "b2"]
PAIRS = ["pair-0", "pair-1", "pair-2"]
CLONES = [f"clone-{i}" for i in range(6)]


# ---------------------------------------------------------------------------
# Synthetic source run
# ---------------------------------------------------------------------------


def _train_partition(n_per_bench: int = 4) -> tuple[list[str], list[str], list[str]]:
    prompts, responses, labels = [], [], []
    for b in BENCHES:
        for i in range(n_per_bench):
            prompts.append(f"{b} prompt {i}")
            responses.append(f"{b} answer {i}")
            labels.append(b)
    return prompts, responses, labels


def _source_history(train_prompts: list[str], train_responses: list[str]) -> list[dict[str, Any]]:
    """Two rounds covering every train row exactly once; soft top-2 pairs."""
    n = len(train_prompts)
    rows_by_round = [list(range(0, n, 2)), list(range(1, n, 2))]
    history = []
    for r, rows in enumerate(rows_by_round):
        # Each prompt is routed to two pairs (soft top-k = 2).
        per_pair: dict[str, list[int]] = {p: [] for p in PAIRS}
        for j, row in enumerate(rows):
            per_pair[PAIRS[j % 3]].append(row)
            per_pair[PAIRS[(j + 1) % 3]].append(row)
        rec = {
            "round": r,
            "routing_mode": "soft",
            "soft_loss": "weighted",
            "positions": {c: [float(r), float(i)] for i, c in enumerate(CLONES)},
            "pair_members": {PAIRS[k]: [CLONES[2 * k], CLONES[2 * k + 1]] for k in range(3)},
            "agent_batch_indices": per_pair,
            "agent_prompts": {p: [train_prompts[i] for i in idx] for p, idx in per_pair.items()},
            "agent_responses": {p: [train_responses[i] for i in idx] for p, idx in per_pair.items()},
            "agent_sample_weights": {p: [0.5 + 0.1 * k for k in range(len(idx))] for p, idx in per_pair.items()},
            "batch_prompts": [train_prompts[i] for i in rows],
            "batch_responses": [train_responses[i] for i in rows],
        }
        if r == 0:
            rec["theory_init"] = {
                # The paired theory init keys this by clone membership
                # (``pair_<a>_<b>``), not by merge-group name; one entry is
                # written in reversed member order on purpose.
                # pair-0 -> axis 2, pair-1 -> axis 0, pair-2 -> axis 1
                "pair_dominant_axis": {
                    f"pair_{CLONES[0]}_{CLONES[1]}": 2,
                    f"pair_{CLONES[3]}_{CLONES[2]}": 0,
                    f"pair_{CLONES[4]}_{CLONES[5]}": 1,
                },
                "sft_merge_groups_resolved": [
                    {"train_as": PAIRS[k], "names": [CLONES[2 * k], CLONES[2 * k + 1]]} for k in range(3)
                ],
            }
        history.append(rec)
    return history


def _write_source_run(tmp_path: Path, history: list[dict[str, Any]]) -> Path:
    run = tmp_path / "source"
    run.mkdir()
    (run / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (run / "resolved_config.yaml").write_text(yaml.safe_dump({
        "task": "closed_loop",
        "agents": [{"name": c} for c in CLONES],
        "policy": "gaussian",
        "sigma_mode": "absolute",
        "sigma": 0.3,
        "closed_loop": {
            "sft_merge_groups": [
                {"train_as": PAIRS[k], "names": [CLONES[2 * k], CLONES[2 * k + 1]]} for k in range(3)
            ],
        },
    }), encoding="utf-8")
    return run


def _write_universal_run(tmp_path: Path, *, final_round: int = 1) -> Path:
    run = tmp_path / "generalist"
    for r in range(final_round + 1):
        d = run / "agents" / "pooled-baseline" / f"round-{r:02d}"
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
        (d / "adapter_model.safetensors").write_bytes(b"")
    (run / "history.json").write_text(
        json.dumps([{"round": r, "positions": {}} for r in range(final_round + 1)]), encoding="utf-8",
    )
    return run


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_modula_res_is_a_known_task_with_its_own_block() -> None:
    assert "modula_res" in KNOWN_TASKS
    assert "modula_res" in TOP_LEVEL_KEYS
    assert {"domain_source", "universal_run_dir", "universal_agent", "universal_round",
            "save_per_round", "rounds", "merge_aliases", "universal_adapter_dir"} == set(MODULA_RES_KEYS)
    assert tasks_mod.TASKS["modula_res"] is tasks_mod.run_modula_res


def test_unknown_modula_res_key_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "arm.yaml"
    p.write_text("task: modula_res\noutput_dir: r\nmodula_res: {bogus: 1}\n", encoding="utf-8")
    with pytest.raises(Exception, match="bogus"):
        load_config(p)


# ---------------------------------------------------------------------------
# Domain batches
# ---------------------------------------------------------------------------


def test_round_batch_indices_is_the_sorted_union() -> None:
    rec = {"agent_batch_indices": {"pair-0": [5, 1], "pair-1": [1, 3]}}
    assert round_batch_indices(rec) == [1, 3, 5]
    assert round_batch_indices({"round": 0}) is None


def test_label_batches_partition_each_round_by_benchmark_and_cover_train_once() -> None:
    prompts, responses, labels = _train_partition()
    history = _source_history(prompts, responses)
    seen: list[str] = []
    for rec in history:
        batches = label_domain_batches(
            rec, train_prompts=prompts, train_responses=responses,
            train_labels=labels, domain_names=BENCHES,
        )
        assert set(batches) == set(BENCHES)
        rows = round_batch_indices(rec)
        assert rows is not None
        for b in BENCHES:
            expected = [prompts[i] for i in rows if labels[i] == b]
            assert batches[b].prompts == expected
            assert batches[b].responses == [responses[i] for i in rows if labels[i] == b]
            assert batches[b].sample_weights is None
            seen += batches[b].prompts
    assert sorted(seen) == sorted(prompts)          # every train row exactly once
    assert len(seen) == len(set(seen))


def test_label_batches_fall_back_to_batch_prompts_when_indices_missing() -> None:
    prompts, responses, labels = _train_partition()
    rec = {"round": 0, "batch_prompts": [prompts[3], prompts[0], prompts[9]]}
    batches = label_domain_batches(
        rec, train_prompts=prompts, train_responses=responses, train_labels=labels, domain_names=BENCHES,
    )
    assert batches["b0"].prompts == [prompts[0], prompts[3]]
    assert batches["b1"].n == 0
    assert batches["b2"].prompts == [prompts[9]]
    with pytest.raises(ValueError, match="not in the train partition"):
        label_domain_batches(
            {"round": 1, "batch_prompts": ["nope"]}, train_prompts=prompts,
            train_responses=responses, train_labels=labels, domain_names=BENCHES,
        )


def test_history_batches_replay_pair_prompts_and_weights() -> None:
    prompts, responses, _ = _train_partition()
    history = _source_history(prompts, responses)
    rec = history[0]
    batches = history_domain_batches(rec, domain_names=PAIRS)
    for p in PAIRS:
        assert batches[p].prompts == rec["agent_prompts"][p]
        assert batches[p].responses == rec["agent_responses"][p]
        assert batches[p].sample_weights == rec["agent_sample_weights"][p]
    # Unit-loss source rounds carry no weights.
    unit = dict(rec, soft_loss="unit")
    assert history_domain_batches(unit, domain_names=PAIRS)["pair-0"].sample_weights is None


# ---------------------------------------------------------------------------
# Universal adapter + source router copy
# ---------------------------------------------------------------------------


def test_resolve_universal_adapter_dir(tmp_path: Path) -> None:
    assert resolve_universal_adapter_dir({"modula_res": {"universal_run_dir": None}}) is None
    assert resolve_universal_adapter_dir({}) is None
    run = _write_universal_run(tmp_path, final_round=2)
    cfg = {"modula_res": {"universal_run_dir": "generalist", "universal_agent": "pooled-baseline"}}
    assert resolve_universal_adapter_dir(cfg, repo_root=tmp_path) == run / "agents" / "pooled-baseline" / "round-02"
    cfg["modula_res"]["universal_round"] = 1
    assert resolve_universal_adapter_dir(cfg, repo_root=tmp_path).name == "round-01"
    cfg["modula_res"]["universal_round"] = 7
    with pytest.raises(FileNotFoundError):
        resolve_universal_adapter_dir(cfg, repo_root=tmp_path)
    cfg["modula_res"]["universal_round"] = "latest"
    with pytest.raises(ValueError, match="universal_round"):
        resolve_universal_adapter_dir(cfg, repo_root=tmp_path)


def test_dominant_axis_by_merge_group_rekeys_membership_keys() -> None:
    """Theory-init ``pair_<a>_<b>`` keys map onto ``train_as`` in any member order."""
    groups = [
        {"train_as": "pair-0", "names": ["clone-5", "clone-10"]},
        {"train_as": "pair-1", "names": ["clone-3", "clone-12"]},
        {"train_as": "pair-2", "names": ["clone-8", "clone-9"]},
    ]
    raw = {
        "pair_clone-5_clone-10": 4,   # same order as the group
        "pair_clone-12_clone-3": 1,   # reversed order
        "pair-2": 6,                  # already keyed by merge-group name
    }
    assert dominant_axis_by_merge_group(raw, groups) == {"pair-0": 4, "pair-1": 1, "pair-2": 6}
    # Unmatched groups are simply absent; an empty input stays empty.
    assert dominant_axis_by_merge_group({"pair_clone-1_clone-2": 0}, groups) == {}
    assert dominant_axis_by_merge_group({}, groups) == {}
    # A partial match is surfaced by the alias step rather than silently
    # falling back to in-order matching.
    partial = dominant_axis_by_merge_group({"pair_clone-5_clone-10": 0}, groups)
    blocks = SourceRouterBlocks(agents=[], merge_groups=groups, pair_dominant_axis=partial)
    with pytest.raises(ValueError, match="lacks 'pair-1'"):
        pair_to_benchmark_aliases(blocks, ["b0", "b1", "b2"])


def test_source_router_blocks_and_aliases(tmp_path: Path) -> None:
    prompts, responses, _ = _train_partition()
    history = _source_history(prompts, responses)
    run = _write_source_run(tmp_path, history)
    blocks = source_router_blocks(run, history)
    assert [a["name"] for a in blocks.agents] == CLONES
    assert blocks.pair_names == PAIRS
    assert blocks.router_keys == {"policy": "gaussian", "sigma_mode": "absolute", "sigma": 0.3}
    assert blocks.pair_dominant_axis == {"pair-0": 2, "pair-1": 0, "pair-2": 1}
    aliases = pair_to_benchmark_aliases(blocks, BENCHES)
    assert aliases == {"pair-0": "b2", "pair-1": "b0", "pair-2": "b1"}
    # Without the theory block the fallback is history pair_members + order.
    (run / "resolved_config.yaml").unlink()
    bare = [dict(history[0]), history[1]]
    bare[0].pop("theory_init")
    blocks2 = source_router_blocks(run, bare)
    assert blocks2.pair_names == PAIRS and blocks2.pair_dominant_axis == {}
    assert pair_to_benchmark_aliases(blocks2, BENCHES) == dict(zip(PAIRS, BENCHES))
    with pytest.raises(ValueError, match="one pair per benchmark"):
        pair_to_benchmark_aliases(blocks, BENCHES[:2])
    with pytest.raises(ValueError, match="one-to-one"):
        pair_to_benchmark_aliases(
            SourceRouterBlocks(agents=[], merge_groups=blocks.merge_groups,
                               pair_dominant_axis={"pair-0": 0, "pair-1": 0, "pair-2": 1}),
            BENCHES,
        )


# ---------------------------------------------------------------------------
# Training loop (stubbed trainer)
# ---------------------------------------------------------------------------


class _FakeSFT:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, agent, prompts, *, responses, cfg, eval_prompts, project, blend,
                 out_dir_override, sample_weights, skip_position_update, frozen_base_adapter_dir):
        self.calls.append({
            "agent": agent.name, "prompts": list(prompts), "responses": responses,
            "out": out_dir_override, "weights": sample_weights,
            "skip": skip_position_update, "frozen": frozen_base_adapter_dir,
        })
        Path(out_dir_override).mkdir(parents=True, exist_ok=True)
        return {"output_dir": out_dir_override, "n_train": len(prompts), "train_loss": 0.1,
                "loaded_prior_lora": None, "frozen_base_adapter": frozen_base_adapter_dir}


def test_train_modula_res_calls_trainer_per_nonempty_cell(tmp_path: Path) -> None:
    prompts, responses, labels = _train_partition()
    history = _source_history(prompts, responses)
    fake = _FakeSFT()
    universal = tmp_path / "U"
    summaries = train_modula_res(
        history, domain_names=BENCHES,
        batches_for_round=lambda rec: label_domain_batches(
            rec, train_prompts=prompts, train_responses=responses, train_labels=labels, domain_names=BENCHES,
        ),
        sft_cfg=object(), project=None, output_dir=tmp_path / "agents",
        universal_adapter_dir=universal,
        initial_positions={b: [0.0, 0.0] for b in BENCHES},
        sft_train=fake,
    )
    assert [s["round"] for s in summaries] == [0, 1]
    assert len(fake.calls) == 6
    assert all(c["frozen"] == str(universal) and c["skip"] is True for c in fake.calls)
    assert fake.calls[0]["out"] == str(tmp_path / "agents" / "b0" / "round-00")
    assert fake.calls[3]["out"] == str(tmp_path / "agents" / "b0" / "round-01")
    assert summaries[0]["n_train_total"] == 6

    # From scratch: frozen adapter is None; empty cells are skipped.
    fake2 = _FakeSFT()
    empty_b1 = lambda rec: {  # noqa: E731
        "b0": DomainBatch(["p"], [None]), "b1": DomainBatch([], []), "b2": DomainBatch(["q"], ["a"]),
    }
    summaries2 = train_modula_res(
        history, domain_names=BENCHES, batches_for_round=empty_b1, sft_cfg=object(), project=None,
        output_dir=tmp_path / "agents2", universal_adapter_dir=None,
        initial_positions={b: [0.0, 0.0] for b in BENCHES}, rounds=[1], sft_train=fake2,
    )
    assert [c["agent"] for c in fake2.calls] == ["b0", "b2"]
    assert all(c["frozen"] is None for c in fake2.calls)
    assert fake2.calls[0]["responses"] is None and fake2.calls[1]["responses"] == ["a"]
    assert summaries2[0]["experts"]["b1"] == {"n_train": 0, "output_dir": None, "skipped": True}


# ---------------------------------------------------------------------------
# End-to-end task with stubs
# ---------------------------------------------------------------------------


class _Split:
    def __init__(self, name: str) -> None:
        self.name = name


class _Space:
    mean = np.zeros(2)

    @staticmethod
    def project(prompts):
        return np.zeros((len(prompts), 2))


def _stub_task_environment(monkeypatch: pytest.MonkeyPatch, train: tuple) -> _FakeSFT:
    prompts, responses, labels = train
    monkeypatch.setattr(tasks_mod, "load_splits", lambda cfg: [_Split(b) for b in BENCHES])
    monkeypatch.setattr(tasks_mod, "make_trait_space", lambda cfg, splits: _Space())
    import infl_ens.data.splits as splits_mod
    monkeypatch.setattr(splits_mod, "load_split_manifest", lambda p: "manifest")
    monkeypatch.setattr(
        splits_mod, "flatten_partition_prompts", lambda s, m, part: (prompts, responses, labels),
    )
    import infl_ens.training.sft_training as sft_mod
    fake = _FakeSFT()
    monkeypatch.setattr(sft_mod, "sft_train_agent", fake)
    return fake


def _task_cfg(tmp_path: Path, source: Path, *, domain_source: str, universal: str | None) -> dict[str, Any]:
    out = tmp_path / "out"
    return {
        "task": "modula_res",
        "seed": 0,
        "repo_root": str(tmp_path),
        "history_path": str(source / "history.json"),
        "output_dir": str(out),
        "benchmarks": [{"kind": b, "path": f"{b}.jsonl"} for b in BENCHES],
        "data_split": {"manifest": "split.json"},
        "sft": {"base_model": "org/m", "output_dir": str(out / "agents"), "lora_r": 8},
        "modula_res": {"domain_source": domain_source, "universal_run_dir": universal},
        "eval": {"after_training": False},
    }


@pytest.mark.parametrize("domain_source", ["labels", "history"])
def test_run_modula_res_writes_history_summary_and_resolved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, domain_source: str,
) -> None:
    train = _train_partition()
    history = _source_history(train[0], train[1])
    source = _write_source_run(tmp_path, history)
    _write_universal_run(tmp_path, final_round=1)
    fake = _stub_task_environment(monkeypatch, train)
    cfg = _task_cfg(tmp_path, source, domain_source=domain_source, universal="generalist")

    assert tasks_mod.run_modula_res(cfg) == 0
    out = Path(cfg["output_dir"])
    experts = BENCHES if domain_source == "labels" else PAIRS
    universal = tmp_path / "generalist" / "agents" / "pooled-baseline" / "round-01"

    # Every trainer call merges the resolved U underneath and never moves positions.
    assert fake.calls and all(c["frozen"] == str(universal) and c["skip"] for c in fake.calls)
    assert sorted({c["agent"] for c in fake.calls}) == sorted(experts)
    outs = [c["out"] for c in fake.calls if c["agent"] == experts[0]]
    assert outs == [str(out / "agents" / experts[0] / f"round-{r:02d}") for r in (0, 1)]
    if domain_source == "history":
        assert all(c["weights"] is not None for c in fake.calls)   # weighted source
    else:
        assert all(c["weights"] is None for c in fake.calls)

    new_hist = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert [r["round"] for r in new_hist] == [0, 1]
    for src, rec in zip(history, new_hist):
        assert rec["positions"] == src["positions"]                # copied, not learned
        assert set(rec["agent_prompts"]) == set(experts)
        assert rec["experts"] == experts
    assert new_hist[0]["pair_members"] == history[0]["pair_members"]

    summary = json.loads((out / "modula_res_summary.json").read_text(encoding="utf-8"))
    assert summary["domain_source"] == domain_source
    assert summary["experts"] == experts
    assert summary["universal_adapter_dir"] == str(universal)
    assert summary["n_rounds"] == 2

    resolved = yaml.safe_load((out / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert [a["name"] for a in resolved["agents"]] == CLONES
    assert [g["train_as"] for g in resolved["closed_loop"]["sft_merge_groups"]] == PAIRS
    assert resolved["policy"] == "gaussian" and resolved["sigma"] == 0.3
    assert resolved["modula_res"]["universal_adapter_dir"] == str(universal)
    if domain_source == "labels":
        assert resolved["modula_res"]["merge_aliases"] == {"pair-0": "b2", "pair-1": "b0", "pair-2": "b1"}
    else:
        assert resolved["modula_res"]["merge_aliases"] == {}


def test_run_modula_res_from_scratch_has_no_universal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train = _train_partition()
    history = _source_history(train[0], train[1])
    source = _write_source_run(tmp_path, history)
    fake = _stub_task_environment(monkeypatch, train)
    cfg = _task_cfg(tmp_path, source, domain_source="labels", universal=None)
    assert tasks_mod.run_modula_res(cfg) == 0
    assert fake.calls and all(c["frozen"] is None for c in fake.calls)
    resolved = yaml.safe_load((Path(cfg["output_dir"]) / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved["modula_res"]["universal_adapter_dir"] is None
    from infl_ens.evaluation.evaluate import universal_adapter_dir_from_config
    assert universal_adapter_dir_from_config(resolved) is None
    resolved["modula_res"]["universal_adapter_dir"] = "/u"
    assert universal_adapter_dir_from_config(resolved) == "/u"
    assert universal_adapter_dir_from_config({"task": "closed_loop", "modula_res": {"universal_adapter_dir": "/u"}}) is None


def test_run_modula_res_rejects_unknown_domain_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="domain_source"):
        tasks_mod.run_modula_res({"modula_res": {"domain_source": "oracle"}, "history_path": "h"})


# ---------------------------------------------------------------------------
# Evaluation-side merge (fake torch / transformers / peft)
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype
        self.merged: list[str] = []

    def to(self, target):
        if isinstance(target, str) and target.startswith("dtype:"):
            self.dtype = target
        return self

    def eval(self):
        return self


class _FakePeftModel:
    merges: list[tuple[str, str]] = []

    def __init__(self, base, path):
        self.base, self.path = base, path

    @classmethod
    def from_pretrained(cls, base, path):
        return cls(base, path)

    def merge_and_unload(self):
        _FakePeftModel.merges.append((self.base.dtype, self.path))
        self.base.merged.append(self.path)
        return self.base

    def eval(self):
        return self


def _install_fake_ml_stack(monkeypatch: pytest.MonkeyPatch, loaded: list[str]) -> None:
    torch = types.ModuleType("torch")
    torch.float32 = "dtype:float32"
    torch.bfloat16 = "dtype:bfloat16"
    torch.device = lambda name: types.SimpleNamespace(type=name)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)
    transformers = types.ModuleType("transformers")

    class _AutoModel:
        @staticmethod
        def from_pretrained(name, dtype=None, torch_dtype=None):
            loaded.append(dtype or torch_dtype)
            return _FakeModel(dtype or torch_dtype)

    class _AutoTok:
        @staticmethod
        def from_pretrained(name):
            return types.SimpleNamespace(pad_token="x", eos_token="x")

    transformers.AutoModelForCausalLM = _AutoModel
    transformers.AutoTokenizer = _AutoTok
    peft = types.ModuleType("peft")
    peft.PeftModel = _FakePeftModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    _FakePeftModel.merges = []


def test_load_base_causal_lm_merges_universal_once_in_fp32(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from infl_ens.evaluation.adapters import load_adapter_model, load_base_causal_lm

    u = tmp_path / "U"
    u.mkdir()
    (u / "adapter_config.json").write_text("{}", encoding="utf-8")
    (u / "adapter_model.safetensors").write_bytes(b"")
    d = tmp_path / "D"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"")

    loaded: list[str] = []
    _install_fake_ml_stack(monkeypatch, loaded)

    model, _tok, _dev = load_base_causal_lm("org/m", universal_adapter_dir=u)
    assert loaded == ["dtype:float32"]                     # fp32 load for the merge
    assert _FakePeftModel.merges == [("dtype:float32", str(u))]
    assert model.dtype == "dtype:float32"                  # cpu inference dtype
    wrapped = load_adapter_model(model, d)
    wrapped2 = load_adapter_model(model, d)
    assert _FakePeftModel.merges == [("dtype:float32", str(u))]   # domain adapters never merge
    assert wrapped.path == str(d) and wrapped2.base is model

    loaded.clear()
    plain, _, _ = load_base_causal_lm("org/m")
    assert loaded == ["dtype:float32"] and plain.merged == []
