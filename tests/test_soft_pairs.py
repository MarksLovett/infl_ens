"""Offline tests for soft (dense) routing over co-located agent pairs.

Covers the pieces that let a closed-loop run put ``2L`` clones at ``L``
co-located theory-Nash positions, route each query softly over the *pairs*,
and train exactly one LoRA per pair at the pair's shared position:

- :func:`infl_ens.inflgame.router.allocation.group_allocation_weights` and
  the equivalence between a group of co-located clones and one agent at the
  shared position,
- :func:`infl_ens.training.merge_training.soft_pair_assignments` /
  :func:`~infl_ens.training.merge_training.soft_pair_position_target` /
  :func:`~infl_ens.training.merge_training.merge_groups_from_theory_pairs`,
- :func:`infl_ens.utils.agent_init.resolve_agent_entries` and the
  ``nearest`` pairing rule,
- de-duplication of soft rounds in
  :func:`infl_ens.training.baseline_replay.pooled_batch_from_round`,
- the ``routing_mode``/``sft_merge_groups`` validation matrix, and
- an end-to-end ``closed_loop`` round with a fake trait space and a stubbed
  SFT trainer, asserting one adapter and one shared position per pair.

Everything here is numpy-only: no ``torch``, no ``trl``, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from infl_ens.data.trait_space import TraitSpace, position_from_corpus
from infl_ens.inflgame.router.allocation import (
    allocation_weights,
    group_allocation_weights,
)
from infl_ens.training.baseline_replay import pooled_batch_from_round
from infl_ens.training.merge_training import (
    group_index_for_merge_groups,
    merge_groups_from_theory_pairs,
    parse_sft_merge_groups,
    soft_pair_assignments,
    soft_pair_position_target,
)
from infl_ens.utils.agent_init import (
    co_locate_theory_pairs,
    nearest_neighbour_pair_indices,
    pair_indices_for_method,
    resolve_agent_entries,
)

# ---------------------------------------------------------------------------
# group allocation
# ---------------------------------------------------------------------------


def test_group_allocation_sums_rows() -> None:
    """Rows are summed per group and query columns still sum to one."""
    G = np.array([
        [0.4, 0.1],
        [0.2, 0.3],
        [0.1, 0.5],
        [0.3, 0.1],
    ])
    out = group_allocation_weights(G, np.array([0, 0, 1, 1]), 2)
    assert out.shape == (2, 2)
    assert np.allclose(out, [[0.6, 0.4], [0.4, 0.6]])
    assert np.allclose(out.sum(axis=0), 1.0)


def test_group_allocation_validates_arguments() -> None:
    """Bad shapes and out-of-range group ids raise."""
    G = np.ones((4, 3)) / 4.0
    with pytest.raises(ValueError):
        group_allocation_weights(G, np.array([0, 0, 1]), 2)
    with pytest.raises(ValueError):
        group_allocation_weights(G, np.array([0, 0, 1, 2]), 2)
    with pytest.raises(ValueError):
        group_allocation_weights(np.ones(4), np.zeros(4, dtype=int), 1)
    with pytest.raises(ValueError):
        group_allocation_weights(G, np.zeros(4, dtype=int), 0)


def test_group_g_equals_single_agent_g_at_shared_position() -> None:
    """Co-located equal-size groups reproduce the L-player allocation."""
    rng = np.random.default_rng(0)
    n_pairs, dim = 4, 3
    pair_pos = rng.random((n_pairs, dim))
    coords = rng.random((11, dim))
    cov = 0.25 ** 2 * np.eye(dim)

    clones = np.repeat(pair_pos, 2, axis=0)          # (2L, dim), co-located
    g_clone = allocation_weights(clones, coords, cov)
    g_group = group_allocation_weights(
        g_clone, np.repeat(np.arange(n_pairs), 2), n_pairs,
    )
    assert np.allclose(g_group, allocation_weights(pair_pos, coords, cov))


# ---------------------------------------------------------------------------
# soft pair assignments
# ---------------------------------------------------------------------------


def _pair_setup(n_pairs: int = 7, m: int = 40, seed: int = 3):
    """Build co-located clones plus a random batch of coordinates."""
    rng = np.random.default_rng(seed)
    pair_pos = rng.random((n_pairs, 2))
    coords = rng.random((m, 2))
    cov = 0.2 ** 2 * np.eye(2)
    clones = np.repeat(pair_pos, 2, axis=0)
    g_clone = allocation_weights(clones, coords, cov)
    group_index = np.repeat(np.arange(n_pairs), 2)
    return g_clone, group_index, pair_pos, coords, cov


def test_soft_pair_assignments_top_k_counts() -> None:
    """top_k groups per query, columns renormalised, idx/weights/mass aligned."""
    g_clone, group_index, _pos, _coords, _cov = _pair_setup()
    n_pairs, m = 7, 40
    W, idx, weights = soft_pair_assignments(g_clone, group_index, n_pairs, 2)

    assert W.shape == (n_pairs, m)
    assert np.all((W > 0).sum(axis=0) == 2)
    assert np.allclose(W.sum(axis=0), 1.0)
    assert sum(len(i) for i in idx) == 2 * m
    for p in range(n_pairs):
        assert weights[p].shape == idx[p].shape
        assert np.allclose(weights[p], W[p, idx[p]])


def test_soft_pair_assignments_full_k_is_dense_group_g() -> None:
    """top_k == n_groups keeps every pair on every query at its share."""
    g_clone, group_index, pair_pos, coords, cov = _pair_setup(n_pairs=5, m=13)
    n_pairs, m = 5, 13
    W, idx, _w = soft_pair_assignments(g_clone, group_index, n_pairs, n_pairs)

    assert np.allclose(W, allocation_weights(pair_pos, coords, cov))
    assert all(len(i) == m for i in idx)


def test_soft_pair_assignments_top_k_one_is_argmax() -> None:
    """top_k == 1 is a deterministic hard assignment to the winning pair."""
    g_clone, group_index, _pos, _coords, _cov = _pair_setup(n_pairs=6, m=20)
    G_group = group_allocation_weights(g_clone, group_index, 6)
    W, _idx, _w = soft_pair_assignments(g_clone, group_index, 6, 1)
    assert np.all((W > 0).sum(axis=0) == 1)
    assert np.array_equal(np.argmax(W, axis=0), np.argmax(G_group, axis=0))


# ---------------------------------------------------------------------------
# pair position target
# ---------------------------------------------------------------------------


def test_soft_pair_position_target_matches_position_from_corpus() -> None:
    """The pair target is the weighted centroid, without re-encoding."""
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.25], [0.9, 0.1]])
    texts = ["a", "b", "c", "d"]
    lookup = {t: coords[i] for i, t in enumerate(texts)}

    def project(queries: Sequence[str]) -> np.ndarray:
        return np.stack([lookup[q] for q in queries], axis=0)

    idx = np.array([0, 1, 3])
    weights = np.array([3.0, 1.0, 2.0])
    got = soft_pair_position_target(coords, idx, weights)
    want = position_from_corpus(
        [texts[i] for i in idx], project, scores=weights.tolist(),
    )
    assert np.allclose(got, want)


def test_soft_pair_position_target_edge_cases() -> None:
    """Empty selections raise; all-zero weights fall back to the mean."""
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        soft_pair_position_target(coords, np.array([], dtype=int), np.array([]))
    with pytest.raises(ValueError):
        soft_pair_position_target(coords, np.array([0, 1]), np.array([1.0]))
    with pytest.raises(ValueError):
        soft_pair_position_target(coords, np.array([0, 1]), np.array([-1.0, 2.0]))
    assert np.allclose(
        soft_pair_position_target(coords, np.array([0, 1]), np.array([0.0, 0.0])),
        [0.5, 0.5],
    )


# ---------------------------------------------------------------------------
# config-driven population and merge groups
# ---------------------------------------------------------------------------


def test_resolve_agent_entries() -> None:
    """A list passes through; a mapping expands to two clones per axis."""
    listed = [{"name": "clone-0"}, {"name": "clone-1"}]
    assert resolve_agent_entries(listed, 7) == listed

    expanded = resolve_agent_entries({"pairs_from_axes": True}, 7)
    assert [e["name"] for e in expanded] == [f"clone-{i}" for i in range(14)]

    prefixed = resolve_agent_entries(
        {"pairs_from_axes": True, "name_prefix": "sft"}, 2,
    )
    assert [e["name"] for e in prefixed] == ["sft-0", "sft-1", "sft-2", "sft-3"]

    with pytest.raises(ValueError):
        resolve_agent_entries({"name_prefix": "clone"}, 7)
    with pytest.raises(ValueError):
        resolve_agent_entries({"pairs_from_axes": True}, 0)
    with pytest.raises(ValueError):
        resolve_agent_entries("clone-0", 7)


def test_merge_groups_from_theory_pairs_partitions_agents() -> None:
    """Theory pairs become groups that parse_sft_merge_groups accepts."""
    names = [f"clone-{i}" for i in range(6)]
    paired = [["clone-1", "clone-4"], ["clone-0", "clone-3"], ["clone-2", "clone-5"]]
    groups = merge_groups_from_theory_pairs(paired)
    assert [g["train_as"] for g in groups] == ["pair-0", "pair-1", "pair-2"]
    assert groups[0]["names"] == ["clone-1", "clone-4"]

    parsed = parse_sft_merge_groups({"sft_merge_groups": groups}, names)
    assert parsed is not None and len(parsed) == 3
    assert np.array_equal(
        group_index_for_merge_groups(parsed, names),
        np.array([1, 0, 2, 1, 0, 2]),
    )

    assert merge_groups_from_theory_pairs(paired, name_prefix="merge")[0][
        "train_as"
    ] == "merge-0"
    with pytest.raises(ValueError):
        merge_groups_from_theory_pairs([["a", "b", "c"]])
    with pytest.raises(ValueError):
        merge_groups_from_theory_pairs([])
    with pytest.raises(ValueError):
        merge_groups_from_theory_pairs([["a", "b"], ["b", "c"]])


def test_parse_sft_merge_groups_rejects_unresolved_sentinel() -> None:
    """The 'from_init' sentinel must be resolved by the driver, not parsed."""
    with pytest.raises(ValueError, match="sentinel"):
        parse_sft_merge_groups(
            {"sft_merge_groups": "from_init"}, ["clone-0", "clone-1"],
        )


def test_group_index_requires_full_cover() -> None:
    """Every router agent must belong to some group."""
    with pytest.raises(ValueError):
        group_index_for_merge_groups(
            [("pair-0", ["clone-0", "clone-1"])],
            ["clone-0", "clone-1", "clone-2", "clone-3"],
        )
    with pytest.raises(ValueError):
        group_index_for_merge_groups(
            [("pair-0", ["clone-0", "ghost"])], ["clone-0", "clone-1"],
        )


# ---------------------------------------------------------------------------
# pairing rules
# ---------------------------------------------------------------------------


def test_nearest_pairing_recovers_interleaved_clusters() -> None:
    """Harm-adjacent pairing mispairs interleaved clusters; nearest does not."""
    # Two tight clusters whose harm coordinates interleave: sorting by harm
    # gives 0, 2, 1, 3 so adjacent pairing crosses the clusters.
    pos = np.array([
        [0.10, 0.90],
        [0.20, 0.10],
        [0.12, 0.90],
        [0.22, 0.10],
    ])
    harm_pairs = [sorted(p.tolist()) for p in pair_indices_for_method(pos)]
    assert harm_pairs == [[0, 2], [1, 3]]

    interleaved = np.array([
        [0.10, 0.90],
        [0.11, 0.10],
        [0.12, 0.90],
        [0.13, 0.10],
    ])
    assert [
        sorted(p.tolist()) for p in pair_indices_for_method(interleaved)
    ] == [[0, 1], [2, 3]]
    assert [
        sorted(p.tolist()) for p in nearest_neighbour_pair_indices(interleaved)
    ] == [[0, 2], [1, 3]]


def test_co_locate_theory_pairs_nearest_is_bitwise() -> None:
    """Partners are bit-identical after co-location under either rule."""
    pos = np.array([
        [0.10, 0.90],
        [0.11, 0.10],
        [0.12, 0.90],
        [0.13, 0.10],
    ])
    out = co_locate_theory_pairs(
        pos, [f"clone-{i}" for i in range(4)], pairing="nearest",
    )
    assert np.array_equal(out[0], out[2])
    assert np.array_equal(out[1], out[3])
    with pytest.raises(ValueError):
        co_locate_theory_pairs(pos, [f"clone-{i}" for i in range(4)], pairing="x")
    with pytest.raises(ValueError):
        nearest_neighbour_pair_indices(np.zeros((3, 2)))


def test_theory_gradient_paired_meta_reports_pre_colocation_spread() -> None:
    """Paired init records the informative pre-co-location distances."""
    from infl_ens.utils.agent_init import init_agents_theory_gradient_paired

    grid = np.array([[0.1, 0.2], [0.9, 0.8], [0.1, 0.8], [0.9, 0.2]])
    space = TraitSpace(
        grid=grid,
        weights=np.ones(4) / 4,
        project=lambda texts: np.zeros((len(texts), 2)),
    )
    cfg = {"agents": [{"name": f"clone-{i}"} for i in range(4)]}
    agents, meta = init_agents_theory_gradient_paired(
        cfg, space, sigma=0.3, seed=0,
        theory_cfg={"n_steps": 20, "pairing": "nearest"},
    )
    assert meta["pairing_method"] == "nearest"
    assert set(meta["within_pair_distance_first_pass"])
    assert set(meta["within_pair_distance_second_pass"])
    assert all(
        v == pytest.approx(0.0)
        for v in meta["within_pair_distance_after_pair"].values()
    )
    assert len(meta["pair_positions"]) == 2
    assert len(meta["pair_dominant_axis"]) == 2

    by_name = {a.name: a.position for a in agents}
    for members in meta["paired_harm_order"]:
        assert np.array_equal(by_name[members[0]], by_name[members[1]])


# ---------------------------------------------------------------------------
# pooled-baseline replay
# ---------------------------------------------------------------------------


def _soft_record() -> dict[str, Any]:
    """A soft top-2 round where two pairs share the same two prompts."""
    return {
        "routing_mode": "soft",
        "soft_top_k": 2,
        "agent_prompts": {"pair-0": ["a", "b"], "pair-1": ["a", "b"]},
        "agent_responses": {"pair-0": ["ra", "rb"], "pair-1": ["ra", "rb"]},
        "agent_batch_indices": {"pair-0": [0, 1], "pair-1": [0, 1]},
    }


def test_pooled_batch_dedupes_soft_rounds() -> None:
    """A soft top-k>1 round contributes each batch row exactly once."""
    record = _soft_record()
    prompts, responses = pooled_batch_from_round(record)
    assert prompts == ["a", "b"]
    assert responses == ["ra", "rb"]

    # Explicit batch log wins when present.
    with_batch = dict(record, batch_prompts=["b", "a"], batch_responses=["rb", "ra"])
    assert pooled_batch_from_round(with_batch)[0] == ["b", "a"]

    # Without indices it falls back to first occurrence by prompt text.
    no_idx = dict(record)
    no_idx.pop("agent_batch_indices")
    assert pooled_batch_from_round(no_idx) == (["a", "b"], ["ra", "rb"])

    # Opting out reproduces the historical concatenation.
    assert pooled_batch_from_round(record, dedupe=False)[0] == ["a", "b", "a", "b"]


def test_pooled_batch_hard_rounds_unchanged() -> None:
    """Hard rounds keep genuinely repeated prompt strings."""
    record = {
        "routing_mode": "hard",
        "agent_prompts": {"clone-0": ["x"], "clone-1": ["x"]},
        "agent_responses": {"clone-0": ["r"], "clone-1": ["r"]},
    }
    assert pooled_batch_from_round(record) == (["x", "x"], ["r", "r"])
    legacy = {"agent_prompts": {"clone-0": ["x"]}, "agent_responses": {}}
    assert pooled_batch_from_round(legacy) == (["x"], [None])


# ---------------------------------------------------------------------------
# validation matrix
# ---------------------------------------------------------------------------


def test_validation_matrix_soft_pairs() -> None:
    """Soft routing over merge groups is allowed; its incompatibilities are not."""
    from infl_ens.training.__main__ import _validate_routing_and_loss_modes as check

    check(
        "G", None, routing_mode="soft", soft_top_k=7, n_agents=14,
        has_merge_groups=True, n_groups=7, merge_mode="fixed",
    )
    with pytest.raises(ValueError, match="merge groups"):
        check(
            "G", None, routing_mode="soft", soft_top_k=8, n_agents=14,
            has_merge_groups=True, n_groups=7, merge_mode="fixed",
        )
    with pytest.raises(ValueError, match="sft_also_train_individual"):
        check(
            "G", None, routing_mode="soft", soft_top_k=2, n_agents=14,
            has_merge_groups=True, n_groups=7, merge_mode="fixed",
            also_train_individual=True,
        )
    with pytest.raises(ValueError, match="proximity"):
        check("G", None, routing_mode="soft", merge_mode="proximity")
    with pytest.raises(ValueError, match="loss_reweight"):
        check("G", "one_minus_G", routing_mode="soft", has_merge_groups=True,
              n_groups=7, merge_mode="fixed")
    with pytest.raises(ValueError, match="loss_reweight"):
        check("G", "position_only", routing_mode="soft", soft_top_k=2,
              n_agents=4)

    # Without merge groups soft_top_k still counts agents.
    check("G", None, routing_mode="soft", soft_top_k=8, n_agents=14)
    # Both soft loss weightings and both centroid arms are valid in soft mode.
    check("G", None, routing_mode="soft", soft_top_k=2, n_agents=4,
          soft_loss="unit")
    check("G", None, routing_mode="soft", soft_top_k=2, n_agents=4,
          soft_loss="unit", position_update="naive")
    # soft_loss is a soft-routing knob only.
    with pytest.raises(ValueError, match="soft_loss"):
        check("G", None, soft_loss="unit")
    with pytest.raises(ValueError, match="soft_loss"):
        check("G", None, routing_mode="soft", soft_top_k=1, n_agents=2,
              soft_loss="bogus")
    with pytest.raises(ValueError, match="position_update"):
        check("G", None, position_update="bogus")
    # The deprecated alias cannot contradict an explicit naive centroid.
    with pytest.raises(ValueError, match="position_only"):
        check("G", "position_only", position_update="naive")
    # The expected-pool centroid is gradient-matched by construction.
    with pytest.raises(ValueError, match="expected_pool"):
        check("G", None, position_update="naive", centroid_mode="expected_pool")
    # ... and is now available under soft routing too.
    check("G", None, routing_mode="soft", soft_top_k=2, n_agents=4,
          centroid_mode="expected_pool")
    # Hard-mode combinations are untouched.
    check("G", "position_only")
    check("G", None, position_update="naive")
    check("G_times_1mG", None)
    check("G_times_1mG", None, position_update="naive")
    with pytest.raises(ValueError):
        check("G_times_1mG", "one_minus_G")


# ---------------------------------------------------------------------------
# end-to-end closed-loop round (fake trait space, stubbed SFT)
# ---------------------------------------------------------------------------


def _install_fake_space(monkeypatch: pytest.MonkeyPatch, prompts: list[str]) -> None:
    """Point the training driver at a deterministic 2-D toy trait space."""
    from infl_ens.data.benchmarks import BenchmarkSplit
    import infl_ens.training.__main__ as driver

    rng = np.random.default_rng(0)
    coords = {p: rng.random(2) for p in prompts}

    def project(queries: Sequence[str]) -> np.ndarray:
        return np.stack([coords[q] for q in queries], axis=0)

    grid = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    space = TraitSpace(
        grid=grid,
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


def _install_stub_sft(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]):
    """Record every SFT call instead of training a LoRA."""
    import infl_ens.training.sft_training as sft_mod

    def fake_sft_train_agent(agent, prompts, responses, cfg, **kwargs):  # noqa: ANN001
        out_dir = Path(kwargs.get("out_dir_override") or cfg.output_dir)
        calls.append({
            "agent": agent.name,
            "n_prompts": len(prompts),
            "sample_weights": list(kwargs.get("sample_weights") or []),
            "skip_position_update": kwargs.get("skip_position_update"),
            "out_dir": str(out_dir),
            "prior": agent.metadata.get("lora_dir"),
        })
        prior = agent.metadata.get("lora_dir")
        agent.metadata["lora_dir"] = str(out_dir)
        return {
            "output_dir": str(out_dir),
            "n_train": len(prompts),
            "log_history": [{"loss": 0.5}],
            "loaded_prior_lora": prior,
        }

    monkeypatch.setattr(sft_mod, "sft_train_agent", fake_sft_train_agent)


def test_closed_loop_soft_pairs_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One adapter and one shared position per pair, top-k over pairs."""
    from infl_ens.training.__main__ import _task_closed_loop

    prompts = [f"q{i}" for i in range(24)]
    _install_fake_space(monkeypatch, prompts)
    calls: list[dict[str, Any]] = []
    _install_stub_sft(monkeypatch, calls)

    out_dir = tmp_path / "run"
    cfg: dict[str, Any] = {
        "task": "closed_loop",
        "seed": 0,
        "output_dir": str(out_dir),
        "policy": "proportional",
        "benchmarks": [{"kind": "fake", "path": "unused"}],
        "agents": {"pairs_from_axes": True},
        "sigma_mode": "absolute",
        "sigma": 0.35,
        "closed_loop": {
            "init_mode": "theory_gradient_paired",
            "init_noise": 0.0,
            "theory_gradient": {"n_steps": 30, "pairing": "nearest"},
            "routing_mode": "soft",
            "soft_top_k": 2,
            "routing_weight": "G",
            "loss_reweight": None,
            # The historical share-weighted centroid; the theory-matched
            # default is exercised in tests/test_topk_matched.py.
            "position_update": "naive",
            "centroid_mode": "batch",
            "blend": 0.5,
            "n_rounds": 2,
            "batch_size": 12,
            "sft_merge_groups": "from_init",
            "save_per_round": True,
            "sft": {"output_dir": str(out_dir / "agents")},
        },
    }
    assert _task_closed_loop(cfg) == 0

    # agents: mapping expanded to 2L clones for the fake L = 2 space.
    assert [e["name"] for e in cfg["agents"]] == [f"clone-{i}" for i in range(4)]
    # groups: one per co-located theory pair, named from the prefix.
    groups = cfg["closed_loop"]["sft_merge_groups"]
    assert [g["train_as"] for g in groups] == ["pair-0", "pair-1"]

    history = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 2

    for record in history:
        assert record["routing_mode"] == "soft"
        assert record["soft_routing_units"] == "pairs"
        assert record["soft_loss"] == "weighted"
        assert record["position_update"] == "naive"
        # The naive arm applies the logged shares directly; nothing extra.
        assert record["agent_position_weights"] == {}
        # top_k == number of pairs here, so every pair sees the whole batch.
        counts = record["merge_prompt_counts"]
        assert set(counts) == {"pair-0", "pair-1"}
        assert sum(counts.values()) == 2 * 12
        # Weights are the renormalised pair shares: each query's total is 1.
        totals = np.zeros(12)
        for name, idxs in record["agent_batch_indices"].items():
            for i, m in enumerate(idxs):
                totals[m] += record["agent_sample_weights"][name][i]
        assert np.allclose(totals, 1.0)
        # Partners share one position exactly.
        positions = record["positions"]
        for group in groups:
            a, b = group["names"]
            assert positions[a] == positions[b]
            assert positions[a] == record["pair_positions"][group["train_as"]]

    # Positions actually moved off the theory init.
    assert history[0]["positions"] != history[1]["positions"]

    # One SFT call per pair per round, cumulative, never per clone.
    assert [c["agent"] for c in calls] == ["pair-0", "pair-1"] * 2
    assert all(c["skip_position_update"] for c in calls)
    assert all(len(c["sample_weights"]) == c["n_prompts"] for c in calls)
    assert calls[2]["prior"] == calls[0]["out_dir"]
    assert calls[0]["out_dir"].endswith("round-00")
    assert calls[2]["out_dir"].endswith("round-01")
    # No clone ever gets its own adapter: the pair adapter is the only one.
    assert not any(c["agent"].startswith("clone-") for c in calls)

    # The resolved config is what evaluation tools should read.
    resolved = out_dir / "resolved_config.yaml"
    if resolved.exists():  # PyYAML present
        import yaml

        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        assert [a["name"] for a in payload["agents"]] == [
            f"clone-{i}" for i in range(4)
        ]
        assert payload["closed_loop"]["sft_merge_groups"] == groups


def test_closed_loop_soft_pairs_rejects_split_partners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Soft pair routing refuses groups whose members are not co-located."""
    from infl_ens.training.__main__ import _task_closed_loop

    prompts = [f"q{i}" for i in range(8)]
    _install_fake_space(monkeypatch, prompts)
    _install_stub_sft(monkeypatch, [])

    cfg: dict[str, Any] = {
        "task": "closed_loop",
        "seed": 0,
        "output_dir": str(tmp_path / "run"),
        "benchmarks": [{"kind": "fake", "path": "unused"}],
        "agents": [{"name": f"clone-{i}"} for i in range(4)],
        "sigma_mode": "absolute",
        "sigma": 0.35,
        "closed_loop": {
            "init_mode": "mean_noise",
            "init_noise": 0.05,
            "routing_mode": "soft",
            "soft_top_k": 2,
            "centroid_mode": "batch",
            "n_rounds": 1,
            "batch_size": 4,
            "sft_merge_groups": [
                {"train_as": "pair-0", "names": ["clone-0", "clone-1"]},
                {"train_as": "pair-1", "names": ["clone-2", "clone-3"]},
            ],
            "sft": {"output_dir": str(tmp_path / "run" / "agents")},
        },
    }
    with pytest.raises(ValueError, match="co-located"):
        _task_closed_loop(cfg)
