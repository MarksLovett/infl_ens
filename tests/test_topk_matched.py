"""Offline tests for the theory-matched soft position update and the top-k winners ablation.

Covers:

- :func:`infl_ens.inflgame.router.allocation.matched_centroid_mass` — the
  dense, un-renormalised :math:`G_i(1-G_i)` centroid mass,
- the gradient-direction verdict in
  :mod:`infl_ens.inflgame.router.verification`: the dense matched soft drift
  is parallel to :math:`\\nabla_{x_i} u_i` (independent of ``top_k``), the
  naive renormalised-share drift is not, co-located clones take identical
  independent steps (pairs persist without a shared update), and the dense
  rule is a lower-variance estimator than the hard ``(1-G)`` rule at equal
  mean,
- ``closed_loop.position_update`` in the driver: theory-matched is the
  default in hard mode (``(1-G)`` centroid, unit loss), ``naive`` restores
  the uniform centroid, and the deprecated ``loss_reweight: position_only``
  alias resolves to the default,
- ``closed_loop.soft_loss: unit`` (top-k winners at unit loss weight) over
  co-located pairs and over individual agents, with the dense
  :math:`G(1-G)` position step over the whole batch logged per round,
  including the ``expected_pool`` centroid under soft routing, and
- pooled-replay de-duplication of unit-loss soft rounds.

Everything here is numpy-only: no ``torch``, no ``trl``, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from infl_ens.data.trait_space import TraitSpace
from infl_ens.inflgame.router.allocation import (
    allocation_weights,
    group_allocation_weights,
    matched_centroid_mass,
    top_k_allocation_weights,
    utility_gradient,
)
from infl_ens.inflgame.router.verification import (
    _build_landscape,
    _cos,
    _expected_drift_canonical_reweighted,
    _expected_drift_dense_matched,
    _expected_drift_naive_pairs,
    _expected_drift_topk_naive,
    _monte_carlo_drift,
)
from infl_ens.training.baseline_replay import pooled_batch_from_round

# ---------------------------------------------------------------------------
# matched_centroid_mass
# ---------------------------------------------------------------------------


def _random_allocation(n_agents: int = 4, m: int = 9, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions = rng.random((n_agents, 2))
    coords = rng.random((m, 2))
    return allocation_weights(positions, coords, 0.2 ** 2 * np.eye(2))


def test_matched_centroid_mass_values() -> None:
    """Mass is G(1-G) on every query, bounded by 1/4, never renormalised."""
    G = _random_allocation()
    M = matched_centroid_mass(G)
    assert M.shape == G.shape
    assert np.allclose(M, G * (1.0 - G))
    assert np.all(M > 0.0)
    assert np.all(M <= 0.25 + 1e-12)
    assert not np.allclose(M.sum(axis=0), 1.0)
    with pytest.raises(ValueError):
        matched_centroid_mass(np.ones(3))


def test_top_k_one_is_hard_argmax_for_training_gate() -> None:
    """top_k=1 keeps one winner per query for the loss-side weights."""
    G = _random_allocation(n_agents=5, m=12, seed=1)
    W = top_k_allocation_weights(G, 1)
    assert np.all((W > 0.0).sum(axis=0) == 1)
    assert np.array_equal(np.argmax(W, axis=0), np.argmax(G, axis=0))


# ---------------------------------------------------------------------------
# gradient-direction verification
# ---------------------------------------------------------------------------


def _landscape():
    """Bimodal landscape with agents well away from any equilibrium.

    The gradient must be non-negligible for a cosine to mean anything, so
    the agents sit between the two resource modes rather than on them.
    """
    grid, weights = _build_landscape()
    cov = 0.20 ** 2 * np.eye(2)
    positions = np.array([
        [0.40, 0.30],
        [0.60, 0.70],
        [0.55, 0.45],
    ])
    return grid, weights, cov, positions


def test_dense_matched_drift_parallel_to_gradient() -> None:
    """The dense matched soft drift IS the utility gradient direction."""
    grid, weights, cov, positions = _landscape()
    grad = utility_gradient(positions, grid, weights, cov)
    assert np.all(np.linalg.norm(grad, axis=1) > 1e-6)
    d_match = _expected_drift_dense_matched(positions, grid, weights, cov)
    for i in range(positions.shape[0]):
        assert _cos(d_match[i], grad[i]) > 0.9999
    # ... and coincides with the hard-routing canonical+(1-G) drift, which is
    # what position_only / the position-only simulator realise in expectation.
    d_rw = _expected_drift_canonical_reweighted(positions, grid, weights, cov)
    assert np.allclose(d_match, d_rw)


def test_naive_topk_drift_less_aligned() -> None:
    """The renormalised-share centroid (position_update: naive) is misaligned at any k."""
    grid, weights, cov, positions = _landscape()
    grad = utility_gradient(positions, grid, weights, cov)
    n = positions.shape[0]
    d_match = _expected_drift_dense_matched(positions, grid, weights, cov)
    cos_match = np.array([_cos(d_match[i], grad[i]) for i in range(n)])
    for k in (n, 2):
        d_naive = _expected_drift_topk_naive(positions, grid, weights, cov, top_k=k)
        cos_naive = np.array([_cos(d_naive[i], grad[i]) for i in range(n)])
        assert np.all(cos_naive <= cos_match + 1e-9)
        assert np.max(cos_match - cos_naive) > 1e-3


def test_colocated_clones_take_identical_independent_steps() -> None:
    """Each clone follows its own gradient; twins get bitwise-identical drifts."""
    grid, weights = _build_landscape()
    cov = 0.20 ** 2 * np.eye(2)
    pair_pos = np.array([[0.40, 0.30], [0.60, 0.70]])
    clones = np.repeat(pair_pos, 2, axis=0)
    group_index = np.repeat(np.arange(2), 2)
    grad = utility_gradient(clones, grid, weights, cov)
    assert np.all(np.linalg.norm(grad, axis=1) > 1e-6)
    d = _expected_drift_dense_matched(clones, grid, weights, cov)
    for c in range(4):
        assert _cos(d[c], grad[c]) > 0.9999
    # Nothing ties the twins together, yet identical inputs give identical
    # steps: the pair persists on its own.
    assert np.array_equal(d[0], d[1])
    assert np.array_equal(d[2], d[3])
    # The naive pair rule (historical arm) is misaligned with the members'
    # gradient.
    d_naive = _expected_drift_naive_pairs(
        clones, grid, weights, cov, group_index, 2, top_k=2,
    )
    for p in range(2):
        assert _cos(d_naive[p], grad[2 * p]) < _cos(d[2 * p], grad[2 * p]) - 1e-3
    # The summed clone share is still the 2-player allocation exactly (this
    # is what the TRAINING gate over pairs relies on).
    G_group = group_allocation_weights(
        allocation_weights(clones, grid, cov), group_index, 2,
    )
    assert np.allclose(G_group, allocation_weights(pair_pos, grid, cov))


def test_dense_rule_is_lower_variance_than_hard_rule() -> None:
    """Same mean direction, smaller finite-batch spread (Rao–Blackwell)."""
    grid, weights, cov, positions = _landscape()
    grad = utility_gradient(positions, grid, weights, cov)
    mc = _monte_carlo_drift(
        positions, grid, weights, cov, batch_size=512, n_trials=60, seed=0,
    )
    for i in range(positions.shape[0]):
        hard = mc["canonical_reweight"][:, i]
        dense = mc["dense_matched"][:, i]
        assert _cos(dense.mean(axis=0), grad[i]) > 0.99
        assert _cos(hard.mean(axis=0), grad[i]) > 0.95
        spread_hard = np.linalg.norm(hard - hard.mean(axis=0), axis=-1).mean()
        spread_dense = np.linalg.norm(dense - dense.mean(axis=0), axis=-1).mean()
        assert spread_dense < spread_hard


# ---------------------------------------------------------------------------
# closed-loop driver (fake trait space, stubbed SFT)
# ---------------------------------------------------------------------------


def _install_fake_space(monkeypatch: pytest.MonkeyPatch, prompts: list[str]) -> None:
    from infl_ens.data.benchmarks import BenchmarkSplit
    import infl_ens.training.closed_loop as driver

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
    """Record every SFT call, keeping ``None`` weights distinguishable from lists."""
    import infl_ens.training.sft_training as sft_mod

    def fake_sft_train_agent(agent, prompts, responses, cfg, **kwargs):  # noqa: ANN001
        out_dir = Path(kwargs.get("out_dir_override") or cfg.output_dir)
        sw = kwargs.get("sample_weights")
        ew = kwargs.get("eval_weights")
        calls.append({
            "agent": agent.name,
            "n_prompts": len(prompts),
            "sample_weights": None if sw is None else list(sw),
            "eval_weights": None if ew is None else list(ew),
            "skip_position_update": kwargs.get("skip_position_update"),
        })
        if not kwargs.get("skip_position_update"):
            agent.update_position_from_corpus(
                list(kwargs["eval_prompts"]),
                kwargs["project"],
                scores=None if ew is None else list(ew),
                blend=kwargs.get("blend", 1.0),
                position_step=kwargs.get("position_step"),
            )
        agent.metadata["lora_dir"] = str(out_dir)
        return {
            "output_dir": str(out_dir),
            "n_train": len(prompts),
            "log_history": [{"loss": 0.5}],
            "loaded_prior_lora": None,
            "position_blend_effective": float(kwargs.get("blend", 1.0)),
        }

    monkeypatch.setattr(sft_mod, "sft_train_agent", fake_sft_train_agent)


def _hard_cfg(out_dir: Path, **closed_loop: Any) -> dict[str, Any]:
    cl: dict[str, Any] = {
        "init_mode": "mean_noise",
        "init_noise": 0.05,
        "routing_mode": "hard",
        "routing_weight": "G",
        "centroid_mode": "batch",
        "blend": 0.5,
        "n_rounds": 1,
        "batch_size": 12,
        "sft": {"output_dir": str(out_dir / "agents")},
    }
    cl.update(closed_loop)
    return {
        "task": "closed_loop",
        "seed": 0,
        "output_dir": str(out_dir),
        "policy": "proportional",
        "benchmarks": [{"kind": "fake", "path": "unused"}],
        "agents": [{"name": "clone-0"}, {"name": "clone-1"}],
        "sigma_mode": "absolute",
        "sigma": 0.35,
        "closed_loop": cl,
    }


def _run_hard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tag: str, **cl: Any):
    from infl_ens.training.closed_loop import run_closed_loop as _task_closed_loop

    prompts = [f"q{i}" for i in range(12)]
    _install_fake_space(monkeypatch, prompts)
    calls: list[dict[str, Any]] = []
    _install_stub_sft(monkeypatch, calls)
    out_dir = tmp_path / tag
    cfg = _hard_cfg(out_dir, **cl)
    assert _task_closed_loop(cfg) == 0
    history = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
    return calls, history[0]


def test_hard_mode_default_is_theory_matched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without any position knob the hard-mode centroid is (1-G)-weighted, loss unit."""
    calls, record = _run_hard(tmp_path, monkeypatch, "default")
    assert record["position_update"] == "theory_matched"
    assert record["loss_reweight"] is None
    assert record["soft_loss"] is None
    assert calls, "at least one agent must have trained"
    for call in calls:
        assert call["sample_weights"] is None
        assert call["eval_weights"] is not None
        assert len(call["eval_weights"]) == call["n_prompts"]
        assert all(0.0 < w < 1.0 for w in call["eval_weights"])
        # The logged (1-G) weights are exactly what the centroid used.
        assert record["agent_sample_weights"][call["agent"]] == pytest.approx(
            call["eval_weights"],
        )

    # The deprecated alias resolves to the very same behaviour.
    alias_calls, alias_record = _run_hard(
        tmp_path, monkeypatch, "alias", loss_reweight="position_only",
    )
    assert alias_record["position_update"] == "theory_matched"
    assert alias_record["loss_reweight"] is None
    assert [c["eval_weights"] for c in alias_calls] == [
        c["eval_weights"] for c in calls
    ]


def test_hard_mode_naive_restores_uniform_centroid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """position_update: naive → unweighted centroid, no (1-G) weights computed."""
    calls, record = _run_hard(
        tmp_path, monkeypatch, "naive", position_update="naive",
    )
    assert record["position_update"] == "naive"
    assert all(c["eval_weights"] is None for c in calls)
    assert all(c["sample_weights"] is None for c in calls)
    assert all(v == [] for v in record["agent_sample_weights"].values())

    with pytest.raises(ValueError, match="position_only"):
        _run_hard(
            tmp_path, monkeypatch, "clash",
            position_update="naive", loss_reweight="position_only",
        )


def test_closed_loop_topk_pairs_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-k winners at unit loss over pairs; clones step independently and stay paired."""
    from infl_ens.training.closed_loop import run_closed_loop as _task_closed_loop

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
            "soft_top_k": 1,          # strict winners: each prompt trains ONE pair
            "soft_loss": "unit",
            "routing_weight": "G",
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
    groups = cfg["closed_loop"]["sft_merge_groups"]
    assert [g["train_as"] for g in groups] == ["pair-0", "pair-1"]

    history = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 2
    for record in history:
        assert record["routing_mode"] == "soft"
        assert record["soft_loss"] == "unit"
        assert record["position_update"] == "theory_matched"
        assert record["loss_reweight"] is None
        # Training: every prompt went to exactly one pair at weight 1.
        counts = record["merge_prompt_counts"]
        assert sum(counts.values()) == 12
        for name, idxs in record["agent_batch_indices"].items():
            assert record["agent_sample_weights"][name] == [1.0] * len(idxs)
        # Position step: every CLONE takes its own dense G_i(1-G_i) step over
        # the WHOLE batch, regardless of which prompts its pair trained on.
        pos_w = record["agent_position_weights"]
        assert set(pos_w) == {f"clone-{i}" for i in range(4)}
        for name in pos_w:
            assert len(pos_w[name]) == 12
            assert all(0.0 < w <= 0.25 + 1e-12 for w in pos_w[name])
        # Nothing writes a shared position, yet partners stay together: they
        # have identical G_i rows and therefore identical steps.
        for group in groups:
            a, b = group["names"]
            assert pos_w[a] == pos_w[b]
            assert record["positions"][a] == record["positions"][b]
            assert record["positions"][a] == record["pair_positions"][group["train_as"]]
            assert record["agent_geometry"]["within_merge_l2"][group["train_as"]] == 0.0
    assert history[0]["positions"] != history[1]["positions"]

    # Unit loss: the stub saw sample_weights=None on every pair call and
    # never performed a position update of its own.
    assert set(c["agent"] for c in calls) <= {"pair-0", "pair-1"}
    assert all(c["sample_weights"] is None for c in calls)
    assert all(c["skip_position_update"] for c in calls)


def test_closed_loop_topk_per_agent_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-k winners at unit loss over individual agents (no merge groups)."""
    from infl_ens.training.closed_loop import run_closed_loop as _task_closed_loop

    prompts = [f"q{i}" for i in range(12)]
    _install_fake_space(monkeypatch, prompts)
    calls: list[dict[str, Any]] = []
    _install_stub_sft(monkeypatch, calls)

    out_dir = tmp_path / "run"
    cfg = _hard_cfg(
        out_dir,
        routing_mode="soft",
        soft_top_k=2,
        soft_loss="unit",
        n_rounds=2,
    )
    cfg["agents"] = [{"name": f"clone-{i}"} for i in range(3)]
    assert _task_closed_loop(cfg) == 0

    history = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
    record = history[0]
    assert record["soft_loss"] == "unit"
    assert record["position_update"] == "theory_matched"
    # Every query reaches exactly two agents for training ...
    assert sum(len(v) for v in record["agent_prompts"].values()) == 2 * 12
    assert record["batch_prompts"] == prompts[:12] or len(record["batch_prompts"]) == 12
    # ... but every agent's position step uses the whole batch with the
    # dense G_i(1-G_i) mass, and SFT performs no position update itself.
    pos_w = record["agent_position_weights"]
    assert set(pos_w) == {"clone-0", "clone-1", "clone-2"}
    for name in pos_w:
        assert len(pos_w[name]) == 12
        assert all(0.0 < w <= 0.25 + 1e-12 for w in pos_w[name])
    for call in calls:
        assert call["sample_weights"] is None
        assert call["eval_weights"] is None
        assert call["skip_position_update"] is True
    assert all(len(v) == 1 for v in record["agent_blend_effective"].values())
    assert history[0]["positions"] != history[1]["positions"]


def test_closed_loop_soft_expected_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Soft routing now accepts the (matched) expected-pool centroid."""
    from infl_ens.training.closed_loop import run_closed_loop as _task_closed_loop

    prompts = [f"q{i}" for i in range(12)]
    _install_fake_space(monkeypatch, prompts)
    calls: list[dict[str, Any]] = []
    _install_stub_sft(monkeypatch, calls)

    out_dir = tmp_path / "run"
    cfg = _hard_cfg(
        out_dir,
        routing_mode="soft",
        soft_top_k=1,
        centroid_mode="expected_pool",
        n_rounds=2,
    )
    assert _task_closed_loop(cfg) == 0
    history = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
    assert history[0]["centroid_mode"] == "expected_pool"
    assert all(c["skip_position_update"] for c in calls)
    assert all(len(v) == 1 for v in history[0]["agent_blend_effective"].values())
    assert history[0]["positions"] != history[1]["positions"]

    with pytest.raises(ValueError, match="expected_pool"):
        cfg = _hard_cfg(
            tmp_path / "bad", routing_mode="soft", soft_top_k=1,
            centroid_mode="expected_pool", position_update="naive",
        )
        _task_closed_loop(cfg)


def test_pooled_batch_dedupes_unit_topk_rounds() -> None:
    """A unit-loss top-k round still replays each source prompt once."""
    record = {
        "routing_mode": "soft",
        "soft_top_k": 2,
        "soft_loss": "unit",
        "position_update": "theory_matched",
        "agent_prompts": {"pair-0": ["a", "b"], "pair-1": ["a", "b"]},
        "agent_responses": {"pair-0": ["ra", "rb"], "pair-1": ["ra", "rb"]},
        "agent_batch_indices": {"pair-0": [0, 1], "pair-1": [0, 1]},
    }
    assert pooled_batch_from_round(record) == (["a", "b"], ["ra", "rb"])
