"""Single CLI entry point for all training in ``infl_ens``.

Per AGENTS.md §4 rule 1, every fine-tune / training script is reached
through this one entry point, with behaviour controlled by a Hydra-style
config. The CLI accepts either:

- a YAML config path via ``--config configs/benchmark/router/safety_truth.yaml``
  with optional ``KEY=VAL`` overrides appended after ``--``,
- or ``--task <name>`` plus task-specific flags for ad-hoc use.

Supported tasks (selected by the ``task`` field of the config):

- ``router_training``: gradient-ascent on agent positions in a fixed
  trait space. Delegates to
  :func:`infl_ens.training.router_training.train_router_positions`.
- ``sft_training``: LoRA SFT of a single :class:`RouterAgent` on a
  benchmark split. Delegates to
  :func:`infl_ens.training.sft_training.sft_train_agent`.
- ``closed_loop``: alternating SFT + position re-estimation across rounds,
  using the influencer-game router to assign queries to agents.

Run with::

    python -m infl_ens.training --config configs/benchmark/router/safety_truth.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from infl_ens.data.benchmarks import (
    BenchmarkSplit,
    build_safety_trait_space,
    load_beavertails,
    load_halueval,
)
from infl_ens.data.encoders import SentenceTransformerEncoder
from infl_ens.data.trait_space import TraitSpace
from infl_ens.inflgame.router import InfluencerRouter, RouterAgent
from infl_ens.training.router_training import (
    RouterTrainingConfig,
    train_router_positions,
)
from infl_ens.utils.resource import gaussian_stability_threshold


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file using PyYAML if available, else a tiny fallback.

    :param path: Path to the YAML file.
    :type path: pathlib.Path
    :returns: Parsed mapping.
    :rtype: dict
    """
    try:
        import yaml
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:  # pragma: no cover
        # Best-effort fallback: only flat key: value supported.
        out: dict[str, Any] = {}
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
        return out


def _apply_overrides(cfg: dict[str, Any], overrides: Sequence[str]) -> None:
    """Apply ``key.subkey=value`` overrides in place.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :param overrides: Sequence of dotted overrides.
    :type overrides: Sequence[str]
    """
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        path = key.split(".")
        node = cfg
        for p in path[:-1]:
            node = node.setdefault(p, {})
        try:
            node[path[-1]] = json.loads(val)
        except json.JSONDecodeError:
            node[path[-1]] = val


def _load_splits(cfg: dict[str, Any]) -> list[BenchmarkSplit]:
    """Load all benchmark splits referenced by ``cfg['benchmarks']``.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :returns: List of loaded :class:`BenchmarkSplit`.
    :rtype: list[BenchmarkSplit]
    """
    splits: list[BenchmarkSplit] = []
    for entry in cfg.get("benchmarks", []):
        kind = entry["kind"]
        path = entry["path"]
        max_records = entry.get("max_records")
        if kind == "beavertails":
            splits.append(load_beavertails(path, max_records=max_records))
        elif kind == "halueval":
            splits.append(load_halueval(
                path,
                tasks=entry.get("tasks"),
                max_records=max_records,
            ))
        else:  # pragma: no cover
            raise ValueError(f"unknown benchmark kind {kind!r}")
    return splits


def _make_trait_space(cfg: dict[str, Any], splits: list[BenchmarkSplit]) -> TraitSpace:
    """Build a :class:`TraitSpace` from benchmark splits and config.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :param splits: Already-loaded benchmark splits.
    :type splits: list[BenchmarkSplit]
    :returns: The constructed trait space.
    :rtype: TraitSpace
    """
    ts_cfg = cfg.get("trait_space", {})
    encoder_name = ts_cfg.get(
        "encoder", "sentence-transformers/all-MiniLM-L6-v2",
    )
    encoder = SentenceTransformerEncoder(model_name=encoder_name)
    return build_safety_trait_space(
        splits, encoder,
        n_grid=int(ts_cfg.get("n_grid", 32)),
        kde_bandwidth=ts_cfg.get("kde_bandwidth"),
        threshold=float(ts_cfg.get("threshold", 0.5)),
    )


def _init_agents(
    cfg: dict[str, Any],
    space: TraitSpace,
    splits: list[BenchmarkSplit],
) -> list[RouterAgent]:
    """Initialise router agents from the config's ``agents`` list.

    An agent entry may name a ``calibration`` field referencing a
    benchmark split, in which case its position is the centroid of that
    split's prompts in trait space; otherwise the position is the
    resource-weighted mean (i.e. a "clone" starting position).

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :param space: Trait space defining the projector.
    :type space: TraitSpace
    :param splits: Already-loaded benchmark splits.
    :type splits: list[BenchmarkSplit]
    :returns: List of :class:`RouterAgent` ready for routing/training.
    :rtype: list[RouterAgent]
    """
    by_name = {s.name: s for s in splits}
    agents: list[RouterAgent] = []
    for entry in cfg.get("agents", []):
        name = entry["name"]
        cal_name = entry.get("calibration")
        if cal_name and cal_name in by_name:
            agents.append(RouterAgent.from_calibration(
                name=name,
                calibration_queries=by_name[cal_name].prompts[:200],
                project=space.project,
            ))
        else:
            agents.append(RouterAgent(name=name, position=space.mean.copy()))
    return agents


def _sigma_from_cfg(
    cfg: dict[str, Any], n_agents: int, space: TraitSpace,
) -> float:
    """Resolve the competitive reach :math:`\\sigma` from the config.

    Supports ``sigma_mode: absolute`` (use ``sigma`` directly) and
    ``sigma_mode: stability_fraction`` (multiplier on :math:`\\sigma_0^*`).

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :param n_agents: Number of agents.
    :type n_agents: int
    :param space: Trait space (for the stability threshold).
    :type space: TraitSpace
    :returns: Scalar :math:`\\sigma`.
    :rtype: float
    """
    mode = cfg.get("sigma_mode", "absolute")
    if mode == "absolute":
        return float(cfg["sigma"])
    if mode == "stability_fraction":
        frac = float(cfg.get("sigma_fraction", 0.8))
        s0 = gaussian_stability_threshold(n_agents, space.grid, space.weights)
        return frac * max(s0, 1e-3)
    raise ValueError(f"unknown sigma_mode {mode!r}")


def _task_router_training(cfg: dict[str, Any]) -> int:
    """Run gradient-ascent training on agent positions.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :returns: Exit code.
    :rtype: int
    """
    splits = _load_splits(cfg)
    space = _make_trait_space(cfg, splits)
    agents = _init_agents(cfg, space, splits)
    sigma = _sigma_from_cfg(cfg, len(agents), space)

    train_cfg = cfg.get("training", {})
    rt_cfg = RouterTrainingConfig(
        sigma=sigma,
        learning_rate=float(train_cfg.get("learning_rate", 5e-3)),
        n_steps=int(train_cfg.get("n_steps", 5000)),
        tol=float(train_cfg.get("tol", 1e-8)),
        clip_to_box=bool(train_cfg.get("clip_to_box", True)),
    )
    info = train_router_positions(space, agents, rt_cfg, seed=int(cfg.get("seed", 0)))

    out = Path(cfg.get("output_dir", "results/router")) / "positions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "sigma": sigma,
                "converged": info["converged"],
                "n_steps": info["n_steps"],
                "positions": {a.name: a.position.tolist() for a in agents},
            },
            fh,
            indent=2,
        )
    print(f"router training: converged={info['converged']} in {info['n_steps']} steps")
    print(f"wrote {out}")
    return 0


def _task_sft_training(cfg: dict[str, Any]) -> int:
    """Run one LoRA SFT round for a single agent.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :returns: Exit code.
    :rtype: int
    """
    from infl_ens.training.sft_training import SFTTrainingConfig, sft_train_agent

    splits = _load_splits(cfg)
    space = _make_trait_space(cfg, splits)
    agents = _init_agents(cfg, space, splits)
    target_name = cfg["sft"]["agent"]
    agent = next(a for a in agents if a.name == target_name)
    split = next(s for s in splits if s.name == cfg["sft"]["benchmark"])

    sft_cfg_dict = dict(cfg.get("sft", {}))
    sft_cfg_dict.pop("agent", None)
    sft_cfg_dict.pop("benchmark", None)
    sft_cfg = SFTTrainingConfig(**sft_cfg_dict)

    result = sft_train_agent(
        agent,
        prompts=split.prompts,
        responses=split.responses or None,
        cfg=sft_cfg,
        eval_prompts=split.prompts[: min(256, split.n)],
        project=space.project,
    )
    print(f"sft training done: {json.dumps(result)}")
    return 0


def _task_closed_loop(cfg: dict[str, Any]) -> int:
    """Alternating route → SFT → position update across ``rounds``.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :returns: Exit code.
    :rtype: int
    """
    from infl_ens.training.sft_training import SFTTrainingConfig, sft_train_agent

    splits = _load_splits(cfg)
    space = _make_trait_space(cfg, splits)
    agents = _init_agents(cfg, space, splits)
    sigma = _sigma_from_cfg(cfg, len(agents), space)
    router = InfluencerRouter(
        space, agents, sigma=sigma,
        policy=cfg.get("policy", "proportional"),
    )

    cl = cfg.get("closed_loop", {})
    n_rounds = int(cl.get("n_rounds", 5))
    batch_size = int(cl.get("batch_size", 256))
    # Per-trait routing-probability weighting. 'G' is the canonical
    # proportional allocation; 'G_times_1mG' matches the strategic gradient
    # in expectation (see allocation.strategic_routing_weights).
    routing_weight = str(cl.get("routing_weight", "G"))
    # When true, persist a separate LoRA adapter per (agent, round) under
    # ``<sft.output_dir>/<agent>/round-NN``. Required by the capability
    # probe (scripts/probe_sft_capability.py). Default off because each
    # adapter is ~25 MB, so a 4-agent / 10-round run adds ~1 GB to disk.
    save_per_round = bool(cl.get("save_per_round", False))
    all_prompts = [p for s in splits for p in s.prompts]
    all_responses = [r for s in splits for r in (s.responses or [""] * s.n)]
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    sft_cfg_dict = dict(cl.get("sft", {}))
    sft_cfg = SFTTrainingConfig(**sft_cfg_dict)
    sft_base_output_dir = Path(sft_cfg.output_dir)

    history: list[dict[str, Any]] = []
    for r in range(n_rounds):
        idx = rng.integers(0, len(all_prompts), size=batch_size)
        batch_prompts = [all_prompts[i] for i in idx]
        batch_responses = [all_responses[i] for i in idx]
        choices = router.route_batch(
            batch_prompts, rng=rng, routing_weight=routing_weight,
        )
        agent_prompts: dict[str, list[str]] = {a.name: [] for a in agents}
        agent_responses: dict[str, list[str]] = {a.name: [] for a in agents}
        agent_sft_logs: dict[str, list[dict[str, Any]]] = {
            a.name: [] for a in agents
        }
        agent_loaded_prior: dict[str, Optional[str]] = {
            a.name: None for a in agents
        }
        for agent in agents:
            mine_p = [q for q, c in zip(batch_prompts, choices) if c.name == agent.name]
            mine_r = [t for t, c in zip(batch_responses, choices) if c.name == agent.name]
            agent_prompts[agent.name] = list(mine_p)
            agent_responses[agent.name] = list(mine_r)
            if not mine_p:
                continue
            # Per-round adapter directory, when requested. Layout:
            #   <sft.output_dir>/<agent>/round-NN/
            out_override = (
                str(sft_base_output_dir / agent.name / f"round-{r:02d}")
                if save_per_round else None
            )
            sft_result = sft_train_agent(
                agent,
                prompts=mine_p,
                responses=mine_r if any(mine_r) else None,
                cfg=sft_cfg,
                eval_prompts=mine_p,
                project=space.project,
                blend=float(cl.get("blend", 0.5)),
                out_dir_override=out_override,
            )
            agent_sft_logs[agent.name] = sft_result.get("log_history", [])
            agent_loaded_prior[agent.name] = sft_result.get("loaded_prior_lora")
        # Per-round diagnostics: grid utility (theory), empirical-pool
        # utility (what proportional routing draws against), and observed
        # share. Divergence between u_grid and u_pool flags a KDE / grid
        # smoothing artefact; divergence between u_pool and share flags
        # Poisson-Binomial noise on this batch.
        from infl_ens.inflgame.router.allocation import (
            empirical_utility,
            strategic_routing_weights,
        )
        pool_coords = space.project(all_prompts)
        observed = np.array(
            [sum(1 for c in choices if c.name == a.name) / max(len(choices), 1)
             for a in agents]
        )
        # Under routing_weight='G' the strategic share equals u_pool only at
        # clone-start; later it diverges. Logging it lets you verify the
        # routing math regardless of which weighting is active this round.
        strategic_share_pool = strategic_routing_weights(
            router.positions, pool_coords, router.cov,
        ).mean(axis=1)
        history.append({
            "round": r,
            "positions": {a.name: a.position.tolist() for a in agents},
            "u_grid": router.expected_utilities().tolist(),
            "u_pool": empirical_utility(
                router.positions, pool_coords, router.cov,
            ).tolist(),
            "strategic_share_pool": strategic_share_pool.tolist(),
            "observed_share": observed.tolist(),
            "routing_weight": routing_weight,
            # New: full per-agent assignments + SFT loss curve, for the
            # capability probe (scripts/probe_sft_capability.py).
            "agent_prompts": agent_prompts,
            "agent_responses": agent_responses,
            "agent_sft_logs": agent_sft_logs,
            # New: which prior adapter (if any) each agent loaded this
            # round. None for fresh-LoRA rounds (the original framework);
            # a path string for cumulative-LoRA rounds.
            "agent_loaded_prior": agent_loaded_prior,
            "save_per_round": save_per_round,
        })

    out = Path(cfg.get("output_dir", "results/closed_loop")) / "history.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"closed-loop done: {n_rounds} rounds, wrote {out}")
    return 0


_TASKS = {
    "router_training": _task_router_training,
    "sft_training": _task_sft_training,
    "closed_loop": _task_closed_loop,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m infl_ens.training``.

    :param argv: Argument vector (defaults to ``sys.argv[1:]``).
    :type argv: Sequence[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        prog="python -m infl_ens.training",
        description="Single CLI for router-position training, SFT, and the closed loop.",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a YAML config under configs/.")
    parser.add_argument("overrides", nargs="*",
                        help="Optional KEY=VALUE config overrides.")
    args = parser.parse_args(argv)

    cfg = _load_yaml(Path(args.config))
    _apply_overrides(cfg, args.overrides)
    task = cfg.get("task")
    if task not in _TASKS:
        print(f"error: unknown task {task!r}; expected one of {sorted(_TASKS)}",
              file=sys.stderr)
        return 2
    return int(_TASKS[task](cfg))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
