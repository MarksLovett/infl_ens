"""Simulate position-only closed-loop dynamics without SFT.

The trait-space update (route → weighted centroid → EMA blend) is
independent of LoRA training. This script runs only that loop on the
safety trait space so position-step policies can be compared in seconds.

Modes:

- **simulate** (default): fresh routing each round from evolving
  positions; canonical ``G`` routing and ``(1-G)``-weighted centroids
  (``position_only`` semantics).
- **replay**: re-apply position updates to corpora logged in an existing
  ``history.json`` (frozen prompts/weights per round). Useful to test a
  new ``position_step`` policy on the same routed batches as a full run.

Run::

    python scripts/simulate_position_only_loop.py \\
        --config configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml \\
        --sigma-fraction 0.25 --seed 0 --n-rounds 20 \\
        --position-step-mode cap_linf --step-cap 0.05 \\
        --output-dir results/position_step_stability_test/mode_cap_linf_0.05/sigma0.25/seed0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]

from infl_ens.data.trait_space_cache import build_or_load_safety_trait_space  # noqa: E402
from infl_ens.data.trait_space import TraitSpace, position_from_corpus  # noqa: E402
from infl_ens.evaluation.benchmarks import load_benchmark_splits  # noqa: E402
from infl_ens.inflgame.router import InfluencerRouter, RouterAgent  # noqa: E402
from infl_ens.inflgame.router.allocation import (  # noqa: E402
    allocation_weights,
    empirical_utility,
    strategic_routing_weights,
)
from infl_ens.utils.position_step import (  # noqa: E402
    apply_position_update,
    blend_for_round,
    expected_pool_centroid,
    parse_position_step,
)
from infl_ens.utils.agent_init import (  # noqa: E402
    init_agents_mean_noise,
    init_agents_pairs_near_reference,
    init_agents_theory_gradient,
    init_agents_theory_gradient_paired,
    resolve_theory_22_reference,
)
from infl_ens.utils.resource import gaussian_stability_threshold  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_splits(cfg: dict[str, Any], repo_root: Path) -> list:
    entries: list[dict[str, Any]] = []
    for entry in cfg.get("benchmarks", []):
        resolved = dict(entry)
        resolved["path"] = str(repo_root / entry["path"])
        entries.append(resolved)
    return load_benchmark_splits(entries)


def _build_trait_space(cfg: dict[str, Any], repo_root: Path) -> TraitSpace:
    # Route through the shared production builder so simulation and
    # training use identical cache, transform, and normalizer geometry.
    return build_or_load_safety_trait_space(cfg, _load_splits(cfg, repo_root))


def _sigma_from_cfg(
    cfg: dict[str, Any],
    n_agents: int,
    space: TraitSpace,
    sigma_fraction_override: Optional[float] = None,
) -> float:
    if sigma_fraction_override is not None:
        s0 = gaussian_stability_threshold(n_agents, space.grid, space.weights)
        return float(sigma_fraction_override) * max(s0, 1e-3)
    mode = cfg.get("sigma_mode", "absolute")
    if mode == "absolute":
        return float(cfg["sigma"])
    if mode == "stability_fraction":
        frac = float(cfg.get("sigma_fraction", 0.8))
        s0 = gaussian_stability_threshold(n_agents, space.grid, space.weights)
        return frac * max(s0, 1e-3)
    raise ValueError(f"unknown sigma_mode {mode!r}")


def pairwise_spread(positions: np.ndarray) -> float:
    """Mean pairwise L2 distance among agents.

    :param positions: ``(N, L)`` array.
    :type positions: numpy.ndarray
    :returns: Mean off-diagonal distance.
    :rtype: float
    """
    n = positions.shape[0]
    if n < 2:
        return 0.0
    dists = [
        float(np.linalg.norm(positions[i] - positions[j]))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return float(np.mean(dists))


def simulate_position_only_loop(
    cfg: dict[str, Any],
    repo_root: Path,
    *,
    seed: int,
    sigma_fraction: float,
    n_rounds: int,
    batch_size: int,
    blend: float,
    position_step: Optional[dict[str, Any]],
    init_noise: float,
    routing_weight: str = "G",
    centroid_mode: str = "batch",
    init_mode: str = "mean_noise",
    theory_ref: Optional[np.ndarray] = None,
    blend_schedule: Optional[str] = None,
    blend_start: Optional[float] = None,
    space: Optional[TraitSpace] = None,
) -> list[dict[str, Any]]:
    """Run routing + weighted-centroid updates only (no SFT).

    :param cfg: Training YAML config.
    :type cfg: dict
    :param repo_root: Repository root for data paths.
    :type repo_root: pathlib.Path
    :param seed: RNG seed for batch sampling and routing.
    :type seed: int
    :param sigma_fraction: Multiplier on :math:`\\sigma_0^*`.
    :type sigma_fraction: float
    :param n_rounds: Number of closed-loop rounds.
    :type n_rounds: int
    :param batch_size: Queries per round.
    :type batch_size: int
    :param blend: EMA blend ceiling.
    :type blend: float
    :param position_step: Adaptive step policy dict.
    :type position_step: dict | None
    :param init_noise: Initial position perturbation std.
    :type init_noise: float
    :param init_mode: ``mean_noise`` (resource mean + noise) or
        ``pairs_near_theory`` (pair anchors from a (2,2) reference).
    :type init_mode: str
    :param theory_ref: ``(N, L)`` reference for ``pairs_near_theory``.
    :type theory_ref: numpy.ndarray | None
    :param routing_weight: ``'G'`` or ``'G_times_1mG'``.
    :type routing_weight: str
    :param centroid_mode: How to form the post-route centroid target.

        - ``batch``: sample ``batch_size`` prompts and route (default).
        - ``full_pool``: use every prompt in the corpus (large-batch MC).
        - ``expected_pool``: deterministic
          :func:`expected_pool_centroid` over the full pool (static limit).
    :type centroid_mode: str
    :param space: Pre-built trait space (avoids reloading the encoder).
    :type space: TraitSpace | None
    :returns: History records (same schema as full closed loop).
    :rtype: list[dict]
    :raises ValueError: If ``centroid_mode`` or ``routing_weight`` is invalid.
    """
    if centroid_mode not in ("batch", "full_pool", "expected_pool"):
        raise ValueError(
            f"centroid_mode must be batch, full_pool, or expected_pool, "
            f"got {centroid_mode!r}"
        )
    if centroid_mode == "expected_pool" and routing_weight != "G":
        raise ValueError(
            "expected_pool centroid_mode requires routing_weight='G' "
            "(position_only canonical semantics)."
        )
    cl = cfg.get("closed_loop", {})
    if space is None:
        space = _build_trait_space(cfg, repo_root)
    cl = cfg.get("closed_loop", {})
    sigma = _sigma_from_cfg(
        cfg, len(cfg.get("agents", [])), space,
        sigma_fraction_override=sigma_fraction,
    )
    theory_init_meta: Optional[dict[str, Any]] = None
    if init_mode == "theory_gradient":
        agents, meta = init_agents_theory_gradient(
            cfg,
            space,
            sigma=sigma,
            seed=seed,
            init_noise=init_noise,
            theory_cfg=cl.get("theory_gradient"),
        )
        theory_init_meta = {
            "init_mode": "theory_gradient",
            "theory_layout": meta["layout"],
            "theory_converged": meta["converged"],
            "theory_n_steps": meta["n_steps"],
            "theory_final_spread": meta["final_spread"],
            "theory_initial": meta["initial"].tolist(),
            "theory_end": meta["theory_end"].tolist(),
        }
    elif init_mode == "theory_gradient_paired":
        agents, meta = init_agents_theory_gradient_paired(
            cfg,
            space,
            sigma=sigma,
            seed=seed,
            init_noise=init_noise,
            theory_cfg=cl.get("theory_gradient"),
        )
        theory_init_meta = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in meta.items()
            if k not in ("initial",)
        }
        if "initial" in meta:
            theory_init_meta["theory_initial"] = meta["initial"].tolist()
        theory_init_meta["theory_end"] = np.asarray(meta["theory_end"]).tolist()
    elif init_mode == "pairs_near_theory":
        if theory_ref is None:
            names = sorted(a["name"] for a in cfg.get("agents", []))
            theory_ref = resolve_theory_22_reference(
                space, names, sigma=sigma, repo_root=repo_root,
                sigma_fraction=sigma_fraction,
            )
        agents = init_agents_pairs_near_reference(
            cfg, space, theory_ref, seed=seed, init_noise=init_noise,
        )
    elif init_mode == "mean_noise":
        agents = init_agents_mean_noise(
            cfg, space, seed=seed, init_noise=init_noise,
        )
    else:
        raise ValueError(
            f"init_mode must be mean_noise, pairs_near_theory, or "
            f"theory_gradient, theory_gradient_paired, got {init_mode!r}",
        )
    router = InfluencerRouter(
        space, agents, sigma=sigma, policy=cfg.get("policy", "proportional"),
    )
    rng = np.random.default_rng(seed)
    all_prompts = [p for s in _load_splits(cfg, repo_root) for p in s.prompts]
    pool_coords = space.project(all_prompts)

    pool_n = len(all_prompts)
    if centroid_mode == "full_pool":
        batch_size = pool_n

    history: list[dict[str, Any]] = []
    for r in range(n_rounds):
        blend_r = blend_for_round(
            r, n_rounds, blend, blend_schedule, blend_start=blend_start,
        )
        agent_blend: dict[str, list[float]] = {a.name: [] for a in agents}
        agent_prompts: dict[str, list[str]] = {a.name: [] for a in agents}
        agent_weights: dict[str, list[float]] = {a.name: [] for a in agents}
        observed = np.zeros(len(agents))

        if centroid_mode == "expected_pool":
            G_pool = allocation_weights(
                router.positions, pool_coords, router.cov,
            )
            for i, agent in enumerate(agents):
                target = expected_pool_centroid(i, pool_coords, G_pool)
                new_pos, beta_eff = apply_position_update(
                    agent.position,
                    target,
                    blend=blend_r,
                    position_step=position_step,
                )
                agent.position = new_pos
                agent_blend[agent.name] = [beta_eff]
            mass = G_pool * (1.0 - G_pool)
            observed = mass.sum(axis=1) / max(mass.sum(), 1e-30)
        else:
            idx = rng.integers(0, pool_n, size=batch_size)
            batch_prompts = [all_prompts[i] for i in idx]
            choices = router.route_batch(
                batch_prompts, rng=rng, routing_weight=routing_weight,
            )
            batch_coords = space.project(batch_prompts)
            G_batch = allocation_weights(
                router.positions, batch_coords, router.cov,
            )
            name_to_idx = {a.name: i for i, a in enumerate(agents)}

            for agent in agents:
                mine_idx = [
                    m for m, c in enumerate(choices) if c.name == agent.name
                ]
                if not mine_idx:
                    continue
                prompts_i = [batch_prompts[m] for m in mine_idx]
                i_idx = name_to_idx[agent.name]
                weights_i = (1.0 - G_batch[i_idx, mine_idx]).tolist()
                target = position_from_corpus(
                    prompts_i, space.project, scores=weights_i,
                )
                new_pos, beta_eff = apply_position_update(
                    agent.position,
                    target,
                    blend=blend_r,
                    position_step=position_step,
                )
                agent.position = new_pos
                agent_prompts[agent.name] = prompts_i
                agent_weights[agent.name] = weights_i
                agent_blend[agent.name] = [beta_eff]

            observed = np.array([
                sum(1 for c in choices if c.name == a.name) / max(len(choices), 1)
                for a in agents
            ])

        rec: dict[str, Any] = {
            "round": r,
            "positions": {a.name: a.position.tolist() for a in agents},
            "u_grid": router.expected_utilities().tolist(),
            "u_pool": empirical_utility(
                router.positions, pool_coords, router.cov,
            ).tolist(),
            "strategic_share_pool": strategic_routing_weights(
                router.positions, pool_coords, router.cov,
            ).mean(axis=1).tolist(),
            "observed_share": observed.tolist(),
            "routing_weight": routing_weight,
            "loss_reweight": "position_only",
            "simulation": "position_only_no_sft",
            "centroid_mode": centroid_mode,
            "batch_size_effective": int(batch_size),
            "pool_size": pool_n,
            "agent_prompts": agent_prompts,
            "agent_sample_weights": agent_weights,
            "agent_blend_effective": agent_blend,
            "position_step": position_step,
            "blend_base": blend,
            "blend_round": blend_r,
            "blend_schedule": blend_schedule,
            "sigma_fraction": sigma_fraction,
            "init_noise": init_noise,
            "init_mode": init_mode,
            "sigma": float(sigma),
            "pairwise_spread": pairwise_spread(router.positions),
        }
        if r == 0 and theory_init_meta is not None:
            rec["theory_init"] = theory_init_meta
        history.append(rec)
    return history


def replay_position_updates(
    history_path: Path,
    space: TraitSpace,
    *,
    blend: float,
    position_step: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Re-run centroid updates on corpora logged in *history_path*.

    Positions at round 0 match the source run; rounds ``1 …`` recompute
    updates from ``agent_prompts`` / ``agent_sample_weights`` only.

    :param history_path: Source ``history.json`` from a full or simulated run.
    :type history_path: pathlib.Path
    :param space: Trait space for projection.
    :type space: TraitSpace
    :param blend: EMA blend ceiling.
    :type blend: float
    :param position_step: New step policy to test.
    :type position_step: dict | None
    :returns: New history with replayed positions.
    :rtype: list[dict]
    """
    with history_path.open(encoding="utf-8") as fh:
        source = json.load(fh)
    if not source:
        raise ValueError(f"{history_path} is empty")

    names = list(source[0]["positions"].keys())
    positions = {
        n: np.asarray(source[0]["positions"][n], dtype=float) for n in names
    }
    out: list[dict[str, Any]] = [dict(source[0])]

    for r in range(1, len(source)):
        rec = source[r]
        agent_blend: dict[str, list[float]] = {n: [] for n in names}
        for name in names:
            prompts = rec.get("agent_prompts", {}).get(name, [])
            weights = rec.get("agent_sample_weights", {}).get(name, [])
            if not prompts:
                continue
            target = position_from_corpus(
                prompts,
                space.project,
                scores=weights if weights else None,
            )
            new_pos, beta_eff = apply_position_update(
                positions[name],
                target,
                blend=blend,
                position_step=position_step,
            )
            positions[name] = new_pos
            agent_blend[name] = [beta_eff]

        new_rec = dict(rec)
        new_rec["positions"] = {n: positions[n].tolist() for n in names}
        new_rec["agent_blend_effective"] = agent_blend
        new_rec["position_step"] = position_step
        new_rec["blend_base"] = blend
        new_rec["simulation"] = "replay_position_only"
        new_rec["replay_source"] = str(history_path)
        pos_arr = np.stack([positions[n] for n in names], axis=0)
        new_rec["pairwise_spread"] = pairwise_spread(pos_arr)
        out.append(new_rec)
    return out


def _position_step_dict(
    mode: str,
    step_cap: float,
    blend_max: float,
) -> dict[str, Any]:
    if mode == "static":
        return {"mode": "static"}
    return {
        "mode": mode,
        "step_cap": step_cap,
        "blend_max": blend_max,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    p.add_argument(
        "--mode", choices=("simulate", "replay"), default="simulate",
        help="simulate: route fresh each round; replay: frozen corpora from --history.",
    )
    p.add_argument("--history", type=Path, default=None,
                   help="Source history for replay mode.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sigma-fraction", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-rounds", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--centroid-mode",
        choices=("batch", "full_pool", "expected_pool"),
        default="batch",
        help="batch: subsample; full_pool: entire corpus + route; "
             "expected_pool: deterministic pool limit (static).",
    )
    p.add_argument(
        "--full-pool",
        action="store_true",
        help="Shorthand for --centroid-mode full_pool.",
    )
    p.add_argument("--blend", type=float, default=0.5)
    p.add_argument("--blend-schedule", type=str, default=None,
                   choices=("linear", "none"))
    p.add_argument("--blend-start", type=float, default=None)
    p.add_argument("--init-noise", type=float, default=None)
    p.add_argument(
        "--init-mode",
        choices=(
            "mean_noise",
            "pairs_near_theory",
            "theory_gradient",
            "theory_gradient_paired",
        ),
        default=None,
        help="mean_noise | pairs_near_theory | theory_gradient (GA from "
             "separated random start, init at theory endpoints) | "
             "theory_gradient_paired (co-located adjacent harm pairs). "
             "Defaults to closed_loop.init_mode or mean_noise.",
    )
    p.add_argument(
        "--theory-ref",
        type=Path,
        default=None,
        help="history.json with (2,2) final positions (per-sigma template).",
    )
    p.add_argument("--position-step-mode", type=str, default="static",
                   choices=("static", "cap_linf", "cap_l2", "trust_box"))
    p.add_argument("--step-cap", type=float, default=0.05)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    :param argv: Optional argument vector.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    args = _build_parser().parse_args(argv)
    if args.full_pool:
        args.centroid_mode = "full_pool"
    cfg = _load_yaml(args.config)
    cl = cfg.get("closed_loop", {})
    n_rounds = int(args.n_rounds if args.n_rounds is not None else cl.get("n_rounds", 10))
    batch_size = int(
        args.batch_size if args.batch_size is not None else cl.get("batch_size", 256)
    )
    centroid_mode = str(args.centroid_mode)
    init_noise = float(
        args.init_noise if args.init_noise is not None else cl.get("init_noise", 1e-4)
    )
    init_mode = str(args.init_mode or cl.get("init_mode", "mean_noise"))
    sigma_frac = float(
        args.sigma_fraction
        if args.sigma_fraction is not None
        else cfg.get("sigma_fraction", 0.5)
    )
    position_step = _position_step_dict(
        args.position_step_mode, args.step_cap, args.blend,
    )
    blend_schedule = args.blend_schedule
    if blend_schedule == "none":
        blend_schedule = None
    blend_start = args.blend_start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "history.json"

    if args.mode == "replay":
        if args.history is None:
            print("--history required for replay mode", file=sys.stderr)
            return 1
        space = _build_trait_space(cfg, args.repo_root)
        history = replay_position_updates(
            args.history,
            space,
            blend=args.blend,
            position_step=position_step,
        )
    else:
        theory_ref = None
        if init_mode == "pairs_near_theory" and args.theory_ref is not None:
            from infl_ens.utils.agent_init import load_reference_from_history
            theory_ref = load_reference_from_history(args.theory_ref)
        history = simulate_position_only_loop(
            cfg,
            args.repo_root,
            seed=args.seed,
            sigma_fraction=sigma_frac,
            n_rounds=n_rounds,
            batch_size=batch_size,
            blend=args.blend,
            position_step=position_step,
            init_noise=init_noise,
            centroid_mode=centroid_mode,
            init_mode=init_mode,
            theory_ref=theory_ref,
            blend_schedule=blend_schedule,
            blend_start=blend_start,
        )

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    final_spread = history[-1].get("pairwise_spread")
    if final_spread is None:
        names = list(history[-1]["positions"].keys())
        pos = np.stack(
            [np.asarray(history[-1]["positions"][n]) for n in names], axis=0,
        )
        final_spread = pairwise_spread(pos)

    print(
        f"wrote {out_path}  "
        f"(mode={args.mode}, centroid={centroid_mode}, "
        f"step={args.position_step_mode}, final_spread={final_spread:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
