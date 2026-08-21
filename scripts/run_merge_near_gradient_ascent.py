#!/usr/bin/env python3
"""Pair init on hypercube edges (or merge-near theory) + grid gradient ascent.

Prints and writes final positions after Nash solve (no matched-pool pre-warmup).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from infl_ens.data.trait_space_cache import build_or_load_safety_trait_space
from infl_ens.evaluation.benchmarks import load_benchmark_splits
from infl_ens.training.merge_training import parse_sft_merge_groups
from infl_ens.training.pool_dynamics import run_gradient_ascent_theory
from infl_ens.utils.agent_init import (
    init_agents_merge_hypercube_edges,
    init_agents_merge_near_theory,
)
from infl_ens.utils.resource import gaussian_stability_threshold


def _sigma_from_cfg(cfg: dict, n_agents: int, space) -> float:
    mode = cfg.get("sigma_mode", "stability_fraction")
    if mode == "absolute":
        return float(cfg["sigma"])
    if mode == "stability_fraction":
        sigma_star = gaussian_stability_threshold(n_agents, space.grid, space.weights)
        return float(cfg.get("sigma_fraction", 0.5)) * sigma_star
    raise ValueError(f"unknown sigma_mode {mode!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmark/router/seven_axis_collapse_near_theory.yaml",
    )
    parser.add_argument(
        "--init",
        choices=("hypercube_edges", "merge_near_theory"),
        default="hypercube_edges",
        help="hypercube_edges: random axis corners like (1,0,0,0,0); "
             "merge_near_theory: Nash anchors + pair offset",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default depends on --init)",
    )
    args = parser.parse_args()

    if args.output is None:
        out_name = (
            "hypercube_edge_gradient_ascent"
            if args.init == "hypercube_edges"
            else "merge_near_gradient_ascent"
        )
        args.output = f"results/{out_name}/fixed_positions.json"

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 0))
    cl = cfg.get("closed_loop", {})
    splits = load_benchmark_splits(cfg.get("benchmarks", []))
    space = build_or_load_safety_trait_space(cfg, splits)
    n_agents = len(cfg.get("agents", []))
    sigma = _sigma_from_cfg(cfg, n_agents, space)
    router_names = [a["name"] for a in cfg["agents"]]
    merge_groups = parse_sft_merge_groups(cl, router_names)
    if not merge_groups:
        raise SystemExit("config requires closed_loop.sft_merge_groups")

    if args.init == "hypercube_edges":
        agents, init_meta = init_agents_merge_hypercube_edges(
            cfg,
            space,
            merge_groups,
            seed=seed,
            init_noise=float(cl.get("init_noise", 0.0)),
            edge_cfg=cl.get("merge_hypercube"),
        )
    else:
        agents, init_meta = init_agents_merge_near_theory(
            cfg,
            space,
            merge_groups,
            sigma=sigma,
            seed=seed,
            init_noise=float(cl.get("init_noise", 0.0)),
            near_cfg=cl.get("merge_near"),
            theory_cfg=cl.get("theory_gradient"),
        )

    names = [a.name for a in agents]
    p0 = np.stack([a.position for a in agents], axis=0)
    tc = cl.get("theory_gradient") or {}
    grad = run_gradient_ascent_theory(
        space,
        p0,
        names,
        sigma=sigma,
        learning_rate=float(tc.get("learning_rate", 5e-3)),
        n_steps=int(tc.get("n_steps", 8000)),
        tol=float(tc.get("tol", 1e-8)),
        seed=seed,
    )
    final = grad["positions"][-1]

    within_pair: dict[str, float] = {}
    for train_as, members in merge_groups:
        i, j = names.index(members[0]), names.index(members[1])
        within_pair[train_as] = float(np.linalg.norm(final[i] - final[j]))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "init_mode": args.init,
        "sigma": sigma,
        "gradient_ascent": {
            "converged": grad["converged"],
            "n_steps": grad["n_steps"],
            "layout": grad["layout"],
            "final_spread": grad["final_spread"],
            "within_pair_distance": within_pair,
        },
        "theory_initial": {n: p0[i].tolist() for i, n in enumerate(names)},
        "positions": {n: final[i].tolist() for i, n in enumerate(names)},
        "init_meta": {
            k: v for k, v in init_meta.items()
            if k != "theory_initial"
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    labels = space.axis_labels or tuple(f"axis-{i}" for i in range(space.L))
    print(f"init={args.init}  sigma={sigma:.4f}  converged={grad['converged']}  steps={grad['n_steps']}")
    print(f"layout={grad['layout']}  spread={grad['final_spread']:.4f}")
    print(f"axes: {labels}")
    if "corner_assignments" in init_meta:
        print("\ncorner assignments (group -> axis, corner):")
        for train_as, info in init_meta["corner_assignments"].items():
            ax = labels[info["axis_index"]]
            c = info["corner"]
            print(f"  {train_as}: axis={ax} corner=[{', '.join(f'{x:.1f}' for x in c)}]")
    print(f"\n=== initial ({args.init}) ===")
    for n in names:
        v = p0[names.index(n)]
        print(f"  {n}: [{', '.join(f'{x:.4f}' for x in v)}]")
    print(f"\n=== after gradient ascent (final) ===")
    for n in names:
        v = final[names.index(n)]
        print(f"  {n}: [{', '.join(f'{x:.4f}' for x in v)}]")
    print(f"\nwithin-pair L2:")
    for k, d in within_pair.items():
        print(f"  {k}: {d:.4f}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
