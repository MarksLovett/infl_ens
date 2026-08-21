#!/usr/bin/env python3
"""Verify realized init spread for spread-calibrated random configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infl_ens.training.__main__ import (  # noqa: E402
    _init_agents_closed_loop,
    _load_splits,
    _load_yaml,
    _make_trait_space,
    _sigma_from_cfg,
)
from infl_ens.training.pool_dynamics import agent_pairwise_geometry  # noqa: E402


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = _load_yaml(cfg_path)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    cl = cfg.get("closed_loop") or {}
    init_noise = float(cl.get("init_noise", 0.0))
    if init_noise <= 0.0 and str(cl.get("init_mode")) == "mean_noise":
        print(f"ERROR: init_noise must be positive for mean_noise ({cfg_path.name})")
        return 1

    splits = _load_splits(cfg)
    space = _make_trait_space(cfg, splits)
    n_agents = len(cfg.get("agents", []))
    sigma = _sigma_from_cfg(cfg, n_agents, space)
    agents, _ = _init_agents_closed_loop(
        cfg, space, splits, cl, sigma=sigma, rng=np.random.default_rng(seed),
    )
    names = [a.name for a in agents]
    pos = np.stack([a.position for a in agents], axis=0)
    merge_groups = [
        ("merge-harm", ["clone-0", "clone-1"]),
        ("merge-hallucination", ["clone-2", "clone-3"]),
        ("merge-privacy", ["clone-4", "clone-5"]),
        ("merge-overrefusal", ["clone-6", "clone-7"]),
        ("merge-policy", ["clone-8", "clone-9"]),
    ]
    geom = agent_pairwise_geometry(pos, names, merge_groups=merge_groups)
    axis_names = getattr(space, "axis_names", None) or [f"axis-{i}" for i in range(pos.shape[1])]
    per_axis_std = {
        str(axis_names[i]): float(np.std(pos[:, i]))
        for i in range(pos.shape[1])
    }
    trait_mean = np.asarray(space.mean, dtype=float).tolist()
    header = cfg_path.read_text(encoding="utf-8").splitlines()
    target = None
    for line in header:
        if line.startswith("# target_mean_pairwise:"):
            target = float(line.split(":")[1].strip())
    row = {
        "config": str(cfg_path.relative_to(ROOT)),
        "seed": seed,
        "init_noise": init_noise,
        "target_mean_pairwise": target,
        "realized_mean_pairwise": geom["mean_pairwise_l2"],
        "per_axis_std": per_axis_std,
        "trait_mean": trait_mean,
        "within_merge_l2": geom["within_merge_l2"],
        "unique_positions": len({tuple(np.round(pos[i], 8)) for i in range(len(names))}),
    }
    print(json.dumps(row, indent=2))
    if target is not None and abs(row["realized_mean_pairwise"] - target) > 0.15:
        print(
            f"WARNING: realized spread {row['realized_mean_pairwise']:.3f} "
            f"far from target {target:.3f} (recalibrate before full sweep)",
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
