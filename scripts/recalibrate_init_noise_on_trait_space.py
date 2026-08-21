#!/usr/bin/env python3
"""Recalibrate init_noise against real trait-space geometry."""

from __future__ import annotations

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


def mean_pairwise(
    cfg_path: Path, init_noise: float, seed: int = 0,
) -> float:
    """Realized mean pairwise L2 at initialization."""
    cfg = _load_yaml(cfg_path)
    cl = dict(cfg.get("closed_loop") or {})
    cl["init_noise"] = init_noise
    splits = _load_splits(cfg)
    space = _make_trait_space(cfg, splits)
    sigma = _sigma_from_cfg(cfg, len(cfg.get("agents", [])), space)
    agents, _ = _init_agents_closed_loop(
        cfg, space, splits, cl, sigma=sigma, rng=np.random.default_rng(seed),
    )
    names = [a.name for a in agents]
    pos = np.stack([a.position for a in agents], axis=0)
    return float(agent_pairwise_geometry(pos, names)["mean_pairwise_l2"])


def solve(cfg_path: Path, target: float, seed: int = 0) -> tuple[float, float]:
    """Binary-search ``init_noise`` for ``target`` mean pairwise."""
    lo, hi = 1e-4, 2.0
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        val = mean_pairwise(cfg_path, mid, seed=seed)
        if val < target:
            lo = mid
        else:
            hi = mid
    realized = mean_pairwise(cfg_path, mid, seed=seed)
    return mid, realized


def main() -> int:
    """CLI entry point."""
    cfg = ROOT / (
        "configs/benchmark/router/attribution_spread_rerun/"
        "random_s09_theory_pre_seed0.yaml"
    )
    s09, r09 = solve(cfg, 0.9)
    s045, r045 = solve(
        ROOT / "configs/benchmark/router/attribution_spread_rerun/"
        "random_s045_theory_pre_seed0.yaml",
        0.45,
    )
    print(f"matched: init_noise={s09:.6f} realized={r09:.4f}")
    print(f"moderate: init_noise={s045:.6f} realized={r045:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
