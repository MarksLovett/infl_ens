#!/usr/bin/env python3
"""Verify oracle-centroid colocated init realizes intended positions."""

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

MERGE_GROUPS = [
    ("merge-harm", ["clone-0", "clone-1"]),
    ("merge-hallucination", ["clone-2", "clone-3"]),
    ("merge-privacy", ["clone-4", "clone-5"]),
    ("merge-overrefusal", ["clone-6", "clone-7"]),
    ("merge-policy", ["clone-8", "clone-9"]),
]


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmark/router/oracle_centroid_shift/ga_theory_pre.yaml",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--metadata",
        default="results/oracle_centroid_shift_init/centroid_metadata.json",
    )
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    cfg = _load_yaml(cfg_path)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    cl = cfg.get("closed_loop") or {}

    splits = _load_splits(cfg)
    space = _make_trait_space(cfg, splits)
    n_agents = len(cfg.get("agents", []))
    sigma = _sigma_from_cfg(cfg, n_agents, space)
    agents, _ = _init_agents_closed_loop(
        cfg, space, splits, cl, sigma=sigma, rng=np.random.default_rng(seed),
    )
    names = [a.name for a in agents]
    pos = np.stack([a.position for a in agents], axis=0)
    geom = agent_pairwise_geometry(pos, names, merge_groups=MERGE_GROUPS)

    meta_path = ROOT / args.metadata
    expected: dict[str, list[float]] = {}
    per_merge_meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = meta.get("positions", {})
        per_merge_meta = meta.get("per_merge", {})

    pair_checks: list[dict] = []
    ok = True
    for merge, (c0, c1) in MERGE_GROUPS:
        p0 = pos[names.index(c0)]
        p1 = pos[names.index(c1)]
        within = float(np.linalg.norm(p0 - p1))
        exp = np.asarray(expected.get(c0, p0), dtype=float) if expected else p0
        delta = float(np.linalg.norm(p0 - exp))
        pair_checks.append({
            "merge": merge,
            "clone_0": c0,
            "clone_1": c1,
            "within_merge_l2": within,
            "delta_from_expected_centroid": delta,
            "position": p0.tolist(),
            "expected_centroid": exp.tolist(),
        })
        if within > 1e-9:
            ok = False
        if expected and delta > 1e-6:
            ok = False

    trait_mean = np.asarray(space.mean, dtype=float).tolist()
    row = {
        "config": str(cfg_path.relative_to(ROOT)),
        "seed": seed,
        "sigma": sigma,
        "trait_mean": trait_mean,
        "trait_dim": space.L,
        "within_merge_l2": geom["within_merge_l2"],
        "mean_pairwise_l2": geom["mean_pairwise_l2"],
        "pair_checks": pair_checks,
        "per_merge_metadata": per_merge_meta,
        "ok": ok,
    }
    print(json.dumps(row, indent=2))
    if not ok:
        print("ERROR: colocated init verification failed")
        return 1
    print("OK: colocated oracle-centroid init verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
