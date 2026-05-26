#!/usr/bin/env python3
"""Evaluate theory-match fixes 2–4 and rank variants.

Fix 2: calibrated blend (low constant or linear schedule).
Fix 3: matched-pool theory dynamics (same update as sim).
Fix 4: gradient-ascent theory from each seed's round-0 (diagnostic).

Runs fast position-only sims (``expected_pool`` + ``init_noise=0.01``), then
scores each variant on:

- ``p_22``: fraction of seeds reaching a (2,2) layout
- ``gap_sim_pool``: mean L2 distance sim end vs matched-pool end (fix 3)
- ``gap_sim_grad``: mean L2 distance sim end vs gradient-ascent end (fix 4)
- ``gap_pool_grad``: mean L2 pool end vs gradient end

Usage::

    python scripts/evaluate_theory_match_fixes.py \\
        --root results/theory_match_fixes \\
        --run-sweeps
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (str(SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from infl_ens.data.trait_space import TraitSpace  # noqa: E402
from infl_ens.training.pool_dynamics import (  # noqa: E402
    classify_layout,
    mean_agent_gap,
    pairwise_spread,
    run_gradient_ascent_theory,
    run_matched_pool_dynamics,
)
from simulate_position_only_loop import simulate_position_only_loop  # noqa: E402

BASELINE_REUSE_ROOT = ROOT / "results" / "pool_and_noise_10seeds"

_SIGMA_RE = re.compile(r"^sigma(?P<val>[0-9.]+)$")
_SEED_RE = re.compile(r"^seed(?P<val>\d+)$")


@dataclass(frozen=True)
class Variant:
    """One fix-2 blend configuration."""

    slug: str
    blend_base: float
    blend_schedule: Optional[str] = None
    blend_start: Optional[float] = None


VARIANTS: list[Variant] = [
    Variant("baseline_blend05", blend_base=0.5),
    Variant("blend_0.10", blend_base=0.10),
    Variant("blend_0.15", blend_base=0.15),
    Variant("blend_linear", blend_base=0.40, blend_schedule="linear", blend_start=0.10),
    Variant("blend_linear_low", blend_base=0.30, blend_schedule="linear", blend_start=0.05),
]


def _load_yaml(path: Path) -> dict:
    import yaml
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _build_space_and_pool(cfg: dict, repo: Path):
    from infl_ens.data.benchmarks import (
        build_safety_trait_space,
        load_beavertails,
        load_halueval,
    )
    from infl_ens.data.encoders import SentenceTransformerEncoder
    from infl_ens.utils.resource import gaussian_stability_threshold

    splits = []
    for entry in cfg.get("benchmarks", []):
        path = repo / entry["path"]
        if entry["kind"] == "beavertails":
            splits.append(load_beavertails(path, max_records=entry.get("max_records")))
        else:
            splits.append(load_halueval(
                path, tasks=entry.get("tasks"), max_records=entry.get("max_records"),
            ))
    ts = cfg.get("trait_space", {})
    enc = SentenceTransformerEncoder(model_name=ts.get(
        "encoder", "sentence-transformers/all-MiniLM-L6-v2",
    ))
    space = build_safety_trait_space(
        splits, enc,
        n_grid=int(ts.get("n_grid", 32)),
        kde_bandwidth=ts.get("kde_bandwidth"),
        threshold=float(ts.get("threshold", 0.5)),
    )
    pool = [p for s in splits for p in s.prompts]
    n_agents = len(cfg.get("agents", []))
    s0 = gaussian_stability_threshold(n_agents, space.grid, space.weights)
    return space, pool, s0


def _sigma_abs(cfg: dict, sigma_frac: float, s0: float) -> float:
    if cfg.get("sigma_mode") == "absolute":
        return float(cfg["sigma"])
    return float(sigma_frac) * max(s0, 1e-3)


def link_baseline_from_pool_noise(
    root: Path,
    sigma_frac: float,
    seed: int,
    *,
    reuse_root: Path = BASELINE_REUSE_ROOT,
) -> Optional[Path]:
    """Symlink baseline_blend05 cell from an existing pool_and_noise run.

    :param root: Sweep root.
    :type root: pathlib.Path
    :param sigma_frac: σ / σ₀*.
    :type sigma_frac: float
    :param seed: RNG seed.
    :type seed: int
    :param reuse_root: Directory with ``sigma*/seed*/history.json``.
    :type reuse_root: pathlib.Path
    :returns: Destination history path if linked, else ``None``.
    :rtype: pathlib.Path | None
    """
    src = reuse_root / f"sigma{sigma_frac}" / f"seed{seed}" / "history.json"
    if not src.is_file():
        return None
    dst = root / "baseline_blend05" / f"sigma{sigma_frac}" / f"seed{seed}" / "history.json"
    if dst.is_file():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())
    return dst


def run_sim_variant(
    variant: Variant,
    *,
    config: Path,
    root: Path,
    sigma_frac: float,
    seed: int,
    n_rounds: int,
    py: str,
    space: Optional[TraitSpace] = None,
    cfg: Optional[dict] = None,
) -> Path:
    """Launch one simulate cell (skip if history exists).

    :param variant: Blend configuration.
    :type variant: Variant
    :param config: YAML config path.
    :type config: pathlib.Path
    :param root: Sweep root.
    :type root: pathlib.Path
    :param sigma_frac: σ / σ₀*.
    :type sigma_frac: float
    :param seed: RNG seed.
    :type seed: int
    :param n_rounds: Closed-loop rounds.
    :type n_rounds: int
    :param py: Python executable.
    :type py: str
    :returns: Path to ``history.json``.
    :rtype: pathlib.Path
    """
    if variant.slug == "baseline_blend05":
        linked = link_baseline_from_pool_noise(root, sigma_frac, seed)
        if linked is not None:
            return linked

    out = root / variant.slug / f"sigma{sigma_frac}" / f"seed{seed}"
    hist = out / "history.json"
    if hist.is_file():
        return hist
    out.mkdir(parents=True, exist_ok=True)
    if cfg is None:
        cfg = _load_yaml(config)
    position_step = {"mode": "static"}
    history = simulate_position_only_loop(
        cfg,
        ROOT,
        seed=seed,
        sigma_fraction=sigma_frac,
        n_rounds=n_rounds,
        batch_size=256,
        blend=variant.blend_base,
        position_step=position_step,
        init_noise=0.01,
        routing_weight="G",
        centroid_mode="expected_pool",
        blend_schedule=variant.blend_schedule,
        blend_start=variant.blend_start,
        space=space,
    )
    hist.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return hist


def diagnose_cell(
    hist_path: Path,
    variant: Variant,
    *,
    cfg: dict,
    space,
    pool: list[str],
    s0: float,
    sigma_frac: float,
    n_rounds: int,
    theory_steps: int,
    theory_lr: float,
    grad_cache: Optional[dict] = None,
) -> dict[str, Any]:
    """Score one sim run against matched-pool and gradient theory.

    :param hist_path: ``history.json`` from sim.
    :type hist_path: pathlib.Path
    :param variant: Blend settings for matched pool.
    :type variant: Variant
    :param cfg: Training config (unused if *space* provided).
    :type cfg: dict
    :param space: Pre-built trait space.
    :type space: TraitSpace
    :param pool: Prompt pool.
    :type pool: list[str]
    :param s0: Stability threshold.
    :type s0: float
    :param sigma_frac: σ fraction.
    :type sigma_frac: float
    :param n_rounds: Sim rounds.
    :type n_rounds: int
    :param theory_steps: Gradient-ascent steps.
    :type theory_steps: int
    :param theory_lr: Gradient learning rate.
    :type theory_lr: float
    :returns: Metrics dict.
    :rtype: dict
    """
    records = json.loads(hist_path.read_text(encoding="utf-8"))
    names = list(records[0]["positions"].keys())
    p0 = np.stack([np.asarray(records[0]["positions"][n]) for n in names])
    sim_end = np.stack([np.asarray(records[-1]["positions"][n]) for n in names])

    sigma = _sigma_abs(cfg, sigma_frac, s0)

    pool_dyn = run_matched_pool_dynamics(
        space, p0, names, pool,
        sigma=sigma,
        n_rounds=n_rounds,
        blend_base=variant.blend_base,
        blend_schedule=variant.blend_schedule,
        blend_start=variant.blend_start,
    )
    if grad_cache is None:
        grad_dyn = run_gradient_ascent_theory(
            space, p0, names,
            sigma=sigma,
            learning_rate=theory_lr,
            n_steps=theory_steps,
            seed=0,
        )
    else:
        grad_dyn = grad_cache
    pool_end = pool_dyn["positions"][-1]
    grad_end = grad_dyn["positions"][-1]

    return {
        "sigma_fraction": sigma_frac,
        "seed": int(_SEED_RE.search(hist_path.parent.name).group("val")),
        "sim_spread": pairwise_spread(sim_end),
        "sim_layout": classify_layout(sim_end),
        "pool_layout": pool_dyn["layout"],
        "grad_layout": grad_dyn["layout"],
        "gap_sim_pool": mean_agent_gap(sim_end, pool_end),
        "gap_sim_grad": mean_agent_gap(sim_end, grad_end),
        "gap_pool_grad": mean_agent_gap(pool_end, grad_end),
        "grad_converged": grad_dyn["converged"],
        "grad_n_steps": grad_dyn["n_steps"],
    }


def aggregate_scores(rows: list[dict], variant_slug: str) -> dict[str, Any]:
    """Aggregate per-cell rows for one variant.

    :param rows: Cell metric dicts.
    :type rows: list[dict]
    :param variant_slug: Variant name.
    :type variant_slug: str
    :returns: Summary statistics.
    :rtype: dict
    """
    n = len(rows)
    p22 = sum(1 for r in rows if r["sim_layout"] == "2,2")
    rows_22 = [r for r in rows if r["sim_layout"] == "2,2"]
    return {
        "variant": variant_slug,
        "n": n,
        "p_22": p22 / max(n, 1),
        "n_22": p22,
        "mean_spread": float(np.mean([r["sim_spread"] for r in rows])),
        "gap_sim_pool_all": float(np.mean([r["gap_sim_pool"] for r in rows])),
        "gap_sim_pool_22": (
            float(np.mean([r["gap_sim_pool"] for r in rows_22])) if rows_22 else float("nan")
        ),
        "gap_sim_grad_all": float(np.mean([r["gap_sim_grad"] for r in rows])),
        "gap_sim_grad_22": (
            float(np.mean([r["gap_sim_grad"] for r in rows_22])) if rows_22 else float("nan")
        ),
        "gap_pool_grad_all": float(np.mean([r["gap_pool_grad"] for r in rows])),
        "grad_reaches_22": sum(1 for r in rows if r["grad_layout"] == "2,2") / max(n, 1),
    }


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional CLI vector.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "results/theory_match_fixes")
    p.add_argument("--config", type=Path,
                   default=ROOT / "configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml")
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.25, 0.75])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--n-rounds", type=int, default=20)
    p.add_argument("--theory-steps", type=int, default=5000)
    p.add_argument("--theory-lr", type=float, default=5e-3)
    p.add_argument("--run-sweeps", action="store_true",
                   help="Run simulate cells before diagnosing.")
    p.add_argument("--py", type=str, default=sys.executable)
    args = p.parse_args(argv)

    cfg = _load_yaml(args.config)
    args.root.mkdir(parents=True, exist_ok=True)

    print("building trait space (once) ...")
    space, pool, s0 = _build_space_and_pool(cfg, ROOT)

    if args.run_sweeps:
        for v in VARIANTS:
            for sigma in args.sigmas:
                for seed in args.seeds:
                    print(f"[sim] {v.slug} sigma={sigma} seed={seed}")
                    run_sim_variant(
                        v, config=args.config, root=args.root,
                        sigma_frac=sigma, seed=seed, n_rounds=args.n_rounds, py=args.py,
                        space=space, cfg=cfg,
                    )

    grad_cache: dict[tuple[float, int], dict] = {}
    all_rows: list[dict] = []
    for v in VARIANTS:
        for sigma in args.sigmas:
            for seed in args.seeds:
                hist = args.root / v.slug / f"sigma{sigma}" / f"seed{seed}" / "history.json"
                if not hist.is_file():
                    print(f"missing {hist}", file=sys.stderr)
                    continue
                key = (sigma, seed)
                if key not in grad_cache:
                    records = json.loads(hist.read_text(encoding="utf-8"))
                    names = list(records[0]["positions"].keys())
                    p0 = np.stack([
                        np.asarray(records[0]["positions"][n]) for n in names
                    ])
                    grad_cache[key] = run_gradient_ascent_theory(
                        space, p0, names,
                        sigma=_sigma_abs(cfg, sigma, s0),
                        learning_rate=args.theory_lr,
                        n_steps=args.theory_steps,
                    )
                row = diagnose_cell(
                    hist, v, cfg=cfg, space=space, pool=pool, s0=s0,
                    sigma_frac=sigma, n_rounds=args.n_rounds,
                    theory_steps=args.theory_steps, theory_lr=args.theory_lr,
                    grad_cache=grad_cache[key],
                )
                row["variant"] = v.slug
                all_rows.append(row)

    summaries = [aggregate_scores(
        [r for r in all_rows if r["variant"] == v.slug], v.slug,
    ) for v in VARIANTS]

    # Rank: primary = p_22, secondary = gap_sim_pool_22 (lower better)
    def score(s: dict) -> tuple:
        g = s["gap_sim_pool_22"]
        gkey = g if g == g else 999.0
        return (-s["p_22"], gkey, -s["mean_spread"])

    summaries.sort(key=score)

    print("\n" + "=" * 72)
    print("  THEORY-MATCH FIX RANKING  (pool_and_noise inits, expected_pool)")
    print("=" * 72)
    print(f"{'variant':<22} {'p(2,2)':>8} {'spread':>8} "
          f"{'gap_s/p':>8} {'gap_s/g':>8} {'gap_p/g':>8} {'grad_22':>8}")
    print("-" * 72)
    for s in summaries:
        print(
            f"{s['variant']:<22} {s['p_22']:7.0%} {s['mean_spread']:8.3f} "
            f"{s['gap_sim_pool_all']:8.3f} {s['gap_sim_grad_all']:8.3f} "
            f"{s['gap_pool_grad_all']:8.3f} {s['grad_reaches_22']:7.0%}"
        )

    best = summaries[0]
    print("\n--- RECOMMENDED CONFIG (fix 2 + existing pool/noise) ---")
    vbest = next(v for v in VARIANTS if v.slug == best["variant"])
    print(f"  centroid_mode: expected_pool")
    print(f"  init_noise: 0.01")
    print(f"  batch_size: 256")
    print(f"  blend: {vbest.blend_base}")
    if vbest.blend_schedule:
        print(f"  blend_schedule: {vbest.blend_schedule}")
        print(f"  blend_start: {vbest.blend_start}")
    print(f"\n  (2,2) rate: {best['p_22']:.0%}  |  "
          f"mean gap(sim,matched_pool) on (2,2) seeds: {best['gap_sim_pool_22']:.3f}")

    out_json = args.root / "ranking.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump({"summaries": summaries, "cells": all_rows}, fh, indent=2)
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
