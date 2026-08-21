"""Compare strategic gradient-ascent equilibria with SFT-driven closed-loop positions.

Example::

    python scripts/compare_theory_vs_sft.py \\
        --config configs/benchmark/router/safety_truth_n4_r10.yaml \\
        --history results/safety_truth_n4_r10/history.json \\
        --output-stem scripts/figures/theory_vs_sft_n4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from infl_ens.inflgame.router import InfluencerRouter, RouterAgent, empirical_utility
from infl_ens.training.pool_dynamics import run_matched_pool_dynamics
from infl_ens.training.theory_vs_sft import (
    build_theory_summary,
    run_strategic_ascent,
    sft_trajectory_from_history,
)
from infl_ens.vis.theory_vs_sft import plot_theory_vs_sft_comparison

FIGS_DIR = Path(__file__).resolve().parent / "figures"


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    :returns: Configured argparse parser.
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="Compare strategic gradient-ascent NE with SFT closed-loop endpoints.",
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--learning-rate", type=float, default=5e-3)
    p.add_argument("--n-steps", type=int, default=5000)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--axis-labels", nargs=2, default=["harm", "hallucination"])
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--output-stem", type=Path, default=None)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--sigma-fraction-override", type=float, default=None)
    p.add_argument("--sigma-override", type=float, default=None)
    p.add_argument(
        "--theory-mode",
        choices=("gradient", "matched_pool", "both"),
        default="gradient",
    )
    p.add_argument("--theory-rounds", type=int, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional CLI argument vector.
    :type argv: list[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    args = _build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if args.sigma_fraction_override is not None and args.sigma_override is not None:
        print(
            "error: --sigma-fraction-override and --sigma-override are mutually exclusive",
            file=sys.stderr,
        )
        return 1
    if args.sigma_fraction_override is not None:
        cfg = dict(cfg)
        cfg["sigma_mode"] = "stability_fraction"
        cfg["sigma_fraction"] = float(args.sigma_fraction_override)
        print(f"  sigma override     = {cfg['sigma_fraction']:.4f} sigma_0*")
    elif args.sigma_override is not None:
        cfg = dict(cfg)
        cfg["sigma_mode"] = "absolute"
        cfg["sigma"] = float(args.sigma_override)
        print(f"  sigma override     = {cfg['sigma']:.4f} (absolute)")

    with args.history.open(encoding="utf-8") as fh:
        hist_records = json.load(fh)
    n_sft_rounds = len(hist_records) - 1
    cl_meta = hist_records[0] if hist_records else {}
    sim_tag = str(cl_meta.get("simulation", ""))
    dyn_label = "position-only" if "position_only" in sim_tag else "SFT"

    print("rebuilding trait space and running strategic gradient ascent ...")
    info = run_strategic_ascent(
        cfg,
        args.history,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        tol=args.tol,
        seed=args.seed,
    )
    info["dyn_label"] = dyn_label
    grad_end = info["positions"][-1]

    pool_end = None
    if args.theory_mode in ("matched_pool", "both"):
        pool_prompts = [p for s in info["splits"] for p in s.prompts]
        n_pool_rounds = args.theory_rounds if args.theory_rounds is not None else n_sft_rounds
        blend_base = float(cl_meta.get("blend_base", cl_meta.get("blend", 0.5)))
        pool_dyn = run_matched_pool_dynamics(
            info["space"],
            info["initial_positions"],
            info["names"],
            pool_prompts,
            sigma=info["sigma"],
            n_rounds=n_pool_rounds,
            blend_base=blend_base,
            blend_schedule=cl_meta.get("blend_schedule"),
            blend_start=cl_meta.get("blend_start"),
        )
        pool_end = pool_dyn["positions"][-1]
        print(f"  matched-pool rounds = {n_pool_rounds}  layout = {pool_dyn['layout']}")

    theo_end_for_plot = pool_end if (
        args.theory_mode == "matched_pool" and pool_end is not None
    ) else grad_end

    print(f"  sigma_0*           = {info['sigma_star']:.4f}")
    print(f"  sigma              = {info['sigma']:.4f}"
          f"  ({info['sigma']/info['sigma_star']:.2f} sigma_0*)")
    print(f"  theory converged   = {info['converged']}  after {info['n_steps']} steps")

    sft_traj = sft_trajectory_from_history(args.history, info["names"])
    print(f"  {dyn_label} trajectory = {sft_traj.shape[0]} rounds, {sft_traj.shape[1]} agents")

    sft_end = sft_traj[-1]
    theo_end = theo_end_for_plot
    mu = info["space"].mean
    hdr = f"\n{'agent':<10} {dyn_label + ' end':>20} {'theory end':>20} {'gap':>8}"
    if pool_end is not None and args.theory_mode == "both":
        hdr += f" {'gap(pool)':>9}"
    print(hdr + f" {'d(SFT,μ)':>10}")
    print("-" * 92)
    for i, name in enumerate(info["names"]):
        gap = float(np.linalg.norm(sft_end[i] - theo_end[i]))
        d_sft = float(np.linalg.norm(sft_end[i] - mu))
        line = (
            f"{name:<10} "
            f"({sft_end[i, 0]:.3f}, {sft_end[i, 1]:.3f})"
            f"   ({theo_end[i, 0]:.3f}, {theo_end[i, 1]:.3f})"
            f"   {gap:.3f}"
        )
        if pool_end is not None and args.theory_mode == "both":
            line += f"   {float(np.linalg.norm(sft_end[i] - pool_end[i])):.3f}"
        line += f"    {d_sft:.3f}"
        print(line)

    pool_corpus = [p for s in info["splits"] for p in s.prompts]
    pool_coords = info["space"].project(pool_corpus)
    router_sft = InfluencerRouter(
        info["space"],
        [RouterAgent(name=n, position=sft_end[i].copy()) for i, n in enumerate(info["names"])],
        sigma=info["sigma"],
        policy="proportional",
    )
    router_theo = InfluencerRouter(
        info["space"],
        [RouterAgent(name=n, position=theo_end[i].copy()) for i, n in enumerate(info["names"])],
        sigma=info["sigma"],
        policy="proportional",
    )
    u_pool_sft = empirical_utility(router_sft.positions, pool_coords, router_sft.cov)
    u_pool_theo = empirical_utility(router_theo.positions, pool_coords, router_theo.cov)
    print(f"\n{'agent':<10} {'u_pool(SFT)':>12} {'u_pool(theory)':>16}")
    print("-" * 42)
    for i, name in enumerate(info["names"]):
        print(f"{name:<10} {u_pool_sft[i]:>12.4f} {u_pool_theo[i]:>16.4f}")

    stem = args.output_stem or (FIGS_DIR / "theory_vs_sft")
    plot_theory_vs_sft_comparison(
        info,
        sft_traj,
        axis_labels=(args.axis_labels[0], args.axis_labels[1]),
        title=args.title,
        dyn_label=dyn_label,
        output_stem=stem,
    )
    for ext in ("pdf", "png"):
        print(f"\nwrote {stem.with_suffix('.' + ext)}")

    if args.summary_json is not None:
        summary = build_theory_summary(
            info,
            sft_end,
            theo_end,
            config_path=args.config,
            history_path=args.history,
            pool_end=pool_end,
            u_pool_sft=u_pool_sft,
            u_pool_theo=u_pool_theo,
        )
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {args.summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
