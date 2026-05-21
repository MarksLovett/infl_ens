"""Compare strategic gradient-ascent equilibria with SFT-driven closed-loop positions.

The closed-loop trainer moves agent positions through **capability drift** —
each agent shifts toward the centroid of queries it actually fine-tuned on.
The strategic Nash solver :func:`train_router_positions` moves positions
through **payoff maximisation** — each agent shifts up the gradient of its
own expected utility under a fixed competitive reach :math:`\\sigma`.

These are different mechanisms. The paper (Lovett & Fu 2024) argues both
produce hierarchical sub-niche structure below :math:`\\sigma_0^*`. This
script lets you check whether they produce the *same* structure on a real
trait space, by:

1. Loading the YAML config used for a closed-loop run and the corresponding
   ``history.json``.
2. Rebuilding the trait space from the same benchmarks, encoder, KDE
   bandwidth, etc.
3. Reading the round-0 positions from ``history.json`` and using them as
   the initial condition for :func:`train_router_positions` — so both
   trajectories share an initial state.
4. Plotting both trajectories side-by-side in trait space, with theoretical
   endpoints marked.

Run with::

    python scripts/compare_theory_vs_sft.py \\
        --config  configs/benchmark/router/safety_truth_n4_r10.yaml \\
        --history results/safety_truth_n4_r10/history.json \\
        --output-stem scripts/figures/theory_vs_sft_n4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infl_ens.data.benchmarks import (
    build_safety_trait_space,
    load_beavertails,
    load_halueval,
)
from infl_ens.data.encoders import SentenceTransformerEncoder
from infl_ens.inflgame.router import (
    InfluencerRouter,
    RouterAgent,
    empirical_utility,
)
from infl_ens.training.router_training import (
    RouterTrainingConfig,
    train_router_positions,
)
from infl_ens.utils.resource import gaussian_stability_threshold


FIGS_DIR = ROOT / "scripts" / "figures"


def _load_yaml(path: Path) -> dict:
    """Load a YAML config, falling back to a tiny parser if PyYAML is absent.

    :param path: Config file path.
    :type path: pathlib.Path
    :returns: Parsed configuration dictionary.
    :rtype: dict
    """
    try:
        import yaml
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ImportError as exc:  # pragma: no cover - PyYAML is a hard dep elsewhere
        raise ImportError(
            "PyYAML is required to read configs; pip install PyYAML"
        ) from exc


def _load_splits(cfg: dict) -> list:
    """Recreate the BenchmarkSplits from the config.

    :param cfg: Parsed config.
    :type cfg: dict
    :returns: List of BenchmarkSplit objects, in declaration order.
    :rtype: list[infl_ens.data.benchmarks.BenchmarkSplit]
    :raises ValueError: For unknown benchmark kinds.
    """
    splits = []
    for entry in cfg.get("benchmarks", []):
        kind = entry["kind"]
        if kind == "beavertails":
            splits.append(load_beavertails(
                Path(entry["path"]),
                max_records=entry.get("max_records"),
            ))
        elif kind == "halueval":
            splits.append(load_halueval(
                Path(entry["path"]),
                tasks=entry.get("tasks"),
                max_records=entry.get("max_records"),
            ))
        else:
            raise ValueError(f"unknown benchmark kind: {kind!r}")
    return splits


def _build_trait_space(cfg: dict, splits: list):
    """Reconstruct the TraitSpace exactly as the original run did.

    :param cfg: Parsed config.
    :type cfg: dict
    :param splits: Loaded benchmark splits.
    :type splits: list[infl_ens.data.benchmarks.BenchmarkSplit]
    :returns: Reconstructed trait space.
    :rtype: infl_ens.data.trait_space.TraitSpace
    """
    ts_cfg = cfg.get("trait_space", {})
    encoder_name = ts_cfg.get("encoder", "sentence-transformers/all-MiniLM-L6-v2")
    encoder = SentenceTransformerEncoder(model_name=encoder_name)
    return build_safety_trait_space(
        splits, lambda xs: encoder(list(xs)),
        n_grid=int(ts_cfg.get("n_grid", 32)),
        kde_bandwidth=ts_cfg.get("kde_bandwidth"),
        threshold=float(ts_cfg.get("threshold", 0.5)),
    )


def _initial_positions_from_history(history_path: Path) -> tuple[list[str], np.ndarray]:
    """Extract round-0 positions from a closed-loop ``history.json``.

    :param history_path: Path to the history file.
    :type history_path: pathlib.Path
    :returns: Tuple of ``(agent_names, positions)`` with positions of shape
        ``(N, L)``.
    :rtype: tuple[list[str], numpy.ndarray]
    :raises ValueError: If the file is empty.
    """
    with history_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    if not records:
        raise ValueError(f"{history_path} contains no rounds")
    r0 = records[0]
    names = list(r0["positions"].keys())
    pos = np.stack([np.asarray(r0["positions"][n]) for n in names], axis=0)
    return names, pos


def _sft_trajectory(history_path: Path, names: Sequence[str]) -> np.ndarray:
    """Stack per-round SFT positions into a ``(T, N, L)`` tensor.

    :param history_path: Path to the history file.
    :type history_path: pathlib.Path
    :param names: Agent name order.
    :type names: Sequence[str]
    :returns: Positions tensor.
    :rtype: numpy.ndarray
    """
    with history_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    return np.stack(
        [
            np.stack([np.asarray(r["positions"][n]) for n in names], axis=0)
            for r in records
        ],
        axis=0,
    )


def _sigma_from_cfg(cfg: dict, n_agents: int, space) -> float:
    """Compute the absolute sigma value from a config.

    Supports the same two modes as ``training/__main__.py``:
    ``sigma_mode='absolute'`` (use ``sigma`` field) or
    ``sigma_mode='stability_fraction'`` (use ``sigma_fraction`` of
    :math:`\\sigma_0^*`).

    :param cfg: Parsed config.
    :type cfg: dict
    :param n_agents: Number of agents.
    :type n_agents: int
    :param space: Trait space (for the stability threshold).
    :type space: infl_ens.data.trait_space.TraitSpace
    :returns: Absolute scalar sigma.
    :rtype: float
    """
    mode = cfg.get("sigma_mode", "stability_fraction")
    if mode == "absolute":
        return float(cfg["sigma"])
    if mode == "stability_fraction":
        sigma_star = gaussian_stability_threshold(n_agents, space.grid, space.weights)
        return float(cfg.get("sigma_fraction", 0.5)) * max(sigma_star, 0.05)
    raise ValueError(f"unknown sigma_mode: {mode!r}")


def run_strategic_ascent(
    cfg: dict,
    history_path: Path,
    *,
    learning_rate: float = 5e-3,
    n_steps: int = 5000,
    tol: float = 1e-8,
    seed: int = 0,
) -> dict:
    """Run the strategic gradient-ascent solver from the SFT run's initial state.

    The trait space is rebuilt from ``cfg`` exactly as the SFT run built
    it. Agent positions are initialised to the round-0 positions read from
    ``history_path``, so both trajectories share an initial condition and
    can be plotted on the same axes.

    :param cfg: Parsed YAML config of the SFT run.
    :type cfg: dict
    :param history_path: Path to that run's ``history.json``.
    :type history_path: pathlib.Path
    :param learning_rate: Step size for gradient ascent.
    :type learning_rate: float
    :param n_steps: Maximum gradient steps.
    :type n_steps: int
    :param tol: Convergence tolerance on max coordinate change.
    :type tol: float
    :param seed: RNG seed forwarded to the trainer.
    :type seed: int
    :returns: Dictionary with keys ``space``, ``agents`` (final),
        ``positions`` ``(T+1, N, L)``, ``utilities`` ``(T, N)``,
        ``converged``, ``n_steps`` (taken), ``sigma``, ``sigma_star``,
        ``initial_positions``.
    :rtype: dict
    """
    splits = _load_splits(cfg)
    space = _build_trait_space(cfg, splits)
    names, init_pos = _initial_positions_from_history(history_path)

    agents = [
        RouterAgent(name=n, position=init_pos[i].copy())
        for i, n in enumerate(names)
    ]
    sigma = _sigma_from_cfg(cfg, len(agents), space)
    sigma_star = gaussian_stability_threshold(
        len(agents), space.grid, space.weights,
    )

    rt_cfg = RouterTrainingConfig(
        sigma=sigma,
        learning_rate=learning_rate,
        n_steps=n_steps,
        tol=tol,
        clip_to_box=True,
    )
    info = train_router_positions(space, agents, rt_cfg, seed=seed)
    info.update({
        "space": space,
        "agents": agents,
        "sigma": sigma,
        "sigma_star": float(sigma_star),
        "initial_positions": init_pos,
        "names": names,
        "splits": splits,
    })
    return info


def plot_comparison(
    info: dict,
    sft_traj: np.ndarray,
    *,
    axis_labels: tuple[str, str] = ("harm", "hallucination"),
    title: Optional[str] = None,
):
    """Render a two-panel comparison figure.

    :param info: Output of :func:`run_strategic_ascent`.
    :type info: dict
    :param sft_traj: SFT trajectory tensor, shape ``(T, N, L)``.
    :type sft_traj: numpy.ndarray
    :param axis_labels: Axis names.
    :type axis_labels: tuple[str, str]
    :param title: Optional suptitle.
    :type title: str | None
    :returns: Matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    names = info["names"]
    n_agents = len(names)
    theo_traj = info["positions"]            # (T_theo+1, N, L)
    sigma = info["sigma"]
    sigma_star = info["sigma_star"]
    space = info["space"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(n_agents, 3)))

    # --- Left: trait-space trajectories side-by-side ---------------------
    ax = axes[0]
    for i, name in enumerate(names):
        # Theoretical (strategic gradient ascent)
        tx, ty = theo_traj[:, i, 0], theo_traj[:, i, 1]
        ax.plot(tx, ty, "-", color=colors[i], lw=1.0, alpha=0.55,
                label=f"{name} (theory)" if i == 0 else None)
        ax.scatter(tx[-1], ty[-1], color=colors[i], marker="X", s=130,
                   edgecolor="black", linewidth=0.7, zorder=4)
        # SFT (closed-loop)
        sx, sy = sft_traj[:, i, 0], sft_traj[:, i, 1]
        ax.plot(sx, sy, "--", color=colors[i], lw=1.8, alpha=0.95,
                label=f"{name} (SFT)" if i == 0 else None)
        ax.scatter(sx[0], sy[0], color=colors[i], marker="o", s=55,
                   edgecolor="black", linewidth=0.6, zorder=3)
        ax.scatter(sx[-1], sy[-1], color=colors[i], marker="*", s=200,
                   edgecolor="black", linewidth=0.7, zorder=4)

    # Resource centroid for visual context
    mu = space.mean
    ax.scatter([mu[0]], [mu[1]], marker="+", s=180, color="black",
               linewidth=1.6, label=r"$\mathbb{E}_B[b]$", zorder=5)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"trajectories  •  ○ start, ★ SFT end, ✕ theory end\n"
        f"σ = {sigma:.3f},  σ₀* = {sigma_star:.3f}  ({sigma/sigma_star:.2f}·σ₀*)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=True)

    # --- Right: bar chart of SFT-end vs theory-end distance to centroid ---
    ax2 = axes[1]
    sft_end = sft_traj[-1]                  # (N, L)
    theo_end = theo_traj[-1]                # (N, L)
    d_sft = np.linalg.norm(sft_end - mu[None, :], axis=1)
    d_theo = np.linalg.norm(theo_end - mu[None, :], axis=1)
    d_pairwise = np.linalg.norm(sft_end - theo_end, axis=1)
    x = np.arange(n_agents)
    w = 0.27
    ax2.bar(x - w, d_theo, w, color="lightgray", edgecolor="black",
            label="theory NE  →  centroid")
    ax2.bar(x,     d_sft,  w, color="steelblue", edgecolor="black",
            label="SFT end    →  centroid")
    ax2.bar(x + w, d_pairwise, w, color="tomato", edgecolor="black",
            label="SFT end  →  theory NE")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=0)
    ax2.set_ylabel("L2 distance in trait space")
    ax2.set_title("specialisation depth & theory ↔ SFT gap")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="best", fontsize=8, frameon=True)

    if title is not None:
        fig.suptitle(title)
    return fig


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    :returns: Configured argparse parser.
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="Compare strategic gradient-ascent NE with SFT closed-loop endpoints."
    )
    p.add_argument("--config", type=Path, required=True,
                   help="YAML config used for the SFT closed-loop run.")
    p.add_argument("--history", type=Path, required=True,
                   help="history.json produced by that run.")
    p.add_argument("--learning-rate", type=float, default=5e-3)
    p.add_argument("--n-steps", type=int, default=5000)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--axis-labels", nargs=2, default=["harm", "hallucination"])
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--output-stem", type=Path, default=None,
                   help="Output filename stem; defaults to "
                        "scripts/figures/theory_vs_sft.")
    p.add_argument("--summary-json", type=Path, default=None,
                   help="Optional JSON summary path with per-agent endpoints.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional CLI argument vector.
    :type argv: list[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    args = _build_parser().parse_args(argv)
    cfg = _load_yaml(args.config)

    print("rebuilding trait space and running strategic gradient ascent ...")
    info = run_strategic_ascent(
        cfg, args.history,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        tol=args.tol,
        seed=args.seed,
    )
    print(f"  sigma_0*           = {info['sigma_star']:.4f}")
    print(f"  sigma              = {info['sigma']:.4f}"
          f"  ({info['sigma']/info['sigma_star']:.2f} sigma_0*)")
    print(f"  theory converged   = {info['converged']}  "
          f"after {info['n_steps']} steps")

    # SFT trajectory
    sft_traj = _sft_trajectory(args.history, info["names"])
    print(f"  SFT trajectory     = {sft_traj.shape[0]} rounds, "
          f"{sft_traj.shape[1]} agents")

    # Summary numbers
    sft_end = sft_traj[-1]
    theo_end = info["positions"][-1]
    mu = info["space"].mean
    print(f"\n{'agent':<10} {'SFT end':>20} {'theory end':>20} "
          f"{'gap':>8} {'d(SFT,μ)':>10} {'d(theo,μ)':>10}")
    print("-" * 82)
    for i, name in enumerate(info["names"]):
        gap = float(np.linalg.norm(sft_end[i] - theo_end[i]))
        d_sft = float(np.linalg.norm(sft_end[i] - mu))
        d_theo = float(np.linalg.norm(theo_end[i] - mu))
        print(f"{name:<10} "
              f"({sft_end[i, 0]:.3f}, {sft_end[i, 1]:.3f})"
              f"   ({theo_end[i, 0]:.3f}, {theo_end[i, 1]:.3f})"
              f"   {gap:.3f}    {d_sft:.3f}     {d_theo:.3f}")

    # u_pool for both endpoints on the full benchmark pool
    pool_corpus = [p for s in info["splits"] for p in s.prompts]
    pool_coords = info["space"].project(pool_corpus)

    router_sft = InfluencerRouter(
        info["space"],
        [RouterAgent(name=n, position=sft_end[i].copy())
         for i, n in enumerate(info["names"])],
        sigma=info["sigma"], policy="proportional",
    )
    router_theo = InfluencerRouter(
        info["space"],
        [RouterAgent(name=n, position=theo_end[i].copy())
         for i, n in enumerate(info["names"])],
        sigma=info["sigma"], policy="proportional",
    )
    u_pool_sft = empirical_utility(router_sft.positions, pool_coords, router_sft.cov)
    u_pool_theo = empirical_utility(router_theo.positions, pool_coords, router_theo.cov)
    print(f"\n{'agent':<10} {'u_pool(SFT)':>12} {'u_pool(theory)':>16}")
    print("-" * 42)
    for i, name in enumerate(info["names"]):
        print(f"{name:<10} {u_pool_sft[i]:>12.4f} {u_pool_theo[i]:>16.4f}")

    # Plot
    fig = plot_comparison(
        info, sft_traj,
        axis_labels=(args.axis_labels[0], args.axis_labels[1]),
        title=args.title,
    )
    stem = args.output_stem or (FIGS_DIR / "theory_vs_sft")
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"\nwrote {pdf_path}")
    print(f"wrote {png_path}")

    # Optional JSON summary
    if args.summary_json is not None:
        summary = {
            "config": str(args.config),
            "history": str(args.history),
            "sigma": float(info["sigma"]),
            "sigma_star": float(info["sigma_star"]),
            "theory_converged": bool(info["converged"]),
            "theory_n_steps": int(info["n_steps"]),
            "agents": [
                {
                    "name": name,
                    "initial": info["initial_positions"][i].tolist(),
                    "sft_end": sft_end[i].tolist(),
                    "theory_end": theo_end[i].tolist(),
                    "gap": float(np.linalg.norm(sft_end[i] - theo_end[i])),
                    "u_pool_sft": float(u_pool_sft[i]),
                    "u_pool_theory": float(u_pool_theo[i]),
                }
                for i, name in enumerate(info["names"])
            ],
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {args.summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
