"""Overlay two closed-loop trajectories on a single trait-space plot.

The use case: after a code change to the position-update path, run a
short smoke-test closed loop (e.g. 5 rounds) and overlay its
trajectory on a longer pre-change run (e.g. 20 rounds) at the same
seed, sigma, and config otherwise. If the change landed, the two
trajectories will diverge from round 1 onward.

The script also computes and marks theory NE positions for each run
(using ``train_router_positions`` on each history's trait space) so
you can see whether the new run is heading toward theory NE while the
old one wasn't.

Run with::

    python scripts/compare_runs.py \\
        --config configs/benchmark/router/<config>.yaml \\
        --history-a results/test_position_only_r5/history.json \\
        --label-a "post-fix r5" \\
        --history-b results/position_only_long_round_sweep/r20/history.json \\
        --label-b "pre-fix r20" \\
        --axis-labels harm hallucination \\
        --title "position_only bug fix smoke test" \\
        --output-stem scripts/figures/test_position_only_r5/compare_runs

Writes ``<output-stem>.pdf`` and ``<output-stem>.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from infl_ens.data.benchmarks import (  # noqa: E402
    BenchmarkSplit,
    build_safety_trait_space,
    load_beavertails,
    load_halueval,
)
from infl_ens.data.encoders import SentenceTransformerEncoder  # noqa: E402
from infl_ens.data.trait_space import TraitSpace  # noqa: E402
from infl_ens.inflgame.router import RouterAgent  # noqa: E402
from infl_ens.training.router_training import (  # noqa: E402
    RouterTrainingConfig,
    train_router_positions,
)
from infl_ens.utils.resource import gaussian_stability_threshold  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file via PyYAML.

    :param path: Path to the config file.
    :type path: pathlib.Path
    :returns: Parsed configuration mapping.
    :rtype: dict
    """
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_splits(cfg: dict[str, Any]) -> list[BenchmarkSplit]:
    """Load all benchmark splits referenced by ``cfg['benchmarks']``.

    :param cfg: Parsed YAML config.
    :type cfg: dict
    :returns: List of loaded benchmark splits.
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
                path, tasks=entry.get("tasks"), max_records=max_records,
            ))
        else:  # pragma: no cover
            raise ValueError(f"unknown benchmark kind {kind!r}")
    return splits


def _build_trait_space(cfg: dict[str, Any], splits: list[BenchmarkSplit]) -> TraitSpace:
    """Reconstruct the closed-loop's trait space from its YAML config.

    :param cfg: Parsed YAML config.
    :type cfg: dict
    :param splits: Loaded benchmark splits.
    :type splits: list[BenchmarkSplit]
    :returns: Trait space matching the config.
    :rtype: TraitSpace
    """
    ts_cfg = cfg.get("trait_space", {})
    encoder = SentenceTransformerEncoder(
        model_name=ts_cfg.get(
            "encoder", "sentence-transformers/all-MiniLM-L6-v2",
        ),
    )
    return build_safety_trait_space(
        splits, encoder,
        n_grid=int(ts_cfg.get("n_grid", 32)),
        kde_bandwidth=ts_cfg.get("kde_bandwidth"),
        threshold=float(ts_cfg.get("threshold", 0.5)),
    )


def _resolve_sigma(
    cfg: dict[str, Any], n_agents: int, space: TraitSpace,
) -> float:
    """Resolve :math:`\\sigma` from the config.

    :param cfg: Parsed YAML config.
    :type cfg: dict
    :param n_agents: Number of agents.
    :type n_agents: int
    :param space: Trait space.
    :type space: TraitSpace
    :returns: Competitive reach.
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


def _trajectories(history: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Extract per-agent position trajectories from ``history.json``.

    :param history: Loaded ``history.json``.
    :type history: list[dict]
    :returns: Mapping ``{agent_name: positions}`` where ``positions``
        has shape ``(R, L)``.
    :rtype: dict[str, numpy.ndarray]
    """
    names = list(history[0]["positions"].keys())
    out: dict[str, np.ndarray] = {}
    for name in names:
        out[name] = np.array(
            [h["positions"][name] for h in history], dtype=float,
        )
    return out


def _theory_ne(
    history: list[dict[str, Any]], space: TraitSpace,
    sigma: float, seed: int, n_steps: int, lr: float,
) -> dict[str, np.ndarray]:
    """Re-run ``train_router_positions`` from the history's round-0 positions.

    :param history: Loaded ``history.json``.
    :type history: list[dict]
    :param space: Trait space.
    :type space: TraitSpace
    :param sigma: Competitive reach.
    :type sigma: float
    :param seed: RNG seed.
    :type seed: int
    :param n_steps: Gradient-ascent steps.
    :type n_steps: int
    :param lr: Learning rate.
    :type lr: float
    :returns: Mapping ``{agent_name: theory_ne_position}``.
    :rtype: dict[str, numpy.ndarray]
    """
    agents = [
        RouterAgent(name=name, position=np.asarray(pos, dtype=float))
        for name, pos in history[0]["positions"].items()
    ]
    rt_cfg = RouterTrainingConfig(
        sigma=sigma, learning_rate=lr, n_steps=n_steps,
        tol=1e-8, clip_to_box=True,
    )
    train_router_positions(space, agents, rt_cfg, seed=seed)
    return {a.name: a.position.copy() for a in agents}


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Optional argument vector; defaults to ``sys.argv[1:]``.
    :type argv: list[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description=(
            "Overlay two closed-loop trajectories on a single trait-space "
            "plot, with theory NE markers, to verify whether a code change "
            "between the two runs affected the position dynamics."
        ),
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="YAML config; both runs assumed to share it.")
    parser.add_argument("--history-a", required=True, type=Path,
                        help="First history.json (e.g. the new / post-fix run).")
    parser.add_argument("--label-a", default="run A",
                        help="Legend label for history A.")
    parser.add_argument("--history-b", required=True, type=Path,
                        help="Second history.json (e.g. the old / pre-fix run).")
    parser.add_argument("--label-b", default="run B",
                        help="Legend label for history B.")
    parser.add_argument("--output-stem", required=True,
                        help="Output stem; <stem>.{pdf,png} are written.")
    parser.add_argument("--axis-labels", nargs=2, default=["dim 0", "dim 1"])
    parser.add_argument("--title", default="")
    parser.add_argument("--theory-steps", type=int, default=5000)
    parser.add_argument("--theory-lr", type=float, default=5e-3)
    args = parser.parse_args(argv)

    cfg = _load_yaml(args.config)
    with args.history_a.open("r", encoding="utf-8") as fh:
        hist_a = json.load(fh)
    with args.history_b.open("r", encoding="utf-8") as fh:
        hist_b = json.load(fh)

    splits = _load_splits(cfg)
    space = _build_trait_space(cfg, splits)
    n_agents = len(hist_a[0]["positions"])
    sigma = _resolve_sigma(cfg, n_agents, space)
    s0 = gaussian_stability_threshold(n_agents, space.grid, space.weights)
    print(f"sigma = {sigma:.4f}  (σ₀* = {s0:.4f})")

    print("extracting trajectories ...")
    traj_a = _trajectories(hist_a)
    traj_b = _trajectories(hist_b)
    print(f"  A: {len(hist_a)} rounds, {len(traj_a)} agents")
    print(f"  B: {len(hist_b)} rounds, {len(traj_b)} agents")

    print(f"computing theory NE (sigma = {sigma:.4f}, "
          f"{args.theory_steps} steps) ...")
    theory_a = _theory_ne(
        hist_a, space, sigma,
        seed=int(cfg.get("seed", 0)),
        n_steps=args.theory_steps, lr=args.theory_lr,
    )

    # Per-agent SFT-end -> theory NE residuals, for both runs.
    summary: dict[str, dict[str, float]] = {}
    for name in traj_a:
        a_end = traj_a[name][-1]
        b_end = traj_b[name][-1] if name in traj_b else None
        ne = theory_a[name]
        summary[name] = {
            "a_end_l2_to_theory": float(np.linalg.norm(a_end - ne)),
            "b_end_l2_to_theory": (
                float(np.linalg.norm(b_end - ne))
                if b_end is not None else float("nan")
            ),
            "a_b_l2": (
                float(np.linalg.norm(a_end - b_end))
                if b_end is not None else float("nan")
            ),
        }

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 8.5))

    # Resource-weighted mean.
    ax.scatter(
        [space.mean[0]], [space.mean[1]],
        marker="P", s=200, c="black", edgecolors="white", linewidths=1.5,
        zorder=5, label=r"$\mathbb{E}_B[b]$",
    )

    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(traj_a):
        c = cmap(i % 10)
        ta = traj_a[name]
        tb = traj_b.get(name)
        ne = theory_a[name]

        # Run B (pre-fix): light, dashed.
        if tb is not None:
            ax.plot(
                tb[:, 0], tb[:, 1],
                linestyle="--", color=c, alpha=0.45, linewidth=1.2,
                zorder=2,
            )
            ax.scatter(
                [tb[-1, 0]], [tb[-1, 1]],
                marker="s", s=80, facecolors="white",
                edgecolors=c, linewidths=1.5, zorder=3,
                label=(f"{name}  {args.label_b}" if i == 0
                       else f"{name}  {args.label_b}"),
            )

        # Run A (post-fix): bold, solid.
        ax.plot(
            ta[:, 0], ta[:, 1],
            linestyle="-", color=c, alpha=0.95, linewidth=1.8,
            zorder=4,
        )
        ax.scatter(
            [ta[0, 0]], [ta[0, 1]],
            marker="o", s=70, facecolors="none",
            edgecolors=c, linewidths=1.6, zorder=4,
        )
        ax.scatter(
            [ta[-1, 0]], [ta[-1, 1]],
            marker="*", s=240, c=[c], edgecolors="black", linewidths=0.7,
            zorder=6,
            label=f"{name}  {args.label_a}",
        )

        # Theory NE.
        ax.scatter(
            [ne[0]], [ne[1]],
            marker="X", s=180, c=[c], edgecolors="black", linewidths=0.7,
            zorder=5,
            label=f"{name}  theory NE",
        )

    ax.set_xlabel(args.axis_labels[0])
    ax.set_ylabel(args.axis_labels[1])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.set_title(
        f"trajectory overlay  (σ = {sigma:.3f},  σ₀* = {s0:.3f})"
        f"\n{args.title}"
        if args.title else
        f"trajectory overlay  (σ = {sigma:.3f},  σ₀* = {s0:.3f})"
    )

    # Legend in a compact layout.
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    dedup_handles: list[Any] = []
    dedup_labels: list[str] = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        dedup_handles.append(h)
        dedup_labels.append(l)
    ax.legend(
        dedup_handles, dedup_labels,
        loc="lower left", fontsize=7, framealpha=0.9, ncol=2,
    )

    fig.tight_layout()
    out = Path(args.output_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.output_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.output_stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output_stem}.{{pdf,png}}")

    # Console summary.
    print()
    print(f"{'agent':<10} "
          f"{args.label_a + ' end → NE':>22} "
          f"{args.label_b + ' end → NE':>22} "
          f"{'A end ↔ B end':>17}")
    for name, s in summary.items():
        print(
            f"{name:<10} "
            f"{s['a_end_l2_to_theory']:>22.4f} "
            f"{s['b_end_l2_to_theory']:>22.4f} "
            f"{s['a_b_l2']:>17.4f}"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
