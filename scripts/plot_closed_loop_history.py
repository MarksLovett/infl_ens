"""Plot closed-loop training history: trait-space trajectories + utility tracking.

Reads the ``history.json`` produced by ``task: closed_loop`` in
:mod:`infl_ens.training.__main__` and renders a two-panel figure:

- **Left**: per-clone trajectory in the 2-D trait space, with start markers,
  arrow-headed lines, and end markers; the resource-weighted mean is shown
  if it can be reconstructed from the configured grid.
- **Right**: per-round overlay of :math:`u_\\text{grid}`, :math:`\\hat u`
  (pool), and the observed routing share, one subplot row per clone.

The figure is saved as both PDF and PNG under ``scripts/figures/`` (AGENTS.md
§4 rule 3). The pure-figure-returning helper :func:`plot_history` is
available for callers that want to embed the figure elsewhere.

Run with::

    python scripts/plot_closed_loop_history.py \\
        --history results/safety_truth/history.json \\
        --axis-labels harm hallucination \\
        --output-stem scripts/figures/safety_truth_round0to4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = ROOT / "scripts" / "figures"


def _load_history(path: Path) -> list[dict]:
    """Load and lightly validate ``history.json``.

    :param path: Path to the history file.
    :type path: pathlib.Path
    :returns: List of per-round dictionaries.
    :rtype: list[dict]
    :raises ValueError: If the file is empty or missing required keys.
    """
    with path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    if not records:
        raise ValueError(f"{path} contains no rounds")
    required = {"round", "positions", "u_grid", "u_pool", "observed_share"}
    missing = required - set(records[0])
    if missing:
        raise ValueError(
            f"{path} is missing required keys: {missing}. "
            "Did you log u_pool / observed_share in the closed-loop dispatcher?"
        )
    return records


def _agent_order(records: Sequence[dict]) -> list[str]:
    """Return agent names in the order they appear in the first round.

    :param records: Loaded history records.
    :type records: Sequence[dict]
    :returns: Ordered agent names.
    :rtype: list[str]
    """
    return list(records[0]["positions"].keys())


def _trajectories(records: Sequence[dict], names: Sequence[str]) -> np.ndarray:
    """Stack positions into a ``(n_rounds, n_agents, L)`` array.

    :param records: Loaded history records.
    :type records: Sequence[dict]
    :param names: Agent name order.
    :type names: Sequence[str]
    :returns: Position trajectory tensor.
    :rtype: numpy.ndarray
    """
    return np.stack(
        [
            np.stack([np.asarray(r["positions"][n]) for n in names], axis=0)
            for r in records
        ],
        axis=0,
    )


def plot_history(
    records: Sequence[dict],
    *,
    axis_labels: tuple[str, str] = ("axis 0", "axis 1"),
    title: Optional[str] = None,
):
    """Render the two-panel diagnostic figure.

    :param records: Loaded history records.
    :type records: Sequence[dict]
    :param axis_labels: Trait-space axis names (e.g. ``("harm", "hallucination")``).
    :type axis_labels: tuple[str, str]
    :param title: Optional figure suptitle.
    :type title: str | None
    :returns: Matplotlib figure handle. Caller is responsible for saving.
    :rtype: matplotlib.figure.Figure
    :raises ValueError: If trait space is not 2-D.
    """
    import matplotlib.pyplot as plt

    names = _agent_order(records)
    n_agents = len(names)
    pos = _trajectories(records, names)         # (T, N, L)
    T, N, L = pos.shape
    if L != 2:
        raise ValueError(f"plot_history requires L=2 trait space, got L={L}")

    u_grid = np.stack([np.asarray(r["u_grid"]) for r in records], axis=0)        # (T, N)
    u_pool = np.stack([np.asarray(r["u_pool"]) for r in records], axis=0)        # (T, N)
    share = np.stack([np.asarray(r["observed_share"]) for r in records], axis=0) # (T, N)
    rounds = np.array([int(r["round"]) for r in records])

    # Optional strategic-pool series, present only in runs logged after
    # strategic-routing support was added.
    has_strategic = "strategic_share_pool" in records[0]
    strat_pool = (
        np.stack([np.asarray(r["strategic_share_pool"]) for r in records], axis=0)
        if has_strategic else None
    )

    fig = plt.figure(figsize=(12, 5 + 1.0 * n_agents), constrained_layout=True)
    gs = fig.add_gridspec(n_agents, 2, width_ratios=[1.0, 1.0])

    # --- Left: trajectory in 2-D trait space, spans all rows -----------------
    ax_traj = fig.add_subplot(gs[:, 0])
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(n_agents, 3)))
    for i, name in enumerate(names):
        xs, ys = pos[:, i, 0], pos[:, i, 1]
        ax_traj.plot(xs, ys, "-", color=colors[i], lw=1.6, alpha=0.7,
                     label=f"{name}")
        ax_traj.scatter(xs[0], ys[0], color=colors[i], marker="o", s=60,
                        edgecolor="black", linewidth=0.6, zorder=3,
                        label=f"{name} start" if i == 0 else None)
        ax_traj.scatter(xs[-1], ys[-1], color=colors[i], marker="*", s=180,
                        edgecolor="black", linewidth=0.6, zorder=3)
        # arrow on the last segment
        if T >= 2:
            ax_traj.annotate(
                "",
                xy=(xs[-1], ys[-1]),
                xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.6, alpha=0.9),
            )
    ax_traj.set_xlim(-0.02, 1.02)
    ax_traj.set_ylim(-0.02, 1.02)
    ax_traj.set_xlabel(axis_labels[0])
    ax_traj.set_ylabel(axis_labels[1])
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.set_title(f"trajectories over {T} round(s)  •  ○ = start, ★ = end")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc="best", fontsize=8, frameon=True)

    # --- Right: utility tracking, one row per clone --------------------------
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(rounds, u_grid[:, i], "-o", color=colors[i],
                lw=1.4, mfc="white", label=r"$u_\mathrm{grid}$")
        ax.plot(rounds, u_pool[:, i], "--s", color=colors[i],
                lw=1.4, mfc=colors[i], label=r"$\hat u_\mathrm{pool}$")
        if strat_pool is not None:
            ax.plot(rounds, strat_pool[:, i], "-.D", color=colors[i],
                    lw=1.0, mfc="none", alpha=0.7,
                    label=r"strategic share pool")
        ax.plot(rounds, share[:, i], ":^", color="black",
                lw=1.1, mfc="white", label="observed share")
        ax.axhline(1.0 / n_agents, color="gray", lw=0.8, alpha=0.5)
        ymax_data = [u_grid, u_pool, share]
        if strat_pool is not None:
            ymax_data.append(strat_pool)
        ax.set_ylim(0.0, max(0.6, float(np.max(ymax_data)) + 0.05))
        ax.set_ylabel(f"{name}\nutility")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8, ncol=2, frameon=True)
        if i == n_agents - 1:
            ax.set_xlabel("round")
        ax.set_xticks(rounds)

    if title is not None:
        fig.suptitle(title)
    return fig


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    :returns: Configured argparse parser.
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="Plot closed-loop trait-space trajectories and utility tracking."
    )
    p.add_argument("--history", type=Path, required=True,
                   help="Path to history.json from a closed_loop run.")
    p.add_argument("--axis-labels", nargs=2, default=["harm", "hallucination"],
                   help="Trait-space axis labels (default: harm hallucination).")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--output-stem", type=Path, default=None,
                   help="Output filename stem (no extension). "
                        "Defaults to scripts/figures/closed_loop_history.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional CLI argument vector.
    :type argv: list[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    args = _build_parser().parse_args(argv)
    records = _load_history(args.history)

    fig = plot_history(
        records,
        axis_labels=(args.axis_labels[0], args.axis_labels[1]),
        title=args.title,
    )

    stem = args.output_stem or (FIGS_DIR / "closed_loop_history")
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
