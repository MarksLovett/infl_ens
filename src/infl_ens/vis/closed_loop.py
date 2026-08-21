"""Closed-loop training history figures (pure: arrays in, Figure out)."""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from infl_ens.vis.save import BboxInches, PathLike, save_figure


def _agent_order(records: Sequence[dict]) -> list[str]:
    """Return agent names in the order they appear in the first round.

    :param records: Loaded history records.
    :type records: Sequence[dict]
    :returns: Ordered agent names.
    :rtype: list[str]
    """
    return list(records[0]["positions"].keys())


def _position_tensor(records: Sequence[dict], names: Sequence[str]) -> np.ndarray:
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
            np.stack([np.asarray(r["positions"][n], dtype=float) for n in names])
            for r in records
        ],
        axis=0,
    )


def plot_history(
    records: Sequence[dict],
    *,
    axis_labels: tuple[str, str] = ("axis 0", "axis 1"),
    title: Optional[str] = None,
    output_stem: Optional[PathLike] = None,
    save_formats: Sequence[str] = ("pdf", "png"),
    save_dpi: int = 200,
    save_bbox_inches: BboxInches = "tight",
):
    """Render the two-panel closed-loop diagnostic figure.

    Left panel: per-clone trajectory in 2-D trait space. Right panel:
    per-round overlay of :math:`u_\\text{grid}`, :math:`\\hat u` (pool),
    and observed routing share, one subplot row per clone.

    :param records: Loaded history records.
    :type records: Sequence[dict]
    :param axis_labels: Trait-space axis names.
    :type axis_labels: tuple[str, str]
    :param title: Optional figure suptitle.
    :type title: str | None
    :param output_stem: If set, write ``.<format>`` files under this stem.
    :type output_stem: str | pathlib.Path | None
    :param save_formats: Extensions to write when ``output_stem`` is set.
    :type save_formats: Sequence[str]
    :param save_dpi: DPI for raster exports.
    :type save_dpi: int
    :param save_bbox_inches: ``bbox_inches`` passed to :func:`save_figure`.
    :type save_bbox_inches: str | None
    :returns: Matplotlib figure handle. Caller may save or display if
        ``output_stem`` is ``None``.
    :rtype: matplotlib.figure.Figure
    :raises ValueError: If trait space is not 2-D.
    """
    import matplotlib.pyplot as plt

    names = _agent_order(records)
    n_agents = len(names)
    pos = _position_tensor(records, names)
    T, N, L = pos.shape
    if L != 2:
        raise ValueError(f"plot_history requires L=2 trait space, got L={L}")

    u_grid = np.stack([np.asarray(r["u_grid"]) for r in records], axis=0)
    u_pool = np.stack([np.asarray(r["u_pool"]) for r in records], axis=0)
    share = np.stack([np.asarray(r["observed_share"]) for r in records], axis=0)
    rounds = np.array([int(r["round"]) for r in records])

    has_strategic = "strategic_share_pool" in records[0]
    strat_pool = (
        np.stack([np.asarray(r["strategic_share_pool"]) for r in records], axis=0)
        if has_strategic else None
    )

    fig = plt.figure(figsize=(12, 5 + 1.0 * n_agents), constrained_layout=True)
    gs = fig.add_gridspec(n_agents, 2, width_ratios=[1.0, 1.0])

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
    if output_stem is not None:
        save_figure(
            fig,
            output_stem,
            formats=save_formats,
            dpi=save_dpi,
            bbox_inches=save_bbox_inches,
        )
    return fig


def plot_pairwise_position_updates(
    records: Sequence[dict],
    *,
    axis_labels: Sequence[str],
    title: Optional[str] = None,
    output_stem: Optional[PathLike] = None,
    save_formats: Sequence[str] = ("pdf", "png"),
    save_dpi: int = 200,
    save_bbox_inches: BboxInches = "tight",
):
    """Render pairwise trajectories plus per-round update magnitudes.

    Intended for multi-axis benchmark spaces (e.g. harm × hallucination ×
    privacy) where :func:`plot_history` requires ``L=2``.

    :param records: Loaded closed-loop history.
    :type records: Sequence[dict]
    :param axis_labels: Human-readable trait labels.
    :type axis_labels: Sequence[str]
    :param title: Optional figure title.
    :type title: str | None
    :param output_stem: If set, write ``.<format>`` files under this stem.
    :type output_stem: str | pathlib.Path | None
    :param save_formats: Extensions to write when ``output_stem`` is set.
    :type save_formats: Sequence[str]
    :param save_dpi: DPI for raster exports.
    :type save_dpi: int
    :param save_bbox_inches: ``bbox_inches`` passed to :func:`save_figure`.
    :type save_bbox_inches: str | None
    :returns: Matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    names = _agent_order(records)
    pos = _position_tensor(records, names)
    rounds = np.asarray([int(r["round"]) for r in records])
    _, n_agents, n_axes = pos.shape
    pairs = list(itertools.combinations(range(n_axes), 2))
    if not pairs:
        raise ValueError("need at least two axes")

    fig = plt.figure(
        figsize=(5.2 * len(pairs), 4.6 + 2.8),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(2, len(pairs), height_ratios=[1.0, 0.62])
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(n_agents, 3)))

    for col, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, col])
        for i, name in enumerate(names):
            xs = pos[:, i, a]
            ys = pos[:, i, b]
            ax.plot(xs, ys, "-o", ms=3, lw=1.4, color=colors[i], label=name)
            ax.scatter(xs[0], ys[0], marker="s", s=48, color=colors[i],
                       edgecolor="black", linewidth=0.5, zorder=3)
            ax.scatter(xs[-1], ys[-1], marker="*", s=135, color=colors[i],
                       edgecolor="black", linewidth=0.5, zorder=4)
            if len(xs) > 1:
                ax.annotate(
                    "",
                    xy=(xs[-1], ys[-1]),
                    xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.2),
                )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(axis_labels[a])
        ax.set_ylabel(axis_labels[b])
        ax.set_title(f"{axis_labels[a]} vs {axis_labels[b]}")
        ax.grid(True, alpha=0.3)
        if col == len(pairs) - 1:
            ax.legend(loc="best", fontsize=8, frameon=True)

    ax_u = fig.add_subplot(gs[1, :])
    deltas = np.linalg.norm(np.diff(pos, axis=0), axis=2)
    for i, name in enumerate(names):
        if deltas.size:
            ax_u.plot(
                rounds[1:],
                deltas[:, i],
                "-o",
                color=colors[i],
                lw=1.5,
                ms=3,
                label=f"{name} update",
            )
    if title:
        fig.suptitle(title)
    if output_stem is not None:
        save_figure(
            fig,
            output_stem,
            formats=save_formats,
            dpi=save_dpi,
            bbox_inches=save_bbox_inches,
        )
    return fig


def plot_trajectory_overlay(
    *,
    resource_mean: np.ndarray,
    traj_a: Mapping[str, np.ndarray],
    traj_b: Mapping[str, np.ndarray],
    theory_a: Mapping[str, np.ndarray],
    label_a: str,
    label_b: str,
    axis_labels: tuple[str, str],
    sigma: float,
    sigma_star: float,
    title: str = "",
    output_stem: Optional[PathLike] = None,
    save_formats: Sequence[str] = ("pdf", "png"),
    save_dpi: int = 150,
    save_bbox_inches: BboxInches = "tight",
):
    """Overlay two closed-loop trajectories on a single trait-space plot.

    Run A is drawn bold and solid; run B is light and dashed. Theory NE
    endpoints from run A are marked with ``X``.

    :param resource_mean: Resource-weighted mean :math:`\\mathbb{E}_B[b]`,
        shape ``(L,)``.
    :type resource_mean: numpy.ndarray
    :param traj_a: Per-agent positions for run A, each ``(R, L)``.
    :type traj_a: Mapping[str, numpy.ndarray]
    :param traj_b: Per-agent positions for run B, each ``(R, L)``.
    :type traj_b: Mapping[str, numpy.ndarray]
    :param theory_a: Theory NE positions for run A, each ``(L,)``.
    :type theory_a: Mapping[str, numpy.ndarray]
    :param label_a: Legend label for run A.
    :type label_a: str
    :param label_b: Legend label for run B.
    :type label_b: str
    :param axis_labels: Trait-space axis names.
    :type axis_labels: tuple[str, str]
    :param sigma: Competitive reach used for the title.
    :type sigma: float
    :param sigma_star: Stability threshold :math:`\\sigma_0^*` for the title.
    :type sigma_star: float
    :param title: Optional subtitle appended below the σ line.
    :type title: str
    :param output_stem: If set, write ``.<format>`` files under this stem.
    :type output_stem: str | pathlib.Path | None
    :param save_formats: Extensions to write when ``output_stem`` is set.
    :type save_formats: Sequence[str]
    :param save_dpi: DPI for raster exports.
    :type save_dpi: int
    :param save_bbox_inches: ``bbox_inches`` passed to :func:`save_figure`.
    :type save_bbox_inches: str | None
    :returns: Matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 8.5))

    ax.scatter(
        [resource_mean[0]], [resource_mean[1]],
        marker="P", s=200, c="black", edgecolors="white", linewidths=1.5,
        zorder=5, label=r"$\mathbb{E}_B[b]$",
    )

    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(traj_a):
        c = cmap(i % 10)
        ta = traj_a[name]
        tb = traj_b.get(name)
        ne = theory_a[name]

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
                label=f"{name}  {label_b}",
            )

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
            label=f"{name}  {label_a}",
        )

        ax.scatter(
            [ne[0]], [ne[1]],
            marker="X", s=180, c=[c], edgecolors="black", linewidths=0.7,
            zorder=5,
            label=f"{name}  theory NE",
        )

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    base_title = f"trajectory overlay  (σ = {sigma:.3f},  σ₀* = {sigma_star:.3f})"
    ax.set_title(f"{base_title}\n{title}" if title else base_title)

    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    dedup_handles: list[Any] = []
    dedup_labels: list[str] = []
    for h, lbl in zip(handles, labels):
        if lbl in seen:
            continue
        seen.add(lbl)
        dedup_handles.append(h)
        dedup_labels.append(lbl)
    ax.legend(
        dedup_handles, dedup_labels,
        loc="lower left", fontsize=7, framealpha=0.9, ncol=2,
    )

    fig.tight_layout()
    if output_stem is not None:
        save_figure(
            fig,
            output_stem,
            formats=save_formats,
            dpi=save_dpi,
            bbox_inches=save_bbox_inches,
        )
    return fig
