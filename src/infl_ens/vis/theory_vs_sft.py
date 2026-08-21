"""Theory-vs-SFT comparison figures (pure: arrays in, Figure out)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from infl_ens.vis.save import BboxInches, PathLike, save_figure


def plot_theory_vs_sft_comparison(
    info: dict,
    sft_traj: np.ndarray,
    *,
    axis_labels: tuple[str, str] = ("harm", "hallucination"),
    title: Optional[str] = None,
    dyn_label: str = "SFT",
    output_stem: Optional[PathLike] = None,
    save_formats: Sequence[str] = ("pdf", "png"),
    save_dpi: int = 200,
    save_bbox_inches: BboxInches = "tight",
):
    """Render a two-panel theory-vs-SFT comparison figure.

    :param info: Output of :func:`infl_ens.training.theory_vs_sft.run_strategic_ascent`.
    :type info: dict
    :param sft_traj: SFT trajectory tensor, shape ``(T, N, L)``.
    :type sft_traj: numpy.ndarray
    :param axis_labels: Axis names.
    :type axis_labels: tuple[str, str]
    :param title: Optional suptitle.
    :type title: str | None
    :param dyn_label: Label for the closed-loop trajectory (e.g. ``SFT``).
    :type dyn_label: str
    :param output_stem: If set, write ``.<format>`` files under this stem.
    :type output_stem: str | pathlib.Path | None
    :param save_formats: Extensions to write when ``output_stem`` is set.
    :type save_formats: Sequence[str]
    :param save_dpi: DPI for raster exports.
    :type save_dpi: int
    :param save_bbox_inches: ``bbox_inches`` passed to :func:`save_figure`.
    :type save_bbox_inches: str | None
    :returns: Matplotlib figure handle.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names = info["names"]
    n_agents = len(names)
    theo_traj = info["positions"]
    sigma = info["sigma"]
    sigma_star = info["sigma_star"]
    space = info["space"]
    theory_start = info.get("theory_start_positions", theo_traj[0])
    sft_start = info.get("sft_start_positions", sft_traj[0])
    has_sep_init = bool(info.get("theory_gradient_init"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(n_agents, 3)))

    ax = axes[0]
    for i, name in enumerate(names):
        tx, ty = theo_traj[:, i, 0], theo_traj[:, i, 1]
        ax.plot(tx, ty, "-", color=colors[i], lw=1.4, alpha=0.75, zorder=2)
        if has_sep_init:
            ax.scatter(
                theory_start[i, 0],
                theory_start[i, 1],
                color=colors[i],
                marker="D",
                s=70,
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
            )
        ax.scatter(
            tx[-1],
            ty[-1],
            color=colors[i],
            marker="X",
            s=130,
            edgecolor="black",
            linewidth=0.7,
            zorder=6,
        )
        sx, sy = sft_traj[:, i, 0], sft_traj[:, i, 1]
        ax.plot(sx, sy, "--", color=colors[i], lw=1.8, alpha=0.95, zorder=3)
        ax.scatter(
            sft_start[i, 0],
            sft_start[i, 1],
            color=colors[i],
            marker="o",
            s=55,
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )
        ax.scatter(
            sx[-1],
            sy[-1],
            color=colors[i],
            marker="*",
            s=200,
            edgecolor="black",
            linewidth=0.7,
            zorder=6,
        )

    mu = space.mean
    ax.scatter([mu[0]], [mu[1]], marker="+", s=180, color="black", linewidth=1.6, zorder=7)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_aspect("equal", adjustable="box")
    init_note = (
        "◇ theory init (separated)" if has_sep_init else f"○ {dyn_label} start (round 0)"
    )
    ax.set_title(
        f"trait-space trajectories\n"
        f"σ = {sigma:.3f},  σ₀* = {sigma_star:.3f}  ({sigma/sigma_star:.2f}·σ₀*)",
    )
    ax.grid(True, alpha=0.3)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
            label=init_note,
        ),
        Line2D([0], [0], color="gray", lw=1.4, label="theory gradient path"),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=9,
            label="theory end (gradient NE)",
        ),
        Line2D([0], [0], color="C0", lw=1.8, ls="--", label=f"{dyn_label} trajectory"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="C0",
            markeredgecolor="black",
            markersize=7,
            label=f"{dyn_label} start (round 0)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="C0",
            markeredgecolor="black",
            markersize=12,
            label=f"{dyn_label} end",
        ),
        Line2D([0], [0], marker="+", color="k", ls="", markersize=10, label=r"$\mathbb{E}_B[b]$"),
    ]
    color_handles = [
        Line2D([0], [0], color=colors[i], lw=2, label=names[i])
        for i in range(n_agents)
    ]
    leg1 = ax.legend(handles=legend_handles, loc="upper left", fontsize=7.5, frameon=True, title="markers")
    ax.add_artist(leg1)
    ax.legend(handles=color_handles, loc="lower right", fontsize=7.5, frameon=True, title="agents")

    ax2 = axes[1]
    sft_end = sft_traj[-1]
    theo_end = theo_traj[-1]
    d_sft = np.linalg.norm(sft_end - mu[None, :], axis=1)
    d_theo = np.linalg.norm(theo_end - mu[None, :], axis=1)
    d_pairwise = np.linalg.norm(sft_end - theo_end, axis=1)
    x = np.arange(n_agents)
    w = 0.27
    ax2.bar(x - w, d_theo, w, color="lightgray", edgecolor="black", label="theory NE  →  centroid")
    ax2.bar(x, d_sft, w, color="steelblue", edgecolor="black", label=f"{dyn_label} end  →  centroid")
    ax2.bar(
        x + w,
        d_pairwise,
        w,
        color="tomato",
        edgecolor="black",
        label=f"{dyn_label} end  →  theory NE",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=0)
    ax2.set_ylabel("L2 distance in trait space")
    ax2.set_title("specialisation depth & theory ↔ SFT gap")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="best", fontsize=8, frameon=True)

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
