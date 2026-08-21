"""Sweep comparison figures (pure: arrays in, Figure out)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from infl_ens.vis.save import BboxInches, PathLike, save_figure


def plot_sweep_grid(
    summaries: list[dict],
    *,
    mode: str,
    axis_labels: tuple[str, str] = ("harm", "hallucination"),
    title: Optional[str] = None,
    with_theory: bool = False,
    output_stem: Optional[PathLike] = None,
    save_formats: Sequence[str] = ("pdf", "png"),
    save_dpi: int = 200,
    save_bbox_inches: BboxInches = "tight",
):
    """Render a grid of per-run trajectory panels with summary text.

    :param summaries: Per-run summaries from
        :func:`infl_ens.training.sweep_aggregate.summarise_flat_sweep_run`.
    :type summaries: list[dict]
    :param mode: Sweep mode (``seeds``, ``sigma``, or ``kde``); used for panel
        titles.
    :type mode: str
    :param axis_labels: Trait-space axis names.
    :type axis_labels: tuple[str, str]
    :param title: Optional figure suptitle.
    :type title: str | None
    :param with_theory: Whether to overlay theoretical Nash endpoints.
    :type with_theory: bool
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

    n_runs = len(summaries)
    n_cols = min(n_runs, 4)
    n_rows = int(np.ceil(n_runs / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 4.0 * n_rows + 0.5),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    label_fmt = {"seeds": "seed={:.0f}", "sigma": "σ_frac={:.2f}",
                 "kde": "h={:.3f}"}[mode]

    for ax, s in zip(axes_flat, summaries):
        names = s["names"]
        pos = s["positions"]
        n_agents = pos.shape[1]
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(n_agents, 3)))
        for i, name in enumerate(names):
            xs, ys = pos[:, i, 0], pos[:, i, 1]
            ax.plot(xs, ys, "--", color=colors[i], lw=1.5, alpha=0.9,
                    label=name if ax is axes_flat[0] else None)
            ax.scatter(xs[0], ys[0], color=colors[i], marker="o",
                       s=40, edgecolor="black", linewidth=0.5, zorder=3)
            ax.scatter(xs[-1], ys[-1], color=colors[i], marker="*",
                       s=160, edgecolor="black", linewidth=0.6, zorder=4)
            if with_theory and s["theory_positions"] is not None:
                tx, ty = s["theory_positions"][i]
                ax.scatter([tx], [ty], color=colors[i], marker="X",
                           s=110, edgecolor="black", linewidth=0.6,
                           alpha=0.65, zorder=4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        eq = "(" + ", ".join(str(x) for x in s["equilibrium_type"]) + ")"
        title_str = f"{label_fmt.format(s['value'])}   →   {eq}"
        if with_theory and s["theory_eq_type"] is not None:
            theo_eq = "(" + ", ".join(str(x) for x in s["theory_eq_type"]) + ")"
            title_str += f"\ntheory: {theo_eq}"
        ax.set_title(title_str, fontsize=10)

    for ax in axes_flat[n_runs:]:
        ax.set_visible(False)

    if summaries:
        axes_flat[0].legend(loc="best", fontsize=8, frameon=True)
        for ax in axes[-1, :]:
            ax.set_xlabel(axis_labels[0])
        for ax in axes[:, 0]:
            ax.set_ylabel(axis_labels[1])

    if title is not None:
        fig.suptitle(title, fontsize=12)
    if output_stem is not None:
        save_figure(
            fig,
            output_stem,
            formats=save_formats,
            dpi=save_dpi,
            bbox_inches=save_bbox_inches,
        )
    return fig


def plot_trajectory_mean_std(
    *,
    sigma_fraction: float,
    names: Sequence[str],
    pos_mean: np.ndarray,
    pos_std: np.ndarray,
    theo_mean: Optional[np.ndarray],
    axis_labels: tuple[str, str],
    title: Optional[str],
    n_seeds: int,
    output_stem: Optional[PathLike] = None,
    save_formats: Sequence[str] = ("pdf", "png"),
    save_dpi: int = 200,
    save_bbox_inches: BboxInches = "tight",
):
    """Trait-space trajectories: mean line and ±1σ band per clone.

    :param sigma_fraction: σ / σ₀* for the legend and title.
    :type sigma_fraction: float
    :param names: Clone names.
    :type names: Sequence[str]
    :param pos_mean: Mean positions ``(T, N, L)``.
    :type pos_mean: numpy.ndarray
    :param pos_std: Std positions ``(T, N, L)``.
    :type pos_std: numpy.ndarray
    :param theo_mean: Optional mean theory endpoints ``(N, L)`` (one point
        per agent).
    :type theo_mean: numpy.ndarray | None
    :param axis_labels: Trait axis names.
    :type axis_labels: tuple[str, str]
    :param title: Figure suptitle.
    :type title: str | None
    :param n_seeds: Number of seeds aggregated (for legend).
    :type n_seeds: int
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
    :raises ValueError: If trait space is not 2-D.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _T, _N, L = pos_mean.shape
    if L != 2:
        raise ValueError(f"requires L=2, got L={L}")

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(_N, 3)))
    sigma_label = f"σ/σ₀* = {sigma_fraction:g}"

    for i, name in enumerate(names):
        xs, ys = pos_mean[:, i, 0], pos_mean[:, i, 1]
        xs_s, ys_s = pos_std[:, i, 0], pos_std[:, i, 1]
        ax.plot(
            xs, ys, "-", color=colors[i], lw=2.0,
            label=f"{name} (mean, {n_seeds} seeds)",
        )
        ax.fill_between(xs, ys - ys_s, ys + ys_s, color=colors[i], alpha=0.2)
        ax.scatter(xs[0], ys[0], color=colors[i], marker="o", s=50,
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.scatter(xs[-1], ys[-1], color=colors[i], marker="*", s=140,
                   edgecolor="black", linewidth=0.5, zorder=3)
        if theo_mean is not None:
            tx, ty = theo_mean[i, 0], theo_mean[i, 1]
            ax.scatter(
                tx, ty, color=colors[i], marker="X", s=120,
                edgecolor="black", linewidth=0.6, zorder=4,
                label="theory NE (mean)" if i == 0 else None,
            )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"trait trajectories — {sigma_label}  •  ○ start, ★ SFT end, ✕ theory",
    )
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="gray", alpha=0.25, label="±1σ across seeds"))
    ax.legend(handles=handles, labels=labels + ["±1σ across seeds"],
              loc="best", fontsize=8, frameon=True)
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


def plot_series_mean_std(
    *,
    rounds: np.ndarray,
    series_mean: np.ndarray,
    series_std: np.ndarray,
    ylabel: str,
    title: str,
    n_seeds: int,
    output_stem: Optional[PathLike] = None,
    save_formats: Sequence[str] = ("pdf", "png"),
    save_dpi: int = 200,
    save_bbox_inches: BboxInches = "tight",
):
    """Single time series with mean line and shaded ±1σ.

    :param rounds: Round indices.
    :type rounds: numpy.ndarray
    :param series_mean: Mean values per round.
    :type series_mean: numpy.ndarray
    :param series_std: Std per round.
    :type series_std: numpy.ndarray
    :param ylabel: Y-axis label.
    :type ylabel: str
    :param title: Panel title (include σ and seed count).
    :type title: str
    :param n_seeds: Seeds aggregated.
    :type n_seeds: int
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

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(rounds, series_mean, "-o", color="C0", lw=2,
            label=f"mean over {n_seeds} seeds")
    ax.fill_between(
        rounds,
        series_mean - series_std,
        series_mean + series_std,
        color="C0",
        alpha=0.25,
        label="±1σ across seeds",
    )
    ax.set_xlabel("round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    if output_stem is not None:
        save_figure(
            fig,
            output_stem,
            formats=save_formats,
            dpi=save_dpi,
            bbox_inches=save_bbox_inches,
        )
    return fig


def plot_overview(
    *,
    x_values: np.ndarray,
    spread_mean: np.ndarray,
    spread_std: np.ndarray,
    margin_mean: np.ndarray,
    margin_std: np.ndarray,
    gap_mean: np.ndarray,
    gap_std: np.ndarray,
    n_seeds: int,
    title: Optional[str],
    xlabel: str = r"$\sigma / \sigma_0^*$",
    output_stem: Optional[PathLike] = None,
    save_formats: Sequence[str] = ("pdf", "png"),
    save_dpi: int = 200,
    save_bbox_inches: BboxInches = "tight",
):
    """Three-panel overview: final spread, final margin, mean theory gap vs *x*.

    :param x_values: Sweep coordinate (σ fraction or ``n_rounds``).
    :type x_values: numpy.ndarray
    :param spread_mean: Mean final pairwise spread per σ.
    :type spread_mean: numpy.ndarray
    :param spread_std: Std of final spread across seeds.
    :type spread_std: numpy.ndarray
    :param margin_mean: Mean final probe margin per σ.
    :type margin_mean: numpy.ndarray
    :param margin_std: Std of final margin.
    :type margin_std: numpy.ndarray
    :param gap_mean: Mean theory gap (L2, averaged over agents) per σ.
    :type gap_mean: numpy.ndarray
    :param gap_std: Std of mean gap across seeds.
    :type gap_std: numpy.ndarray
    :param n_seeds: Seeds per σ cell.
    :type n_seeds: int
    :param title: Optional suptitle.
    :type title: str | None
    :param xlabel: X-axis label for all panels.
    :type xlabel: str
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

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = [
        (spread_mean, spread_std, "final pairwise spread", "trait separation"),
        (margin_mean, margin_std, r"final probe margin $\mu(r)$", "cross-batch NLL margin"),
        (gap_mean, gap_std, "mean theory↔SFT gap", r"$\|\mathbf{x}_\mathrm{SFT}-\mathbf{x}_\mathrm{theo}\|_2$"),
    ]
    for ax, (ym, ys, ylab, subt) in zip(axes, panels):
        ax.errorbar(
            x_values, ym, yerr=ys, fmt="o-", capsize=4, lw=1.8,
            label=f"mean ± std ({n_seeds} seeds)",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        ax.set_title(subt)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

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


def plot_spread_by_mode_sigma(
    rows: Sequence[dict],
    *,
    output_stem: Optional[PathLike] = None,
    suptitle: str = "final pairwise spread by mode and sigma",
    collapse_thresh: float = 0.45,
    mode_label_fn=None,
    rotate_xticks: float = 0.0,
    save_formats: Sequence[str] = ("pdf", "png"),
    save_dpi: int = 200,
    save_bbox_inches: BboxInches = "tight",
):
    """Grouped bar chart of mean final spread by mode and sigma.

    :param rows: Summary rows with ``mode``, ``sigma_fraction``,
        ``final_spread``.
    :type rows: Sequence[dict]
    :param output_stem: If set, write figure files under this stem.
    :type output_stem: str | pathlib.Path | None
    :param suptitle: Figure suptitle.
    :type suptitle: str
    :param collapse_thresh: Horizontal reference line for collapse.
    :type collapse_thresh: float
    :param mode_label_fn: Optional formatter for mode tick labels.
    :type mode_label_fn: Callable[[str], str] | None
    :param rotate_xticks: X tick label rotation in degrees.
    :type rotate_xticks: float
    :returns: Matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if mode_label_fn is None:
        mode_label_fn = lambda m: m  # noqa: E731

    modes = sorted({r["mode"] for r in rows})
    sigmas = sorted({r["sigma_fraction"] for r in rows})
    fig, axes = plt.subplots(
        1, len(sigmas), figsize=(5 * len(sigmas), 4), squeeze=False,
    )

    for j, sigma in enumerate(sigmas):
        ax = axes[0, j]
        labels, means, stds = [], [], []
        for mode in modes:
            vals = [
                r["final_spread"]
                for r in rows
                if r["mode"] == mode and r["sigma_fraction"] == sigma
            ]
            if not vals:
                continue
            labels.append(mode_label_fn(mode))
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, alpha=0.88)
        ax.axhline(
            collapse_thresh, color="red", ls="--", lw=1,
            label=f"collapse < {collapse_thresh}",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels,
            fontsize=8,
            rotation=rotate_xticks,
            ha="right" if rotate_xticks else "center",
        )
        ax.set_ylabel("final pairwise spread")
        ax.set_title(f"σ/σ₀* = {sigma:g}")
        ax.grid(True, axis="y", alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)

    fig.suptitle(suptitle)
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
