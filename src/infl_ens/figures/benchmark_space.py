"""Pairwise trait-space resource heatmaps (pure: arrays in, Figure out)."""

from __future__ import annotations

import itertools
from typing import Any, Optional, Sequence

import numpy as np

from infl_ens.figures.save import BboxInches, PathLike, save_figure


def _axis_values(grid: np.ndarray, dim: int) -> np.ndarray:
    """Return sorted grid coordinates for one axis.

    :param grid: Trait-space grid, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param dim: Axis index.
    :type dim: int
    :returns: Sorted unique grid values.
    :rtype: numpy.ndarray
    """
    return np.unique(grid[:, dim])


def _pairwise_marginal(
    grid: np.ndarray,
    weights: np.ndarray,
    pair: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marginalise ``weights`` onto a pair of trait axes.

    :param grid: Regular trait-space grid, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param weights: Resource weights, shape ``(K,)``.
    :type weights: numpy.ndarray
    :param pair: Axis indices to retain.
    :type pair: tuple[int, int]
    :returns: ``(x_values, y_values, heatmap)`` where heatmap is
        ``(len(x_values), len(y_values))``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :raises ValueError: If the grid is not a regular Cartesian product.
    """
    L = grid.shape[1]
    axes = [_axis_values(grid, d) for d in range(L)]
    shape = tuple(len(a) for a in axes)
    if int(np.prod(shape)) != weights.size:
        raise ValueError("resource heatmaps require a regular Cartesian grid")
    tensor = weights.reshape(shape)
    collapse = tuple(d for d in range(L) if d not in pair)
    heat = tensor.sum(axis=collapse) if collapse else tensor
    return axes[pair[0]], axes[pair[1]], heat


def _pairwise_empirical_marginal(
    coords: np.ndarray,
    pair: tuple[int, int],
    *,
    n_bins: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram projected prompts onto a trait-axis pair.

    :param coords: Projected prompt coordinates, shape ``(M, L)``.
    :type coords: numpy.ndarray
    :param pair: Axis indices to retain.
    :type pair: tuple[int, int]
    :param n_bins: Number of histogram bins per axis.
    :type n_bins: int
    :returns: ``(x_centers, y_centers, heatmap)`` where heatmap is
        ``(n_bins, n_bins)`` and normalised to sum to one.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :raises ValueError: If ``n_bins < 2`` or ``coords`` is empty.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if coords.size == 0:
        raise ValueError("coords must be non-empty")
    heat, x_edges, y_edges = np.histogram2d(
        coords[:, pair[0]],
        coords[:, pair[1]],
        bins=n_bins,
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    total = float(heat.sum())
    if total > 0.0:
        heat = heat / total
    return x_centers, y_centers, heat


def _gaussian_smooth_2d(heat: np.ndarray, *, sigma: float) -> np.ndarray:
    """Apply a separable Gaussian blur to a 2-D histogram.

    :param heat: Input grid, shape ``(nx, ny)``.
    :type heat: numpy.ndarray
    :param sigma: Kernel standard deviation in bin units. ``0`` returns
        ``heat`` unchanged.
    :type sigma: float
    :returns: Smoothed grid with the same shape as ``heat``.
    :rtype: numpy.ndarray
    """
    if sigma <= 0.0:
        return heat
    radius = max(1, int(round(3.0 * sigma)))
    ax = np.arange(-radius, radius + 1, dtype=float)
    k1 = np.exp(-0.5 * (ax / sigma) ** 2)
    k1 /= k1.sum()
    kernel = np.outer(k1, k1)
    padded = np.pad(heat, radius, mode="constant")
    out = np.zeros_like(heat, dtype=float)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            patch = padded[i:i + 2 * radius + 1, j:j + 2 * radius + 1]
            out[i, j] = float((patch * kernel).sum())
    total = float(out.sum())
    if total > 0.0:
        out *= float(heat.sum()) / total
    return out


def _prepare_heat_display(
    heat: np.ndarray,
    *,
    mass_norm: str,
    smooth_sigma: float = 0.0,
    vmax_percentile: float = 92.0,
) -> tuple[np.ndarray, float, float, Optional[Any]]:
    """Rescale a pairwise heatmap for visibility.

    :param heat: Raw mass grid, shape ``(nx, ny)``.
    :type heat: numpy.ndarray
    :param mass_norm: One of ``global``, ``per_panel``, ``per_panel_sqrt``,
        ``per_panel_log``, ``per_panel_power``.
    :type mass_norm: str
    :param smooth_sigma: Gaussian smoothing radius in bin units applied
        before colour scaling.
    :type smooth_sigma: float
    :param vmax_percentile: When using per-panel modes, clip the display
        ceiling to this percentile of smoothed, non-zero bin mass.
    :type vmax_percentile: float
    :returns: ``(display_grid, vmin, vmax, norm)`` suitable for
        :func:`matplotlib.axes.Axes.imshow`.
    :rtype: tuple[numpy.ndarray, float, float, Any | None]
    :raises ValueError: If ``mass_norm`` is unknown.
    """
    from matplotlib.colors import LogNorm, PowerNorm

    h = np.asarray(heat, dtype=float)
    if smooth_sigma > 0.0:
        h = _gaussian_smooth_2d(h, sigma=smooth_sigma)
    if mass_norm == "global":
        return h, 0.0, float(h.max()) if h.size else 1.0, None

    positive = h[h > 0]
    if positive.size == 0:
        return h, 0.0, 1.0, None
    peak = float(np.percentile(positive, vmax_percentile))
    peak = max(peak, float(positive.min()))
    if peak <= 0.0:
        return h, 0.0, 1.0, None

    if mass_norm == "per_panel":
        return np.clip(h / peak, 0.0, 1.0), 0.0, 1.0, None
    if mass_norm == "per_panel_sqrt":
        disp = np.sqrt(np.clip(h / peak, 0.0, 1.0))
        return disp, 0.0, 1.0, None
    if mass_norm == "per_panel_log":
        eps = max(peak * 1e-4, 1e-8)
        vmin = max(float(positive.min()), eps)
        return np.clip(h, eps, peak), vmin, peak, LogNorm(vmin=vmin, vmax=peak)
    if mass_norm == "per_panel_power":
        disp = np.clip(h / peak, 0.0, 1.0)
        return disp, 0.0, 1.0, PowerNorm(gamma=0.45, vmin=0.0, vmax=1.0)

    raise ValueError(
        "mass_norm must be one of "
        "global, per_panel, per_panel_sqrt, per_panel_log, per_panel_power; "
        f"got {mass_norm!r}"
    )


def plot_pairwise_heatmaps(
    grid: np.ndarray,
    weights: np.ndarray,
    *,
    axis_labels: Sequence[str],
    prompt_coords: Optional[np.ndarray] = None,
    positions: Optional[dict[str, np.ndarray]] = None,
    trajectories: Optional[dict[str, np.ndarray]] = None,
    partition_labels: Optional[np.ndarray] = None,
    max_scatter: int = 1500,
    title: Optional[str] = None,
    density_mode: str = "kde",
    pairwise_bins: int = 32,
    mass_norm: str = "per_panel_power",
    scatter_alpha: float = 0.10,
    smooth_sigma: float = 2.0,
    vmax_percentile: float = 92.0,
    output_stem: Optional[PathLike] = None,
    save_formats: Sequence[str] = ("png", "pdf"),
    save_dpi: int = 220,
    save_bbox_inches: BboxInches = None,
):
    """Render all two-axis resource heatmaps.

    :param grid: Trait-space grid, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param weights: Resource weights on the grid, shape ``(K,)``.
    :type weights: numpy.ndarray
    :param axis_labels: Human-readable axis labels.
    :type axis_labels: Sequence[str]
    :param prompt_coords: Optional actual prompt projections to overlay.
    :type prompt_coords: numpy.ndarray | None
    :param positions: Optional trained router positions to overlay.
    :type positions: dict[str, numpy.ndarray] | None
    :param trajectories: Optional per-agent position trajectories to overlay.
    :type trajectories: dict[str, numpy.ndarray] | None
    :param partition_labels: Optional per-prompt partition ids aligned with
        ``prompt_coords`` (0=train, 1=val, 2=test). When set, scatter is
        drawn in partition colours instead of a single overlay colour.
    :type partition_labels: numpy.ndarray | None
    :param max_scatter: Maximum prompt points to overlay; ``0`` disables.
    :type max_scatter: int
    :param title: Optional figure title.
    :type title: str | None
    :param density_mode: ``'kde'`` marginalises the trait-space KDE;
        ``'empirical'`` histograms projected prompts directly.
    :type density_mode: str
    :param pairwise_bins: Histogram bins per axis when
        ``density_mode='empirical'``.
    :type pairwise_bins: int
    :param mass_norm: How to scale heatmap colours.
    :type mass_norm: str
    :param scatter_alpha: Alpha for overlaid prompt scatter points.
    :type scatter_alpha: float
    :param smooth_sigma: Gaussian smoothing of each 2-D histogram, in bin
        units, before colour mapping.
    :type smooth_sigma: float
    :param vmax_percentile: Per-panel colour ceiling percentile on smoothed
        non-zero bins.
    :type vmax_percentile: float
    :param output_stem: If set, write ``.<format>`` files under this stem.
    :type output_stem: str | pathlib.Path | None
    :param save_formats: Extensions to write when ``output_stem`` is set.
    :type save_formats: Sequence[str]
    :param save_dpi: DPI for raster exports.
    :type save_dpi: int
    :param save_bbox_inches: ``bbox_inches`` for :func:`save_figure`; ``None``
        keeps the default Matplotlib crop (preferred with ``constrained_layout``).
    :type save_bbox_inches: str | None
    :returns: Matplotlib figure handle.
    :rtype: matplotlib.figure.Figure
    :raises ValueError: If ``density_mode`` is unknown or empirical mode
        is requested without prompt coordinates.
    """
    import matplotlib.pyplot as plt

    L = grid.shape[1]
    pairs = list(itertools.combinations(range(L), 2))
    if not pairs:
        raise ValueError("need at least two trait-space axes")

    n_cols = min(4, len(pairs))
    n_rows = (len(pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 4.5 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    rng = np.random.default_rng(0)
    scatter = prompt_coords
    part_labels = partition_labels
    if scatter is not None and max_scatter > 0 and scatter.shape[0] > max_scatter:
        idx = rng.choice(scatter.shape[0], size=max_scatter, replace=False)
        scatter = scatter[idx]
        if part_labels is not None:
            part_labels = part_labels[idx]
    elif max_scatter <= 0:
        scatter = None
        part_labels = None

    if density_mode == "empirical":
        if prompt_coords is None:
            raise ValueError("empirical heatmaps require prompt_coords")
        marginals = [
            (pair, *_pairwise_empirical_marginal(
                prompt_coords, pair, n_bins=pairwise_bins,
            ))
            for pair in pairs
        ]
        cbar_label = "empirical mass"
    elif density_mode == "kde":
        marginals = [
            (pair, *_pairwise_marginal(grid, weights, pair))
            for pair in pairs
        ]
        cbar_label = "marginal mass"
    else:
        raise ValueError(
            f"density_mode must be 'kde' or 'empirical', got {density_mode!r}"
        )
    if mass_norm == "global":
        global_vmax = max(float(heat.max()) for _, _, _, heat in marginals)
    else:
        global_vmax = None
    cbar_label = {
        "global": cbar_label,
        "per_panel": "normalised mass (panel max = 1)",
        "per_panel_sqrt": "sqrt-normalised mass",
        "per_panel_log": "log mass",
        "per_panel_power": "normalised mass (γ=0.45)",
    }.get(mass_norm, cbar_label)
    trajectory_items = list((trajectories or {}).items())
    trajectory_colors = plt.get_cmap("tab10")(
        np.linspace(0, 1, max(len(trajectory_items), 3)),
    )
    position_items = list((positions or {}).items())
    position_colors = plt.get_cmap("tab20")(
        np.linspace(0, 1, max(len(position_items), 1)),
    )

    for ax, (pair, x, y, heat) in zip(axes.ravel(), marginals):
        disp, vmin, vmax, norm = _prepare_heat_display(
            heat,
            mass_norm=mass_norm if mass_norm != "global" else "global",
            smooth_sigma=smooth_sigma,
            vmax_percentile=vmax_percentile,
        )
        if mass_norm == "global":
            vmax = global_vmax if global_vmax is not None else vmax
        imshow_kwargs: dict[str, Any] = {
            "origin": "lower",
            "extent": (
                float(x.min()), float(x.max()),
                float(y.min()), float(y.max()),
            ),
            "aspect": "auto",
            "cmap": "inferno",
        }
        if norm is None:
            imshow_kwargs["vmin"] = vmin
            imshow_kwargs["vmax"] = vmax
        else:
            imshow_kwargs["norm"] = norm
        im = ax.imshow(disp.T, **imshow_kwargs)
        if scatter is not None:
            if part_labels is not None:
                if part_labels.shape[0] != scatter.shape[0]:
                    raise ValueError(
                        "partition_labels must align with prompt_coords rows"
                    )
                part_styles = (
                    (0, "#1f77b4", "train (70%)"),
                    (1, "#ff7f0e", "val (10%)"),
                    (2, "#d62728", "test (20%)"),
                )
                for part_id, color, label in part_styles:
                    mask = part_labels == part_id
                    if not np.any(mask):
                        continue
                    ax.scatter(
                        scatter[mask, pair[0]],
                        scatter[mask, pair[1]],
                        s=4,
                        c=color,
                        alpha=max(scatter_alpha, 0.12),
                        linewidths=0,
                        label=label,
                        zorder=3,
                    )
            else:
                ax.scatter(
                    scatter[:, pair[0]],
                    scatter[:, pair[1]],
                    s=3,
                    c="cyan",
                    alpha=scatter_alpha,
                    linewidths=0,
                    label="sampled prompts",
                )
        if positions:
            for i, (name, pos) in enumerate(position_items):
                color = position_colors[i]
                ax.scatter(
                    pos[pair[0]],
                    pos[pair[1]],
                    marker="*",
                    s=110,
                    c=[color],
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=7,
                    label=name,
                )
        for i, (name, path) in enumerate(trajectory_items):
            xs = path[:, pair[0]]
            ys = path[:, pair[1]]
            color = trajectory_colors[i]
            ax.plot(xs, ys, "-o", ms=2.4, lw=1.2, color=color, label=name, zorder=4)
            ax.scatter(
                xs[0],
                ys[0],
                marker="s",
                s=42,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=5,
            )
            ax.scatter(
                xs[-1],
                ys[-1],
                marker="*",
                s=130,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=6,
            )
            if len(xs) > 1:
                ax.annotate(
                    "",
                    xy=(xs[-1], ys[-1]),
                    xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
                    zorder=6,
                )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(axis_labels[pair[0]])
        ax.set_ylabel(axis_labels[pair[1]])
        ax.set_title(f"{axis_labels[pair[0]]} vs {axis_labels[pair[1]]}")
        ax.grid(True, color="white", alpha=0.18, linewidth=0.6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    for ax in axes.ravel()[len(marginals):]:
        ax.set_visible(False)
    if positions or trajectory_items:
        from matplotlib.lines import Line2D

        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        for i, (name, _) in enumerate(position_items):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color=position_colors[i],
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    markersize=10,
                    linestyle="None",
                )
            )
            legend_labels.append(name)
        for i, (name, path) in enumerate(trajectory_items):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=trajectory_colors[i],
                    markersize=5,
                    linewidth=1.2,
                    label=name,
                )
            )
            legend_labels.append(name)
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=max(1, min(len(legend_labels), 7)),
                fontsize=8,
                frameon=True,
            )
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
