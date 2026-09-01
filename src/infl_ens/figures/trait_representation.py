"""How benchmark prompts are represented in the learned trait space.

Compares the legacy clipped calibration
(``clip((w·e - lo)/(hi - lo), 0, 1)`` per axis, then the clipped residual
rescale, then the concave stretch) against the current quantile
normalization (unclipped affine scores → residualize → per-axis empirical
CDF → stretch → clip).  Both representations are derived from ONE encode of
the same prompts with the same learned axes, so the figures isolate the
normalization stage.

The functions here are pure (coordinates in, figures/stats out);
:mod:`infl_ens.figures.render` does the encoding and the disk I/O.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from matplotlib.figure import Figure

from infl_ens.data.benchmarks.safety_trait_space import _raw_coordinate_matrix

_SATURATION_TOL = 1e-6


def stratified_sample(
    splits: Sequence[Any],
    max_prompts: int,
    seed: int,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Sample prompts evenly across benchmarks.

    :param splits: Loaded benchmark splits.
    :type splits: Sequence[BenchmarkSplit]
    :param max_prompts: Total prompt budget across all splits.
    :type max_prompts: int
    :param seed: RNG seed for reproducible sampling.
    :type seed: int
    :returns: ``(prompts, split_ids, split_names)``.
    :rtype: tuple[list[str], numpy.ndarray, list[str]]
    """
    rng = np.random.default_rng(seed)
    per_split = max(1, max_prompts // max(len(splits), 1))
    prompts: list[str] = []
    ids: list[int] = []
    names = [s.name for s in splits]
    for idx, split in enumerate(splits):
        pool = list(split.prompts)
        if len(pool) > per_split:
            pick = rng.choice(len(pool), size=per_split, replace=False)
            pool = [pool[int(i)] for i in pick]
        prompts.extend(pool)
        ids.extend([idx] * len(pool))
    return prompts, np.asarray(ids, dtype=int), names


def legacy_coordinates(
    embeddings: np.ndarray,
    axes: Sequence[Any],
    gammas: np.ndarray,
) -> np.ndarray:
    """Reconstruct the pre-quantile clipped coordinate pipeline.

    :param embeddings: Sentence embeddings, shape ``(N, D)``.
    :type embeddings: numpy.ndarray
    :param axes: Learned axes.
    :type axes: Sequence[LearnedAxis]
    :param gammas: Per-axis stretch exponents.
    :type gammas: numpy.ndarray
    :returns: Legacy coordinates in ``[0, 1]^L``.
    :rtype: numpy.ndarray
    """
    base = np.clip(_raw_coordinate_matrix(embeddings, axes), 0.0, 1.0)
    coords = base.copy()
    for j, axis in enumerate(axes):
        if axis.residual_coef is None or j == 0:
            continue
        pred = axis.residual_intercept + coords[:, :j] @ axis.residual_coef
        resid = base[:, j] - pred
        lo = axis.residual_lo if axis.residual_lo is not None else axis.lo
        hi = axis.residual_hi if axis.residual_hi is not None else axis.hi
        span = max(float(hi) - float(lo), 1e-12)
        coords[:, j] = np.clip((resid - float(lo)) / span, 0.0, 1.0)
    if not np.all(gammas == 1.0):
        coords = 1.0 - np.power(1.0 - np.clip(coords, 0.0, 1.0), gammas)
    return np.clip(coords, 0.0, 1.0)


def ks_vs_uniform(column: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic of one column against U[0, 1].

    :param column: Sample values in ``[0, 1]``.
    :type column: numpy.ndarray
    :returns: The KS statistic (``nan`` for an empty column).
    :rtype: float
    """
    v = np.sort(np.asarray(column, dtype=float))
    n = v.shape[0]
    if n == 0:
        return float("nan")
    grid = np.arange(1, n + 1) / n
    return float(np.max(np.maximum(np.abs(grid - v), np.abs(v - (grid - 1.0 / n)))))


def representation_stats(coords: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    """Per-axis saturation, uniformity, and correlation diagnostics.

    :param coords: Coordinates in ``[0, 1]^L``, shape ``(N, L)``.
    :type coords: numpy.ndarray
    :param labels: Axis names.
    :type labels: Sequence[str]
    :returns: JSON-safe summary statistics.
    :rtype: dict
    """
    x = np.asarray(coords, dtype=float)
    per_axis = []
    for j, name in enumerate(labels):
        col = x[:, j]
        at_zero = float(np.mean(col <= _SATURATION_TOL))
        at_one = float(np.mean(col >= 1.0 - _SATURATION_TOL))
        per_axis.append(
            {
                "axis": name,
                "frac_at_zero": at_zero,
                "frac_at_one": at_one,
                "frac_saturated": at_zero + at_one,
                "ks_vs_uniform": ks_vs_uniform(col),
                "mean": float(col.mean()),
                "std": float(col.std()),
            },
        )
    corr = np.corrcoef(x, rowvar=False) if x.shape[0] > 1 else np.eye(x.shape[1])
    corr = np.atleast_2d(corr)
    off = corr - np.diag(np.diag(corr))
    return {
        "n_prompts": int(x.shape[0]),
        "in_unit_box": bool(np.all((x >= 0.0) & (x <= 1.0))),
        "max_frac_saturated": max(a["frac_saturated"] for a in per_axis),
        "max_ks_vs_uniform": max(a["ks_vs_uniform"] for a in per_axis),
        "mean_abs_offdiag_corr": float(np.mean(np.abs(off))),
        "per_axis": per_axis,
    }


def plot_marginals(
    legacy: np.ndarray,
    new: np.ndarray,
    labels: Sequence[str],
    *,
    stats_legacy: dict[str, Any],
    stats_new: dict[str, Any],
    pre_stretch: Optional[np.ndarray] = None,
    stats_pre_stretch: Optional[dict[str, Any]] = None,
    title: Optional[str] = None,
) -> Figure:
    """Per-axis marginal histograms, one row per representation.

    When the config enables a post-normalization stretch, the pure-CDF
    row is drawn between the legacy and final rows so the stretch's effect
    on the marginals is visible rather than conflated with the normalizer's.

    :param legacy: Legacy coordinates, shape ``(N, L)``.
    :type legacy: numpy.ndarray
    :param new: Final coordinates the router sees, shape ``(N, L)``.
    :type new: numpy.ndarray
    :param labels: Axis names.
    :type labels: Sequence[str]
    :param stats_legacy: Summary stats for the legacy arm.
    :type stats_legacy: dict
    :param stats_new: Summary stats for the final arm.
    :type stats_new: dict
    :param pre_stretch: Optional pure-CDF coordinates before stretch.
    :type pre_stretch: numpy.ndarray | None
    :param stats_pre_stretch: Summary stats for the pure-CDF arm.
    :type stats_pre_stretch: dict | None
    :param title: Optional figure suptitle.
    :type title: str | None
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    rows = [("legacy: clip to [0,1]", legacy, stats_legacy, "#d62728")]
    if pre_stretch is not None and stats_pre_stretch is not None:
        rows.append(("quantile CDF only", pre_stretch, stats_pre_stretch, "#2ca02c"))
        rows.append(("CDF + stretch (routed)", new, stats_new, "#1f77b4"))
    else:
        rows.append(("new: quantile CDF", new, stats_new, "#1f77b4"))

    n_axes = len(labels)
    fig, axarr = plt.subplots(
        len(rows), n_axes, figsize=(2.6 * n_axes, 2.8 * len(rows)), sharex=True, squeeze=False,
    )
    bins = np.linspace(0.0, 1.0, 41)
    for row, (row_label, coords, stats, color) in enumerate(rows):
        for j, name in enumerate(labels):
            ax = axarr[row][j]
            ax.hist(coords[:, j], bins=bins, color=color, alpha=0.85, density=True, edgecolor="none")
            ax.axhline(1.0, color="0.35", lw=0.9, ls="--", zorder=3)
            info = stats["per_axis"][j]
            ax.set_title(
                f"{name}\nsat={info['frac_saturated']:.1%}  KS={info['ks_vs_uniform']:.3f}",
                fontsize=8,
            )
            ax.set_xlim(-0.02, 1.02)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(row_label, fontsize=9)
    fig.suptitle(
        title or "Trait-space representation: clipped calibration vs quantile normalization",
        fontsize=11,
    )
    fig.text(
        0.5, 0.005,
        "dashed line = uniform density. sat = fraction of mass pinned at 0 or 1; "
        "KS = deviation from U[0,1].",
        ha="center", fontsize=8, color="0.3",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig


def plot_pair_comparison(
    legacy: np.ndarray,
    new: np.ndarray,
    labels: Sequence[str],
    *,
    pairs: Sequence[tuple[int, int]],
    bins: int = 48,
    title: Optional[str] = None,
) -> Figure:
    """Side-by-side 2-D empirical densities for selected axis pairs.

    :param legacy: Legacy coordinates, shape ``(N, L)``.
    :type legacy: numpy.ndarray
    :param new: Quantile-normalized coordinates, shape ``(N, L)``.
    :type new: numpy.ndarray
    :param labels: Axis names.
    :type labels: Sequence[str]
    :param pairs: Axis index pairs to render.
    :type pairs: Sequence[tuple[int, int]]
    :param bins: Histogram resolution per axis.
    :type bins: int
    :param title: Optional figure suptitle.
    :type title: str | None
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n_pairs = len(pairs)
    fig, axarr = plt.subplots(2, n_pairs, figsize=(2.9 * n_pairs, 6.0), squeeze=False)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = (("legacy", legacy), ("new", new))
    for row, (row_label, coords) in enumerate(rows):
        for k, (i, j) in enumerate(pairs):
            ax = axarr[row][k]
            heat, _, _ = np.histogram2d(coords[:, i], coords[:, j], bins=(edges, edges))
            total = heat.sum()
            if total > 0:
                heat = heat / total
            vmax = np.percentile(heat[heat > 0], 98) if np.any(heat > 0) else 1.0
            ax.imshow(
                heat.T, origin="lower", extent=(0, 1, 0, 1),
                aspect="equal", cmap="magma", vmin=0.0, vmax=max(vmax, 1e-12),
            )
            ax.set_xlabel(labels[i], fontsize=8)
            ax.set_ylabel(f"{row_label}\n{labels[j]}" if k == 0 else labels[j], fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(
        title or "Pairwise prompt density: clipped calibration (top) vs quantile (bottom)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_dataset_composition(
    split_ids: np.ndarray,
    split_names: Sequence[str],
    *,
    title: Optional[str] = None,
) -> Figure:
    """Sampled prompt counts per benchmark.

    :param split_ids: Per-prompt benchmark index, shape ``(N,)``.
    :type split_ids: numpy.ndarray
    :param split_names: Benchmark names.
    :type split_names: Sequence[str]
    :param title: Optional figure title.
    :type title: str | None
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    counts = [int(np.sum(split_ids == i)) for i in range(len(split_names))]
    fig, ax = plt.subplots(figsize=(1.3 * len(split_names) + 2.5, 3.6))
    ax.bar(range(len(split_names)), counts, color="#4c72b0")
    ax.set_xticks(range(len(split_names)))
    ax.set_xticklabels(split_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("prompts sampled", fontsize=9)
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=8)
    ax.set_title(title or "Sampled prompts per benchmark", fontsize=10)
    fig.tight_layout()
    return fig


__all__ = [
    "ks_vs_uniform",
    "legacy_coordinates",
    "plot_dataset_composition",
    "plot_marginals",
    "plot_pair_comparison",
    "representation_stats",
    "stratified_sample",
]
