"""Grouped bar chart comparing benchmark NLL across models."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from infl_ens.vis.save import save_figure


def _setup_latex_style() -> None:
    """Use Computer Modern–style math text when full LaTeX is unavailable."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def plot_benchmark_nll_comparison(
    benchmarks: Sequence[str],
    benchmark_labels: Mapping[str, str],
    base_nll: Mapping[str, float],
    adapter_nll: Mapping[str, Mapping[str, float]],
    adapter_std: Mapping[str, Mapping[str, float]] | None = None,
    *,
    agents: Sequence[str] | None = None,
    include_base: bool = True,
    title: str | None = None,
    ylabel: str = r"Mean token NLL $\downarrow$",
    output_stem: str | Path | None = None,
    save_formats: Sequence[str] = ("pdf", "png"),
    save_dpi: int = 200,
    save_bbox_inches: str | None = "tight",
) -> Figure:
    """Bar chart of mean NLL: base model vs specialist adapters.

    :param benchmarks: Benchmark ids in plot order.
    :type benchmarks: Sequence[str]
    :param benchmark_labels: Display label per benchmark id.
    :type benchmark_labels: Mapping[str, str]
    :param base_nll: Base-model mean NLL per benchmark.
    :type base_nll: Mapping[str, float]
    :param adapter_nll: ``adapter_nll[benchmark][agent]`` means.
    :type adapter_nll: Mapping[str, Mapping[str, float]]
    :param adapter_std: Optional standard deviations for error bars.
    :type adapter_std: Mapping[str, Mapping[str, float]] | None
    :param agents: Agent ids in bar order (default: sorted union).
    :type agents: Sequence[str] | None
    :param include_base: Whether to include the base-model bar series.
    :type include_base: bool
    :param title: Optional figure title.
    :type title: str | None
    :param ylabel: Y-axis label (LaTeX allowed).
    :type ylabel: str
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
    _setup_latex_style()

    if agents is None:
        seen: list[str] = []
        for bench in benchmarks:
            for agent in adapter_nll.get(bench, {}):
                if agent not in seen:
                    seen.append(agent)
        agents = sorted(seen)

    series_labels = (["Base"] if include_base else []) + list(agents)
    n_series = len(series_labels)
    n_bench = len(benchmarks)
    x = np.arange(n_bench, dtype=float)
    width = 0.8 / n_series

    colors = {
        "Base": "#4d4d4d",
        "clone-0": "#2166ac",
        "clone-1": "#b2182b",
        "clone-2": "#d6604d",
        "clone-3": "#67a9cf",
        "clone-4": "#1b7837",
        "clone-5": "#a6dba0",
        "generalist": "#7b3294",
    }
    legend_labels = {
        "Base": r"Base (Qwen2.5-1.5B)",
        "clone-0": r"Clone 0 (high)",
        "clone-1": r"Clone 1 (low)",
        "clone-2": r"Clone 2 (low)",
        "clone-3": r"Clone 3 (high)",
        "clone-4": r"Clone 4",
        "clone-5": r"Clone 5",
        "generalist": r"Generalist",
    }

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for j, label in enumerate(series_labels):
        offset = (j - (n_series - 1) / 2.0) * width
        heights: list[float] = []
        yerr: list[float] | None = [] if adapter_std is not None else None
        for bench in benchmarks:
            if label == "Base":
                heights.append(float(base_nll[bench]))
                if yerr is not None:
                    yerr.append(0.0)
            else:
                heights.append(float(adapter_nll[bench][label]))
                if yerr is not None:
                    yerr.append(float(adapter_std[bench].get(label, 0.0)))
        ax.bar(
            x + offset,
            heights,
            width=width * 0.95,
            label=legend_labels.get(label, label),
            color=colors.get(label, "#888888"),
            edgecolor="white",
            linewidth=0.6,
            yerr=yerr,
            capsize=2.5,
            error_kw={"elinewidth": 0.9},
        )

    ax.legend(ncol=3, loc="upper right", framealpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels([benchmark_labels[b] for b in benchmarks])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0.0, ymax * 1.08)
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
