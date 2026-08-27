"""Pure plotting: arrays and records in, :class:`matplotlib.figure.Figure` out."""

from __future__ import annotations

from infl_ens.vis.benchmark_nll_bar import plot_benchmark_nll_comparison
from infl_ens.vis.benchmark_space import plot_pairwise_heatmaps
from infl_ens.vis.closed_loop import (
    plot_history,
    plot_pairwise_position_updates,
    plot_trajectory_overlay,
)
from infl_ens.vis.save import save_figure

__all__ = [
    "plot_benchmark_nll_comparison",
    "plot_pairwise_heatmaps",
    "plot_history",
    "plot_pairwise_position_updates",
    "plot_trajectory_overlay",
    "save_figure",
]
