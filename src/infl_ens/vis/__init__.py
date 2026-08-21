"""Visualization helpers (pure: arrays in, :class:`matplotlib.figure.Figure` out)."""

from __future__ import annotations

from infl_ens.vis.benchmark_nll_bar import plot_benchmark_nll_comparison
from infl_ens.vis.benchmark_space import plot_pairwise_heatmaps
from infl_ens.vis.capability_probe import plot_probe
from infl_ens.vis.closed_loop import (
    plot_history,
    plot_pairwise_position_updates,
    plot_trajectory_overlay,
)
from infl_ens.vis.save import save_figure
from infl_ens.vis.sweeps import (
    plot_overview,
    plot_series_mean_std,
    plot_sweep_grid,
    plot_trajectory_mean_std,
)
from infl_ens.vis.theory_vs_sft import plot_theory_vs_sft_comparison

__all__ = [
    "plot_benchmark_nll_comparison",
    "plot_pairwise_heatmaps",
    "plot_probe",
    "plot_history",
    "plot_pairwise_position_updates",
    "plot_trajectory_overlay",
    "plot_theory_vs_sft_comparison",
    "plot_overview",
    "plot_series_mean_std",
    "plot_sweep_grid",
    "plot_trajectory_mean_std",
    "save_figure",
]
