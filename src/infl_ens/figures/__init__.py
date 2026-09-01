"""Figures and tables of an experiment.

Two layers:

- Pure plotting and table builders (records or arrays in, a
  :class:`matplotlib.figure.Figure`, a string of TeX or a table out):
  :mod:`~infl_ens.figures.closed_loop`, :mod:`~infl_ens.figures.pair_positions`,
  :mod:`~infl_ens.figures.benchmark_space`, :mod:`~infl_ens.figures.benchmark_nll_bar`,
  :mod:`~infl_ens.figures.trait_representation`, :mod:`~infl_ens.figures.pgf_tex`,
  :mod:`~infl_ens.figures.per_round_tables`, :mod:`~infl_ens.figures.cross_arm_report`.
- :mod:`~infl_ens.figures.render`, the only module that reads run
  artifacts, with the :data:`~infl_ens.figures.render.FIGURES` registry
  behind ``python -m infl_ens.figures --config <experiment>``.

Outputs go to the experiment's ``figures_dir`` (``figures/<experiment>/``).
"""

from __future__ import annotations

from infl_ens.figures.benchmark_nll_bar import plot_benchmark_nll_comparison
from infl_ens.figures.benchmark_space import plot_pairwise_heatmaps
from infl_ens.figures.closed_loop import (
    plot_history,
    plot_pairwise_position_updates,
    plot_trajectory_overlay,
)
from infl_ens.figures.pair_positions import plot_final_positions, plot_within_pair
from infl_ens.figures.pgf_tex import arm_comparison_tex, oracle_routing_tex
from infl_ens.figures.save import save_figure
from infl_ens.figures.scale_family import (
    CellNLL,
    plot_family_scale_nll,
    write_family_scale_table,
)
from infl_ens.figures.style import BENCHMARK_LABELS, BENCHMARK_ORDER, apply_paper_style
from infl_ens.figures.trait_representation import (
    plot_dataset_composition,
    plot_marginals,
    plot_pair_comparison,
)

__all__ = [
    "BENCHMARK_LABELS",
    "BENCHMARK_ORDER",
    "CellNLL",
    "apply_paper_style",
    "arm_comparison_tex",
    "oracle_routing_tex",
    "plot_benchmark_nll_comparison",
    "plot_dataset_composition",
    "plot_family_scale_nll",
    "plot_final_positions",
    "plot_history",
    "plot_marginals",
    "plot_pair_comparison",
    "plot_pairwise_heatmaps",
    "plot_pairwise_position_updates",
    "plot_trajectory_overlay",
    "plot_within_pair",
    "save_figure",
    "write_family_scale_table",
]
