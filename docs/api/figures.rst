:mod:`infl_ens.figures`
=======================

Figures and tables of an experiment. The plot and table builders are
pure (records or arrays in, a :class:`matplotlib.figure.Figure`, TeX
source or a table out); :mod:`~infl_ens.figures.render` is the only
module that reads run artifacts, and its
:data:`~infl_ens.figures.render.FIGURES` registry backs

.. code-block:: bash

   python -m infl_ens.figures --config configs/experiments/seven_axis_3arm.yaml [--only a,b] [--list]

Top-level re-exports
--------------------

- :func:`~infl_ens.figures.closed_loop.plot_history`
- :func:`~infl_ens.figures.pair_positions.plot_final_positions`
- :func:`~infl_ens.figures.pair_positions.plot_within_pair`
- :func:`~infl_ens.figures.benchmark_space.plot_pairwise_heatmaps`
- :func:`~infl_ens.figures.benchmark_nll_bar.plot_benchmark_nll_comparison`
- :func:`~infl_ens.figures.trait_representation.plot_marginals`
- :func:`~infl_ens.figures.pgf_tex.oracle_routing_tex`
- :func:`~infl_ens.figures.pgf_tex.arm_comparison_tex`
- :func:`~infl_ens.figures.save.save_figure`
- :func:`~infl_ens.figures.style.apply_paper_style`

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.figures.style
   infl_ens.figures.save
   infl_ens.figures.closed_loop
   infl_ens.figures.pair_positions
   infl_ens.figures.benchmark_space
   infl_ens.figures.benchmark_nll_bar
   infl_ens.figures.trait_representation
   infl_ens.figures.pgf_tex
   infl_ens.figures.per_round_tables
   infl_ens.figures.cross_arm_report
   infl_ens.figures.render
