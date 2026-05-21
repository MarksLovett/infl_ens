Scripts
=======

One-off scripts live in ``scripts/`` at the repo root (per AGENTS.md
§3). They are not part of the importable package. The table below
mirrors the corresponding section of :doc:`structure`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Script
     - Role
   * - ``download_beavertails.py``
     - Downloads ``PKU-Alignment/BeaverTails`` to ``data/beavertails/``
       via the ``datasets`` library.
   * - ``download_halueval.py``
     - Downloads HaluEval task JSON files from ``RUCAIBox/HaluEval`` to
       ``data/halueval/``.
   * - ``build_safety_trait_space.py``
     - Convenience wrapper around
       ``python -m infl_ens.data build-safety-trait-space``.
   * - ``compare_utility_estimators.py``
     - Side-by-side comparison of grid :math:`u_i`, empirical-pool
       :math:`\hat u_i`, and finite-batch proportional share.
       ``--mode {toy,safety}``.
   * - ``compare_theory_vs_sft.py``
     - Rebuilds the trait space from a closed-loop run's config,
       initialises agents from ``history.json`` round 0, runs
       :func:`~infl_ens.training.train_router_positions`, and compares
       the strategic-Nash endpoints with the SFT trajectory in trait
       space.
   * - ``plot_closed_loop_history.py``
     - Reads ``history.json`` from a closed-loop run and renders
       trajectories + utility tracking to PDF/PNG under
       ``scripts/figures/``.
   * - ``run_sweep.sh``
     - Bash launcher that sweeps one parameter (seeds, sigma_fraction,
       or kde_bandwidth) over the closed-loop trainer.
   * - ``plot_sweep.py``
     - Aggregates a sweep root directory into one figure: per-run
       trajectory panels, equilibrium-type classification by
       single-linkage clustering, optional overlay of theoretical Nash
       endpoints, CSV summary.
   * - ``probe_sft_capability.py``
     - Capability probe: per-agent SFT loss curves and cross-perplexity
       matrix from a run with ``save_per_round: true``.
   * - ``closed_loop_demo.py``
     - Toy hash-bag closed-loop simulation (no external deps).
   * - ``smoke_test.py``
     - End-to-end pipeline sanity check.

Invoking a script
-----------------

All scripts assume the repo root is the working directory and that
``src/`` is importable:

.. code-block:: bash

   PYTHONPATH=src python scripts/smoke_test.py
   PYTHONPATH=src python scripts/closed_loop_demo.py
   bash scripts/run_sweep.sh --help
