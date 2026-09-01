Pipeline
========

One command takes an experiment from raw benchmarks to figures:

.. code-block:: bash

   python -m infl_ens.pipeline --config configs/experiments/seven_axis_3arm.yaml

Run it from the repository root. The experiment file names the arms
(each a run config under ``configs/arms/``), the stages to run, the
evaluation window and the figure options; see :doc:`configs`.

Stages
------

Stages run in this order and are individually re-runnable: a stage whose
outputs already exist is skipped unless ``--force`` is given, so a
failure late in the pipeline costs no GPU time on the next launch.

.. list-table::
   :header-rows: 1
   :widths: 14 56 30

   * - Stage
     - What it does
     - Reads / writes
   * - ``download``
     - *(opt-in)* Fetches every benchmark named by the first arm that is
       missing on disk (:mod:`infl_ens.data.download`).
     - ``data/<benchmark>/``
   * - ``manifest``
     - Builds the stratified train/val/test split and the exact-coverage
       batch plan from the first specialist arm's ``data_split`` block.
       Skipped when the manifest file exists.
     - ``data/splits/<manifest>.json``
   * - ``train``
     - Runs every arm's task (``closed_loop`` or ``baseline_replay``) in
       experiment order through :data:`infl_ens.training.tasks.TASKS`.
     - ``<output_dir>/history.json``, ``resolved_config.yaml``,
       ``data_split.json``, ``agents/<pair>/round-NN/``, ``eval_<partition>/``
   * - ``perround``
     - Scores each specialist arm's archived adapters on
       ``eval.perround_partition`` at ``eval.perround_rounds`` and writes
       the held-out NLL by pair tables.
     - ``<output_dir>/eval_val/eval_results.json``,
       ``<output_dir>/tables/pair_nll_by_round.{csv,md,tex,json}``
   * - ``routing``
     - Route-then-score on ``eval.routing_partition``: pooled generalist
       vs learned routing (expected / sampled / argmax :math:`G`) vs the
       oracle ceiling, per specialist arm.
     - ``<output_dir>/routing_ensemble_diagnostics.json``
   * - ``figures``
     - Renders the experiment's figures and the cross-arm report
       (:mod:`infl_ens.figures.render`).
     - ``figures/<experiment>/``
   * - ``prune``
     - *(opt-in)* Deletes intermediate ``round-NN`` adapters, keeping the
       final round (:func:`infl_ens.utils.checkpoints.prune_intermediate_adapters`).
     - ``<output_dir>/agents/``

Useful invocations:

.. code-block:: bash

   # Validate every config and print the plan (no GPU, no torch import).
   python -m infl_ens.pipeline --config configs/experiments/seven_axis_3arm.yaml --dry-run

   # Cheap gate: the pytest subset plus two-round runs of the smoke arms.
   python -m infl_ens.pipeline --config ... --smoke

   # Re-run only the analysis stages after pulling results back.
   python -m infl_ens.pipeline --config ... --stages routing,figures

   # Train one arm only.
   python -m infl_ens.pipeline --config ... --stages train --only-arm soft

Progress goes to stdout and ``<results_dir>/pipeline.log``;
``<results_dir>/stage_status.json`` records when each stage started,
finished and whether it succeeded.

Per-stage CLIs
--------------

The pipeline calls the same entry points you can run by hand:

.. code-block:: bash

   python -m infl_ens.training   --config configs/arms/soft_topk3_pairs.yaml -- closed_loop.n_rounds=2
   python -m infl_ens.evaluation --config results/<run>/resolved_config.yaml -- eval.partitions='["val"]'
   python -m infl_ens.figures    --config configs/experiments/seven_axis_3arm.yaml --only pair_positions

``python -m infl_ens.figures --list`` prints the figure registry; figures
flagged *(gpu)* need the encoder and are rendered only with ``--gpu``.

Running on the GPU host
-----------------------

``scripts/run_on_doob.sh`` is the only shell script in the repository. It
copies ``src/``, ``configs/`` and ``tests/`` to the remote, then drives
the Python pipeline under ``tmux``:

.. code-block:: bash

   MODE=smoke  bash scripts/run_on_doob.sh              # gate first, in the foreground
   bash scripts/run_on_doob.sh                          # queue every stage under tmux
   STAGES=routing,figures bash scripts/run_on_doob.sh   # re-run the analysis stages
   MODE=status bash scripts/run_on_doob.sh              # tmux, log tail, stage status, GPU
   MODE=pull   bash scripts/run_on_doob.sh              # copy results + figures back

Environment variables: ``REMOTE`` (default ``mslovett@doob.dartmouth.edu``),
``REMOTE_REPO``, ``EXPERIMENT``, ``GPU``, ``STAGES``, ``ONLY_ARM``,
``FORCE=1``, ``SKIP_SYNC=1``, ``FORCE_GPU=1``, ``PY``.

The first log lines of a real launch should show the trait-space cache
being **loaded**, not built: the layered configs resolve to the same
``benchmarks`` + ``trait_space`` blocks as the cached encode
(fingerprint ``3b42c68a8dd334c5``), which ``tests/test_config_fingerprint.py``
guards.
