Configs
=======

Hydra/YAML configs live under ``configs/`` at the repo root and are
loaded by the single training CLI:

.. code-block:: bash

   python -m infl_ens.training --config configs/benchmark/router/safety_truth.yaml

Per AGENTS.md §4 rule 1, *adding a new training scenario means adding a
config file*, not adding a new entry-point script.

Model configs
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Role
   * - ``configs/model/qwen2_5_1_5b.yaml``
     - Base-model + LoRA hyperparameters for the SFT trainer.

Data configs
------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Role
   * - ``configs/data/beavertails.yaml``
     - Static BeaverTails loader settings.
   * - ``configs/data/halueval.yaml``
     - Static HaluEval loader settings.

Benchmark / router configs
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - File
     - Role
   * - ``configs/benchmark/router/example.yaml``
     - Original synthetic three-anchor example.
   * - ``configs/benchmark/router/safety_truth.yaml``
     - 2-D BeaverTails + HaluEval closed-loop config.
   * - ``configs/benchmark/router/safety_truth_n4_r10_strategic.yaml``
     - Strategic-gradient routing variant
       (``routing_weight: G_times_1mG``).
   * - ``configs/benchmark/router/safety_truth_n4_r10_strategic_long.yaml``
     - Same routing, pushes the SFT step harder (3 epochs, batch 512).
   * - ``configs/benchmark/router/safety_truth_n4_r20_strategic_long.yaml``
     - 20-round variant — stability of the strategic ``(2,2)`` basin.
   * - ``configs/benchmark/router/safety_truth_n4_r40_strategic_long.yaml``
     - 40-round variant — long-horizon stability + overfitting check.
   * - ``configs/benchmark/router/safety_truth_n4_r{10,20,40}_strategic_long_cum.yaml``
     - Cumulative-LoRA variants: each agent loads its prior adapter and
       continues training rather than restarting from the base model.
   * - ``configs/benchmark/router/beavertails_only.yaml``
     - 1-D harm-axis ablation.
   * - ``configs/benchmark/router/halueval_only.yaml``
     - 1-D hallucination-axis ablation.
