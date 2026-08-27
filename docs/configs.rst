Configs
=======

Every CLI reads YAML through :mod:`infl_ens.config`. A file may compose
other files with an ``includes:`` list (paths relative to the including
file); fragments are deep-merged in order and the including file's own
keys win. Dotted ``KEY=VALUE`` overrides given after ``--`` on the command
line are applied on top, and the result is validated against the key
tables in :mod:`infl_ens.config` — an unknown key fails with the file
that introduced it.

Layout
------

.. code-block:: text

   configs/
   ├── encoders/     qwen3_embedding_8b_awq.yaml (default), bge_large_en_v1_5.yaml (template)
   ├── trait_space/  seven_axis.yaml           geometry knobs; includes the encoder preset
   ├── data/         seven_axis_safety.yaml    the seven benchmarks + the train/val/test split
   ├── models/       qwen2_5_1_5b_instruct.yaml base LM + LoRA hyperparameters (sft block)
   ├── arms/         _closed_loop_base.yaml    everything the specialist arms share
   │                 soft_full_pairs.yaml, soft_topk3_pairs.yaml, topk3_unit_pairs.yaml,
   │                 hard_topk3_pairs.yaml, hard_pairs_matched.yaml, generalist_replay.yaml
   └── experiments/  seven_axis_3arm.yaml      arms, stages, evaluation window, figures, smoke

An arm file is short because it only says what differs:

.. code-block:: yaml

   # configs/arms/soft_topk3_pairs.yaml
   includes:
     - _closed_loop_base.yaml
   output_dir: results/seven_axis_soft_topk3_pairs/seed0
   closed_loop:
     routing_mode: soft
     soft_top_k: 3
     soft_loss: weighted

Changing an experiment
----------------------

- **A new encoder**: copy ``configs/encoders/bge_large_en_v1_5.yaml``,
  set ``trait_space.encoder`` (the model id) and the ``encoder`` block
  (pooling, padding side, dtype, placement), and include it from a
  ``configs/trait_space/*.yaml``. Any Hugging Face ``AutoModel`` that
  returns ``last_hidden_state`` works
  (:class:`infl_ens.data.encoders.HuggingFaceEncoder`).
- **A new arm**: add a file under ``configs/arms/`` that includes
  ``_closed_loop_base.yaml`` and overrides the routing knobs
  (``routing_mode``, ``soft_top_k``, ``soft_loss``, ``soft_select``,
  ``position_update``, ...; see
  :data:`infl_ens.config.CLOSED_LOOP_KEYS`), then list it under ``arms:``
  in the experiment file.
- **A different base model**: add ``configs/models/<name>.yaml`` with an
  ``sft`` block and include it instead of the Qwen2.5 one. A closed-loop
  config may still override individual fields under ``closed_loop.sft``.
- **Quick variants** without new files: ``python -m infl_ens.training
  --config configs/arms/soft_topk3_pairs.yaml -- closed_loop.n_rounds=2
  data_split=null``.

The trait-space cache contract
------------------------------

The resolved ``benchmarks`` list and ``trait_space`` block are hashed into
the trait-space cache fingerprint
(:func:`infl_ens.data.trait_space_cache.trait_space_fingerprint`). The
GPU host holds the 28k-prompt encode under
``data/trait_space_cache/3b42c68a8dd334c5``; every arm in
``configs/arms/`` resolves to exactly those blocks, and
``tests/test_config_fingerprint.py`` fails if an edit changes that.

Two keys are deliberately *outside* the fingerprint:

- ``trait_space.encoder_batch_size`` (throughput) and the cache location
  keys, so a host can tune them freely;
- the top-level ``encoder`` block with the
  :class:`~infl_ens.data.encoders.HuggingFaceEncoder` keyword arguments.
  Changing the *model id* (``trait_space.encoder``) changes the
  fingerprint and triggers a fresh encode; changing only pooling or
  ``max_length`` for the same id does not, so point ``trait_space.cache_dir``
  somewhere new in that case.

Experiment files
----------------

.. code-block:: yaml

   name: seven_axis_3arm
   results_dir: results/seven_axis_3arm     # pipeline.log + stage_status.json
   figures_dir: figures/seven_axis_3arm
   arms:
     - {name: soft, label: "Soft k=3 wtd", title: "Soft top-3 pairs",
        role: specialist, config: ../arms/soft_topk3_pairs.yaml}
     - {name: generalist, label: "Pooled generalist",
        role: generalist, config: ../arms/generalist_replay.yaml}
   stages: [manifest, train, perround, routing, figures]
   eval:
     perround_rounds: [4, final]   # held-out NLL at these rounds
     perround_partition: val
     routing_partition: test
     max_eval_records: 1000
   figures:
     axis_labels: [harm, hallucination, jailbreak, privacy, overrefusal, injection, policy]
     include: [oracle_routing, arm_comparison, pair_positions, within_pair, cross_arm_report]
   smoke:
     tests: [tests/test_soft_pairs.py, ...]
     arms: [soft, hard]
     overrides: {data_split: null, eval: null, closed_loop.n_rounds: 2}

See :class:`infl_ens.experiment.ExperimentConfig` for every key.

Resolved configs
----------------

Every run writes ``<output_dir>/resolved_config.yaml``: the flattened
config with ``agents`` and ``sft_merge_groups`` expanded to literal lists
and the merged ``sft`` block. The evaluation, routing and figure stages
read that file, never the arm YAML, so they always see what the run
actually used.
