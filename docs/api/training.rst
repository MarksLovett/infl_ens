:mod:`infl_ens.training`
========================

The closed loop (route → SFT → position update) and the pooled replay
baseline. Per AGENTS.md rule 1 there is a single training CLI:

.. code-block:: bash

   python -m infl_ens.training --config configs/arms/soft_topk3_pairs.yaml [-- KEY=VAL ...]

which dispatches on the config's ``task`` field through
:data:`~infl_ens.training.tasks.TASKS` (``closed_loop`` or
``baseline_replay``). The router trainer (gradient ascent on agent
positions) is exported *eagerly*; the LoRA SFT helpers *lazily* via
``__getattr__`` so importing ``infl_ens.training`` does not pull in
:mod:`torch` or :mod:`transformers`.

Top-level re-exports
--------------------

- :class:`~infl_ens.training.router_training.RouterTrainingConfig`
- :func:`~infl_ens.training.router_training.train_router_positions`
- :class:`~infl_ens.training.sft_training.SFTTrainingConfig`
- :func:`~infl_ens.training.sft_training.sft_train_agent`

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.training.tasks
   infl_ens.training.closed_loop
   infl_ens.training.setup
   infl_ens.training.agent_init
   infl_ens.training.position_step
   infl_ens.training.router_training
   infl_ens.training.sft_training
   infl_ens.training.merge_training
   infl_ens.training.baseline_replay
   infl_ens.training.data_split
   infl_ens.training.closed_loop_eval
   infl_ens.training.pool_dynamics
