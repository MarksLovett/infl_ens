:mod:`infl_ens.training`
========================

Training entry point and trainers. Per AGENTS.md §4 rule 1 there is a
single CLI:

.. code-block:: bash

   python -m infl_ens.training --config <path>

which dispatches on the config's ``task`` field. The router trainer
(gradient ascent on agent positions) is exported *eagerly*; the LoRA
SFT helpers are exported *lazily* via :func:`__getattr__` so importing
``infl_ens.training`` does not pull in :mod:`torch` or
:mod:`transformers`.

Top-level re-exports
--------------------

.. autosummary::
   :nosignatures:

   infl_ens.training.RouterTrainingConfig
   infl_ens.training.train_router_positions
   infl_ens.training.SFTTrainingConfig
   infl_ens.training.sft_train_agent

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.training.router_training
   infl_ens.training.sft_training
