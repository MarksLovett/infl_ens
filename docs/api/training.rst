:mod:`infl_ens.training`
========================

Training entry point and trainers. Per AGENTS.md §4 rule 1 there is a
single CLI:

.. code-block:: bash

   python -m infl_ens.training --config <path>

which dispatches on the config's ``task`` field. The router trainer
(gradient ascent on agent positions) and the LoRA SFT trainer live as
submodules.

.. currentmodule:: infl_ens.training

Top-level re-exports
--------------------

.. autosummary::
   :nosignatures:

   RouterTrainingConfig
   train_router_positions

The SFT helpers are lazily re-exported (``infl_ens.training`` avoids
importing :mod:`torch` and :mod:`transformers` at package import time):

.. autosummary::
   :nosignatures:

   SFTTrainingConfig
   sft_train_agent

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.training.router_training
   infl_ens.training.sft_training
