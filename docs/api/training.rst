:mod:`infl_ens.training`
========================

Training entry point and trainers. Per AGENTS.md §4 rule 1 there is a
single CLI:

.. code-block:: bash

   python -m infl_ens.training --config <path>

which dispatches on the config's ``task`` field. The router trainer
(gradient ascent on agent positions) is intended to be exported
*eagerly*; the LoRA SFT helpers *lazily* via ``__getattr__`` so
importing ``infl_ens.training`` does not pull in :mod:`torch` or
:mod:`transformers`.

Top-level re-exports
--------------------

Each link below jumps to the **canonical** definition:

- :class:`~infl_ens.training.router_training.RouterTrainingConfig`
- :func:`~infl_ens.training.router_training.train_router_positions`
- :class:`~infl_ens.training.sft_training.SFTTrainingConfig`
- :func:`~infl_ens.training.sft_training.sft_train_agent`

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.training.router_training
   infl_ens.training.sft_training
