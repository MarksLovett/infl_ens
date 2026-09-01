:mod:`infl_ens.config` and :mod:`infl_ens.experiment`
=====================================================

Layered YAML loading shared by every CLI, and the experiment files that
describe a study (its arms, stages and analysis settings).

- :func:`~infl_ens.config.load_config` — includes, dotted overrides,
  key validation, flat ``dict`` out.
- :func:`~infl_ens.config.resolve_sft_block` — the merged base-model +
  LoRA settings of a run.
- :func:`~infl_ens.experiment.load_experiment` —
  :class:`~infl_ens.experiment.ExperimentConfig` with one
  :class:`~infl_ens.experiment.ArmSpec` per arm.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.config
   infl_ens.experiment
