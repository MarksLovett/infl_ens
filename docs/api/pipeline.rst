:mod:`infl_ens.pipeline`
========================

The end-to-end experiment runner behind

.. code-block:: bash

   python -m infl_ens.pipeline --config configs/experiments/seven_axis_3arm.yaml

Stages are functions of a :class:`~infl_ens.pipeline.stages.PipelineContext`
registered in :data:`~infl_ens.pipeline.stages.STAGES`; see :doc:`../pipeline`
for what each one reads and writes.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.pipeline.stages
