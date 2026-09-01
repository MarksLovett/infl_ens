:mod:`infl_ens.evaluation`
===========================

Score saved LoRA adapters on the safety benchmarks (mean per-token NLL on
chat-formatted ``(prompt, response)`` pairs, the same objective as SFT)
and run the route-then-score diagnostics of a routed ensemble. Benchmark
*loading* lives in :mod:`infl_ens.data.benchmarks.loading`.

Single CLI:

.. code-block:: bash

   python -m infl_ens.evaluation --config results/<run>/resolved_config.yaml -- eval.partitions='["val"]'

Top-level re-exports
--------------------

- :class:`~infl_ens.evaluation.evaluate.AdapterEvalConfig`
- :class:`~infl_ens.evaluation.evaluate.BenchmarkEvalResult`
- :class:`~infl_ens.evaluation.evaluate.EvalJobConfig`
- :func:`~infl_ens.evaluation.evaluate.evaluate_adapter_on_splits`
- :func:`~infl_ens.evaluation.evaluate.evaluate_run_adapters`
- :func:`~infl_ens.evaluation.evaluate.run_eval_job`
- :func:`~infl_ens.evaluation.evaluate.run_unified_eval`
- :func:`~infl_ens.evaluation.adapters.discover_adapters`

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.evaluation.evaluate
   infl_ens.evaluation.routing_eval
   infl_ens.evaluation.adapters
   infl_ens.evaluation.metrics
   infl_ens.evaluation.benchmarks
