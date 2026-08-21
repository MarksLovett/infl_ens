:mod:`infl_ens.evaluation`
===========================

Evaluate saved LoRA adapters on the BeaverTails and HaluEval benchmarks.
Benchmark *loading* remains in :mod:`infl_ens.data.benchmarks`; this
subpackage scores adapters with mean per-token NLL on chat-formatted
``(prompt, response)`` pairs (the same objective as SFT training).

Single CLI::

   python -m infl_ens.evaluation --config configs/evaluation/adapter_on_benchmarks.yaml

Top-level re-exports
--------------------

- :class:`~infl_ens.evaluation.evaluate.AdapterEvalConfig`
- :class:`~infl_ens.evaluation.evaluate.BenchmarkEvalResult`
- :class:`~infl_ens.evaluation.evaluate.EvalJobConfig`
- :func:`~infl_ens.evaluation.evaluate.evaluate_adapter_on_splits`
- :func:`~infl_ens.evaluation.evaluate.evaluate_run_adapters`
- :func:`~infl_ens.evaluation.evaluate.run_eval_job`
- :func:`~infl_ens.evaluation.benchmarks.load_benchmark_splits`
- :func:`~infl_ens.evaluation.adapters.discover_adapters`

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.evaluation.adapters
   infl_ens.evaluation.benchmarks
   infl_ens.evaluation.metrics
   infl_ens.evaluation.evaluate
