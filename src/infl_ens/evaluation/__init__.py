"""Evaluate saved LoRA adapters on BeaverTails and HaluEval.

Per AGENTS.md module placement, benchmark *loading* lives in
:mod:`infl_ens.data.benchmarks`; this subpackage owns *scoring* adapters
against those corpora (mean per-token NLL on chat-formatted examples).

Single CLI::

    python -m infl_ens.evaluation --config <path>

Heavy dependencies (:mod:`torch`, :mod:`transformers`, :mod:`peft`) are
imported only inside :mod:`infl_ens.evaluation.adapters` and
:mod:`infl_ens.evaluation.metrics`, so ``import infl_ens.evaluation`` stays
lightweight.

Public surface (eager)
----------------------

- :data:`BENCHMARK_KINDS`, :func:`load_benchmark_splits`,
  :func:`subsample_split` from :mod:`infl_ens.evaluation.benchmarks`.
- :func:`is_adapter_dir`, :func:`resolve_adapter_dir`,
  :func:`discover_adapters`, :class:`AdapterRef` from
  :mod:`infl_ens.evaluation.adapters`.
- :class:`AdapterEvalConfig`, :class:`BenchmarkEvalResult`,
  :class:`EvalJobConfig`, :func:`evaluate_adapter_on_split`,
  :func:`evaluate_adapter_on_splits`, :func:`evaluate_run_adapters`,
  :func:`run_eval_job`, :func:`write_eval_report` from
  :mod:`infl_ens.evaluation.evaluate`.

Lazy:

- :func:`format_chat_example`, :func:`mean_token_nll` from
  :mod:`infl_ens.evaluation.metrics` (via ``__getattr__``).
- Compare helpers from :mod:`infl_ens.evaluation.compare` and
  :mod:`infl_ens.evaluation.capability_probe` (via ``__getattr__``).
"""

from __future__ import annotations

from typing import Any

from infl_ens.evaluation.adapters import (
    AdapterRef,
    discover_adapters,
    is_adapter_dir,
    resolve_adapter_dir,
)
from infl_ens.evaluation.benchmarks import (
    BENCHMARK_KINDS,
    load_benchmark_splits,
    subsample_split,
)
from infl_ens.evaluation.aggregate import (
    AggregatedEvalMetric,
    EvalMatrix,
    aggregate_eval_across_seeds,
    build_eval_matrix,
    format_eval_matrix_csv,
    format_eval_matrix_markdown,
    load_aggregated_report,
    write_aggregated_eval_report,
    write_eval_matrix_outputs,
)
from infl_ens.evaluation.evaluate import (
    AdapterEvalConfig,
    BenchmarkEvalResult,
    EvalJobConfig,
    evaluate_adapter_on_split,
    evaluate_adapter_on_splits,
    evaluate_run_adapters,
    run_eval_job,
    write_eval_report,
)

_LAZY_METRIC_NAMES: frozenset[str] = frozenset({
    "format_chat_example",
    "mean_token_nll",
    "split_to_texts",
})

_LAZY_BASE_EVAL_NAMES: frozenset[str] = frozenset({
    "BaseEvalResult",
    "evaluate_base_model",
    "write_base_eval_report",
})

_LAZY_COMPARE_NAMES: frozenset[str] = frozenset({
    "CornerMergeRecord",
    "DEFAULT_SAFETY_BENCHMARKS",
    "ModelScore",
    "aggregate_merge_by_corner",
    "assign_corner_roles",
    "compare_all_models",
    "compare_baseline_vs_specialists",
    "corner_centroid",
    "eval_adapter",
    "parse_merge_members",
    "process_merge_seed",
    "resolve_adapter_at",
})

_LAZY_CAPABILITY_PROBE_NAMES: frozenset[str] = frozenset({
    "cross_batch_margin",
    "probe_run",
    "write_probe_csv",
})

__all__ = [
    "AdapterEvalConfig",
    "AdapterRef",
    "AggregatedEvalMetric",
    "BENCHMARK_KINDS",
    "EvalMatrix",
    "BenchmarkEvalResult",
    "EvalJobConfig",
    "aggregate_eval_across_seeds",
    "build_eval_matrix",
    "discover_adapters",
    "format_eval_matrix_csv",
    "format_eval_matrix_markdown",
    "load_aggregated_report",
    "evaluate_adapter_on_split",
    "evaluate_adapter_on_splits",
    "evaluate_run_adapters",
    "format_chat_example",
    "is_adapter_dir",
    "load_benchmark_splits",
    "mean_token_nll",
    "resolve_adapter_dir",
    "run_eval_job",
    "split_to_texts",
    "subsample_split",
    "write_aggregated_eval_report",
    "write_eval_matrix_outputs",
    "write_eval_report",
    "BaseEvalResult",
    "evaluate_base_model",
    "write_base_eval_report",
    "CornerMergeRecord",
    "DEFAULT_SAFETY_BENCHMARKS",
    "ModelScore",
    "aggregate_merge_by_corner",
    "assign_corner_roles",
    "compare_all_models",
    "compare_baseline_vs_specialists",
    "corner_centroid",
    "cross_batch_margin",
    "eval_adapter",
    "parse_merge_members",
    "probe_run",
    "process_merge_seed",
    "resolve_adapter_at",
    "write_probe_csv",
]


def __getattr__(name: str) -> Any:
    """Resolve lazy metric exports on first access.

    :param name: Attribute name.
    :type name: str
    :returns: Requested symbol from :mod:`infl_ens.evaluation.metrics`.
    :rtype: Any
    :raises AttributeError: If ``name`` is not a lazy export.
    """
    if name in _LAZY_METRIC_NAMES:
        from infl_ens.evaluation import metrics
        value = getattr(metrics, name)
        globals()[name] = value
        return value
    if name in _LAZY_BASE_EVAL_NAMES:
        from infl_ens.evaluation import base_eval
        value = getattr(base_eval, name)
        globals()[name] = value
        return value
    if name in _LAZY_COMPARE_NAMES:
        from infl_ens.evaluation import compare
        value = getattr(compare, name)
        globals()[name] = value
        return value
    if name in _LAZY_CAPABILITY_PROBE_NAMES:
        from infl_ens.evaluation import capability_probe
        value = getattr(capability_probe, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy metric names to :func:`dir` and tab completion.

    :returns: Public attribute names.
    :rtype: list[str]
    """
    return sorted(
        set(globals())
        | _LAZY_METRIC_NAMES
        | _LAZY_BASE_EVAL_NAMES
        | _LAZY_COMPARE_NAMES
        | _LAZY_CAPABILITY_PROBE_NAMES
    )
