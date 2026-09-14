"""Score saved LoRA adapters on the safety benchmarks.

Per AGENTS.md module placement, benchmark *loading* lives in
:mod:`infl_ens.data.benchmarks`; this subpackage owns *scoring* adapters
against those corpora (mean per-token NLL on chat-formatted examples) and
the route-then-score diagnostics of a routed ensemble.

Single CLI::

    python -m infl_ens.evaluation --config <training yaml>

Heavy dependencies (:mod:`torch`, :mod:`transformers`, :mod:`peft`) are
imported only inside :mod:`infl_ens.evaluation.adapters`,
:mod:`infl_ens.evaluation.metrics` and :mod:`infl_ens.evaluation.routing_eval`,
so ``import infl_ens.evaluation`` stays lightweight.

Public surface (eager)
----------------------

- :data:`BENCHMARK_KINDS`, :func:`load_benchmark_splits`,
  :func:`subsample_split` (re-exported from
  :mod:`infl_ens.data.benchmarks.loading`).
- :func:`is_adapter_dir`, :func:`resolve_adapter_dir`,
  :func:`discover_adapters`, :class:`AdapterRef` from
  :mod:`infl_ens.evaluation.adapters`.
- :class:`AdapterEvalConfig`, :class:`BenchmarkEvalResult`,
  :class:`EvalJobConfig`, :func:`evaluate_adapter_on_split`,
  :func:`evaluate_adapter_on_splits`, :func:`evaluate_run_adapters`,
  :func:`run_eval_job`, :func:`run_unified_eval`,
  :func:`final_round_from_history`, :func:`write_eval_report` from
  :mod:`infl_ens.evaluation.evaluate`.

Lazy:

- :func:`format_chat_example`, :func:`mean_token_nll`,
  :func:`split_to_texts` from :mod:`infl_ens.evaluation.metrics` (via
  ``__getattr__``).
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
from infl_ens.evaluation.evaluate import (
    AdapterEvalConfig,
    BenchmarkEvalResult,
    EvalJobConfig,
    evaluate_adapter_on_split,
    evaluate_adapter_on_splits,
    evaluate_run_adapters,
    final_round_from_history,
    run_eval_job,
    run_unified_eval,
    write_eval_report,
)
from infl_ens.evaluation.nll_artifact import (
    NLL_ARTIFACT_SCHEMA_VERSION,
    NllMatrixArtifact,
    checkpoint_sha256,
    ordered_records_sha256,
)
from infl_ens.evaluation.routers import (
    LinearRouter,
    build_validation_router_suite,
    fit_best_linear_router,
    fit_linear_router,
    fit_simplex_stacking,
    gaussian_router,
    hindsight_oracle_router,
    low_support_mask,
    metadata_router,
    nearest_centroid_router,
    normalize_routing_weights,
    permute_router,
    uniform_router,
    validation_selected_permutation,
)
from infl_ens.evaluation.routing_eval import RoutingWeightScores, score_routing_weights

_LAZY_METRIC_NAMES: frozenset[str] = frozenset({
    "format_chat_example",
    "mean_token_nll",
    "split_to_texts",
})

__all__ = [
    "AdapterEvalConfig",
    "AdapterRef",
    "BENCHMARK_KINDS",
    "BenchmarkEvalResult",
    "EvalJobConfig",
    "LinearRouter",
    "NLL_ARTIFACT_SCHEMA_VERSION",
    "NllMatrixArtifact",
    "RoutingWeightScores",
    "checkpoint_sha256",
    "build_validation_router_suite",
    "discover_adapters",
    "evaluate_adapter_on_split",
    "evaluate_adapter_on_splits",
    "evaluate_run_adapters",
    "final_round_from_history",
    "fit_linear_router",
    "fit_best_linear_router",
    "fit_simplex_stacking",
    "format_chat_example",
    "gaussian_router",
    "hindsight_oracle_router",
    "is_adapter_dir",
    "load_benchmark_splits",
    "low_support_mask",
    "mean_token_nll",
    "metadata_router",
    "nearest_centroid_router",
    "normalize_routing_weights",
    "ordered_records_sha256",
    "permute_router",
    "resolve_adapter_dir",
    "run_eval_job",
    "run_unified_eval",
    "score_routing_weights",
    "split_to_texts",
    "subsample_split",
    "uniform_router",
    "validation_selected_permutation",
    "write_eval_report",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy metric names to :func:`dir` and tab completion.

    :returns: Public attribute names.
    :rtype: list[str]
    """
    return sorted(set(globals()) | _LAZY_METRIC_NAMES)
