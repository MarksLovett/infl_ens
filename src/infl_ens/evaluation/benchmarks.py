"""Benchmark loading for evaluation (re-export of the shared loader).

The loader lives in :mod:`infl_ens.data.benchmarks.loading` so training,
evaluation and the pipeline read the same ``benchmarks`` config shape.
This module keeps the historical import path working.
"""

from __future__ import annotations

from infl_ens.data.benchmarks.loading import (
    BENCHMARK_KINDS,
    load_benchmark_split,
    load_benchmark_splits,
    load_benchmark_splits_with_partition,
    subsample_split,
)

__all__ = [
    "BENCHMARK_KINDS",
    "load_benchmark_split",
    "load_benchmark_splits",
    "load_benchmark_splits_with_partition",
    "subsample_split",
]
