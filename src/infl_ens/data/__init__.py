"""AI-safety benchmark loaders and learned trait-space construction.

This subpackage groups one module per benchmark used as a trait-space
axis, plus a combined builder. Public surface:

- :class:`BenchmarkSplit`: uniform container of (prompt, score) records.
- :func:`load_beavertails`: BeaverTails loader (harm axis).
- :func:`load_halueval`: HaluEval loader (hallucination axis).
- :func:`build_safety_trait_space`: combined :math:`N`-axis trait space.
- :class:`LearnedAxis`: a single scoring axis fit from labelled prompts.
- :data:`BEAVERTAILS_CATEGORIES`, :data:`HALUEVAL_TASKS`: dataset
  taxonomy constants exposed for downstream filtering.

All loaders return :class:`BenchmarkSplit` instances so downstream code
(trait-space builders, trainers, evaluation scripts) handles every
benchmark through one interface.
"""

from __future__ import annotations

from infl_ens.data.benchmarks.base import BenchmarkSplit
from infl_ens.data.benchmarks.beavertails import (
    BEAVERTAILS_CATEGORIES,
    load_beavertails,
)
from infl_ens.data.benchmarks.halueval import (
    HALUEVAL_TASKS,
    load_halueval,
)
from infl_ens.data.benchmarks.safety_trait_space import (
    LearnedAxis,
    build_safety_trait_space,
)

__all__ = [
    "BEAVERTAILS_CATEGORIES",
    "BenchmarkSplit",
    "HALUEVAL_TASKS",
    "LearnedAxis",
    "build_safety_trait_space",
    "load_beavertails",
    "load_halueval",
]
