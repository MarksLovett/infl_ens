"""Trait-space construction, embedding wrappers, and benchmark loaders.

This subpackage groups the *data side* of the influencer-game pipeline:

- :class:`TraitSpace` and :func:`build_trait_space` define the
  :math:`L`-dimensional trait space :math:`\\mathbb{B}` (a unit box or
  probability simplex) and the empirical resource distribution
  :math:`B(b)` over it.
- :func:`position_from_corpus` projects a fresh corpus of queries into a
  pre-built trait space (used by closed-loop trainers to refresh agent
  positions between rounds).
- :class:`HuggingFaceEncoder` is the direct Hugging Face embedding backend.
  Heavy imports (``torch`` and ``transformers``) are deferred until the
  encoder is *constructed*, so importing this subpackage stays cheap.
- :mod:`infl_ens.data.benchmarks` provides labelled benchmark loaders
  for BeaverTails (harm axis) and HaluEval (hallucination axis).

Per AGENTS.md §3 this subpackage is *code only*: no disk reads, no global
state, no side effects at import time.
"""

from __future__ import annotations

from infl_ens.data import behavioral, benchmarks
from infl_ens.data.behavioral import (
    BehavioralCase,
    BehavioralSuite,
    ChatMessage,
    audit_contamination,
    load_behavioral_suite,
    load_behavioral_suites,
)
from infl_ens.data.encoders import HuggingFaceEncoder
from infl_ens.data.splits import flatten_partition_records
from infl_ens.data.trait_normalize import QuantileNormalizer
from infl_ens.data.trait_space import (
    TraitSpace,
    build_trait_space,
    positive_simplex_grid,
    position_from_corpus,
    softmax_to_simplex,
)

__all__ = [
    "BehavioralCase",
    "BehavioralSuite",
    "ChatMessage",
    "HuggingFaceEncoder",
    "QuantileNormalizer",
    "TraitSpace",
    "audit_contamination",
    "behavioral",
    "benchmarks",
    "build_trait_space",
    "flatten_partition_records",
    "load_behavioral_suite",
    "load_behavioral_suites",
    "position_from_corpus",
    "positive_simplex_grid",
    "softmax_to_simplex",
]
