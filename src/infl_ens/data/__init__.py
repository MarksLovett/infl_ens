"""Trait-space construction, embedding wrappers, and benchmark loaders.

This subpackage groups the *data side* of the influencer-game pipeline:

- :class:`TraitSpace` and :func:`build_trait_space` define the
  :math:`L`-dimensional trait space :math:`\\mathbb{B}` and the empirical
  resource distribution :math:`B(b)` over it.
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

from infl_ens.data import benchmarks
from infl_ens.data.encoders import HuggingFaceEncoder
from infl_ens.data.trait_linear_transform import FrozenLinearTransform
from infl_ens.data.trait_normalize import QuantileNormalizer
from infl_ens.data.trait_space import (
    TraitSpace,
    build_trait_space,
    position_from_corpus,
)

__all__ = [
    "FrozenLinearTransform",
    "HuggingFaceEncoder",
    "QuantileNormalizer",
    "TraitSpace",
    "benchmarks",
    "build_trait_space",
    "position_from_corpus",
]
