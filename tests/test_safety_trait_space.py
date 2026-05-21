"""Offline tests for :mod:`infl_ens.data.benchmarks.safety_trait_space`.

Uses a deterministic toy encoder so no model downloads are needed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from infl_ens.data.benchmarks import (
    BenchmarkSplit,
    build_safety_trait_space,
)


def _toy_encoder(texts: Sequence[str]) -> np.ndarray:
    """Hash-bag encoder that produces a stable 8-D embedding per token bag."""
    out = []
    for t in texts:
        words = t.lower().split()
        if not words:
            out.append(np.zeros(8))
            continue
        vecs = []
        for w in words:
            r = np.random.default_rng(abs(hash(w)) % (2 ** 32))
            vecs.append(r.standard_normal(8))
        out.append(np.mean(vecs, axis=0))
    return np.stack(out, axis=0)


def _make_split(name: str, axis: str, prompts: list[str], scores: list[float]) -> BenchmarkSplit:
    """Convenience constructor for a labelled :class:`BenchmarkSplit`."""
    return BenchmarkSplit(
        name=name,
        prompts=prompts,
        responses=[p + " response" for p in prompts],
        scores=np.asarray(scores, dtype=float),
        axis_name=axis,
    )


def test_build_safety_trait_space_two_axes() -> None:
    """Two splits produce a 2-D trait space with the expected axis labels."""
    harm_prompts = (
        ["very harmful query about violence"] * 6
        + ["safe query about cooking"] * 6
    )
    harm_scores = [1.0] * 6 + [0.0] * 6
    halu_prompts = (
        ["plausible-but-false claim about astronomy"] * 6
        + ["accurate factual claim about chemistry"] * 6
    )
    halu_scores = [1.0] * 6 + [0.0] * 6

    splits = [
        _make_split("beavertails", "harm", harm_prompts, harm_scores),
        _make_split("halueval", "hallucination", halu_prompts, halu_scores),
    ]
    space = build_safety_trait_space(splits, _toy_encoder, n_grid=8)
    assert space.L == 2
    assert space.K == 8 ** 2
    assert space.axis_labels == ("harm", "hallucination")
    # Grid coordinates lie in [0, 1] and weights normalise.
    assert space.grid.min() >= 0.0
    assert space.grid.max() <= 1.0
    assert np.isclose(space.weights.sum(), 1.0)
