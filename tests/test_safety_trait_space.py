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
from infl_ens.data.benchmarks.safety_trait_space import (
    build_safety_trait_space_bundle,
    project_pre_normalizer_coordinates,
)
from infl_ens.data.trait_linear_transform import fit_whiten


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


def _unique_prompts(theme: str, n: int, seed: int) -> list[str]:
    """Generate ``n`` unique prompts sharing a theme word bag."""
    rng = np.random.default_rng(seed)
    vocab = [f"tok{k}" for k in range(60)]
    return [
        f"{theme} u{seed}x{i} " + " ".join(rng.choice(vocab, size=3))
        for i in range(n)
    ]


def _unique_splits() -> tuple[list[BenchmarkSplit], list[str]]:
    """Two labelled splits of unique prompts plus their combined corpus."""
    harm_prompts = (
        _unique_prompts("harmful violent attack", 60, seed=10)
        + _unique_prompts("safe gentle cooking", 60, seed=11)
    )
    halu_prompts = (
        _unique_prompts("false invented claim", 60, seed=12)
        + _unique_prompts("accurate verified fact", 60, seed=13)
    )
    scores = [1.0] * 60 + [0.0] * 60
    splits = [
        _make_split("beavertails", "harm", harm_prompts, scores),
        _make_split("halueval", "hallucination", halu_prompts, scores),
    ]
    return splits, harm_prompts + halu_prompts


def _max_ks_vs_uniform(coords: np.ndarray) -> float:
    """Max per-axis KS statistic of columns against U[0, 1]."""
    n = coords.shape[0]
    grid = np.arange(1, n + 1) / n
    worst = 0.0
    for j in range(coords.shape[1]):
        v = np.sort(coords[:, j])
        ks = float(np.max(np.maximum(np.abs(grid - v), np.abs(v - (grid - 1.0 / n)))))
        worst = max(worst, ks)
    return worst


def test_project_bounded_and_uniform_bare_bones() -> None:
    """Bare-bones pipeline projects into [0,1]^L with near-uniform marginals."""
    splits, corpus = _unique_splits()
    space = build_safety_trait_space(splits, _toy_encoder, n_grid=6)
    coords = space.project(corpus)
    assert coords.shape == (len(corpus), 2)
    assert coords.min() >= 0.0
    assert coords.max() <= 1.0
    assert _max_ks_vs_uniform(coords) < 0.1


def test_project_bounded_and_uniform_with_whiten() -> None:
    """The [0,1]^L guarantee holds with an unbounded whiten transform active."""
    splits, corpus = _unique_splits()
    bundle = build_safety_trait_space_bundle(splits, _toy_encoder, n_grid=6)
    pre = project_pre_normalizer_coordinates(
        _toy_encoder, list(bundle.axes), corpus,
    )
    transform = fit_whiten(pre, fit_source="test corpus, label-blind")
    space = build_safety_trait_space(
        splits, _toy_encoder, n_grid=6, linear_transform=transform,
    )
    coords = space.project(corpus)
    assert coords.min() >= 0.0
    assert coords.max() <= 1.0
    assert _max_ks_vs_uniform(coords) < 0.1


def test_optional_transforms_toggle() -> None:
    """Neutral toggle values reproduce bare-bones; active ones change outputs."""
    splits, corpus = _unique_splits()
    probe = corpus[::10]
    base = build_safety_trait_space(splits, _toy_encoder, n_grid=6)
    neutral = build_safety_trait_space(
        splits,
        _toy_encoder,
        n_grid=6,
        coordinate_residualize=False,
        mode_alignment_weight=0.0,
        coordinate_stretch_gamma=1.0,
    )
    assert np.allclose(neutral.project(probe), base.project(probe))

    stretched = build_safety_trait_space(
        splits, _toy_encoder, n_grid=6, coordinate_stretch_gamma=2.0,
    )
    assert not np.allclose(stretched.project(probe), base.project(probe))
    assert stretched.project(probe).max() <= 1.0

    residualized = build_safety_trait_space(
        splits, _toy_encoder, n_grid=6, coordinate_residualize=True,
    )
    assert not np.allclose(residualized.project(probe), base.project(probe))

    mode_aligned = build_safety_trait_space(
        splits, _toy_encoder, n_grid=6, mode_alignment_weight=0.5,
    )
    assert not np.allclose(mode_aligned.project(probe), base.project(probe))
