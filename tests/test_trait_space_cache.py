"""Tests for trait-space embedding memoisation and on-disk cache."""

from __future__ import annotations

from typing import Sequence

from pathlib import Path

import numpy as np

from infl_ens.data.benchmarks import BenchmarkSplit
from infl_ens.data.benchmarks.safety_trait_space import build_safety_trait_space_bundle
from infl_ens.data.trait_space_cache import (
    load_safety_trait_space_cache,
    save_safety_trait_space_cache,
    trait_space_fingerprint,
)


def _toy_encoder(texts: Sequence[str]) -> np.ndarray:
    """Deterministic hash-bag encoder."""
    out = []
    for t in texts:
        words = t.lower().split()
        if not words:
            out.append(np.zeros(8))
            continue
        r = np.random.default_rng(abs(hash(t)) % (2 ** 32))
        out.append(r.standard_normal(8))
    return np.stack(out, axis=0)


class _CountingEncoder:
    """Encoder wrapper that counts underlying batch calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        return _toy_encoder(texts)


def _make_split(name: str, axis: str, prompts: list[str], scores: list[float]) -> BenchmarkSplit:
    return BenchmarkSplit(
        name=name,
        prompts=prompts,
        responses=[p + " response" for p in prompts],
        scores=np.asarray(scores, dtype=float),
        axis_name=axis,
    )


def test_build_bundle_deduplicates_embeddings() -> None:
    """Unique prompts/responses are encoded once even with residualization."""
    shared = ["shared calibration prompt"] * 4
    harm_prompts = shared + ["harmful prompt"] * 4 + ["safe prompt"] * 4
    halu_prompts = shared + ["false claim"] * 4 + ["true claim"] * 4
    splits = [
        _make_split("beavertails", "harm", harm_prompts, [1.0] * 8 + [0.0] * 4),
        _make_split("halueval", "hallucination", halu_prompts, [1.0] * 8 + [0.0] * 4),
    ]
    encoder = _CountingEncoder()
    build_safety_trait_space_bundle(
        splits,
        encoder,
        n_grid=4,
        coordinate_residualize=True,
        mode_alignment_weight=0.25,
    )
    assert encoder.calls <= 4


def test_trait_space_disk_cache_roundtrip(tmp_path: Path) -> None:
    """Saved cache reloads to the same grid, weights, and projections."""
    harm_prompts = ["harm text"] * 6 + ["safe text"] * 6
    halu_prompts = ["false text"] * 6 + ["true text"] * 6
    splits = [
        _make_split("beavertails", "harm", harm_prompts, [1.0] * 6 + [0.0] * 6),
        _make_split("halueval", "hallucination", halu_prompts, [1.0] * 6 + [0.0] * 6),
    ]
    encoder = _CountingEncoder()
    bundle = build_safety_trait_space_bundle(splits, encoder, n_grid=6)
    cfg = {
        "benchmarks": [{"kind": "beavertails"}, {"kind": "halueval"}],
        "trait_space": {"n_grid": 6},
    }
    fingerprint = trait_space_fingerprint(cfg)
    cache_path = tmp_path / fingerprint
    save_safety_trait_space_cache(
        cache_path,
        bundle,
        fingerprint=fingerprint,
        encoder_name="toy-encoder",
    )
    reloaded = load_safety_trait_space_cache(
        cache_path,
        encoder,
        expected_fingerprint=fingerprint,
        expected_encoder="toy-encoder",
    )
    assert np.allclose(reloaded.grid, bundle.space.grid)
    assert np.allclose(reloaded.weights, bundle.space.weights)
    queries = ["harm text", "safe text", "false text"]
    assert np.allclose(
        reloaded.project(queries),
        bundle.space.project(queries),
    )
