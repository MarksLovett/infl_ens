"""Tests for trait-space embedding memoisation and on-disk cache."""

from __future__ import annotations

import json

from typing import Sequence

from pathlib import Path

import numpy as np

from infl_ens.data.benchmarks import BenchmarkSplit
from infl_ens.data.benchmarks.safety_trait_space import (
    build_safety_trait_space_bundle,
    project_pre_normalizer_coordinates,
)
from infl_ens.data.trait_space_cache import (
    _trait_space_build_kwargs,
    coordinate_chain_from_cache,
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


def _unique_prompts(theme: str, n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    vocab = [f"tok{k}" for k in range(40)]
    return [
        f"{theme} u{seed}x{i} " + " ".join(rng.choice(vocab, size=3))
        for i in range(n)
    ]


def _unique_splits() -> list[BenchmarkSplit]:
    harm = _unique_prompts("harm bad", 20, seed=1) + _unique_prompts("safe good", 20, seed=2)
    halu = _unique_prompts("false wrong", 20, seed=3) + _unique_prompts("true right", 20, seed=4)
    scores = [1.0] * 20 + [0.0] * 20
    return [
        _make_split("beavertails", "harm", harm, scores),
        _make_split("halueval", "hallucination", halu, scores),
    ]


def test_full_chain_cache_roundtrip(tmp_path: Path) -> None:
    """Residualize + stretch survive a cache roundtrip bit-exactly."""
    splits = _unique_splits()
    corpus = [p for s in splits for p in s.prompts]
    encoder = _CountingEncoder()

    bundle = build_safety_trait_space_bundle(
        splits,
        encoder,
        n_grid=4,
        coordinate_residualize=True,
        coordinate_stretch_gamma=2.0,
        quantile_knots=101,
    )
    cache_path = tmp_path / "full_chain"
    save_safety_trait_space_cache(
        cache_path, bundle, fingerprint="fp", encoder_name="toy",
    )

    manifest = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 3
    assert manifest["quantile_knots"] == 101
    assert "linear_transform" not in manifest
    with np.load(cache_path / "arrays.npz") as arrays:
        assert "qnorm_axis_0_knots" in arrays
        assert "qnorm_axis_1_cdf" in arrays

    reloaded = load_safety_trait_space_cache(
        cache_path, encoder, expected_fingerprint="fp", expected_encoder="toy",
    )
    queries = corpus[::7]
    got = reloaded.project(queries)
    assert np.allclose(got, bundle.space.project(queries))
    assert got.min() >= 0.0
    assert got.max() <= 1.0


def test_fingerprint_ignores_throughput_and_location_keys() -> None:
    """Batch size and cache location must not change the cache identity."""
    cfg = {
        "benchmarks": [{"kind": "beavertails"}],
        "trait_space": {
            "encoder": "some/model",
            "encoder_batch_size": 2,
            "cache": True,
            "cache_dir": "data/trait_space_cache",
            "n_grid": 3,
        },
    }
    base = trait_space_fingerprint(cfg)

    for key, value in (
        ("encoder_batch_size", 64),
        ("cache", False),
        ("cache_dir", "/tmp/elsewhere"),
        ("cache_path", "/tmp/explicit"),
    ):
        variant = {**cfg, "trait_space": {**cfg["trait_space"], key: value}}
        assert trait_space_fingerprint(variant) == base, key

    # Nested encoder mappings: batch_size / device_map are also throughput.
    mapping = {
        "benchmarks": cfg["benchmarks"],
        "trait_space": {
            "encoder": {"model_name": "some/model", "batch_size": 2},
            "n_grid": 3,
        },
    }
    faster = {
        "benchmarks": cfg["benchmarks"],
        "trait_space": {
            "encoder": {
                "model_name": "some/model",
                "batch_size": 64,
                "device_map": "cuda:0",
            },
            "n_grid": 3,
        },
    }
    assert trait_space_fingerprint(mapping) == trait_space_fingerprint(faster)

    # Geometry-affecting settings still must change the fingerprint.
    for key, value in (
        ("n_grid", 8),
        ("coordinate_residualize", True),
        ("coordinate_stretch_gamma", 2.0),
        ("quantile_knots", 51),
    ):
        variant = {**cfg, "trait_space": {**cfg["trait_space"], key: value}}
        assert trait_space_fingerprint(variant) != base, key


def test_build_kwargs_defaults() -> None:
    """Config extraction includes the normalizer keys."""
    kwargs = _trait_space_build_kwargs({"trait_space": {}})
    assert kwargs["quantile_knots"] == 1001
    assert "linear_transform" not in kwargs
    kwargs = _trait_space_build_kwargs({"trait_space": {"quantile_knots": 51}})
    assert kwargs["quantile_knots"] == 51


def test_coordinate_chain_matches_projector(tmp_path: Path) -> None:
    """The cached chain equals CDF∘stretch on pre-normalizer coords."""
    splits = _unique_splits()
    corpus = [p for s in splits for p in s.prompts]
    encoder = _CountingEncoder()

    bundle = build_safety_trait_space_bundle(
        splits,
        encoder,
        n_grid=4,
        coordinate_stretch_gamma=1.5,
        quantile_knots=101,
    )
    cache_path = tmp_path / "chain"
    save_safety_trait_space_cache(
        cache_path, bundle, fingerprint="fp", encoder_name="toy",
    )
    cfg = {"trait_space": {"cache": True, "cache_path": str(cache_path)}}
    chain = coordinate_chain_from_cache(cfg)

    queries = corpus[::5]
    pre_nt = project_pre_normalizer_coordinates(
        encoder, list(bundle.axes), queries,
    )
    assert np.allclose(chain(pre_nt), bundle.space.project(queries))
    single = chain(pre_nt[0])
    assert single.shape == (2,)
    assert np.allclose(single, bundle.space.project([queries[0]])[0])
