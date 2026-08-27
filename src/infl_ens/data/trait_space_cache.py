"""Persist and reload benchmark-derived :class:`TraitSpace` artifacts.

Builds a content-addressed cache under ``data/trait_space_cache/`` so
theory-only router runs can skip repeated Hugging Face encoding
when benchmark and trait-space settings are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from infl_ens.data.benchmarks.base import BenchmarkSplit
from infl_ens.data.benchmarks.safety_trait_space import (
    LearnedAxis,
    SafetyTraitSpaceBundle,
    _make_learned_projector,
    build_safety_trait_space_bundle,
)
from infl_ens.data.encoders import HuggingFaceEncoder, make_encoder
from infl_ens.data.trait_linear_transform import (
    FrozenLinearTransform,
    load_transform_from_cfg,
)
from infl_ens.data.trait_normalize import AxisQuantileMap, QuantileNormalizer
from infl_ens.data.trait_space import TraitSpace

_CACHE_VERSION = 3
_MANIFEST_NAME = "manifest.json"
_ARRAYS_NAME = "arrays.npz"

#: ``trait_space`` keys that control *where* the cache lives or *how fast*
#: the encode runs, but not the geometry of the resulting trait space.
#: Excluding them from the fingerprint keeps one cache entry shared across
#: configs that differ only in throughput or storage location, so tuning
#: the batch size never forces a re-encode.
_FINGERPRINT_IGNORED_KEYS = frozenset(
    {
        "encoder_batch_size",
        "cache",
        "cache_dir",
        "cache_path",
    },
)


def _fingerprint_trait_space_block(ts_cfg: dict[str, Any]) -> dict[str, Any]:
    """Strip throughput/location-only keys from a ``trait_space`` block.

    Batch size and cache location change how quickly the encode runs and
    where its output is stored, never the coordinates it produces, so
    they must not participate in the cache identity.

    :param ts_cfg: The config's ``trait_space`` block.
    :type ts_cfg: dict
    :returns: Copy carrying only geometry-affecting settings.
    :rtype: dict
    """
    trimmed = {
        k: v for k, v in ts_cfg.items() if k not in _FINGERPRINT_IGNORED_KEYS
    }
    encoder = trimmed.get("encoder")
    if isinstance(encoder, dict):
        trimmed["encoder"] = {
            k: v
            for k, v in encoder.items()
            if k not in ("batch_size", "device_map")
        }
    return trimmed


def trait_space_fingerprint(cfg: dict[str, Any]) -> str:
    """Hash the benchmark list and trait-space block of a router config.

    Only settings that change the resulting geometry are hashed:
    :data:`_FINGERPRINT_IGNORED_KEYS` (batch size and cache location) are
    excluded, as are the encoder mapping's ``batch_size`` / ``device_map``
    entries. Tuning throughput therefore reuses an existing cache instead
    of forcing a fresh encode.

    When ``trait_space.linear_transform`` names a transform file, the
    file *content* is hashed too, so editing the transform JSON in place
    invalidates the cache rather than silently reusing stale geometry.

    :param cfg: Full router or training config.
    :type cfg: dict
    :returns: Hex digest prefix used as a cache directory name.
    :rtype: str
    """
    payload = {
        "benchmarks": cfg.get("benchmarks", []),
        "trait_space": _fingerprint_trait_space_block(cfg.get("trait_space") or {}),
    }
    rel = (cfg.get("trait_space") or {}).get("linear_transform")
    if rel:
        path = Path(str(rel))
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        try:
            payload["linear_transform_content"] = path.read_text(encoding="utf-8")
        except OSError:
            payload["linear_transform_content"] = None
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def trait_space_cache_path(cfg: dict[str, Any]) -> Path:
    """Resolve the on-disk cache directory for a config.

    Honors ``trait_space.cache_path`` when set; otherwise uses
    ``trait_space.cache_dir / <fingerprint>``.

    :param cfg: Router or training config.
    :type cfg: dict
    :returns: Cache directory path (may not exist yet).
    :rtype: pathlib.Path
    """
    ts_cfg = cfg.get("trait_space", {})
    explicit = ts_cfg.get("cache_path")
    if explicit:
        return Path(str(explicit))
    root = Path(str(ts_cfg.get("cache_dir", "data/trait_space_cache")))
    return root / trait_space_fingerprint(cfg)


def _axis_to_manifest(axis: LearnedAxis) -> dict[str, Any]:
    """Serialise scalar :class:`LearnedAxis` fields for JSON storage.

    :param axis: Learned axis.
    :type axis: LearnedAxis
    :returns: JSON-safe mapping without ndarray payloads.
    :rtype: dict
    """
    return {
        "name": axis.name,
        "lo": float(axis.lo),
        "hi": float(axis.hi),
        "residual_intercept": float(axis.residual_intercept),
        "residual_lo": axis.residual_lo,
        "residual_hi": axis.residual_hi,
        "has_residual_coef": axis.residual_coef is not None,
    }


def _axis_from_manifest(entry: dict[str, Any], arrays: np.lib.npyio.NpzFile, idx: int) -> LearnedAxis:
    """Reconstruct one :class:`LearnedAxis` from manifest metadata and arrays.

    :param entry: Manifest record for the axis.
    :type entry: dict
    :param arrays: Loaded ``arrays.npz`` handle.
    :type arrays: numpy.lib.npyio.NpzFile
    :param idx: Axis index.
    :type idx: int
    :returns: Reconstructed axis.
    :rtype: LearnedAxis
    """
    direction = np.asarray(arrays[f"axis_{idx}_direction"], dtype=float)
    residual_coef = None
    if entry.get("has_residual_coef"):
        residual_coef = np.asarray(arrays[f"axis_{idx}_residual_coef"], dtype=float)
    return LearnedAxis(
        direction=direction,
        lo=float(entry["lo"]),
        hi=float(entry["hi"]),
        name=str(entry["name"]),
        residual_coef=residual_coef,
        residual_intercept=float(entry.get("residual_intercept", 0.0)),
        residual_lo=entry.get("residual_lo"),
        residual_hi=entry.get("residual_hi"),
    )


def _normalizer_from_cache(
    manifest: dict[str, Any],
    arrays: np.lib.npyio.NpzFile,
    n_axes: int,
) -> QuantileNormalizer:
    """Reconstruct the quantile normalizer from manifest metadata and arrays.

    :param manifest: Loaded cache manifest.
    :type manifest: dict
    :param arrays: Loaded ``arrays.npz`` handle.
    :type arrays: numpy.lib.npyio.NpzFile
    :param n_axes: Number of trait axes.
    :type n_axes: int
    :returns: Reconstructed normalizer.
    :rtype: QuantileNormalizer
    """
    maps = tuple(
        AxisQuantileMap(
            knots=np.asarray(arrays[f"qnorm_axis_{idx}_knots"], dtype=float),
            cdf=np.asarray(arrays[f"qnorm_axis_{idx}_cdf"], dtype=float),
        )
        for idx in range(n_axes)
    )
    return QuantileNormalizer(
        axis_maps=maps,
        n_knots=int(manifest["quantile_knots"]),
        fit_n=int(manifest["normalizer_fit_n"]),
    )


def save_safety_trait_space_cache(
    path: Path,
    bundle: SafetyTraitSpaceBundle,
    *,
    fingerprint: str,
    encoder_name: str,
) -> None:
    """Write a trait-space bundle to ``path``.

    :param path: Cache directory to create or overwrite.
    :type path: pathlib.Path
    :param bundle: Built trait-space artifacts.
    :type bundle: SafetyTraitSpaceBundle
    :param fingerprint: Config fingerprint stored for validation.
    :type fingerprint: str
    :param encoder_name: Hugging Face model identifier used at build time.
    :type encoder_name: str
    :returns: ``None``.
    :rtype: None
    """
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": _CACHE_VERSION,
        "fingerprint": fingerprint,
        "encoder": encoder_name,
        "axis_labels": list(bundle.space.axis_labels or ()),
        "coordinate_stretch_gamma": float(bundle.coordinate_stretch_gamma),
        "coordinate_stretch_gammas": bundle.coordinate_stretch_gammas,
        "axes": [_axis_to_manifest(ax) for ax in bundle.axes],
        "quantile_knots": int(bundle.normalizer.n_knots),
        "normalizer_fit_n": int(bundle.normalizer.fit_n),
        "linear_transform": (
            bundle.linear_transform.to_dict()
            if bundle.linear_transform is not None
            else None
        ),
    }
    array_payload: dict[str, np.ndarray] = {
        "grid": np.asarray(bundle.space.grid, dtype=float),
        "weights": np.asarray(bundle.space.weights, dtype=float),
    }
    for idx, axis in enumerate(bundle.axes):
        array_payload[f"axis_{idx}_direction"] = np.asarray(axis.direction, dtype=float)
        if axis.residual_coef is not None:
            array_payload[f"axis_{idx}_residual_coef"] = np.asarray(
                axis.residual_coef, dtype=float,
            )
    for idx, axis_map in enumerate(bundle.normalizer.axis_maps):
        array_payload[f"qnorm_axis_{idx}_knots"] = np.asarray(axis_map.knots, dtype=float)
        array_payload[f"qnorm_axis_{idx}_cdf"] = np.asarray(axis_map.cdf, dtype=float)
    with (path / _MANIFEST_NAME).open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    np.savez_compressed(path / _ARRAYS_NAME, **array_payload)


def load_safety_trait_space_cache(
    path: Path,
    encoder: Callable[[Sequence[str]], np.ndarray],
    *,
    expected_fingerprint: Optional[str] = None,
    expected_encoder: Optional[str] = None,
) -> TraitSpace:
    """Reload a cached :class:`TraitSpace` and rebuild its ``project`` callable.

    :param path: Cache directory written by :func:`save_safety_trait_space_cache`.
    :type path: pathlib.Path
    :param encoder: Sentence encoder used at routing time.
    :type encoder: Callable[[Sequence[str]], numpy.ndarray]
    :param expected_fingerprint: Optional fingerprint that must match the
        manifest.
    :type expected_fingerprint: str | None
    :param expected_encoder: Optional encoder name that must match the manifest.
    :type expected_encoder: str | None
    :returns: Trait space with a live ``project`` closure.
    :rtype: TraitSpace
    :raises FileNotFoundError: If ``path`` is missing required files.
    :raises ValueError: If the manifest fails validation.
    """
    manifest_path = path / _MANIFEST_NAME
    arrays_path = path / _ARRAYS_NAME
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"incomplete trait-space cache at {path}")

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if int(manifest.get("version", -1)) != _CACHE_VERSION:
        raise ValueError(
            f"unsupported trait-space cache version {manifest.get('version')!r} "
            f"at {path}"
        )
    if expected_fingerprint is not None and manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            f"trait-space cache fingerprint mismatch at {path}: "
            f"expected {expected_fingerprint!r}, got {manifest.get('fingerprint')!r}"
        )
    if expected_encoder is not None and manifest.get("encoder") != expected_encoder:
        raise ValueError(
            f"trait-space cache encoder mismatch at {path}: "
            f"expected {expected_encoder!r}, got {manifest.get('encoder')!r}"
        )

    with np.load(arrays_path, allow_pickle=False) as arrays:
        axes = tuple(
            _axis_from_manifest(entry, arrays, idx)
            for idx, entry in enumerate(manifest["axes"])
        )
        grid = np.asarray(arrays["grid"], dtype=float)
        weights = np.asarray(arrays["weights"], dtype=float)
        normalizer = _normalizer_from_cache(manifest, arrays, len(axes))

    transform_payload = manifest.get("linear_transform")
    linear_transform = (
        FrozenLinearTransform.from_dict(transform_payload)
        if transform_payload
        else None
    )
    project = _make_learned_projector(
        encoder,
        list(axes),
        normalizer=normalizer,
        linear_transform=linear_transform,
        coordinate_stretch_gamma=float(manifest["coordinate_stretch_gamma"]),
        coordinate_stretch_gammas=manifest.get("coordinate_stretch_gammas"),
    )
    axis_labels = tuple(manifest.get("axis_labels") or tuple(ax.name for ax in axes))
    return TraitSpace(
        grid=grid,
        weights=weights,
        project=project,
        axis_labels=axis_labels,
    )


def _trait_space_build_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract :func:`build_safety_trait_space_bundle` kwargs from config.

    :param cfg: Router or training config.
    :type cfg: dict
    :returns: Keyword arguments for the builder.
    :rtype: dict
    """
    ts_cfg = cfg.get("trait_space", {})
    return {
        "n_grid": int(ts_cfg.get("n_grid", 32)),
        "kde_bandwidth": ts_cfg.get("kde_bandwidth"),
        "threshold": float(ts_cfg.get("threshold", 0.5)),
        "coordinate_residualize": bool(ts_cfg.get("coordinate_residualize", False)),
        "mode_alignment_weight": float(ts_cfg.get("mode_alignment_weight", 0.0)),
        "mode_alignment_weights": ts_cfg.get("mode_alignment_weights"),
        "coordinate_stretch_gamma": float(ts_cfg.get("coordinate_stretch_gamma", 1.0)),
        "coordinate_stretch_gammas": ts_cfg.get("coordinate_stretch_gammas"),
        "linear_transform": load_transform_from_cfg(cfg),
        "quantile_knots": int(ts_cfg.get("quantile_knots", 1001)),
    }



def build_or_load_safety_trait_space(
    cfg: dict[str, Any],
    splits: list[BenchmarkSplit],
) -> TraitSpace:
    """Build a safety trait space, optionally loading from disk cache.

    When ``trait_space.cache`` is ``True``, reads ``trait_space.cache_path``
    or ``trait_space.cache_dir / <fingerprint>`` before rebuilding. On a
    miss, builds the space, writes the cache, and returns the result.

    :param cfg: Router or training config.
    :type cfg: dict
    :param splits: Loaded benchmark splits.
    :type splits: list[BenchmarkSplit]
    :returns: Trait space for routing and theory solves.
    :rtype: TraitSpace
    """
    ts_cfg = cfg.get("trait_space", {})
    encoder = make_encoder(cfg)
    build_kwargs = _trait_space_build_kwargs(cfg)
    use_cache = bool(ts_cfg.get("cache", False))
    if not use_cache:
        return build_safety_trait_space_bundle(
            splits, encoder, **build_kwargs,
        ).space

    fingerprint = trait_space_fingerprint(cfg)
    cache_path = trait_space_cache_path(cfg)
    if cache_path.is_dir():
        try:
            return load_safety_trait_space_cache(
                cache_path,
                encoder,
                expected_fingerprint=fingerprint,
                expected_encoder=encoder.model_name,
            )
        except (FileNotFoundError, ValueError, OSError, KeyError):
            pass

    bundle = build_safety_trait_space_bundle(splits, encoder, **build_kwargs)
    save_safety_trait_space_cache(
        cache_path,
        bundle,
        fingerprint=fingerprint,
        encoder_name=encoder.model_name,
    )
    return bundle.space


@dataclass(frozen=True)
class CachedTraitArtifacts:
    """Frozen pipeline artifacts reloaded from a trait-space cache.

    :param axes: Learned axis directions and calibration metadata.
    :type axes: tuple[LearnedAxis, ...]
    :param normalizer: Fitted per-axis quantile normalizer.
    :type normalizer: QuantileNormalizer
    :param linear_transform: Optional frozen standardize/whiten transform.
    :type linear_transform: FrozenLinearTransform | None
    :param gammas: Per-axis post-normalization stretch exponents.
    :type gammas: numpy.ndarray
    :param axis_labels: Axis names in config order.
    :type axis_labels: tuple[str, ...]
    """

    axes: tuple[LearnedAxis, ...]
    normalizer: QuantileNormalizer
    linear_transform: Optional[FrozenLinearTransform]
    gammas: np.ndarray
    axis_labels: tuple[str, ...]


def artifacts_from_bundle(bundle: SafetyTraitSpaceBundle) -> CachedTraitArtifacts:
    """Adapt a freshly built bundle to the cached-artifacts container.

    Lets callers treat cached and uncached builds uniformly.

    :param bundle: Bundle returned by
        :func:`~infl_ens.data.benchmarks.safety_trait_space.build_safety_trait_space_bundle`.
    :type bundle: SafetyTraitSpaceBundle
    :returns: The same artifacts in cache-loader form.
    :rtype: CachedTraitArtifacts
    """
    from infl_ens.data.benchmarks.safety_trait_space import _coordinate_stretch_vector

    gammas = _coordinate_stretch_vector(
        bundle.axes,
        coordinate_stretch_gamma=bundle.coordinate_stretch_gamma,
        coordinate_stretch_gammas=bundle.coordinate_stretch_gammas,
    )
    return CachedTraitArtifacts(
        axes=tuple(bundle.axes),
        normalizer=bundle.normalizer,
        linear_transform=bundle.linear_transform,
        gammas=gammas,
        axis_labels=tuple(
            bundle.space.axis_labels or tuple(a.name for a in bundle.axes)
        ),
    )


def load_cache_artifacts(cfg: dict[str, Any]) -> CachedTraitArtifacts:
    """Reload learned axes, normalizer, transform, and stretch from cache.

    Lets analysis tools reconstruct any stage of the projection pipeline
    without re-encoding text or rebuilding the trait space.

    :param cfg: Router or training config with ``trait_space.cache``.
    :type cfg: dict
    :returns: The frozen pipeline artifacts.
    :rtype: CachedTraitArtifacts
    :raises FileNotFoundError: If the cache directory is missing or
        incomplete.
    :raises ValueError: If the cache version is unsupported.
    """
    path = trait_space_cache_path(cfg)
    manifest_path = path / _MANIFEST_NAME
    arrays_path = path / _ARRAYS_NAME
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(
            "load_cache_artifacts needs a built trait-space cache at "
            f"{path}; run the build once with trait_space.cache "
            "enabled to create it"
        )
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if int(manifest.get("version", -1)) != _CACHE_VERSION:
        raise ValueError(
            f"unsupported trait-space cache version {manifest.get('version')!r} "
            f"at {path}"
        )
    with np.load(arrays_path, allow_pickle=False) as arrays:
        axes = tuple(
            _axis_from_manifest(entry, arrays, idx)
            for idx, entry in enumerate(manifest["axes"])
        )
        normalizer = _normalizer_from_cache(manifest, arrays, len(axes))
    transform_payload = manifest.get("linear_transform")
    transform = (
        FrozenLinearTransform.from_dict(transform_payload)
        if transform_payload
        else None
    )
    names = [str(entry["name"]) for entry in manifest["axes"]]
    default_gamma = float(manifest["coordinate_stretch_gamma"])
    overrides = manifest.get("coordinate_stretch_gammas") or {}
    gammas = np.asarray(
        [float(overrides.get(name, default_gamma)) for name in names],
        dtype=float,
    )
    return CachedTraitArtifacts(
        axes=axes,
        normalizer=normalizer,
        linear_transform=transform,
        gammas=gammas,
        axis_labels=tuple(manifest.get("axis_labels") or names),
    )


def coordinate_chain_from_cache(
    cfg: dict[str, Any],
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the transform → CDF → stretch chain from a built cache.

    Used to migrate legacy pre-normalization coordinates (e.g. stored
    ``fixed_positions`` files) into the final normalized geometry without
    re-encoding any text. Requires the config's trait-space cache to have
    been built at the current cache version.

    :param cfg: Router or training config with ``trait_space.cache``.
    :type cfg: dict
    :returns: Callable mapping ``(L,)`` or ``(M, L)`` pre-normalizer
        coordinates into ``[0, 1]^L``.
    :rtype: Callable[[numpy.ndarray], numpy.ndarray]
    :raises FileNotFoundError: If the cache directory is missing or
        incomplete.
    :raises ValueError: If the cache version is unsupported.
    """
    artifacts = load_cache_artifacts(cfg)
    transform = artifacts.linear_transform
    normalizer = artifacts.normalizer
    gammas = artifacts.gammas

    def chain(coords: np.ndarray) -> np.ndarray:
        x = np.asarray(coords, dtype=float)
        if transform is not None:
            x = transform.apply(x)
        x = normalizer.transform(x)
        if not np.all(gammas == 1.0):
            x = 1.0 - np.power(1.0 - np.clip(x, 0.0, 1.0), gammas)
        return np.clip(x, 0.0, 1.0)

    return chain
