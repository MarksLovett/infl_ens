"""Persist and reload benchmark-derived :class:`TraitSpace` artifacts.

Builds a content-addressed cache under ``data/trait_space_cache/`` so
theory-only router runs can skip repeated sentence-transformer encoding
when benchmark and trait-space settings are unchanged.
"""

from __future__ import annotations

import hashlib
import json
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
from infl_ens.data.encoders import SentenceTransformerEncoder
from infl_ens.data.trait_linear_transform import load_transform_from_cfg
from infl_ens.data.trait_space import TraitSpace

_CACHE_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_ARRAYS_NAME = "arrays.npz"


def trait_space_fingerprint(cfg: dict[str, Any]) -> str:
    """Hash the benchmark list and trait-space block of a router config.

    :param cfg: Full router or training config.
    :type cfg: dict
    :returns: Hex digest prefix used as a cache directory name.
    :rtype: str
    """
    payload = {
        "benchmarks": cfg.get("benchmarks", []),
        "trait_space": cfg.get("trait_space", {}),
    }
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
    :param encoder_name: Sentence-transformer identifier used at build time.
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

    project = _make_learned_projector(
        encoder,
        list(axes),
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
    }


def make_trait_space_encoder(cfg: dict[str, Any]) -> SentenceTransformerEncoder:
    """Construct the sentence encoder named in ``trait_space`` config.

    :param cfg: Router or training config.
    :type cfg: dict
    :returns: Encoder with optional ``encoder_batch_size`` override.
    :rtype: SentenceTransformerEncoder
    """
    ts_cfg = cfg.get("trait_space", {})
    return SentenceTransformerEncoder(
        model_name=ts_cfg.get(
            "encoder", "sentence-transformers/all-MiniLM-L6-v2",
        ),
        batch_size=int(ts_cfg.get("encoder_batch_size", 256)),
    )


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
    encoder = make_trait_space_encoder(cfg)
    build_kwargs = _trait_space_build_kwargs(cfg)
    use_cache = bool(ts_cfg.get("cache", False))
    if not use_cache:
        space = build_safety_trait_space_bundle(
            splits, encoder, **build_kwargs,
        ).space
        transform = load_transform_from_cfg(cfg)
        if transform is not None:
            space = transform.apply_trait_space(space)
        return space

    fingerprint = trait_space_fingerprint(cfg)
    cache_path = trait_space_cache_path(cfg)
    if cache_path.is_dir():
        try:
            space = load_safety_trait_space_cache(
                cache_path,
                encoder,
                expected_fingerprint=fingerprint,
                expected_encoder=encoder.model_name,
            )
            transform = load_transform_from_cfg(cfg)
            if transform is not None:
                space = transform.apply_trait_space(space)
            return space
        except (FileNotFoundError, ValueError, OSError, KeyError):
            pass

    bundle = build_safety_trait_space_bundle(splits, encoder, **build_kwargs)
    save_safety_trait_space_cache(
        cache_path,
        bundle,
        fingerprint=fingerprint,
        encoder_name=encoder.model_name,
    )
    space = bundle.space
    transform = load_transform_from_cfg(cfg)
    if transform is not None:
        space = transform.apply_trait_space(space)
    return space
