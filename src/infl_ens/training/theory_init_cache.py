"""Disk cache for the closed-loop theory initialization.

Each arm opens with an 8000-step grid-Nash gradient ascent that runs on the CPU
while the GPU sits idle. Measured at production settings it costs ~15.9 min at
N = 14 (it does *not* converge early there -- both passes hit the 8000-step cap)
and ~55 s at N = 4, where it does. Across a ten-arm replicate that is ~2.6 h of
dead GPU time, and ~16 h over the six replicates still to run.

The solve is a pure function of its inputs and is bitwise reproducible: two
calls in one process and two calls in *separate* processes under heavy host load
produce byte-identical position arrays (verified sha256 over ``theory_end``,
``initial`` and the stacked agent positions). So it is safe to compute once,
store, and reload -- and, more usefully, to precompute a whole replicate's arms
in parallel ahead of time so training never waits for them.

Structure deliberately mirrors :mod:`infl_ens.data.trait_space_cache`:
``<root>/<fingerprint>/{manifest.json,arrays.npz}``, an explicit version int,
and rebuild-on-any-mismatch so a stale or corrupt entry degrades to a fresh
solve rather than a wrong answer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# Bump whenever the stored payload's meaning changes, so old entries are
# ignored rather than misread.
_CACHE_VERSION = 1

_MANIFEST = "manifest.json"
_ARRAYS = "arrays.npz"

DEFAULT_CACHE_ROOT = "data/theory_init_cache"


def theory_init_fingerprint(
    *,
    names: Sequence[str],
    sigma: float,
    grid: np.ndarray,
    weights: np.ndarray,
    seed: int,
    learning_rate: float,
    n_steps: int,
    tol: float,
    min_pairwise: float,
    pairing: str,
    skip_initial_theory: bool,
) -> str:
    """Hash every input the solve actually depends on.

    The list is exhaustive by design -- a missing input would silently serve a
    result computed for different settings, which is far worse than a cache
    miss. Included because they feed the computation: agent count, ``sigma``,
    the trait-space grid and weights, the run ``seed`` (it picks the random
    start, hence the equilibrium basin), the five ``theory_gradient`` knobs and
    ``skip_initial_theory`` (it selects a different branch entirely).

    ``names`` are included even though the arithmetic never reads them: they
    order the position rows and become the keys of ``paired_harm_order``,
    ``pair_positions`` and ``pair_dominant_axis`` in the returned metadata, and
    ``paired_harm_order`` decides which agents share a LoRA adapter.

    Deliberately excluded: ``init_noise`` (applied as jitter *after* the solve,
    so it is re-applied cheaply on load), ``policy``, ``snap_collapsed_pairs``
    and ``collapse_merge_threshold`` (all consumed strictly post-init).

    :returns: 16 hex characters identifying this input set.
    :rtype: str
    """
    payload = {
        "version": _CACHE_VERSION,
        "names": list(names),
        "sigma": float(sigma),
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "n_steps": int(n_steps),
        "tol": float(tol),
        "min_pairwise": float(min_pairwise),
        "pairing": str(pairing),
        "skip_initial_theory": bool(skip_initial_theory),
        # Hash the raw bytes rather than the values: the grid is large, and the
        # bytes are what the solve actually consumes.
        "grid_sha": hashlib.sha256(
            np.ascontiguousarray(grid, dtype=np.float64).tobytes()
        ).hexdigest(),
        "weights_sha": hashlib.sha256(
            np.ascontiguousarray(weights, dtype=np.float64).tobytes()
        ).hexdigest(),
        "grid_shape": list(np.shape(grid)),
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def theory_init_cache_path(fingerprint: str, root: str = DEFAULT_CACHE_ROOT) -> Path:
    """Directory holding one cached solve."""
    return Path(root) / fingerprint


def save_theory_init(meta: dict[str, Any], fingerprint: str,
                     root: str = DEFAULT_CACHE_ROOT) -> Path:
    """Persist a solve's metadata.

    Arrays go to ``arrays.npz``; everything else goes to ``manifest.json``. The
    split is by type rather than by a fixed key list, so a future field added to
    ``meta`` is stored rather than silently dropped.
    """
    path = theory_init_cache_path(fingerprint, root)
    path.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    plain: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            plain[key] = value

    manifest = {
        "version": _CACHE_VERSION,
        "fingerprint": fingerprint,
        "array_keys": sorted(arrays),
        "meta": plain,
    }
    # Write to temporaries and move, so a crash mid-write cannot leave a
    # half-written entry that later reads as valid.
    tmp_arrays = path / (_ARRAYS + ".tmp")
    tmp_manifest = path / (_MANIFEST + ".tmp")
    # Pass a handle, not a path: ``savez_compressed`` appends ".npz" to any
    # filename that lacks it, which would write "arrays.npz.tmp.npz" instead.
    with open(tmp_arrays, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    with open(tmp_manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, sort_keys=True, default=str)
    tmp_arrays.replace(path / _ARRAYS)
    tmp_manifest.replace(path / _MANIFEST)
    return path


def load_theory_init(fingerprint: str,
                     root: str = DEFAULT_CACHE_ROOT) -> Optional[dict[str, Any]]:
    """Return a cached solve's metadata, or ``None`` on any miss or mismatch.

    Every failure path returns ``None`` rather than raising: a cache is an
    optimisation, and the caller must always be able to fall back to solving.
    """
    path = theory_init_cache_path(fingerprint, root)
    try:
        with open(path / _MANIFEST, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        if int(manifest.get("version", -1)) != _CACHE_VERSION:
            return None
        if str(manifest.get("fingerprint")) != fingerprint:
            return None
        meta: dict[str, Any] = dict(manifest.get("meta", {}))
        with np.load(path / _ARRAYS) as npz:
            for key in manifest.get("array_keys", []):
                meta[key] = npz[key]
        if "theory_end" not in meta:
            return None
        return meta
    except (FileNotFoundError, ValueError, OSError, KeyError, TypeError):
        return None
