"""Per-axis quantile (empirical-CDF) normalization to :math:`[0, 1]^L`.

The always-on final stage of the trait-space pipeline: after axis learning
and any optional transforms (residualization, standardize/whiten), every
coordinate axis is mapped through a frozen empirical CDF fitted on the
calibration corpus. The output is guaranteed to lie in :math:`[0, 1]^L`
with near-uniform marginals on the calibration distribution, regardless of
which upstream transforms are active.

Fitting must use **unclipped** upstream scores — fitting a CDF on already
clipped values piles ties at 0/1 and destroys marginal uniformity.

Maps are frozen at fit time and never see oracle labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AxisQuantileMap:
    """Monotone interpolated empirical CDF for a single trait axis.

    :param knots: Strictly increasing coordinate knots, shape ``(k,)``.
    :type knots: numpy.ndarray
    :param cdf: Nondecreasing CDF values in ``[0, 1]`` at each knot,
        shape ``(k,)``.
    :type cdf: numpy.ndarray
    """

    knots: np.ndarray
    cdf: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Map raw axis values through the fitted CDF.

        Values below the first knot map to the first CDF value and values
        above the last knot map to the last (``np.interp`` endpoint
        clamping), so out-of-range queries saturate smoothly at 0/1.

        :param values: Raw axis values, any shape.
        :type values: numpy.ndarray
        :returns: CDF-normalized values in ``[0, 1]``, same shape.
        :rtype: numpy.ndarray
        """
        return np.interp(np.asarray(values, dtype=float), self.knots, self.cdf)


@dataclass(frozen=True)
class QuantileNormalizer:
    """Frozen per-axis empirical-CDF normalizer to :math:`[0, 1]^L`.

    :param axis_maps: One fitted :class:`AxisQuantileMap` per trait axis.
    :type axis_maps: tuple[AxisQuantileMap, ...]
    :param n_knots: Requested quantile-grid size at fit time.
    :type n_knots: int
    :param fit_n: Number of calibration coordinates used to fit.
    :type fit_n: int
    """

    axis_maps: tuple[AxisQuantileMap, ...]
    n_knots: int
    fit_n: int

    @property
    def L(self) -> int:
        """Trait dimension."""
        return len(self.axis_maps)

    def transform(self, coords: np.ndarray) -> np.ndarray:
        """Normalize coordinate rows into :math:`[0, 1]^L`.

        :param coords: Shape ``(M, L)`` or ``(L,)``.
        :type coords: numpy.ndarray
        :returns: Normalized coordinates, same rank as input.
        :rtype: numpy.ndarray
        :raises ValueError: If the trailing dimension is not ``L``.
        """
        x = np.asarray(coords, dtype=float)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.L:
            raise ValueError(
                f"coords must have trailing dimension {self.L}, "
                f"got shape {np.asarray(coords).shape}"
            )
        out = np.stack(
            [m.transform(x[:, j]) for j, m in enumerate(self.axis_maps)],
            axis=1,
        )
        return out[0] if single else out

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable payload."""
        return {
            "n_knots": int(self.n_knots),
            "fit_n": int(self.fit_n),
            "axes": [
                {"knots": m.knots.tolist(), "cdf": m.cdf.tolist()}
                for m in self.axis_maps
            ],
            "oracle_labels_used": False,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> QuantileNormalizer:
        """Load from a JSON-compatible mapping."""
        if payload.get("oracle_labels_used"):
            raise ValueError(
                "normalizer fit must not use oracle labels; "
                f"got oracle_labels_used={payload['oracle_labels_used']!r}",
            )
        maps = tuple(
            AxisQuantileMap(
                knots=np.asarray(entry["knots"], dtype=float),
                cdf=np.asarray(entry["cdf"], dtype=float),
            )
            for entry in payload["axes"]
        )
        return cls(
            axis_maps=maps,
            n_knots=int(payload["n_knots"]),
            fit_n=int(payload["fit_n"]),
        )

    def save(self, path: Path) -> None:
        """Write normalizer JSON to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> QuantileNormalizer:
        """Load normalizer from JSON."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _fit_axis_quantile_map(col: np.ndarray, n_knots: int) -> AxisQuantileMap:
    """Fit one axis's interpolated empirical CDF.

    Knots are placed at a fixed quantile grid (deterministic, bounded
    size). Tied knot values are collapsed to a single knot whose CDF value
    is the mean of the tied grid positions (midpoint-of-jump convention).
    Sentinel knots just outside the observed range pin the CDF to exactly
    0 and 1 so out-of-range values clamp continuously.

    :param col: Raw axis values, shape ``(N,)``.
    :type col: numpy.ndarray
    :param n_knots: Quantile-grid size before tie collapsing.
    :type n_knots: int
    :returns: Fitted axis map.
    :rtype: AxisQuantileMap
    """
    n = col.shape[0]
    q_grid = np.linspace(0.0, 1.0, min(int(n_knots), max(2, n)))
    knots = np.quantile(col, q_grid)
    uniq, inverse = np.unique(knots, return_inverse=True)
    if uniq.shape[0] == 1:
        return AxisQuantileMap(knots=uniq, cdf=np.asarray([0.5]))
    counts = np.bincount(inverse)
    cdf = np.bincount(inverse, weights=q_grid) / counts
    span = float(uniq[-1] - uniq[0])
    eps = max(1e-9, 1e-6 * span)
    return AxisQuantileMap(
        knots=np.concatenate(([uniq[0] - eps], uniq, [uniq[-1] + eps])),
        cdf=np.concatenate(([0.0], cdf, [1.0])),
    )


def fit_quantile_normalizer(
    coords: np.ndarray,
    *,
    n_knots: int = 1001,
) -> QuantileNormalizer:
    """Fit a per-axis empirical-CDF normalizer on calibration coordinates.

    :param coords: Unclipped calibration coordinates, shape ``(N, L)``.
    :type coords: numpy.ndarray
    :param n_knots: Quantile-grid size per axis (before tie collapsing).
    :type n_knots: int
    :returns: Frozen normalizer mapping into :math:`[0, 1]^L`.
    :rtype: QuantileNormalizer
    :raises ValueError: If ``coords`` is not 2-D or ``n_knots < 2``.
    """
    x = np.asarray(coords, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError(f"coords must be a non-empty (N, L) array, got shape {x.shape}")
    if int(n_knots) < 2:
        raise ValueError(f"n_knots must be >= 2, got {n_knots}")
    maps = tuple(
        _fit_axis_quantile_map(x[:, j], int(n_knots)) for j in range(x.shape[1])
    )
    return QuantileNormalizer(
        axis_maps=maps,
        n_knots=int(n_knots),
        fit_n=int(x.shape[0]),
    )
