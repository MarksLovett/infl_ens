"""Frozen unsupervised linear transforms on calibrated trait coordinates.

Applies a fixed affine map :math:`x' = (x - \\mu) W^\\top` to **both**
the trait-space KDE grid and live ``project`` outputs so theory dynamics
and Gaussian routing :math:`G` operate in the same geometry.

Oracle merge labels must never enter transform fitting — only trait-vector
second moments from held-out split prompts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from infl_ens.data.trait_space import TraitSpace

_EPS = 1e-8


@dataclass(frozen=True)
class FrozenLinearTransform:
    """Unsupervised affine trait-space map frozen at fit time.

    :param mean: Column mean :math:`\\mu`, shape ``(L,)``.
    :type mean: numpy.ndarray
    :param matrix: Right-multiply matrix :math:`W`, shape ``(L, L)``;
        transforms as ``(x - mean) @ matrix.T``.
    :type matrix: numpy.ndarray
    :param mode: One of ``standardize`` or ``whiten``.
    :type mode: str
    :param fit_source: Provenance string (must not reference oracle labels).
    :type fit_source: str
    :param fit_n: Number of trait vectors used to fit.
    :type fit_n: int
  """

    mean: np.ndarray
    matrix: np.ndarray
    mode: str
    fit_source: str
    fit_n: int

    @property
    def L(self) -> int:
        """Trait dimension."""
        return int(self.mean.shape[0])

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """Apply the frozen map to coordinate rows.

        :param coords: Shape ``(M, L)`` or ``(L,)``.
        :type coords: numpy.ndarray
        :returns: Transformed coordinates, same rank as input.
        :rtype: numpy.ndarray
        """
        x = np.asarray(coords, dtype=float)
        if x.ndim == 1:
            return (x - self.mean) @ self.matrix.T
        return (x - self.mean) @ self.matrix.T

    def apply_trait_space(self, space: TraitSpace) -> TraitSpace:
        """Return a trait space whose grid and ``project`` use this map.

        Both the KDE resource grid **and** live query projections are
        transformed so closed-loop position updates and router
        :func:`~infl_ens.inflgame.router.allocation.allocation_weights`
        share one coordinate system.

        :param space: Base calibrated trait space.
        :type space: TraitSpace
        :returns: Wrapped trait space in transformed coordinates.
        :rtype: TraitSpace
        """
        grid_t = self.apply(space.grid)
        base_project = space.project

        def project(queries: Sequence[str]) -> np.ndarray:
            return self.apply(base_project(queries))

        return TraitSpace(
            grid=grid_t,
            weights=space.weights,
            project=project,
            axis_labels=space.axis_labels,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable payload."""
        return {
            "mode": self.mode,
            "fit_source": self.fit_source,
            "fit_n": self.fit_n,
            "mean": self.mean.tolist(),
            "matrix": self.matrix.tolist(),
            "oracle_labels_used": False,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrozenLinearTransform:
        """Load from a JSON-compatible mapping."""
        if payload.get("oracle_labels_used"):
            raise ValueError(
                "transform fit must not use oracle labels; "
                f"got oracle_labels_used={payload['oracle_labels_used']!r}",
            )
        return cls(
            mean=np.asarray(payload["mean"], dtype=float),
            matrix=np.asarray(payload["matrix"], dtype=float),
            mode=str(payload["mode"]),
            fit_source=str(payload["fit_source"]),
            fit_n=int(payload["fit_n"]),
        )

    def save(self, path: Path) -> None:
        """Write transform JSON to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FrozenLinearTransform:
        """Load transform from JSON."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def fit_standardize(
    coords: np.ndarray,
    *,
    fit_source: str,
) -> FrozenLinearTransform:
    """Per-axis mean-center and divide by std (no decorrelation).

    :param coords: Trait vectors, shape ``(N, L)``.
    :type coords: numpy.ndarray
    :param fit_source: Provenance label stored in the artifact.
    :type fit_source: str
    :returns: Frozen standardize transform.
    :rtype: FrozenLinearTransform
    """
    x = np.asarray(coords, dtype=float)
    mu = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < _EPS, 1.0, std)
    w = np.diag(1.0 / std)
    return FrozenLinearTransform(
        mean=mu,
        matrix=w,
        mode="standardize",
        fit_source=fit_source,
        fit_n=int(x.shape[0]),
    )


def fit_whiten(
    coords: np.ndarray,
    *,
    fit_source: str,
    method: str = "zca",
) -> FrozenLinearTransform:
    """ZCA whitening: decorrelate and equalise variance.

    :param coords: Trait vectors, shape ``(N, L)``.
    :type coords: numpy.ndarray
    :param fit_source: Provenance label stored in the artifact.
    :type fit_source: str
    :param method: Only ``zca`` is supported.
    :type method: str
    :returns: Frozen whitening transform.
    :rtype: FrozenLinearTransform
    :raises ValueError: If ``method`` is unsupported.
    """
    if method != "zca":
        raise ValueError(f"method must be 'zca', got {method!r}")
    x = np.asarray(coords, dtype=float)
    n, _l = x.shape
    mu = x.mean(axis=0)
    xc = x - mu
    cov = (xc.T @ xc) / max(n - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(evals, _EPS))
    w = evecs @ np.diag(inv_sqrt) @ evecs.T
    return FrozenLinearTransform(
        mean=mu,
        matrix=w,
        mode="whiten",
        fit_source=fit_source,
        fit_n=int(n),
    )


def matrix_distance(a: FrozenLinearTransform, b: FrozenLinearTransform) -> dict[str, float]:
    """Frobenius and operator-norm distance between two transforms.

    :param a: First transform.
    :type a: FrozenLinearTransform
    :param b: Second transform.
    :type b: FrozenLinearTransform
    :returns: Distance metrics on the composite matrix ``W_a - W_b``.
    :rtype: dict[str, float]
    """
    delta = a.matrix - b.matrix
    fro = float(np.linalg.norm(delta, ord="fro"))
    op = float(np.linalg.norm(delta, ord=2))
    return {"frobenius": fro, "operator_2": op}


def load_transform_from_cfg(
    cfg: dict[str, Any],
    repo_root: Path | None = None,
) -> FrozenLinearTransform | None:
    """Load optional ``trait_space.linear_transform`` from config.

    :param cfg: Router or training config.
    :type cfg: dict
    :param repo_root: Repository root for relative paths.
    :type repo_root: pathlib.Path | None
    :returns: Transform, or ``None`` when unset.
    :rtype: FrozenLinearTransform | None
    """
    rel = (cfg.get("trait_space") or {}).get("linear_transform")
    if not rel:
        return None
    path = Path(str(rel))
    root = repo_root or Path(__file__).resolve().parents[3]
    if not path.is_absolute():
        path = root / path
    return FrozenLinearTransform.load(path)


def realized_moments(coords: np.ndarray) -> dict[str, Any]:
    """Per-axis variance and mean absolute off-diagonal correlation.

    :param coords: Transformed trait vectors, shape ``(N, L)``.
    :type coords: numpy.ndarray
    :returns: Summary statistics for VERIFY_ONLY checks.
    :rtype: dict
    """
    x = np.asarray(coords, dtype=float)
    var = np.var(x, axis=0)
    if x.shape[0] < 2:
        corr = np.eye(x.shape[1])
    else:
        corr = np.corrcoef(x, rowvar=False)
    off = corr - np.diag(np.diag(corr))
    return {
        "per_axis_variance": var.tolist(),
        "mean_abs_offdiag_corr": float(np.mean(np.abs(off))),
        "max_abs_offdiag_corr": float(np.max(np.abs(off))),
    }
