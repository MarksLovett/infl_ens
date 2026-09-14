"""Concrete log-concave influence-kernel families."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from infl_ens.inflgame.kernels.base import InfluenceKernel, validate_kernel_inputs


def _positive_sigma(value: float) -> float:
    sigma = float(value)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be positive and finite, got {value!r}")
    return sigma


def _special() -> tuple[Any, Any, Any]:
    """Load SciPy special functions only for kernels that require them."""
    try:
        from scipy.special import digamma, gammaln, polygamma
    except ImportError as exc:  # pragma: no cover - environment-level
        raise ImportError(
            "Dirichlet and product-Beta kernels require scipy; install infl_ens[ml]"
        ) from exc
    return digamma, gammaln, polygamma


@dataclass(frozen=True)
class GaussianKernel(InfluenceKernel):
    """Multivariate Gaussian influence kernel.

    :param dimension: Trait dimension.
    :type dimension: int
    :param sigma: Scalar reach multiplying the shape matrix.
    :type sigma: float
    :param shape: Symmetric positive-definite shape matrix.
    :type shape: numpy.ndarray
    """

    dimension: int
    sigma: float
    shape: np.ndarray
    _precision: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", _positive_sigma(self.sigma))
        shape = np.asarray(self.shape, dtype=float)
        if shape.shape != (self.dimension, self.dimension):
            raise ValueError("Gaussian shape has the wrong dimension")
        object.__setattr__(self, "shape", shape.copy())
        object.__setattr__(self, "_precision", np.linalg.inv(shape) / self.sigma**2)

    @property
    def kind(self) -> str:
        """Return ``"gaussian"``.

        :returns: Kernel name.
        :rtype: str
        """
        return "gaussian"

    @property
    def covariance(self) -> np.ndarray:
        """Gaussian covariance :math:`\\sigma^2 C`.

        :returns: Covariance matrix.
        :rtype: numpy.ndarray
        """
        return self.sigma**2 * self.shape

    def log_influence(self, positions: np.ndarray, resources: np.ndarray) -> np.ndarray:
        """Evaluate unnormalized Gaussian log influence.

        :param positions: Agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Resource coordinates, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Log influences, shape ``(N, M)``.
        :rtype: numpy.ndarray
        """
        x, b = validate_kernel_inputs(positions, resources, self.dimension)
        diff = x[:, None, :] - b[None, :, :]
        return -0.5 * np.einsum("nml,lk,nmk->nm", diff, self._precision, diff)

    def score(self, positions: np.ndarray, resources: np.ndarray) -> np.ndarray:
        """Evaluate the Gaussian position score.

        :param positions: Agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Resource coordinates, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Position scores, shape ``(N, M, L)``.
        :rtype: numpy.ndarray
        """
        x, b = validate_kernel_inputs(positions, resources, self.dimension)
        return np.einsum("lk,nmk->nml", self._precision, b[None, :, :] - x[:, None, :])

    def log_hessian(self, positions: np.ndarray, resources: np.ndarray) -> np.ndarray:
        """Evaluate the constant Gaussian log Hessian.

        :param positions: Agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Resource coordinates, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Log Hessians, shape ``(N, M, L, L)``.
        :rtype: numpy.ndarray
        """
        x, b = validate_kernel_inputs(positions, resources, self.dimension)
        return np.broadcast_to(-self._precision, (len(x), len(b), self.dimension, self.dimension)).copy()

    def with_sigma(self, sigma: float) -> "GaussianKernel":
        """Return a Gaussian kernel with another reach.

        :param sigma: Positive competitive reach.
        :type sigma: float
        :returns: Reparameterized Gaussian kernel.
        :rtype: GaussianKernel
        """
        return GaussianKernel(self.dimension, sigma, self.shape)

    def to_config(self) -> dict[str, Any]:
        """Return serializable Gaussian metadata.

        :returns: Resolved kernel fields.
        :rtype: dict[str, Any]
        """
        return {"kind": self.kind, "sigma": self.sigma, "shape": self.shape.tolist()}


@dataclass(frozen=True)
class HyperbolicKernel(InfluenceKernel):
    """Smoothed multivariate hyperbolic influence kernel.

    :param dimension: Trait dimension.
    :type dimension: int
    :param sigma: Positive reach.
    :type sigma: float
    :param shape: Symmetric positive-definite Mahalanobis shape.
    :type shape: numpy.ndarray
    :param delta: Positive smoothing radius.
    :type delta: float
    """

    dimension: int
    sigma: float
    shape: np.ndarray
    delta: float = 0.1
    _precision: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", _positive_sigma(self.sigma))
        delta = float(self.delta)
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError(f"hyperbolic delta must be positive, got {self.delta!r}")
        object.__setattr__(self, "delta", delta)
        shape = np.asarray(self.shape, dtype=float)
        if shape.shape != (self.dimension, self.dimension):
            raise ValueError("hyperbolic shape has the wrong dimension")
        object.__setattr__(self, "shape", shape.copy())
        object.__setattr__(self, "_precision", np.linalg.inv(shape))

    @property
    def kind(self) -> str:
        """Return ``"hyperbolic"``.

        :returns: Kernel name.
        :rtype: str
        """
        return "hyperbolic"

    def _geometry(self, positions: np.ndarray, resources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, b = validate_kernel_inputs(positions, resources, self.dimension)
        d = b[None, :, :] - x[:, None, :]
        pd = np.einsum("lk,nmk->nml", self._precision, d)
        radius = np.sqrt(self.delta**2 + np.einsum("nml,nml->nm", d, pd))
        return pd, radius

    def log_influence(self, positions: np.ndarray, resources: np.ndarray) -> np.ndarray:
        """Evaluate hyperbolic log influence.

        :param positions: Agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Resource coordinates, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Log influences, shape ``(N, M)``.
        :rtype: numpy.ndarray
        """
        _pd, radius = self._geometry(positions, resources)
        return -radius / self.sigma

    def score(self, positions: np.ndarray, resources: np.ndarray) -> np.ndarray:
        """Evaluate the hyperbolic position score.

        :param positions: Agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Resource coordinates, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Position scores, shape ``(N, M, L)``.
        :rtype: numpy.ndarray
        """
        pd, radius = self._geometry(positions, resources)
        return pd / (self.sigma * radius[..., None])

    def log_hessian(self, positions: np.ndarray, resources: np.ndarray) -> np.ndarray:
        """Evaluate the strictly negative hyperbolic log Hessian.

        :param positions: Agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Resource coordinates, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Log Hessians, shape ``(N, M, L, L)``.
        :rtype: numpy.ndarray
        """
        pd, radius = self._geometry(positions, resources)
        base = -self._precision[None, None, :, :] / (self.sigma * radius[..., None, None])
        outer = pd[..., :, None] * pd[..., None, :]
        return base + outer / (self.sigma * radius[..., None, None] ** 3)

    def with_sigma(self, sigma: float) -> "HyperbolicKernel":
        """Return a hyperbolic kernel with another reach.

        :param sigma: Positive competitive reach.
        :type sigma: float
        :returns: Reparameterized hyperbolic kernel.
        :rtype: HyperbolicKernel
        """
        return HyperbolicKernel(self.dimension, sigma, self.shape, self.delta)

    def to_config(self) -> dict[str, Any]:
        """Return serializable hyperbolic metadata.

        :returns: Resolved kernel fields.
        :rtype: dict[str, Any]
        """
        return {
            "kind": self.kind,
            "sigma": self.sigma,
            "shape": self.shape.tolist(),
            "delta": self.delta,
        }


@dataclass(frozen=True)
class DirichletKernel(InfluenceKernel):
    """Mode-parameterized Dirichlet influence kernel.

    The concentration vector is :math:`\\alpha=1+x/\\sigma`. For a simplex
    position ``x`` this gives ``sum(alpha) - L = 1 / sigma`` and hence the
    Dirichlet mode is exactly ``x``.

    :param dimension: Simplex ambient dimension.
    :type dimension: int
    :param sigma: Positive inverse mode concentration.
    :type sigma: float
    """

    dimension: int
    sigma: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", _positive_sigma(self.sigma))

    @property
    def kind(self) -> str:
        """Return ``"dirichlet"``.

        :returns: Kernel name.
        :rtype: str
        """
        return "dirichlet"

    def validate_domain(self, domain: str) -> None:
        """Require the native probability simplex.

        :param domain: Trait-space coordinate domain.
        :type domain: str
        :returns: ``None``.
        :rtype: None
        :raises ValueError: If ``domain`` is not ``"simplex"``.
        """
        if domain != "simplex":
            raise ValueError("the mode-parameterized Dirichlet kernel requires a simplex")

    def _inputs(self, positions: np.ndarray, resources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, b = validate_kernel_inputs(positions, resources, self.dimension)
        if np.any(x < -1e-12) or not np.allclose(x.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Dirichlet positions must lie on the probability simplex")
        if np.any(b <= 0.0) or not np.allclose(b.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Dirichlet resources must lie in the open probability simplex")
        return x, b

    def log_influence(self, positions: np.ndarray, resources: np.ndarray) -> np.ndarray:
        """Evaluate the normalized Dirichlet log density.

        :param positions: Simplex agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Open-simplex resources, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Log densities, shape ``(N, M)``.
        :rtype: numpy.ndarray
        """
        _digamma, gammaln, _polygamma = _special()
        x, b = self._inputs(positions, resources)
        alpha = 1.0 + x / self.sigma
        normalizer = gammaln(alpha.sum(axis=1)) - gammaln(alpha).sum(axis=1)
        return normalizer[:, None] + np.einsum(
            "nl,ml->nm", alpha - 1.0, np.log(b)
        )

    def score(self, positions: np.ndarray, resources: np.ndarray) -> np.ndarray:
        """Evaluate the full ambient Dirichlet position score.

        :param positions: Simplex agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Open-simplex resources, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Ambient position scores, shape ``(N, M, L)``.
        :rtype: numpy.ndarray
        """
        digamma, _gammaln, _polygamma = _special()
        x, b = self._inputs(positions, resources)
        alpha = 1.0 + x / self.sigma
        common = digamma(alpha.sum(axis=1))[:, None, None]
        return (
            np.log(b)[None, :, :]
            - digamma(alpha)[:, None, :]
            + common
        ) / self.sigma

    def log_hessian(self, positions: np.ndarray, resources: np.ndarray) -> np.ndarray:
        """Evaluate the full ambient Dirichlet log Hessian.

        :param positions: Simplex agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Open-simplex resources, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Ambient log Hessians, shape ``(N, M, L, L)``.
        :rtype: numpy.ndarray
        """
        _digamma, _gammaln, polygamma = _special()
        x, b = self._inputs(positions, resources)
        alpha = 1.0 + x / self.sigma
        common = polygamma(1, alpha.sum(axis=1))[:, None, None]
        h = np.broadcast_to(common, (len(x), self.dimension, self.dimension)).copy()
        diagonal = np.arange(self.dimension)
        h[:, diagonal, diagonal] -= polygamma(1, alpha)
        h /= self.sigma**2
        return np.broadcast_to(h[:, None, :, :], (len(x), len(b), self.dimension, self.dimension)).copy()

    def with_sigma(self, sigma: float) -> "DirichletKernel":
        """Return a mode-parameterized Dirichlet kernel with another reach.

        :param sigma: Positive inverse concentration.
        :type sigma: float
        :returns: Reparameterized Dirichlet kernel.
        :rtype: DirichletKernel
        """
        return DirichletKernel(self.dimension, sigma)

    def to_config(self) -> dict[str, Any]:
        """Return serializable Dirichlet metadata.

        :returns: Resolved kernel fields.
        :rtype: dict[str, Any]
        """
        return {"kind": self.kind, "sigma": self.sigma, "parameterization": "mode"}


@dataclass(frozen=True)
class ProductBetaKernel(InfluenceKernel):
    """Product of mode-parameterized beta influence factors.

    :param dimension: Trait dimension.
    :type dimension: int
    :param sigma: Positive inverse concentration.
    :type sigma: float
    """

    dimension: int
    sigma: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", _positive_sigma(self.sigma))

    @property
    def kind(self) -> str:
        """Return ``"product_beta"``.

        :returns: Kernel name.
        :rtype: str
        """
        return "product_beta"

    def _inputs(self, positions: np.ndarray, resources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, b = validate_kernel_inputs(positions, resources, self.dimension)
        if np.any(x < -1e-12) or np.any(x > 1.0 + 1e-12):
            raise ValueError("product-Beta positions must lie in [0, 1]^L")
        if np.any(b <= 0.0) or np.any(b >= 1.0):
            raise ValueError("product-Beta resources must lie in the open unit cube")
        return x, b

    def _parameters(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return 1.0 + x / self.sigma, 1.0 + (1.0 - x) / self.sigma

    def log_influence(self, positions: np.ndarray, resources: np.ndarray) -> np.ndarray:
        """Evaluate normalized product-Beta log influence.

        :param positions: Agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Open-cube resources, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Log influences, shape ``(N, M)``.
        :rtype: numpy.ndarray
        """
        _digamma, gammaln, _polygamma = _special()
        x, b = self._inputs(positions, resources)
        alpha, beta = self._parameters(x)
        log_norm = gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
        return (
            log_norm.sum(axis=1)[:, None]
            + np.einsum("nl,ml->nm", alpha - 1.0, np.log(b))
            + np.einsum("nl,ml->nm", beta - 1.0, np.log1p(-b))
        )

    def score(self, positions: np.ndarray, resources: np.ndarray) -> np.ndarray:
        """Evaluate the product-Beta position score.

        :param positions: Agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Open-cube resources, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Position scores, shape ``(N, M, L)``.
        :rtype: numpy.ndarray
        """
        digamma, _gammaln, _polygamma = _special()
        x, b = self._inputs(positions, resources)
        alpha, beta = self._parameters(x)
        return (
            np.log(b)[None, :, :]
            - np.log1p(-b)[None, :, :]
            - digamma(alpha)[:, None, :]
            + digamma(beta)[:, None, :]
        ) / self.sigma

    def log_hessian(self, positions: np.ndarray, resources: np.ndarray) -> np.ndarray:
        """Evaluate the diagonal product-Beta log Hessian.

        :param positions: Agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Open-cube resources, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Log Hessians, shape ``(N, M, L, L)``.
        :rtype: numpy.ndarray
        """
        _digamma, _gammaln, polygamma = _special()
        x, b = self._inputs(positions, resources)
        alpha, beta = self._parameters(x)
        diagonal_values = -(polygamma(1, alpha) + polygamma(1, beta)) / self.sigma**2
        h = np.zeros((len(x), self.dimension, self.dimension), dtype=float)
        diagonal = np.arange(self.dimension)
        h[:, diagonal, diagonal] = diagonal_values
        return np.broadcast_to(h[:, None, :, :], (len(x), len(b), self.dimension, self.dimension)).copy()

    def with_sigma(self, sigma: float) -> "ProductBetaKernel":
        """Return a product-Beta kernel with another reach.

        :param sigma: Positive inverse concentration.
        :type sigma: float
        :returns: Reparameterized product-Beta kernel.
        :rtype: ProductBetaKernel
        """
        return ProductBetaKernel(self.dimension, sigma)

    def to_config(self) -> dict[str, Any]:
        """Return serializable product-Beta metadata.

        :returns: Resolved kernel fields.
        :rtype: dict[str, Any]
        """
        return {"kind": self.kind, "sigma": self.sigma, "parameterization": "mode"}
