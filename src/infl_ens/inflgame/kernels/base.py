"""Abstract interface for influence kernels used by the router game."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np


class InfluenceKernel(ABC):
    """Shared interface for a differentiable multivariate influence kernel.

    Kernel methods are vectorised over ``N`` agent positions and ``M``
    resources.  Scores and Hessians are derivatives with respect to the
    corresponding agent position, never with respect to the resource.
    Domain constraints are applied by the game dynamics rather than by the
    kernel itself.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        """Stable configuration name for the kernel family.

        :returns: Kernel-family name.
        :rtype: str
        """

    dimension: int
    sigma: float

    @abstractmethod
    def log_influence(
        self,
        positions: np.ndarray,
        resources: np.ndarray,
    ) -> np.ndarray:
        """Evaluate log influence for every agent-resource pair.

        :param positions: Agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Resource coordinates, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Log influences, shape ``(N, M)``.
        :rtype: numpy.ndarray
        """

    @abstractmethod
    def score(
        self,
        positions: np.ndarray,
        resources: np.ndarray,
    ) -> np.ndarray:
        """Evaluate :math:`\\nabla_x\\log f(x,b)`.

        :param positions: Agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Resource coordinates, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Scores, shape ``(N, M, L)``.
        :rtype: numpy.ndarray
        """

    @abstractmethod
    def log_hessian(
        self,
        positions: np.ndarray,
        resources: np.ndarray,
    ) -> np.ndarray:
        """Evaluate :math:`\\nabla_x^2\\log f(x,b)`.

        :param positions: Agent positions, shape ``(N, L)``.
        :type positions: numpy.ndarray
        :param resources: Resource coordinates, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Hessians, shape ``(N, M, L, L)``.
        :rtype: numpy.ndarray
        """

    @abstractmethod
    def with_sigma(self, sigma: float) -> "InfluenceKernel":
        """Return the same kernel family with another reach.

        :param sigma: Positive competitive reach.
        :type sigma: float
        :returns: Reparameterized immutable kernel.
        :rtype: InfluenceKernel
        """

    @abstractmethod
    def to_config(self) -> dict[str, Any]:
        """Return serializable resolved kernel metadata.

        :returns: Configuration mapping.
        :rtype: dict[str, Any]
        """

    def validate_domain(self, domain: str) -> None:
        """Validate a trait-space domain for this kernel.

        :param domain: ``"box"`` or ``"simplex"``.
        :type domain: str
        :returns: ``None``.
        :rtype: None
        :raises ValueError: If the domain is unsupported.
        """
        if domain not in ("box", "simplex"):
            raise ValueError(f"coordinate domain must be 'box' or 'simplex', got {domain!r}")


def validate_kernel_inputs(
    positions: np.ndarray,
    resources: np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize and validate vectorized kernel inputs.

    :param positions: Candidate position matrix.
    :type positions: numpy.ndarray
    :param resources: Candidate resource matrix.
    :type resources: numpy.ndarray
    :param dimension: Required trailing dimension.
    :type dimension: int
    :returns: Float position and resource arrays.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    :raises ValueError: If shapes or values are invalid.
    """
    x = np.asarray(positions, dtype=float)
    b = np.asarray(resources, dtype=float)
    if x.ndim != 2 or x.shape[1] != dimension:
        raise ValueError(f"positions must have shape (N, {dimension}), got {x.shape}")
    if b.ndim != 2 or b.shape[1] != dimension:
        raise ValueError(f"resources must have shape (M, {dimension}), got {b.shape}")
    if not np.isfinite(x).all() or not np.isfinite(b).all():
        raise ValueError("positions and resources must be finite")
    return x, b


def shape_matrix_from_config(
    value: Any,
    dimension: int,
) -> np.ndarray:
    """Resolve an identity or explicit SPD kernel-shape matrix.

    :param value: ``None``, ``"identity"``, or an ``L`` by ``L`` sequence.
    :type value: Any
    :param dimension: Required matrix dimension.
    :type dimension: int
    :returns: Symmetric positive-definite matrix.
    :rtype: numpy.ndarray
    :raises ValueError: If the value is not an SPD matrix.
    """
    if value is None or (isinstance(value, str) and value == "identity"):
        return np.eye(dimension, dtype=float)
    if isinstance(value, str):
        raise ValueError("kernel.shape string value must be 'identity'")
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"kernel.shape must be 'identity' or shape ({dimension}, {dimension}), "
            f"got {matrix.shape}"
        )
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-10):
        raise ValueError("kernel.shape must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("kernel.shape must be positive definite") from exc
    return matrix


def build_kernel(
    config: Mapping[str, Any] | None,
    *,
    sigma: float,
    dimension: int,
) -> InfluenceKernel:
    """Build an influence kernel from a resolved configuration block.

    A missing block resolves to a Gaussian kernel for API convenience. The
    training driver separately preserves its untouched legacy Gaussian path
    when the block is absent.

    :param config: Optional top-level ``kernel`` configuration.
    :type config: Mapping[str, Any] | None
    :param sigma: Positive competitive reach.
    :type sigma: float
    :param dimension: Trait-space dimensionality.
    :type dimension: int
    :returns: Configured kernel.
    :rtype: InfluenceKernel
    :raises ValueError: If the family or its parameters are invalid.
    """
    from infl_ens.inflgame.kernels.families import (
        DirichletKernel,
        GaussianKernel,
        HyperbolicKernel,
        ProductBetaKernel,
    )

    cfg = dict(config or {})
    kind = str(cfg.get("kind", "gaussian")).lower().replace("-", "_")
    if kind == "gaussian":
        return GaussianKernel(
            dimension=dimension,
            sigma=float(sigma),
            shape=shape_matrix_from_config(cfg.get("shape"), dimension),
        )
    if kind == "hyperbolic":
        return HyperbolicKernel(
            dimension=dimension,
            sigma=float(sigma),
            shape=shape_matrix_from_config(cfg.get("shape"), dimension),
            delta=float(cfg.get("delta", 0.1)),
        )
    if kind == "dirichlet":
        return DirichletKernel(dimension=dimension, sigma=float(sigma))
    if kind in ("product_beta", "beta"):
        return ProductBetaKernel(dimension=dimension, sigma=float(sigma))
    raise ValueError(
        "kernel.kind must be gaussian, hyperbolic, dirichlet, or product_beta, "
        f"got {kind!r}"
    )
