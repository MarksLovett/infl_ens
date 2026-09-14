"""Kernel-agnostic allocation and position dynamics for the influencer game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from infl_ens.data.trait_space import TraitSpace
from infl_ens.inflgame.kernels import InfluenceKernel


def kernel_allocation_weights(
    positions: np.ndarray,
    resources: np.ndarray,
    kernel: InfluenceKernel,
) -> np.ndarray:
    """Compute proportional allocations for an arbitrary influence kernel.

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param resources: Resource coordinates, shape ``(M, L)``.
    :type resources: numpy.ndarray
    :param kernel: Configured influence kernel.
    :type kernel: InfluenceKernel
    :returns: Allocation matrix, shape ``(N, M)``, with unit column sums.
    :rtype: numpy.ndarray
    """
    log_f = np.asarray(kernel.log_influence(positions, resources), dtype=float)
    if log_f.ndim != 2 or not np.isfinite(log_f).all():
        raise ValueError("kernel.log_influence must return a finite 2-D matrix")
    shifted = log_f - np.max(log_f, axis=0, keepdims=True)
    influence = np.exp(shifted)
    totals = influence.sum(axis=0, keepdims=True)
    if np.any(totals <= 0.0) or not np.isfinite(totals).all():
        raise FloatingPointError("kernel allocation normalization failed")
    return influence / totals


def kernel_expected_utilities(
    positions: np.ndarray,
    resources: np.ndarray,
    weights: np.ndarray,
    kernel: InfluenceKernel,
) -> np.ndarray:
    """Evaluate expected utilities under a discrete resource measure.

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param resources: Resource support, shape ``(M, L)``.
    :type resources: numpy.ndarray
    :param weights: Nonnegative resource masses, shape ``(M,)``.
    :type weights: numpy.ndarray
    :param kernel: Configured influence kernel.
    :type kernel: InfluenceKernel
    :returns: Per-agent expected utility.
    :rtype: numpy.ndarray
    """
    w = _normalized_weights(weights, len(resources))
    return kernel_allocation_weights(positions, resources, kernel) @ w


def game_utility_gradient(
    positions: np.ndarray,
    resources: np.ndarray,
    kernel: InfluenceKernel,
    *,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute each player's own exact utility gradient.

    Implements

    .. math::

        \\nabla_{x_i}u_i = \\sum_m w_m G_i(b_m)(1-G_i(b_m))
        \\nabla_{x_i}\\log f_\\kappa(x_i,b_m).

    Passing no weights gives the empirical distribution on the observed
    batch.  This is the online estimator used by ``position_update:
    game_gradient``; offline theory passes ``TraitSpace.weights`` instead.

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param resources: Resource coordinates, shape ``(M, L)``.
    :type resources: numpy.ndarray
    :param kernel: Configured influence kernel.
    :type kernel: InfluenceKernel
    :param weights: Optional resource masses. Uniform when omitted.
    :type weights: numpy.ndarray | None
    :returns: Utility gradients, shape ``(N, L)``.
    :rtype: numpy.ndarray
    :raises ValueError: If the resource set is empty or weights are invalid.
    """
    b = np.asarray(resources, dtype=float)
    if b.ndim != 2 or len(b) == 0:
        raise ValueError("resources must be a non-empty 2-D matrix")
    w = (
        np.full(len(b), 1.0 / len(b), dtype=float)
        if weights is None
        else _normalized_weights(weights, len(b))
    )
    allocations = kernel_allocation_weights(positions, b, kernel)
    score = np.asarray(kernel.score(positions, b), dtype=float)
    expected_shape = (len(positions), len(b), kernel.dimension)
    if score.shape != expected_shape:
        raise ValueError(
            f"kernel.score must return shape {expected_shape}, got {score.shape}"
        )
    mass = w[None, :] * allocations * (1.0 - allocations)
    return np.einsum("nm,nml->nl", mass, score)


@dataclass(frozen=True)
class GameGradientStep:
    """Diagnostics from one simultaneous projected game-gradient step.

    :param positions: Projected positions after the step.
    :type positions: numpy.ndarray
    :param gradient: Raw kernel game gradient.
    :type gradient: numpy.ndarray
    :param projected_gradient: Domain-tangent gradient.
    :type projected_gradient: numpy.ndarray
    :param effective_learning_rate: One scalar applied to every agent.
    :type effective_learning_rate: float
    :param max_gradient_norm: Maximum projected per-agent gradient norm.
    :type max_gradient_norm: float
    :param tangent_residual: Maximum removed normal-component norm.
    :type tangent_residual: float
    """

    positions: np.ndarray
    gradient: np.ndarray
    projected_gradient: np.ndarray
    effective_learning_rate: float
    max_gradient_norm: float
    tangent_residual: float


def projected_game_gradient_step(
    positions: np.ndarray,
    resources: np.ndarray,
    kernel: InfluenceKernel,
    trait_space: TraitSpace,
    *,
    weights: Optional[np.ndarray] = None,
    learning_rate: float = 1.0,
    max_step_norm: Optional[float] = 0.05,
) -> GameGradientStep:
    """Apply one simultaneous, domain-projected game-gradient step.

    A single effective learning rate is shared by the joint population. This
    retains the joint vector-field direction while bounding the largest
    per-agent step across kernels with different score scales.

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param resources: Observed or offline resource support, shape ``(M, L)``.
    :type resources: numpy.ndarray
    :param kernel: Configured influence kernel.
    :type kernel: InfluenceKernel
    :param trait_space: Domain and projection owner.
    :type trait_space: TraitSpace
    :param weights: Optional resource masses; omitted for an empirical batch.
    :type weights: numpy.ndarray | None
    :param learning_rate: Positive base learning rate.
    :type learning_rate: float
    :param max_step_norm: Optional positive maximum per-agent L2 step.
    :type max_step_norm: float | None
    :returns: Updated positions and step diagnostics.
    :rtype: GameGradientStep
    :raises ValueError: If step controls or dimensions are invalid.
    """
    eta = float(learning_rate)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("position_learning_rate must be positive and finite")
    if kernel.dimension != trait_space.L:
        raise ValueError("kernel and trait-space dimensions differ")
    kernel.validate_domain(trait_space.coordinate_domain)
    gradient = game_utility_gradient(positions, resources, kernel, weights=weights)
    projected = trait_space.project_tangent(gradient)
    normal = gradient - projected
    tangent_residual = float(np.max(np.linalg.norm(normal, axis=1)))
    norms = np.linalg.norm(projected, axis=1)
    max_norm = float(np.max(norms)) if len(norms) else 0.0
    eta_effective = eta
    if max_step_norm is not None:
        cap = float(max_step_norm)
        if not np.isfinite(cap) or cap <= 0.0:
            raise ValueError("position_max_step_norm must be positive and finite")
        if max_norm > 0.0:
            eta_effective = min(eta, cap / max_norm)
    candidate = np.asarray(positions, dtype=float) + eta_effective * projected
    updated = trait_space.project_positions(candidate)
    return GameGradientStep(
        positions=updated,
        gradient=gradient,
        projected_gradient=projected,
        effective_learning_rate=float(eta_effective),
        max_gradient_norm=max_norm,
        tangent_residual=tangent_residual,
    )


def _normalized_weights(weights: np.ndarray, size: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.shape != (size,):
        raise ValueError(f"weights must have shape ({size},), got {w.shape}")
    if not np.isfinite(w).all() or np.any(w < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    return w / total
