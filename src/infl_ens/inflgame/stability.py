"""Kernel-aware symmetric equilibria and first-bifurcation thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from infl_ens.data.trait_space import TraitSpace, softmax_to_simplex
from infl_ens.inflgame.kernels import InfluenceKernel, build_kernel


@dataclass(frozen=True)
class StabilityResult:
    """Resolved first-bifurcation threshold and numerical diagnostics.

    :param sigma_star: Critical reach below which the symmetric equilibrium
        is unstable.
    :type sigma_star: float
    :param equilibrium: Symmetric equilibrium at the refined threshold.
    :type equilibrium: numpy.ndarray
    :param equilibrium_residual: Tangent first-order residual norm.
    :type equilibrium_residual: float
    :param eigenvalue_residual: Largest antisymmetric eigenvalue at the root.
    :type eigenvalue_residual: float
    :param method: Solver identifier.
    :type method: str
    """

    sigma_star: float
    equilibrium: np.ndarray
    equilibrium_residual: float
    eigenvalue_residual: float
    method: str


def tangent_basis(dimension: int) -> np.ndarray:
    """Return an orthonormal basis for the simplex tangent space.

    :param dimension: Simplex ambient dimension.
    :type dimension: int
    :returns: Matrix ``Q`` of shape ``(dimension, dimension - 1)``.
    :rtype: numpy.ndarray
    """
    if dimension < 2:
        raise ValueError("simplex dimension must be at least two")
    spanning = np.vstack([np.eye(dimension - 1), -np.ones((1, dimension - 1))])
    return np.linalg.qr(spanning, mode="reduced")[0]


def solve_symmetric_equilibrium(
    kernel: InfluenceKernel,
    space: TraitSpace,
    *,
    initial: Optional[np.ndarray] = None,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """Solve the symmetric first-order condition against offline KDE mass.

    At a symmetric population allocation is ``1/N``, so the stationary
    condition reduces to the domain-tangent expectation of the kernel score.

    :param kernel: Kernel at the candidate reach.
    :type kernel: InfluenceKernel
    :param space: Offline resource support and coordinate domain.
    :type space: TraitSpace
    :param initial: Optional warm-start position.
    :type initial: numpy.ndarray | None
    :param tolerance: Least-squares termination tolerance.
    :type tolerance: float
    :returns: Equilibrium position and tangent residual norm.
    :rtype: tuple[numpy.ndarray, float]
    :raises RuntimeError: If SciPy is unavailable or the solve fails.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - environment-level
        raise RuntimeError("kernel stability matching requires scipy") from exc

    weights = np.asarray(space.weights, dtype=float)
    weights = weights / weights.sum()
    start = np.asarray(space.mean if initial is None else initial, dtype=float)
    if space.coordinate_domain == "simplex":
        q = tangent_basis(space.L)
        clipped = np.maximum(space.project_positions(start), 1e-12)
        clipped /= clipped.sum()
        z0 = np.log(clipped[:-1]) - np.log(clipped[-1])

        def unpack(z: np.ndarray) -> np.ndarray:
            return softmax_to_simplex(np.concatenate([z, np.zeros(1)]))

        def residual(z: np.ndarray) -> np.ndarray:
            x = unpack(z)
            score = kernel.score(x[None, :], space.grid)[0]
            return q.T @ np.einsum("m,ml->l", weights, score)

        solved = least_squares(
            residual,
            z0,
            xtol=tolerance,
            ftol=tolerance,
            gtol=tolerance,
            max_nfev=1000,
        )
        position = unpack(solved.x)
        norm = float(np.linalg.norm(residual(solved.x)))
    else:
        def residual_box(x: np.ndarray) -> np.ndarray:
            score = kernel.score(x[None, :], space.grid)[0]
            return np.einsum("m,ml->l", weights, score)

        solved = least_squares(
            residual_box,
            np.clip(start, 1e-9, 1.0 - 1e-9),
            bounds=(0.0, 1.0),
            xtol=tolerance,
            ftol=tolerance,
            gtol=tolerance,
            max_nfev=1000,
        )
        position = solved.x
        norm = float(np.linalg.norm(residual_box(solved.x)))
    final_score = kernel.score(position[None, :], space.grid)[0]
    score_scale = float(
        np.sqrt(np.einsum("m,ml,ml->", weights, final_score, final_score))
    )
    relative_norm = norm / max(1.0, score_scale)
    if not solved.success or (
        norm > max(1e-6, 100.0 * tolerance) and relative_norm > 1e-7
    ):
        raise RuntimeError(
            f"symmetric equilibrium solve failed: {solved.message}; residual={norm:.3e}, "
            f"relative={relative_norm:.3e}"
        )
    return np.asarray(position, dtype=float), norm


def antisymmetric_stability_matrix(
    kernel: InfluenceKernel,
    space: TraitSpace,
    n_agents: int,
    equilibrium: np.ndarray,
) -> np.ndarray:
    """Evaluate the antisymmetric linearization at a symmetric equilibrium.

    :param kernel: Configured influence kernel.
    :type kernel: InfluenceKernel
    :param space: Offline resource measure.
    :type space: TraitSpace
    :param n_agents: Number of competing agents, at least two.
    :type n_agents: int
    :param equilibrium: Symmetric equilibrium position.
    :type equilibrium: numpy.ndarray
    :returns: Symmetric stability matrix in active domain coordinates.
    :rtype: numpy.ndarray
    """
    if n_agents < 2:
        raise ValueError("n_agents must be at least two")
    weights = np.asarray(space.weights, dtype=float)
    weights = weights / weights.sum()
    x = np.asarray(equilibrium, dtype=float)[None, :]
    score = kernel.score(x, space.grid)[0]
    hessian = kernel.log_hessian(x, space.grid)[0]
    e_h = np.einsum("m,mij->ij", weights, hessian)
    e_ss = np.einsum("m,mi,mj->ij", weights, score, score)
    matrix = ((n_agents - 1) * e_h + (n_agents - 2) * e_ss) / n_agents**2
    matrix = 0.5 * (matrix + matrix.T)
    if space.coordinate_domain == "simplex":
        q = tangent_basis(space.L)
        matrix = q.T @ matrix @ q
    return 0.5 * (matrix + matrix.T)


def hyperbolic_stability_threshold(
    kernel_config: Mapping[str, Any],
    space: TraitSpace,
    n_agents: int,
) -> StabilityResult:
    """Evaluate the hyperbolic threshold by generalized eigenvalue.

    The symmetric hyperbolic equilibrium and its unscaled score geometry are
    independent of reach. Consequently the first bifurcation is the largest
    generalized eigenvalue of the score-scatter and log-curvature matrices,
    restricted to the simplex tangent when necessary.

    :param kernel_config: Hyperbolic kernel configuration without ``sigma``.
    :type kernel_config: Mapping[str, Any]
    :param space: Offline KDE resource distribution and coordinate domain.
    :type space: TraitSpace
    :param n_agents: Population size, at least two.
    :type n_agents: int
    :returns: Analytic threshold and equilibrium diagnostics.
    :rtype: StabilityResult
    :raises ValueError: If the kernel is not hyperbolic or the population is
        too small.
    :raises RuntimeError: If the active curvature is not positive definite.
    """
    if n_agents < 2:
        raise ValueError("n_agents must be at least two")
    kernel = build_kernel(kernel_config, sigma=1.0, dimension=space.L)
    if kernel.kind != "hyperbolic":
        raise ValueError("hyperbolic_stability_threshold requires a hyperbolic kernel")
    kernel.validate_domain(space.coordinate_domain)
    equilibrium, residual = solve_symmetric_equilibrium(kernel, space)
    weights = np.asarray(space.weights, dtype=float)
    weights /= weights.sum()
    x = equilibrium[None, :]
    score = kernel.score(x, space.grid)[0]
    curvature = -np.einsum(
        "m,mij->ij", weights, kernel.log_hessian(x, space.grid)[0]
    )
    scatter = np.einsum("m,mi,mj->ij", weights, score, score)
    if space.coordinate_domain == "simplex":
        q = tangent_basis(space.L)
        curvature = q.T @ curvature @ q
        scatter = q.T @ scatter @ q
    curvature = 0.5 * (curvature + curvature.T)
    scatter = 0.5 * (scatter + scatter.T)
    try:
        chol = np.linalg.cholesky(curvature)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("hyperbolic active curvature is not positive definite") from exc
    left = np.linalg.solve(chol, scatter)
    transformed = np.linalg.solve(chol, left.T).T
    transformed = 0.5 * (transformed + transformed.T)
    ratio = float(np.linalg.eigvalsh(transformed)[-1])
    sigma_star = max(0.0, (n_agents - 2) / (n_agents - 1) * ratio)
    if sigma_star > 0.0:
        at_root = kernel.with_sigma(sigma_star)
        eigen_residual = float(
            np.linalg.eigvalsh(
                antisymmetric_stability_matrix(
                    at_root, space, n_agents, equilibrium
                )
            )[-1]
        )
    else:
        eigen_residual = 0.0
    return StabilityResult(
        sigma_star=sigma_star,
        equilibrium=equilibrium,
        equilibrium_residual=float(residual),
        eigenvalue_residual=eigen_residual,
        method="analytic_hyperbolic_generalized_eigen",
    )


def numerical_stability_threshold(
    kernel_config: Mapping[str, Any],
    space: TraitSpace,
    n_agents: int,
    *,
    sigma_min: float = 1e-6,
    sigma_max: float = 1e3,
    scan_points: int = 161,
) -> StabilityResult:
    """Numerically find a unique positive-to-negative stability crossing.

    :param kernel_config: Explicit kernel configuration without ``sigma``.
    :type kernel_config: Mapping[str, Any]
    :param space: Offline KDE resource distribution and domain.
    :type space: TraitSpace
    :param n_agents: Population size.
    :type n_agents: int
    :param sigma_min: Smallest scanned reach.
    :type sigma_min: float
    :param sigma_max: Largest scanned reach.
    :type sigma_max: float
    :param scan_points: Number of logarithmically spaced probes.
    :type scan_points: int
    :returns: Refined threshold and diagnostics.
    :rtype: StabilityResult
    :raises RuntimeError: If no unique first-bifurcation crossing is resolved.
    """
    try:
        from scipy.optimize import brentq
    except ImportError as exc:  # pragma: no cover - environment-level
        raise RuntimeError("kernel stability matching requires scipy") from exc

    if not 0.0 < sigma_min < sigma_max or scan_points < 3:
        raise ValueError("invalid numerical sigma scan")
    sigmas = np.geomspace(sigma_min, sigma_max, int(scan_points))
    values: list[float] = []
    equilibria: list[np.ndarray] = []
    residuals: list[float] = []
    warm: Optional[np.ndarray] = None
    bracket: tuple[int, int] | None = None
    for sigma in sigmas:
        kernel = build_kernel(kernel_config, sigma=float(sigma), dimension=space.L)
        kernel.validate_domain(space.coordinate_domain)
        try:
            equilibrium, residual = solve_symmetric_equilibrium(
                kernel, space, initial=warm
            )
        except RuntimeError:
            if bracket is not None:
                break
            raise
        matrix = antisymmetric_stability_matrix(kernel, space, n_agents, equilibrium)
        values.append(float(np.linalg.eigvalsh(matrix)[-1]))
        equilibria.append(equilibrium)
        residuals.append(residual)
        warm = equilibrium
        if len(values) >= 2 and values[-2] > 0.0 and values[-1] <= 0.0:
            if bracket is not None:
                raise RuntimeError("multiple positive-to-negative stability crossings found")
            bracket = (len(values) - 2, len(values) - 1)
        if bracket is not None and len(values) - 1 >= bracket[1] + 3:
            break

    if bracket is None:
        raise RuntimeError(
            "stability_fraction requires one positive-to-negative crossing; "
            f"found none over [{sigma_min:g}, {sigma_max:g}]"
        )
    lo_idx, hi_idx = bracket
    warm_root = equilibria[hi_idx]
    root_position = warm_root
    root_equilibrium_residual = residuals[hi_idx]

    def eigenvalue(sigma: float) -> float:
        nonlocal root_position, root_equilibrium_residual
        kernel = build_kernel(kernel_config, sigma=float(sigma), dimension=space.L)
        root_position, root_equilibrium_residual = solve_symmetric_equilibrium(
            kernel, space, initial=root_position
        )
        matrix = antisymmetric_stability_matrix(
            kernel, space, n_agents, root_position
        )
        return float(np.linalg.eigvalsh(matrix)[-1])

    sigma_star = float(
        brentq(
            eigenvalue,
            float(sigmas[lo_idx]),
            float(sigmas[hi_idx]),
            xtol=1e-10,
            rtol=1e-10,
            maxiter=100,
        )
    )
    eigen_residual = eigenvalue(sigma_star)
    return StabilityResult(
        sigma_star=sigma_star,
        equilibrium=root_position.copy(),
        equilibrium_residual=float(root_equilibrium_residual),
        eigenvalue_residual=float(eigen_residual),
        method="numerical_log_scan_brent",
    )
