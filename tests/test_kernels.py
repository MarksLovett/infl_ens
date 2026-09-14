"""Kernel-family, simplex-domain, and numerical-stability tests."""

from __future__ import annotations

import numpy as np
import pytest

from infl_ens.data.trait_space import (
    TraitSpace,
    positive_simplex_grid,
    softmax_to_simplex,
)
from infl_ens.inflgame.dynamics import kernel_allocation_weights
from infl_ens.inflgame.kernels import (
    DirichletKernel,
    GaussianKernel,
    HyperbolicKernel,
    ProductBetaKernel,
    build_kernel,
)
from infl_ens.inflgame.router.allocation import allocation_weights
from infl_ens.inflgame.stability import (
    hyperbolic_stability_threshold,
    numerical_stability_threshold,
)


def _positions_and_resources() -> tuple[np.ndarray, np.ndarray]:
    positions = np.array([[0.2, 0.3, 0.5], [0.4, 0.2, 0.4]])
    resources = np.array([[0.1, 0.3, 0.6], [0.2, 0.5, 0.3]])
    return positions, resources


@pytest.mark.parametrize(
    "kernel",
    [
        GaussianKernel(3, 0.3, np.eye(3)),
        HyperbolicKernel(3, 0.3, np.eye(3), 0.1),
        DirichletKernel(3, 0.3),
        ProductBetaKernel(3, 0.3),
    ],
)
def test_kernel_score_matches_tangent_finite_difference(kernel: object) -> None:
    positions, resources = _positions_and_resources()
    direction = np.array([0.3, -0.2, -0.1])
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    plus = positions.copy()
    minus = positions.copy()
    plus[0] += eps * direction
    minus[0] -= eps * direction
    numeric = (
        kernel.log_influence(plus, resources)[0]
        - kernel.log_influence(minus, resources)[0]
    ) / (2.0 * eps)
    analytic = np.einsum("ml,l->m", kernel.score(positions, resources)[0], direction)
    assert np.allclose(analytic, numeric, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize(
    "kernel",
    [
        GaussianKernel(3, 0.3, np.eye(3)),
        HyperbolicKernel(3, 0.3, np.eye(3), 0.1),
        DirichletKernel(3, 0.3),
        ProductBetaKernel(3, 0.3),
    ],
)
def test_kernel_hessian_matches_score_direction(kernel: object) -> None:
    positions, resources = _positions_and_resources()
    direction = np.array([0.2, -0.3, 0.1])
    eps = 1e-6
    plus = positions.copy()
    minus = positions.copy()
    plus[0] += eps * direction
    minus[0] -= eps * direction
    numeric = (
        kernel.score(plus, resources)[0]
        - kernel.score(minus, resources)[0]
    ) / (2.0 * eps)
    analytic = np.einsum(
        "mij,j->mi", kernel.log_hessian(positions, resources)[0], direction
    )
    assert np.allclose(analytic, numeric, atol=3e-6, rtol=3e-5)


def test_mode_parameterized_dirichlet_has_position_as_mode() -> None:
    position = np.array([[0.2, 0.3, 0.5]])
    resources = np.array(
        [position[0], [0.1, 0.4, 0.5], [0.25, 0.25, 0.5]], dtype=float
    )
    kernel = DirichletKernel(3, 0.2)
    log_density = kernel.log_influence(position, resources)[0]
    assert int(np.argmax(log_density)) == 0
    assert kernel.to_config()["parameterization"] == "mode"


def test_kernel_allocations_and_legacy_gaussian_agree() -> None:
    positions, resources = _positions_and_resources()
    sigma = 0.3
    kernel = GaussianKernel(3, sigma, np.eye(3))
    actual = kernel_allocation_weights(positions, resources, kernel)
    legacy = allocation_weights(positions, resources, sigma**2 * np.eye(3))
    assert np.allclose(actual, legacy, atol=1e-14, rtol=1e-14)
    assert np.allclose(actual.sum(axis=0), 1.0)


def test_simplex_grid_and_softmax_geometry() -> None:
    grid = positive_simplex_grid(7, 14)
    assert grid.shape == (1716, 7)
    assert np.all(grid > 0.0)
    assert np.allclose(grid.sum(axis=1), 1.0)
    projected = softmax_to_simplex(np.array([[0.0, 0.5, 1.0]]))
    assert np.all(projected > 0.0)
    assert np.allclose(projected.sum(axis=1), 1.0)


def test_dirichlet_factory_rejects_box_domain() -> None:
    kernel = build_kernel({"kind": "dirichlet"}, sigma=0.2, dimension=3)
    with pytest.raises(ValueError, match="simplex"):
        kernel.validate_domain("box")


def test_numerical_dirichlet_stability_crossing() -> None:
    grid = positive_simplex_grid(3, 6)
    weights = np.exp(-10.0 * np.sum((grid - np.array([0.2, 0.3, 0.5])) ** 2, axis=1))
    weights /= weights.sum()
    space = TraitSpace(
        grid=grid,
        weights=weights,
        project=lambda queries: np.tile(np.array([0.2, 0.3, 0.5]), (len(queries), 1)),
        coordinate_domain="simplex",
        simplex_resolution=6,
    )
    result = numerical_stability_threshold(
        {"kind": "dirichlet"}, space, 6, scan_points=61
    )
    assert 0.01 < result.sigma_star < 1.0
    assert result.equilibrium_residual < 1e-6
    assert abs(result.eigenvalue_residual) < 1e-7


def test_hyperbolic_analytic_threshold_matches_numerical_root() -> None:
    grid = positive_simplex_grid(3, 6)
    weights = np.exp(
        -10.0 * np.sum((grid - np.array([0.2, 0.3, 0.5])) ** 2, axis=1)
    )
    weights /= weights.sum()
    space = TraitSpace(
        grid=grid,
        weights=weights,
        project=lambda queries: np.tile(
            np.array([0.2, 0.3, 0.5]), (len(queries), 1)
        ),
        coordinate_domain="simplex",
        simplex_resolution=6,
    )
    config = {"kind": "hyperbolic", "shape": "identity", "delta": 0.1}
    analytic = hyperbolic_stability_threshold(config, space, 6)
    numeric = numerical_stability_threshold(config, space, 6, scan_points=61)
    assert analytic.method == "analytic_hyperbolic_generalized_eigen"
    assert np.isclose(analytic.sigma_star, numeric.sigma_star, rtol=2e-7)
    assert abs(analytic.eigenvalue_residual) < 1e-7
