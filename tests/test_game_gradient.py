"""Exact online game-gradient and legacy-compatibility tests."""

from __future__ import annotations

import numpy as np
import pytest

from infl_ens.data.trait_space import TraitSpace, positive_simplex_grid
from infl_ens.inflgame.dynamics import (
    game_utility_gradient,
    kernel_expected_utilities,
    projected_game_gradient_step,
)
from infl_ens.inflgame.kernels import (
    DirichletKernel,
    GaussianKernel,
    HyperbolicKernel,
    ProductBetaKernel,
)
from infl_ens.inflgame.router import RouterAgent
from infl_ens.training.router_training import RouterTrainingConfig, train_router_positions


def _space() -> TraitSpace:
    grid = positive_simplex_grid(3, 6)
    weights = np.linspace(1.0, 3.0, len(grid))
    weights /= weights.sum()
    return TraitSpace(
        grid=grid,
        weights=weights,
        project=lambda queries: np.tile(np.array([0.2, 0.3, 0.5]), (len(queries), 1)),
        coordinate_domain="simplex",
        simplex_resolution=6,
    )


@pytest.mark.parametrize(
    "kernel",
    [
        GaussianKernel(3, 0.3, np.eye(3)),
        HyperbolicKernel(3, 0.3, np.eye(3), 0.1),
        DirichletKernel(3, 0.3),
        ProductBetaKernel(3, 0.3),
    ],
)
def test_game_gradient_matches_own_utility_finite_difference(kernel: object) -> None:
    space = _space()
    positions = np.array([[0.2, 0.3, 0.5], [0.4, 0.2, 0.4]])
    direction = np.array([0.2, -0.1, -0.1])
    direction /= np.linalg.norm(direction)
    gradient = game_utility_gradient(
        positions, space.grid, kernel, weights=space.weights
    )
    eps = 1e-6
    plus = positions.copy()
    minus = positions.copy()
    plus[0] += eps * direction
    minus[0] -= eps * direction
    numeric = (
        kernel_expected_utilities(plus, space.grid, space.weights, kernel)[0]
        - kernel_expected_utilities(minus, space.grid, space.weights, kernel)[0]
    ) / (2.0 * eps)
    assert np.isclose(gradient[0] @ direction, numeric, atol=3e-6, rtol=3e-5)


def test_observed_batch_and_offline_kde_are_intentionally_distinct() -> None:
    space = _space()
    positions = np.array([[0.2, 0.3, 0.5], [0.4, 0.2, 0.4]])
    kernel = DirichletKernel(3, 0.3)
    offline = game_utility_gradient(
        positions, space.grid, kernel, weights=space.weights
    )
    observed = game_utility_gradient(positions, space.grid[:3], kernel)
    assert not np.allclose(offline, observed)


def test_population_wide_cap_preserves_one_step_scalar() -> None:
    space = _space()
    positions = np.array([[0.2, 0.3, 0.5], [0.4, 0.2, 0.4]])
    kernel = DirichletKernel(3, 0.05)
    result = projected_game_gradient_step(
        positions,
        space.grid[:4],
        kernel,
        space,
        learning_rate=10.0,
        max_step_norm=0.01,
    )
    assert result.effective_learning_rate < 10.0
    intended = result.effective_learning_rate * result.projected_gradient
    assert np.max(np.linalg.norm(intended, axis=1)) <= 0.0100000001
    assert np.all(result.positions >= 0.0)
    assert np.allclose(result.positions.sum(axis=1), 1.0)


def test_kernel_aware_theory_trainer_uses_offline_weights() -> None:
    space = _space()
    start = np.array([[0.2, 0.3, 0.5], [0.4, 0.2, 0.4]])
    agents = [
        RouterAgent("a", start[0].copy()),
        RouterAgent("b", start[1].copy()),
    ]
    kernel = DirichletKernel(3, 0.3)
    expected = projected_game_gradient_step(
        start,
        space.grid,
        kernel,
        space,
        weights=space.weights,
        learning_rate=0.1,
        max_step_norm=None,
    ).positions
    train_router_positions(
        space,
        agents,
        RouterTrainingConfig(
            sigma=0.3,
            learning_rate=0.1,
            n_steps=1,
            kernel=kernel,
            max_step_norm=None,
        ),
    )
    assert np.allclose(np.stack([agent.position for agent in agents]), expected)
