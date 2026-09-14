"""Train router-agent positions via gradient ascent on influencer-game utility.

This trainer specializes the canonical adaptive-dynamics loop for the router
use case. It consumes a :class:`TraitSpace` and a list of
:class:`RouterAgent`, then runs simultaneous gradient ascent. Explicit
kernels use their exact score-function game gradient and the configured
domain projection; configurations without a kernel retain the historical
Gaussian update

.. math::

    x_i^{(t+1)} \\;=\\; \\mathrm{Proj}_{[0,1]^L}\\!\\Big(
        x_i^{(t)} + \\eta\\, \\nabla_{x_i} u_i(\\mathbf{x}^{(t)})
    \\Big)

until convergence and writes the equilibrium positions back into each agent
in place.

Per AGENTS.md §4 rule 1, this module is not a standalone CLI: training is
launched via ``python -m infl_ens.training`` with a config under
``configs/arms/``. Game reward and gradient math remain owned by
:mod:`infl_ens.inflgame`.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from infl_ens.data.trait_space import TraitSpace
from infl_ens.inflgame.dynamics import (
    kernel_allocation_weights,
    projected_game_gradient_step,
)
from infl_ens.inflgame.kernels import InfluenceKernel
from infl_ens.inflgame.router.agents import RouterAgent
from infl_ens.inflgame.router.allocation import (
    allocation_weights,
    utility_gradient,
)


@dataclass
class RouterTrainingConfig:
    """Hyperparameters for router-position gradient ascent.

    :param sigma: Competitive reach. Scalar (isotropic) or shape ``(L,)``
        (axis-aligned anisotropic). Held fixed during training.
    :type sigma: float | numpy.ndarray
    :param learning_rate: Step size :math:`\\eta` for gradient ascent.
    :type learning_rate: float
    :param n_steps: Maximum number of gradient steps.
    :type n_steps: int
    :param tol: Convergence tolerance on the max coordinate change per step.
    :type tol: float
    :param clip_to_box: Whether to project positions back to
        :math:`[0, 1]^L` after each step.
    :type clip_to_box: bool
    :param kernel: Explicit kernel. ``None`` selects the legacy Gaussian
        implementation.
    :type kernel: InfluenceKernel | None
    :param max_step_norm: Optional population-wide cap on the largest
        explicit-kernel per-agent L2 step.
    :type max_step_norm: float | None
    """

    sigma: Union[float, np.ndarray]
    learning_rate: float = 5e-3
    n_steps: int = 5000
    tol: float = 1e-8
    clip_to_box: bool = True
    kernel: InfluenceKernel | None = None
    max_step_norm: float | None = None


def train_router_positions(
    trait_space: TraitSpace,
    agents: list[RouterAgent],
    cfg: RouterTrainingConfig,
    *,
    seed: int = 0,
) -> dict:
    """Train agent positions to a (local) Nash equilibrium via gradient ascent.

    Agent positions are updated in place. The function additionally returns
    a trajectory dictionary suitable for plotting in :mod:`infl_ens.figures`.

    :param trait_space: Discretised trait space and resource distribution.
    :type trait_space: TraitSpace
    :param agents: Agents to optimise in place. Their starting positions
        are used as the initial condition of the dynamics.
    :type agents: list[RouterAgent]
    :param cfg: Training configuration.
    :type cfg: RouterTrainingConfig
    :param seed: RNG seed forwarded to :func:`infl_ens.utils.seeding.seed_all`.
    :type seed: int
    :returns: Dictionary with keys

        - ``positions``: ``(T+1, N, L)`` position trajectory,
        - ``utilities``: ``(T, N)`` utility trajectory,
        - ``converged``: ``bool``,
        - ``n_steps``: number of steps actually taken.
    :rtype: dict
    """
    # Avoid hard import-time dep so the module is importable without
    # the full utils stack present in editable installs.
    try:
        from infl_ens.utils.seeding import seed_all
        seed_all(seed)
    except ImportError:
        np.random.seed(seed)

    L = trait_space.L
    cov: np.ndarray | None = None
    if cfg.kernel is None:
        sigma_arr = np.atleast_1d(np.asarray(cfg.sigma, dtype=float))
        if sigma_arr.size == 1:
            cov = float(sigma_arr.item()) ** 2 * np.eye(L)
        elif sigma_arr.shape == (L,):
            cov = np.diag(sigma_arr ** 2)
        else:
            raise ValueError(
                f"sigma must be scalar or shape ({L},), got {sigma_arr.shape}"
            )
    else:
        cfg.kernel.validate_domain(trait_space.coordinate_domain)

    pos = np.stack([a.position for a in agents], axis=0).copy()
    grid = trait_space.grid
    weights = trait_space.weights

    traj_pos = [pos.copy()]
    traj_u: list[np.ndarray] = []
    converged = False
    t = 0

    for t in range(1, cfg.n_steps + 1):
        if cfg.kernel is None:
            assert cov is not None
            grad = utility_gradient(pos, grid, weights, cov)
            new_pos = pos + cfg.learning_rate * grad
            if cfg.clip_to_box:
                new_pos = np.clip(new_pos, 0.0, 1.0)
        else:
            result = projected_game_gradient_step(
                pos,
                grid,
                cfg.kernel,
                trait_space,
                weights=weights,
                learning_rate=cfg.learning_rate,
                max_step_norm=cfg.max_step_norm,
            )
            new_pos = result.positions

        step = float(np.max(np.abs(new_pos - pos)))
        pos = new_pos
        G = (
            kernel_allocation_weights(pos, grid, cfg.kernel)
            if cfg.kernel is not None
            else allocation_weights(pos, grid, cov)
        )
        traj_pos.append(pos.copy())
        traj_u.append(G @ weights)

        if step < cfg.tol:
            converged = True
            break

    for a, p in zip(agents, pos):
        a.position = p

    return {
        "positions": np.stack(traj_pos, axis=0),
        "utilities": (
            np.stack(traj_u, axis=0) if traj_u else np.zeros((0, len(agents)))
        ),
        "converged": converged,
        "n_steps": t,
    }
