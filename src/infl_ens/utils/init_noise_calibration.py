"""Calibrate ``closed_loop.init_noise`` for mean-noise agent initialization."""

from __future__ import annotations

from math import gamma, sqrt

import numpy as np

from infl_ens.utils.agent_init import init_agents_mean_noise


def expected_pairwise_two_agent(sigma: float, n_dims: int) -> float:
    """Expected L2 distance between two i.i.d. mean-noise agents.

    For positions :math:`\\mu + \\sigma\\epsilon_i` with
    :math:`\\epsilon_i \\sim \\mathcal{N}(0, I_L)`, the difference has law
    :math:`\\mathcal{N}(0, 2\\sigma^2 I_L)` and

    .. math::

       \\mathbb{E}\\|\\Delta\\| = \\sigma\\sqrt{2}\\,
       \\Gamma\\!\\left(\\frac{L+1}{2}\\right) / \\Gamma\\!\\left(\\frac{L}{2}\\right).

    :param sigma: ``init_noise`` scale.
    :type sigma: float
    :param n_dims: Trait-space dimension ``L``.
    :type n_dims: int
    :returns: Expected pairwise L2 for one pair.
    :rtype: float
    """
    if sigma <= 0.0 or n_dims < 1:
        raise ValueError("sigma must be positive and n_dims >= 1")
    return sigma * sqrt(2.0) * gamma((n_dims + 1) / 2.0) / gamma(n_dims / 2.0)


def mean_pairwise_spread(
    sigma: float,
    *,
    n_agents: int,
    n_dims: int,
    seed: int = 0,
    n_draws: int = 200,
) -> float:
    """Monte Carlo mean pairwise L2 over ``n_agents`` mean-noise inits.

    :param sigma: ``init_noise`` scale.
    :type sigma: float
    :param n_agents: Number of router agents.
    :type n_agents: int
    :param n_dims: Trait-space dimension.
    :type n_dims: int
    :param seed: Base RNG seed for draws.
    :type seed: int
    :param n_draws: Number of independent init draws.
    :type n_draws: int
    :returns: Mean of per-draw mean pairwise L2.
    :rtype: float
    """
    cfg = {"agents": [{"name": f"clone-{i}"} for i in range(n_agents)]}
    resource_mean = np.zeros(n_dims, dtype=float)

    class _Space:
        L = n_dims
        mean = resource_mean

    space = _Space()
    vals: list[float] = []
    for draw in range(n_draws):
        agents = init_agents_mean_noise(
            cfg, space, seed=seed + draw, init_noise=sigma,
        )
        pos = np.stack([a.position for a in agents], axis=0)
        dists = [
            float(np.linalg.norm(pos[i] - pos[j]))
            for i in range(n_agents)
            for j in range(i + 1, n_agents)
        ]
        vals.append(float(np.mean(dists)))
    return float(np.mean(vals))


def solve_init_noise(
    target_mean_pairwise: float,
    *,
    n_agents: int = 10,
    n_dims: int = 5,
    tol: float = 0.01,
    max_iter: int = 48,
) -> float:
    """Solve ``init_noise`` so mean pairwise L2 matches ``target_mean_pairwise``.

    Uses binary search on empirical mean pairwise spread (``n_agents`` i.i.d.
  Gaussian perturbations). The two-agent closed form is a lower bound;
  multi-agent mean pairwise is higher because each pair draws independent noise.

    :param target_mean_pairwise: Desired mean pairwise L2 at initialization.
    :type target_mean_pairwise: float
    :param n_agents: Number of agents in the ensemble.
    :type n_agents: int
    :param n_dims: Trait-space dimension.
    :type n_dims: int
    :param tol: Acceptable absolute error on mean pairwise.
    :type tol: float
    :param max_iter: Binary-search iterations.
    :type max_iter: int
    :returns: Calibrated ``init_noise``.
    :rtype: float
    """
    if target_mean_pairwise <= 0.0:
        raise ValueError("target_mean_pairwise must be positive")
    lo, hi = 1e-6, 10.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        realized = mean_pairwise_spread(
            mid, n_agents=n_agents, n_dims=n_dims, n_draws=120,
        )
        if abs(realized - target_mean_pairwise) <= tol:
            return mid
        if realized < target_mean_pairwise:
            lo = mid
        else:
            hi = mid
    return mid
