"""Matched pool-centroid dynamics for theory comparison (fix 3/4).

Runs the same expected-pool position update rule as ``centroid_mode:
expected_pool`` in the closed loop, starting from a given initial state.
Also wraps grid gradient-ascent for side-by-side diagnostics.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from infl_ens.inflgame.router import InfluencerRouter, RouterAgent
from infl_ens.inflgame.router.allocation import allocation_weights
from infl_ens.data.trait_space import TraitSpace
from infl_ens.training.router_training import (
    RouterTrainingConfig,
    train_router_positions,
)
from infl_ens.utils.position_step import (
    apply_position_update,
    blend_for_round,
    expected_pool_centroid,
)


def mean_agent_gap(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Mean per-agent L2 distance between two position sets.

    :param pos_a: ``(N, L)`` positions.
    :type pos_a: numpy.ndarray
    :param pos_b: ``(N, L)`` positions.
    :type pos_b: numpy.ndarray
    :returns: Mean gap.
    :rtype: float
    """
    return float(np.mean(np.linalg.norm(pos_a - pos_b, axis=1)))


def pairwise_spread(pos: np.ndarray) -> float:
    """Mean pairwise L2 spread among agents.

    :param pos: ``(N, L)`` positions.
    :type pos: numpy.ndarray
    :returns: Mean off-diagonal distance.
    :rtype: float
    """
    n = pos.shape[0]
    if n < 2:
        return 0.0
    return float(np.mean([
        np.linalg.norm(pos[i] - pos[j])
        for i in range(n) for j in range(i + 1, n)
    ]))


def classify_layout(pos: np.ndarray, *, spread_thresh: float = 0.45) -> str:
    """Classify as ``2,2`` or ``collapsed``.

    :param pos: ``(N, 2)`` positions (expects 4 agents).
    :type pos: numpy.ndarray
    :param spread_thresh: Collapse threshold on pairwise spread.
    :type spread_thresh: float
    :returns: ``'2,2'`` or ``'collapsed'``.
    :rtype: str
    """
    if pairwise_spread(pos) < spread_thresh:
        return "collapsed"
    harm = pos[:, 0]
    order = np.argsort(harm)
    low, high = set(order[:2].tolist()), set(order[2:].tolist())
    if len(low) != 2:
        return "collapsed"
    harm_sep = float(np.mean(pos[list(high), 0]) - np.mean(pos[list(low), 0]))
    return "2,2" if harm_sep >= 0.35 else "collapsed"


def run_matched_pool_dynamics(
    space: TraitSpace,
    initial_positions: np.ndarray,
    names: Sequence[str],
    pool_prompts: Sequence[str],
    *,
    sigma: float,
    n_rounds: int,
    blend_base: float = 0.5,
    blend_schedule: Optional[str] = None,
    blend_start: Optional[float] = None,
    position_step: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Integrate expected-pool centroid updates for ``n_rounds``.

    :param space: Trait space.
    :type space: TraitSpace
    :param initial_positions: Round-0 positions ``(N, L)``.
    :type initial_positions: numpy.ndarray
    :param names: Agent names.
    :type names: Sequence[str]
    :param pool_prompts: Full prompt pool for projection.
    :type pool_prompts: Sequence[str]
    :param sigma: Competitive reach.
    :type sigma: float
    :param n_rounds: Number of discrete rounds.
    :type n_rounds: int
    :param blend_base: Terminal / static blend.
    :type blend_base: float
    :param blend_schedule: Optional ``linear`` schedule.
    :type blend_schedule: str | None
    :param blend_start: Start blend for linear schedule.
    :type blend_start: float | None
    :param position_step: Optional adaptive step policy.
    :type position_step: dict | None
    :returns: Dict with ``positions`` ``(n_rounds+1, N, L)``, ``final_spread``,
        ``layout``.
    :rtype: dict
    """
    agents = [
        RouterAgent(name=n, position=initial_positions[i].copy())
        for i, n in enumerate(names)
    ]
    router = InfluencerRouter(space, agents, sigma=sigma, policy="proportional")
    pool_coords = space.project(list(pool_prompts))
    traj = [router.positions.copy()]

    for r in range(n_rounds):
        blend_r = blend_for_round(
            r, n_rounds, blend_base, blend_schedule, blend_start=blend_start,
        )
        G_pool = allocation_weights(router.positions, pool_coords, router.cov)
        for i, agent in enumerate(agents):
            target = expected_pool_centroid(i, pool_coords, G_pool)
            agent.position, _ = apply_position_update(
                agent.position,
                target,
                blend=blend_r,
                position_step=position_step,
            )
        traj.append(router.positions.copy())

    final = traj[-1]
    return {
        "positions": np.stack(traj, axis=0),
        "final_spread": pairwise_spread(final),
        "layout": classify_layout(final),
    }


def run_gradient_ascent_theory(
    space: TraitSpace,
    initial_positions: np.ndarray,
    names: Sequence[str],
    *,
    sigma: float,
    learning_rate: float = 5e-3,
    n_steps: int = 5000,
    tol: float = 1e-8,
    seed: int = 0,
) -> dict[str, Any]:
    """Grid gradient-ascent Nash solver from the same initial state (fix 4).

    :param space: Trait space.
    :type space: TraitSpace
    :param initial_positions: ``(N, L)`` round-0 positions.
    :type initial_positions: numpy.ndarray
    :param names: Agent names.
    :type names: Sequence[str]
    :param sigma: Competitive reach.
    :type sigma: float
    :param learning_rate: Gradient step size.
    :type learning_rate: float
    :param n_steps: Max steps.
    :type n_steps: int
    :param tol: Convergence tolerance.
    :type tol: float
    :param seed: RNG seed.
    :type seed: int
    :returns: Dict with trajectory, convergence, layout.
    :rtype: dict
    """
    agents = [
        RouterAgent(name=n, position=initial_positions[i].copy())
        for i, n in enumerate(names)
    ]
    info = train_router_positions(
        space,
        agents,
        RouterTrainingConfig(
            sigma=sigma,
            learning_rate=learning_rate,
            n_steps=n_steps,
            tol=tol,
            clip_to_box=True,
        ),
        seed=seed,
    )
    traj = np.concatenate(
        [initial_positions[None, ...], info["positions"]], axis=0,
    )
    final = traj[-1]
    return {
        "positions": traj,
        "final_spread": pairwise_spread(final),
        "layout": classify_layout(final),
        "converged": bool(info["converged"]),
        "n_steps": int(info["n_steps"]),
    }
