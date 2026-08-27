"""Matched pool-centroid dynamics for theory comparison (fix 3/4).

Runs the same expected-pool position update rule as ``centroid_mode:
expected_pool`` in the closed loop, starting from a given initial state.
Also wraps grid gradient-ascent for side-by-side diagnostics.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from infl_ens.inflgame.router import RouterAgent
from infl_ens.data.trait_space import TraitSpace
from infl_ens.training.router_training import (
    RouterTrainingConfig,
    train_router_positions,
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
    """Classify an even-agent harm-axis layout or collapse.

    Four-agent layouts retain the historical ``'2,2'`` label. Larger even
    layouts are labelled ``'<N/2>x2'`` when adjacent harm-sorted pairs are
    separated enough to be interpreted as paired corners.

    :param pos: ``(N, L)`` positions.
    :type pos: numpy.ndarray
    :param spread_thresh: Collapse threshold on pairwise spread.
    :type spread_thresh: float
    :returns: ``'2,2'``, ``'<N/2>x2'``, or ``'collapsed'``.
    :rtype: str
    """
    if pairwise_spread(pos) < spread_thresh:
        return "collapsed"
    n = pos.shape[0]
    if n < 2 or n % 2 != 0:
        return "collapsed"
    harm = pos[:, 0]
    order = np.argsort(harm)
    pair_means = np.array([
        float(np.mean(harm[order[i : i + 2]]))
        for i in range(0, n, 2)
    ])
    if len(pair_means) == 1:
        return "1x2"
    min_sep = float(np.min(np.diff(pair_means)))
    if min_sep < 0.15:
        return "collapsed"
    return "2,2" if n == 4 else f"{n // 2}x2"


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


def agent_pairwise_geometry(
    positions: np.ndarray,
    names: Sequence[str],
    *,
    merge_groups: Optional[Sequence[tuple[str, Sequence[str]]]] = None,
) -> dict[str, Any]:
    """Summarize inter-agent L2 geometry for attribution diagnostics.

    :param positions: Agent positions ``(N, L)``.
    :type positions: numpy.ndarray
    :param names: Agent names aligned with rows of ``positions``.
    :type names: Sequence[str]
    :param merge_groups: Optional ``(train_as, members)`` SFT merge pairs.
    :type merge_groups: Sequence[tuple[str, Sequence[str]]] | None
    :returns: Pairwise and within-merge distance summaries.
    :rtype: dict
    """
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    if n != len(names):
        raise ValueError("positions rows must match names length")
    pairwise: dict[str, float] = {}
    off_diag: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            pairwise[f"{names[i]}|{names[j]}"] = dist
            off_diag.append(dist)
    within_merge: dict[str, float] = {}
    if merge_groups:
        by_name = {name: idx for idx, name in enumerate(names)}
        for train_as, members in merge_groups:
            members = list(members)
            if len(members) != 2:
                continue
            i, j = by_name[members[0]], by_name[members[1]]
            within_merge[train_as] = float(np.linalg.norm(pos[i] - pos[j]))
    return {
        "pairwise_l2": pairwise,
        "within_merge_l2": within_merge,
        "mean_pairwise_l2": float(np.mean(off_diag)) if off_diag else 0.0,
        "min_pairwise_l2": float(np.min(off_diag)) if off_diag else 0.0,
        "max_pairwise_l2": float(np.max(off_diag)) if off_diag else 0.0,
    }
