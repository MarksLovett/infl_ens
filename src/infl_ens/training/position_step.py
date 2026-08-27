"""Closed-loop position-step helpers (schedules and pool centroids).

Adaptive EMA blend toward a corpus centroid lives in
:mod:`infl_ens.data.position_blend` so :mod:`infl_ens.inflgame.router`
does not import this utils module.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from infl_ens.data.position_blend import (
    VALID_BLEND_MODES,
    apply_position_update,
    effective_blend,
    parse_position_step,
)

VALID_CENTROID_MODES = frozenset({"batch", "expected_pool"})

__all__ = [
    "VALID_BLEND_MODES",
    "VALID_CENTROID_MODES",
    "apply_position_update",
    "blend_for_round",
    "effective_blend",
    "expected_pool_centroid",
    "parse_position_step",
]


def expected_pool_centroid(
    agent_idx: int,
    coords: np.ndarray,
    G: np.ndarray,
) -> np.ndarray:
    """Deterministic pool centroid for canonical routing + ``(1-G)`` weights.

    In the ``M \\to \\infty`` limit with routing :math:`p_i \\propto G_i`,
    the weighted centroid for agent ``i`` converges to

    .. math::

        \\hat x_i =
        \\frac{\\sum_m G_{im}(1-G_{im})\\, b_m}
             {\\sum_m G_{im}(1-G_{im})}.

    :param agent_idx: Agent index ``i``.
    :type agent_idx: int
    :param coords: Projected prompts ``(M, L)``.
    :type coords: numpy.ndarray
    :param G: Allocation weights ``(N, M)``.
    :type G: numpy.ndarray
    :returns: Target position ``(L,)``.
    :rtype: numpy.ndarray
    """
    g = G[agent_idx]
    w = g * (1.0 - g)
    total = float(w.sum())
    if total <= 1e-30:
        return np.asarray(coords, dtype=float).mean(axis=0)
    return (w[:, None] * coords).sum(axis=0) / total


def blend_for_round(
    round_idx: int,
    n_rounds: int,
    blend_base: float,
    schedule: Optional[str],
    *,
    blend_start: Optional[float] = None,
) -> float:
    """Resolve EMA blend for a closed-loop round.

    :param round_idx: Current round ``0 … n_rounds-1``.
    :type round_idx: int
    :param n_rounds: Total rounds.
    :type n_rounds: int
    :param blend_base: Terminal / static blend.
    :type blend_base: float
    :param schedule: ``None`` or ``'linear'`` (ramp from ``blend_start``).
    :type schedule: str | None
    :param blend_start: Starting blend when ``schedule='linear'``.
    :type blend_start: float | None
    :returns: Blend in ``[0, 1]``.
    :rtype: float
    """
    if schedule is None or schedule == "none":
        return float(blend_base)
    if schedule == "linear":
        start = float(blend_start if blend_start is not None else blend_base * 0.25)
        if n_rounds <= 1:
            return float(blend_base)
        t = round_idx / max(n_rounds - 1, 1)
        return float(start + t * (blend_base - start))
    raise ValueError(f"unknown blend_schedule {schedule!r}")
