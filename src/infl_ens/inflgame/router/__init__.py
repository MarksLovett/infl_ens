"""Public surface of the router: :class:`InfluencerRouter` and helpers.

Top-level exports:

- :class:`InfluencerRouter` — wraps a trait space and a list of agents,
  provides ``route(query)`` and ``route_batch(queries)`` for passive
  routing and exposes the allocation math for closed-loop trainers.
- :class:`RouterAgent` — dataclass for a single candidate model, with
  :py:meth:`RouterAgent.from_calibration` for centroid-based init and
  :py:meth:`RouterAgent.update_position_from_corpus` for post-training
  position refresh.
- :func:`allocation_weights` — :math:`G_i(\\mathbf{x}, b)`.
- :func:`expected_utilities` — :math:`u_i` on a discrete trait space.
- :func:`empirical_utility` — :math:`\\hat u_i` from a finite query batch.
- :func:`strategic_routing_weights` — :math:`p_i^{\\mathrm{strat}}`,
  the strategic-gradient routing weight :math:`G_i (1 - G_i)`.
- :func:`top_k_allocation_weights` — top-:math:`k` sparsified, per-query
  renormalised allocation weights for soft (dense) routing.
- :func:`sampled_top_k_mask` — draw :math:`k` distinct agents per query
  without replacement from :math:`G` (Gumbel-top-:math:`k`), the stochastic
  counterpart of the deterministic top-:math:`k` gate.
- :func:`matched_centroid_mass` — gradient-matched centroid mass
  :math:`G_i (1 - G_i)`, dense over every query, for the theory-matched
  position update under soft routing.
- :func:`group_allocation_weights` — per-group allocation
  :math:`G_p = \\sum_{i \\in p} G_i`, for soft routing over merge groups.
- :func:`utility_gradient` — :math:`\\nabla_{x_i} u_i(\\mathbf{x})`.
"""

from __future__ import annotations

from infl_ens.inflgame.router.agents import RouterAgent
from infl_ens.inflgame.router.allocation import (
    allocation_weights,
    empirical_utility,
    expected_utilities,
    group_allocation_weights,
    matched_centroid_mass,
    sampled_top_k_mask,
    strategic_routing_weights,
    top_k_allocation_weights,
    utility_gradient,
)
from infl_ens.inflgame.router.core import InfluencerRouter

__all__ = [
    "InfluencerRouter",
    "RouterAgent",
    "allocation_weights",
    "empirical_utility",
    "expected_utilities",
    "group_allocation_weights",
    "strategic_routing_weights",
    "top_k_allocation_weights",
    "matched_centroid_mass",
    "sampled_top_k_mask",
    "utility_gradient",
]
