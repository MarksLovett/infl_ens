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
- :func:`utility_gradient` — :math:`\\nabla_{x_i} u_i(\\mathbf{x})`.
"""

from __future__ import annotations

from infl_ens.inflgame.router.agents import RouterAgent
from infl_ens.inflgame.router.allocation import (
    allocation_weights,
    empirical_utility,
    expected_utilities,
    strategic_routing_weights,
    utility_gradient,
)
from infl_ens.inflgame.router.core import InfluencerRouter

__all__ = [
    "InfluencerRouter",
    "RouterAgent",
    "allocation_weights",
    "empirical_utility",
    "expected_utilities",
    "strategic_routing_weights",
    "utility_gradient",
]
