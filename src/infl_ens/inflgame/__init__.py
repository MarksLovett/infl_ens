"""Influencer's-game environment: kernels, dynamics, and equilibrium math.

This subpackage hosts everything that defines the *game* rather than
the *learners*:

- :func:`~infl_ens.inflgame.router.allocation_weights`     —
  :math:`G_i(\\mathbf{x}, b)`, the proportional-allocation rule.
- :func:`~infl_ens.inflgame.router.expected_utilities`     —
  :math:`u_i(\\mathbf{x}) = \\int G_i(\\mathbf{x}, b)\\, B(b)\\, db`.
- :func:`~infl_ens.inflgame.router.utility_gradient`       —
  :math:`\\nabla_{x_i} u_i(\\mathbf{x})`.
- :class:`~infl_ens.inflgame.router.InfluencerRouter`      —
  public router class wrapping the math for trainers.

Per AGENTS.md §4 rule 2, *the environment owns the reward*: trainers
consume this subpackage and do not reimplement payoff math.
"""

from __future__ import annotations

from infl_ens.inflgame import router

__all__ = ["router"]
