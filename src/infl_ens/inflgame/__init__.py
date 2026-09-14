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

from infl_ens.inflgame import kernels, router
from infl_ens.inflgame.dynamics import (
    GameGradientStep,
    game_utility_gradient,
    kernel_allocation_weights,
    kernel_expected_utilities,
    projected_game_gradient_step,
)
from infl_ens.inflgame.kernels import (
    DirichletKernel,
    GaussianKernel,
    HyperbolicKernel,
    InfluenceKernel,
    ProductBetaKernel,
    build_kernel,
)
from infl_ens.inflgame.stability import (
    StabilityResult,
    antisymmetric_stability_matrix,
    hyperbolic_stability_threshold,
    numerical_stability_threshold,
    solve_symmetric_equilibrium,
)

__all__ = [
    "DirichletKernel",
    "GameGradientStep",
    "GaussianKernel",
    "HyperbolicKernel",
    "InfluenceKernel",
    "ProductBetaKernel",
    "StabilityResult",
    "antisymmetric_stability_matrix",
    "build_kernel",
    "game_utility_gradient",
    "hyperbolic_stability_threshold",
    "kernel_allocation_weights",
    "kernel_expected_utilities",
    "kernels",
    "numerical_stability_threshold",
    "projected_game_gradient_step",
    "router",
    "solve_symmetric_equilibrium",
]
