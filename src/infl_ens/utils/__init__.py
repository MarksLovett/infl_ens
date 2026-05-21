"""Cross-cutting helpers for the influencer-game pipeline.

Per AGENTS.md §4 rule 4, this subpackage does *not* import any sibling
subpackages — utilities here are pure helpers usable from anywhere in
the codebase. At present the public surface is:

- :func:`weighted_mean` and :func:`weighted_covariance`: moments of an
  empirical resource distribution on a discrete trait-space grid.
- :func:`gaussian_stability_threshold`: closed-form first-bifurcation
  threshold

  .. math::

     \\sigma_0^* \\;=\\; \\sqrt{\\frac{N - 2}{N - 1}}\\, \\sigma_B

  (Corollary 8, Lovett & Fu 2024). Picking the competitive reach
  :math:`\\sigma` above or below :math:`\\sigma_0^*` decides whether the
  symmetric Nash equilibrium is locally stable.
"""

from __future__ import annotations

from infl_ens.utils.resource import (
    gaussian_stability_threshold,
    weighted_covariance,
    weighted_mean,
)

__all__ = [
    "gaussian_stability_threshold",
    "weighted_covariance",
    "weighted_mean",
]
