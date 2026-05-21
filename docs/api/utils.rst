:mod:`infl_ens.utils`
=====================

Cross-cutting helpers — weighted moments of the resource distribution
and the closed-form stability threshold
:math:`\sigma_0^* = \sqrt{(N-2)/(N-1)}\,\sigma_B` (Corollary 8,
Lovett & Fu 2024). Per AGENTS.md §4 rule 4, ``utils/`` does not import
sibling subpackages.

.. currentmodule:: infl_ens.utils

Top-level re-exports
--------------------

.. autosummary::
   :nosignatures:

   weighted_mean
   weighted_covariance
   gaussian_stability_threshold

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.utils.resource
