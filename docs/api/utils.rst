:mod:`infl_ens.utils`
=====================

Cross-cutting helpers — weighted moments of the resource distribution
and the closed-form stability threshold

.. math::

   \sigma_0^* \;=\; \sqrt{\frac{N - 2}{N - 1}}\, \sigma_B

(Corollary 8, Lovett & Fu 2024). Per AGENTS.md §4 rule 4, ``utils/``
does not import sibling subpackages.

Top-level re-exports
--------------------

Each link below jumps to the **canonical** definition:

- :func:`~infl_ens.utils.resource.weighted_mean`
- :func:`~infl_ens.utils.resource.weighted_covariance`
- :func:`~infl_ens.utils.resource.gaussian_stability_threshold`

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.utils.resource
