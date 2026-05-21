:mod:`infl_ens.inflgame`
========================

Game environment: routing agents, allocation math
(:math:`G_i, u_i, \nabla_{x_i} u_i, p_i^{\mathrm{strat}}`), and the
public :class:`~infl_ens.inflgame.router.InfluencerRouter` class.

.. currentmodule:: infl_ens.inflgame.router

Top-level re-exports
--------------------

The following symbols are available directly off
``infl_ens.inflgame.router``
(see ``src/infl_ens/inflgame/router/__init__.py``):

.. autosummary::
   :nosignatures:

   RouterAgent
   InfluencerRouter
   allocation_weights
   expected_utilities
   empirical_utility
   strategic_routing_weights
   utility_gradient

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.inflgame.router.agents
   infl_ens.inflgame.router.allocation
   infl_ens.inflgame.router.core
