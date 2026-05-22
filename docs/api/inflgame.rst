:mod:`infl_ens.inflgame`
========================

Game environment: routing agents, allocation math
(:math:`G_i,\, u_i,\, \nabla_{x_i} u_i,\, p_i^{\mathrm{strat}}`), and
the public :class:`~infl_ens.inflgame.router.InfluencerRouter` class.

Top-level re-exports
--------------------

The following symbols are available directly off
``infl_ens.inflgame.router``
(see ``src/infl_ens/inflgame/router/__init__.py``):

.. autosummary::
   :nosignatures:

   infl_ens.inflgame.router.RouterAgent
   infl_ens.inflgame.router.InfluencerRouter
   infl_ens.inflgame.router.allocation_weights
   infl_ens.inflgame.router.expected_utilities
   infl_ens.inflgame.router.empirical_utility
   infl_ens.inflgame.router.strategic_routing_weights
   infl_ens.inflgame.router.utility_gradient

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.inflgame.router.agents
   infl_ens.inflgame.router.allocation
   infl_ens.inflgame.router.core
