:mod:`infl_ens.inflgame`
========================

Game environment: routing agents, allocation math
(:math:`G_i,\, u_i,\, \nabla_{x_i} u_i,\, p_i^{\mathrm{strat}}`), and
the public :class:`~infl_ens.inflgame.router.core.InfluencerRouter`
class.

Top-level re-exports
--------------------

The following symbols are intended to be available directly off
``infl_ens.inflgame.router``
(see ``src/infl_ens/inflgame/router/__init__.py``). Each link below
jumps to the **canonical** definition:

- :class:`~infl_ens.inflgame.router.agents.RouterAgent`
- :class:`~infl_ens.inflgame.router.core.InfluencerRouter`
- :func:`~infl_ens.inflgame.router.allocation.allocation_weights`
- :func:`~infl_ens.inflgame.router.allocation.expected_utilities`
- :func:`~infl_ens.inflgame.router.allocation.empirical_utility`
- :func:`~infl_ens.inflgame.router.allocation.strategic_routing_weights`
- :func:`~infl_ens.inflgame.router.allocation.utility_gradient`

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.inflgame.router.agents
   infl_ens.inflgame.router.allocation
   infl_ens.inflgame.router.core
