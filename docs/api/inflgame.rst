:mod:`infl_ens.inflgame`
========================

Game environment: influence kernels, routing agents, allocation math
(:math:`G_i,\, u_i,\, \nabla_{x_i} u_i,\, p_i^{\mathrm{strat}}`), projected
game dynamics, and numerical stability matching.

Top-level re-exports
--------------------

The following core symbols are available directly from
``infl_ens.inflgame`` or ``infl_ens.inflgame.router``. Each link jumps to
the **canonical** definition:

- :class:`~infl_ens.inflgame.router.agents.RouterAgent`
- :class:`~infl_ens.inflgame.router.core.InfluencerRouter`
- :func:`~infl_ens.inflgame.router.allocation.allocation_weights`
- :func:`~infl_ens.inflgame.router.allocation.expected_utilities`
- :func:`~infl_ens.inflgame.router.allocation.empirical_utility`
- :func:`~infl_ens.inflgame.router.allocation.strategic_routing_weights`
- :func:`~infl_ens.inflgame.router.allocation.utility_gradient`
- :class:`~infl_ens.inflgame.kernels.base.InfluenceKernel`
- :class:`~infl_ens.inflgame.kernels.families.GaussianKernel`
- :class:`~infl_ens.inflgame.kernels.families.HyperbolicKernel`
- :class:`~infl_ens.inflgame.kernels.families.DirichletKernel`
- :class:`~infl_ens.inflgame.kernels.families.ProductBetaKernel`
- :func:`~infl_ens.inflgame.dynamics.game_utility_gradient`
- :func:`~infl_ens.inflgame.dynamics.projected_game_gradient_step`
- :func:`~infl_ens.inflgame.stability.hyperbolic_stability_threshold`
- :func:`~infl_ens.inflgame.stability.numerical_stability_threshold`

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   infl_ens.inflgame.router.agents
   infl_ens.inflgame.router.allocation
   infl_ens.inflgame.router.core
   infl_ens.inflgame.kernels.base
   infl_ens.inflgame.kernels.families
   infl_ens.inflgame.dynamics
   infl_ens.inflgame.stability
