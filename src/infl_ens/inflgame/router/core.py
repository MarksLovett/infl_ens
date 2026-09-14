"""Main :class:`InfluencerRouter` class.

The router is the consumer-facing object: configure it once with a trait
space, a list of agents, and a competitive reach :math:`\\sigma`, then call
:meth:`route` or :meth:`route_batch` per query.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from infl_ens.data.trait_space import TraitSpace
from infl_ens.inflgame.dynamics import (
    kernel_allocation_weights,
    kernel_expected_utilities,
)
from infl_ens.inflgame.kernels import GaussianKernel, InfluenceKernel
from infl_ens.inflgame.router.agents import RouterAgent
from infl_ens.inflgame.router.allocation import (
    allocation_weights,
    expected_utilities,
    strategic_routing_weights,
)


class InfluencerRouter:
    """Route queries to agents via the influencer-game allocation rule.

    Given ``N`` agents with strategic positions :math:`x_1, \\dots, x_N` in
    a trait space :math:`\\mathbb{B}`, an incoming query is embedded at
    :math:`b^* \\in \\mathbb{B}` and routed according to a policy:

    - ``'argmax'``: deterministic, picks
      :math:`\\arg\\max_i f_i(x_i, b^*)`. Corresponds to the
      :math:`\\sigma \\to 0` winner-takes-all limit of the influencer game.
    - ``'proportional'``: stochastic, samples agent :math:`i` with
      probability :math:`G_i(\\mathbf{x}, b^*)`. Matches the game's payoff
      semantics exactly and provides natural load balancing.

    Agent positions can be set manually, initialised from calibration sets
    (see :meth:`RouterAgent.from_calibration`), or optimised by the trainer
    in :mod:`infl_ens.training.router_training`.

    :param trait_space: Discretised trait space and resource distribution.
    :type trait_space: TraitSpace
    :param agents: Router agents. Each ``position`` must have shape
        ``(trait_space.L,)``.
    :type agents: Sequence[RouterAgent]
    :param sigma: Competitive reach. Scalar (isotropic) or shape ``(L,)``
        (axis-aligned anisotropic). Shared across agents to preserve the
        kernel-symmetry assumption of Theorems 1–5.
    :type sigma: float | numpy.ndarray
    :param policy: ``'argmax'`` or ``'proportional'``.
    :type policy: str
    :param kernel: Explicit influence kernel. ``None`` preserves the legacy
        Gaussian covariance implementation.
    :type kernel: InfluenceKernel | None
    :raises ValueError: If ``policy`` is unrecognised, ``agents`` is empty,
        ``sigma`` has the wrong shape, or any agent's position has the wrong
        dimensionality.
    """

    _VALID_POLICIES = ("argmax", "proportional")

    def __init__(
        self,
        trait_space: TraitSpace,
        agents: Sequence[RouterAgent],
        sigma: Union[float, np.ndarray, None] = None,
        policy: str = "argmax",
        *,
        kernel: Optional[InfluenceKernel] = None,
    ) -> None:
        if policy not in self._VALID_POLICIES:
            raise ValueError(
                f"policy must be one of {self._VALID_POLICIES}, got {policy!r}"
            )
        agents = list(agents)
        if not agents:
            raise ValueError("need at least one agent")
        L = trait_space.L
        for a in agents:
            if a.position.shape != (L,):
                raise ValueError(
                    f"agent {a.name!r} has position shape {a.position.shape},"
                    f" expected ({L},)"
                )

        if kernel is None:
            if sigma is None:
                raise ValueError("sigma is required for the legacy Gaussian router")
            sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=float))
            if sigma_arr.size == 1:
                cov: Optional[np.ndarray] = float(sigma_arr.item()) ** 2 * np.eye(L)
            elif sigma_arr.shape == (L,):
                cov = np.diag(sigma_arr ** 2)
            else:
                raise ValueError(
                    f"sigma must be scalar or shape ({L},), got {sigma_arr.shape}"
                )
        else:
            if kernel.dimension != L:
                raise ValueError(
                    f"kernel dimension {kernel.dimension} does not match trait dimension {L}"
                )
            kernel.validate_domain(trait_space.coordinate_domain)
            cov = kernel.covariance if isinstance(kernel, GaussianKernel) else None

        self.trait_space = trait_space
        self.agents = agents
        self.policy = policy
        self.cov = cov
        self.kernel = kernel

    @property
    def positions(self) -> np.ndarray:
        """Stacked agent positions.

        :returns: Matrix of shape ``(N, L)``.
        :rtype: numpy.ndarray
        """
        return np.stack([a.position for a in self.agents], axis=0)

    @property
    def sigma_scalar(self) -> float:
        """Geometric-mean competitive reach across axes.

        For isotropic ``cov`` this is exactly :math:`\\sigma`; for anisotropic
        diagonal ``cov`` it is the geometric mean of the axis reaches, used
        as a single-number summary in :meth:`is_stable`.

        :returns: Effective scalar :math:`\\sigma`.
        :rtype: float
        """
        if self.kernel is not None:
            return float(self.kernel.sigma)
        assert self.cov is not None
        return float(np.exp(0.5 * np.mean(np.log(np.diag(self.cov)))))

    def allocation_weights(self, resources: np.ndarray) -> np.ndarray:
        """Evaluate allocations at already-projected resource coordinates.

        :param resources: Resource coordinates, shape ``(M, L)``.
        :type resources: numpy.ndarray
        :returns: Clone-level allocation matrix, shape ``(N, M)``.
        :rtype: numpy.ndarray
        """
        if self.kernel is not None:
            return kernel_allocation_weights(self.positions, resources, self.kernel)
        assert self.cov is not None
        return allocation_weights(self.positions, resources, self.cov)

    def expected_utilities(self) -> np.ndarray:
        """Expected utility :math:`u_i(\\mathbf{x})` for each agent.

        :returns: Utility vector, shape ``(N,)``. Sums to one in expectation.
        :rtype: numpy.ndarray
        """
        if self.kernel is not None:
            return kernel_expected_utilities(
                self.positions,
                self.trait_space.grid,
                self.trait_space.weights,
                self.kernel,
            )
        assert self.cov is not None
        return expected_utilities(
            self.positions, self.trait_space.grid, self.trait_space.weights, self.cov
        )

    def is_stable(self) -> bool:
        """Whether the configured :math:`\\sigma` clears the symmetric stability threshold.

        Explicit kernels use their numerical antisymmetric bifurcation root.
        The legacy Gaussian path retains the existing analytic multivariate
        threshold. This local test does not certify any particular asymmetric
        configuration.

        :returns: ``True`` if the configured reach exceeds the threshold.
        :rtype: bool
        """
        if self.kernel is not None:
            from infl_ens.inflgame.stability import numerical_stability_threshold

            config = self.kernel.to_config()
            config.pop("sigma", None)
            config.pop("parameterization", None)
            result = numerical_stability_threshold(
                config, self.trait_space, len(self.agents)
            )
            return self.sigma_scalar > result.sigma_star
        # Local import keeps the legacy implementation decoupled from utils.
        from infl_ens.utils.resource import gaussian_stability_threshold
        thresh = gaussian_stability_threshold(
            len(self.agents),
            self.trait_space.grid,
            self.trait_space.weights,
        )
        return self.sigma_scalar > thresh

    def route(
        self,
        query: str,
        rng: Optional[np.random.Generator] = None,
    ) -> RouterAgent:
        """Route a single query to one agent.

        Ties under the ``'argmax'`` policy are broken uniformly at random
        rather than by index order. This matters at clone-start: when all
        positions are identical, :math:`G_i \\equiv 1/N` and every agent
        ties; deterministic argmax would always pick agent 0 and break the
        intended symmetric initial condition.

        :param query: Raw query text.
        :type query: str
        :param rng: Optional generator for proportional sampling and argmax
            tiebreaking. Defaults to ``numpy.random.default_rng()``;
            call-site is responsible for seeding via
            :func:`infl_ens.utils.seeding.seed_all`.
        :type rng: numpy.random.Generator | None
        :returns: Selected agent.
        :rtype: RouterAgent
        """
        b_star = self.trait_space.project([query])                       # (1, L)
        G_b = self.allocation_weights(b_star)[:, 0] # (N,)
        rng = rng if rng is not None else np.random.default_rng()
        if self.policy == "argmax":
            top = G_b.max()
            ties = np.flatnonzero(G_b >= top - 1e-12)
            idx = int(rng.choice(ties))
        else:
            idx = int(rng.choice(len(self.agents), p=G_b))
        return self.agents[idx]

    def route_batch(
        self,
        queries: Sequence[str],
        rng: Optional[np.random.Generator] = None,
        *,
        routing_weight: str = "G",
    ) -> list[RouterAgent]:
        """Route a batch of queries.

        Ties under the ``'argmax'`` policy are broken uniformly at random
        per query; see :meth:`route` for rationale.

        :param queries: Raw query texts.
        :type queries: Sequence[str]
        :param rng: Optional generator for sampling / tiebreaks.
        :type rng: numpy.random.Generator | None
        :param routing_weight: Per-trait routing-probability weighting.

            - ``'G'`` (default): standard proportional allocation
              :math:`p_i \\propto G_i`. This is the canonical Lovett & Fu
              (2024) allocation rule and what every prior call assumes.
            - ``'G_times_1mG'``: strategic-gradient-matched weighting
              :math:`p_i \\propto G_i(1 - G_i)`. Routes more queries to
              contested traits and fewer to traits where one agent already
              dominates. In expectation over a batch sampled from
              :math:`B`, the resulting position drift matches the
              strategic gradient :math:`\\nabla_{x_i} u_i` (up to
              :math:`\\Sigma^{-1}` and a normaliser). Reduces to ``'G'`` at
              clone-start (where :math:`G_i \\equiv 1/N`).

            Only honoured under ``policy='proportional'``; ``policy='argmax'``
            always uses :math:`G` directly.
        :type routing_weight: str
        :returns: List of selected agents, one per query, in input order.
        :rtype: list[RouterAgent]
        :raises ValueError: If ``routing_weight`` is not a recognised mode.
        """
        if not queries:
            return []
        if routing_weight not in ("G", "G_times_1mG"):
            raise ValueError(
                f"routing_weight must be 'G' or 'G_times_1mG', got {routing_weight!r}"
            )
        b_star = self.trait_space.project(list(queries))                 # (M, L)
        G = self.allocation_weights(b_star)                              # (N, M)
        rng = rng if rng is not None else np.random.default_rng()
        N, M = G.shape
        if self.policy == "argmax":
            idx = np.empty(M, dtype=int)
            for m in range(M):
                col = G[:, m]
                top = col.max()
                ties = np.flatnonzero(col >= top - 1e-12)
                idx[m] = int(rng.choice(ties))
        else:
            if routing_weight == "G_times_1mG":
                if self.kernel is not None:
                    P = G * (1.0 - G)
                    totals = P.sum(axis=0, keepdims=True)
                    P = np.divide(P, totals, out=G.copy(), where=totals > 1e-12)
                else:
                    assert self.cov is not None
                    P = strategic_routing_weights(
                        self.positions, b_star, self.cov,
                    )                                                     # (N, M)
            else:
                P = G
            idx = np.fromiter(
                (rng.choice(N, p=P[:, m]) for m in range(M)),
                dtype=int,
                count=M,
            )
        return [self.agents[int(i)] for i in idx]

    def dispatch(self, query: str, rng: Optional[np.random.Generator] = None) -> str:
        """Route a query and invoke the chosen agent's ``call``.

        :param query: Raw query text.
        :type query: str
        :param rng: Optional RNG.
        :type rng: numpy.random.Generator | None
        :returns: Response from the chosen agent.
        :rtype: str
        :raises RuntimeError: If the chosen agent has no ``call`` callable.
        """
        agent = self.route(query, rng=rng)
        if agent.call is None:
            raise RuntimeError(
                f"agent {agent.name!r} has no `call` callable;"
                " use `route` to select without invoking"
            )
        return agent.call(query)
