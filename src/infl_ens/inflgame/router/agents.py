"""Router-agent data structures.

A :class:`RouterAgent` is one participant in the influencer game played by
the router: it bundles a strategic position :math:`x_i \\in \\mathbb{B}` with
an identifier and an optional invocation callable used at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class RouterAgent:
    """A single router agent (e.g. an SLM endpoint).

    :param name: Human-readable identifier (e.g. ``"qwen2.5-3b-coder"``).
    :type name: str
    :param position: Strategic position :math:`x_i \\in [0, 1]^L`,
        shape ``(L,)``. Determines the agent's specialty in trait space.
    :type position: numpy.ndarray
    :param call: Optional callable ``query -> response`` invoked by
        :meth:`InfluencerRouter.dispatch`. Not used by the equilibrium
        analysis or position training; only needed at inference.
    :type call: Callable[[str], str] | None
    :param metadata: Arbitrary attached metadata (cost, latency, benchmarks).
    :type metadata: dict
    :raises ValueError: If ``position`` is not 1-D.
    """

    name: str
    position: np.ndarray
    call: Optional[Callable[[str], str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        if self.position.ndim != 1:
            raise ValueError(
                f"position must be 1-D, got shape {self.position.shape}"
            )

    @classmethod
    def from_calibration(
        cls,
        name: str,
        calibration_queries: list[str],
        project: Callable[[list[str]], np.ndarray],
        **kwargs: Any,
    ) -> "RouterAgent":
        """Initialise an agent's position from a calibration query set.

        The agent's position is taken as the centroid in trait space of
        queries it is known to handle well. This sets ``x_i`` to the
        agent's "centre of mass" in trait space before any game-theoretic
        refinement.

        :param name: Agent identifier.
        :type name: str
        :param calibration_queries: Queries the agent handles well.
        :type calibration_queries: list[str]
        :param project: Trait-space projector from
            :class:`infl_ens.data.trait_space.TraitSpace`.
        :type project: Callable[[list[str]], numpy.ndarray]
        :param kwargs: Forwarded to the :class:`RouterAgent` constructor.
        :returns: New agent with position set to the centroid.
        :rtype: RouterAgent
        :raises ValueError: If ``calibration_queries`` is empty.
        """
        if not calibration_queries:
            raise ValueError("calibration_queries must be non-empty")
        coords = project(list(calibration_queries))
        position = coords.mean(axis=0)
        return cls(name=name, position=position, **kwargs)

    def update_position_from_corpus(
        self,
        queries: list[str],
        project: Callable[[list[str]], np.ndarray],
        *,
        scores: Optional[list[float]] = None,
        blend: float = 1.0,
        position_step: Optional[dict[str, Any]] = None,
    ) -> float:
        """Re-estimate this agent's position from a fresh corpus.

        Used after each external training round (or each closed-loop step)
        to refresh the trait-space position with current capability.

        :param queries: Queries to embed (e.g. the agent's last training
            batch, or an eval set).
        :type queries: list[str]
        :param project: Trait-space projector from
            :class:`infl_ens.data.trait_space.TraitSpace`.
        :type project: Callable[[list[str]], numpy.ndarray]
        :param scores: Optional per-query non-negative weights.
        :type scores: list[float] | None
        :param blend: Linear blend ceiling in ``[0, 1]``. With
            ``position_step.mode: static``, this is the EMA coefficient
            :math:`\\beta` in
            :math:`x \\leftarrow (1 - \\beta) x + \\beta \\hat x`.
        :type blend: float
        :param position_step: Optional adaptive step policy from
            ``closed_loop.position_step`` (see
            :func:`infl_ens.utils.position_step.apply_position_update`).
        :type position_step: dict | None
        :returns: Effective blend coefficient used for this update.
        :rtype: float
        :raises ValueError: If ``blend`` is outside ``[0, 1]``.
        """
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"blend must be in [0, 1], got {blend}")
        # Local import to avoid a circular dep through inflgame.router.__init__.
        from infl_ens.data.trait_space import position_from_corpus
        from infl_ens.utils.position_step import apply_position_update

        new_pos = position_from_corpus(queries, project, scores=scores)
        self.position, beta_eff = apply_position_update(
            self.position,
            new_pos,
            blend=blend,
            position_step=position_step,
        )
        return beta_eff
