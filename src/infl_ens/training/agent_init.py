"""Agent position initialization helpers for closed-loop and simulators."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from infl_ens.data.trait_space import TraitSpace
from infl_ens.inflgame.router import RouterAgent


def harm_pair_indices(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split four positions into low-harm and high-harm pairs.

    :param pos: ``(4, L)`` positions.
    :type pos: numpy.ndarray
    :returns: Index arrays ``(low_pair, high_pair)`` each length 2.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    order = np.argsort(pos[:, 0])
    return order[:2], order[2:]


def adjacent_harm_pair_indices(pos: np.ndarray) -> list[np.ndarray]:
    """Pair an even number of positions by adjacent harm-axis order.

    Positions are sorted by axis ``0`` (harm), then grouped as adjacent
    pairs: lowest two, next two, and so on. This generalizes the original
    four-agent low/high harm pairing to any even number of agents.

    :param pos: ``(N, L)`` positions with even ``N``.
    :type pos: numpy.ndarray
    :returns: List of length ``N/2``; each entry has two row indices.
    :rtype: list[numpy.ndarray]
    :raises ValueError: If ``N`` is odd or smaller than two.
    """
    if pos.ndim != 2:
        raise ValueError(f"positions must be a 2-D array, got shape {pos.shape}")
    n_agents = pos.shape[0]
    if n_agents < 2 or n_agents % 2 != 0:
        raise ValueError(
            f"pairwise theory init requires an even number of agents >= 2, "
            f"got {n_agents}"
        )
    order = np.argsort(pos[:, 0])
    return [order[i : i + 2] for i in range(0, n_agents, 2)]


def nearest_neighbour_pair_indices(pos: np.ndarray) -> list[np.ndarray]:
    """Pair an even number of positions by greedy nearest-neighbour matching.

    All pairwise L2 distances are sorted ascending; each closest still-unused
    pair is taken in turn. Unlike :func:`adjacent_harm_pair_indices` this does
    not assume that partners are adjacent in harm-axis order, so it recovers
    the true pairing when a theory solve leaves clusters whose harm
    coordinates interleave. Pairs are returned ordered by the harm (axis 0)
    coordinate of their centroid so downstream naming stays deterministic.

    :param pos: ``(N, L)`` positions with even ``N``.
    :type pos: numpy.ndarray
    :returns: List of length ``N/2``; each entry holds two row indices.
    :rtype: list[numpy.ndarray]
    :raises ValueError: If ``pos`` is not 2-D, or ``N`` is odd or below two.
    """
    if pos.ndim != 2:
        raise ValueError(f"positions must be a 2-D array, got shape {pos.shape}")
    n_agents = pos.shape[0]
    if n_agents < 2 or n_agents % 2 != 0:
        raise ValueError(
            f"pairwise theory init requires an even number of agents >= 2, "
            f"got {n_agents}"
        )
    dists = [
        (float(np.linalg.norm(pos[i] - pos[j])), i, j)
        for i in range(n_agents)
        for j in range(i + 1, n_agents)
    ]
    dists.sort(key=lambda t: (t[0], t[1], t[2]))
    used: set[int] = set()
    pairs: list[np.ndarray] = []
    for _d, i, j in dists:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        pairs.append(np.array([i, j], dtype=int))
    pairs.sort(key=lambda pr: float(pos[pr].mean(axis=0)[0]))
    return pairs


#: Supported pairing rules for paired theory initialization.
PAIRING_METHODS: tuple[str, ...] = ("harm_adjacent", "nearest")


def pair_indices_for_method(
    pos: np.ndarray,
    pairing: str = "harm_adjacent",
) -> list[np.ndarray]:
    """Dispatch to a pairing rule by name.

    :param pos: ``(N, L)`` positions with even ``N``.
    :type pos: numpy.ndarray
    :param pairing: ``'harm_adjacent'`` (default, historical) or
        ``'nearest'`` (greedy nearest-neighbour matching).
    :type pairing: str
    :returns: List of index pairs.
    :rtype: list[numpy.ndarray]
    :raises ValueError: If ``pairing`` is not a known method.
    """
    if pairing == "harm_adjacent":
        return adjacent_harm_pair_indices(pos)
    if pairing == "nearest":
        return nearest_neighbour_pair_indices(pos)
    raise ValueError(
        f"pairing must be one of {PAIRING_METHODS}, got {pairing!r}"
    )


def trait_box_bounds(space: TraitSpace) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds from the trait grid.

    :param space: Trait space.
    :type space: TraitSpace
    :returns: ``(lo, hi)`` each shape ``(L,)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    lo = np.min(space.grid, axis=0)
    hi = np.max(space.grid, axis=0)
    return lo, hi


def random_separated_initial_positions(
    space: TraitSpace,
    n_agents: int,
    seed: int,
    *,
    min_pairwise: float = 0.2,
    max_tries: int = 2000,
) -> np.ndarray:
    """Sample ``n_agents`` positions in trait space with pairwise separation.

    Draws uniformly in the grid bounding box until every pair is at least
  ``min_pairwise`` apart in L2 (rejection sampling).

    :param space: Trait space.
    :type space: TraitSpace
    :param n_agents: Number of agents.
    :type n_agents: int
    :param seed: RNG seed (per-run reproducibility).
    :type seed: int
    :param min_pairwise: Minimum L2 distance between any two agents.
    :type min_pairwise: float
    :param max_tries: Attempts per agent before structured fallback.
    :type max_tries: int
    :returns: Initial positions ``(n_agents, L)``.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    lo, hi = trait_box_bounds(space)
    positions: list[np.ndarray] = []

    for _ in range(n_agents):
        placed = False
        for _ in range(max_tries):
            cand = rng.uniform(lo, hi)
            if not positions:
                positions.append(cand)
                placed = True
                break
            if min(np.linalg.norm(cand - p) for p in positions) >= min_pairwise:
                positions.append(cand)
                placed = True
                break
        if not placed:
            # Fallback: radial spokes from resource mean (well separated).
            return _radial_separated_positions(space, n_agents, seed)

    return np.stack(positions, axis=0)


def _radial_separated_positions(
    space: TraitSpace,
    n_agents: int,
    seed: int,
) -> np.ndarray:
    """Place agents on a circle around :math:`\\mathbb{E}_B[b]` inside the trait box.

    :param space: Trait space.
    :type space: TraitSpace
    :param n_agents: Number of agents.
    :type n_agents: int
    :param seed: RNG seed.
    :type seed: int
    :returns: Positions ``(n_agents, L)``.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    lo, hi = trait_box_bounds(space)
    span = float(np.max(hi - lo))
    radius = 0.22 * span
    angles = np.linspace(0.0, 2.0 * np.pi, n_agents, endpoint=False)
    angles = angles + rng.uniform(-0.15, 0.15, size=n_agents)
    out = []
    for a in angles:
        direction = np.zeros(space.L)
        direction[0] = np.cos(a)
        if space.L > 1:
            direction[1] = np.sin(a)
        else:
            direction[0] = np.cos(a)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction = direction / norm
        pos = space.mean + radius * direction
        out.append(np.clip(pos, lo, hi))
    return np.stack(out, axis=0)


def run_theory_gradient_positions(
    space: TraitSpace,
    agent_names: Sequence[str],
    *,
    sigma: float,
    seed: int,
    learning_rate: float = 5e-3,
    n_steps: int = 5000,
    tol: float = 1e-8,
    min_pairwise: float = 0.2,
) -> dict[str, Any]:
    """Gradient-ascent Nash from a random separated state.

    :param space: Trait space.
    :type space: TraitSpace
    :param agent_names: Agent names (order matches returned rows).
    :type agent_names: Sequence[str]
    :param sigma: Competitive reach.
    :type sigma: float
    :param seed: RNG seed for separated start.
    :type seed: int
    :param learning_rate: Gradient step size.
    :type learning_rate: float
    :param n_steps: Maximum gradient steps.
    :type n_steps: int
    :param tol: Convergence tolerance.
    :type tol: float
    :param min_pairwise: Minimum separation at random start.
    :type min_pairwise: float
    :returns: Dict with ``initial``, ``theory_end``, ``layout``, ``converged``,
        ``n_steps``, ``final_spread``.
    :rtype: dict
    """
    from infl_ens.training.pool_dynamics import run_gradient_ascent_theory

    n = len(agent_names)
    p0 = random_separated_initial_positions(
        space, n, seed, min_pairwise=min_pairwise,
    )
    grad = run_gradient_ascent_theory(
        space,
        p0,
        list(agent_names),
        sigma=sigma,
        learning_rate=learning_rate,
        n_steps=n_steps,
        tol=tol,
        seed=seed,
    )
    theory_end = grad["positions"][-1]
    return {
        "initial": p0,
        "theory_end": theory_end,
        "layout": grad["layout"],
        "converged": grad["converged"],
        "n_steps": grad["n_steps"],
        "final_spread": grad["final_spread"],
    }


def init_agents_at_positions(
    cfg: dict[str, Any],
    positions: np.ndarray,
    space: TraitSpace,
    *,
    seed: int,
    init_noise: float = 0.0,
) -> list[RouterAgent]:
    """Create agents at given coordinates (optional jitter).

    :param cfg: Training config with ``agents`` list.
    :type cfg: dict
    :param positions: ``(N, L)`` rows aligned with ``cfg['agents']`` order.
    :type positions: numpy.ndarray
    :param space: Trait space (for ``L``).
    :type space: TraitSpace
    :param seed: RNG seed for jitter.
    :type seed: int
    :param init_noise: Gaussian noise after placing at ``positions``.
    :type init_noise: float
    :returns: Initialized agents.
    :rtype: list[RouterAgent]
    """
    agents_cfg = cfg.get("agents", [])
    if positions.shape[0] != len(agents_cfg):
        raise ValueError(
            f"positions rows {positions.shape[0]} != {len(agents_cfg)} agents",
        )
    rng = np.random.default_rng(seed)
    agents: list[RouterAgent] = []
    for entry, base in zip(agents_cfg, positions):
        pos = np.asarray(base, dtype=float).copy()
        if init_noise > 0.0:
            pos = pos + init_noise * rng.standard_normal(space.L)
        agents.append(RouterAgent(name=entry["name"], position=pos))
    return agents


def resolve_agent_entries(
    agents_cfg: Any,
    n_axes: int,
    *,
    default_prefix: str = "clone",
) -> list[dict[str, Any]]:
    """Expand the config ``agents`` block into a concrete list of entries.

    A list is returned unchanged. A mapping of the form
    ``{pairs_from_axes: true, name_prefix: <str>}`` expands to
    ``2 * n_axes`` entries named ``<prefix>-0`` through ``<prefix>-{2L-1}``:
    two clones per trait axis, the population the pair-merge experiments use.
    Resolving this once in the task driver keeps every downstream consumer
    (``len(cfg["agents"])``, the initializers, the evaluation scripts) seeing
    an ordinary list.

    :param agents_cfg: Value of the top-level ``agents`` config key.
    :type agents_cfg: list | dict | None
    :param n_axes: Trait-space dimensionality :math:`L`.
    :type n_axes: int
    :param default_prefix: Name prefix when the mapping omits ``name_prefix``.
    :type default_prefix: str
    :returns: Concrete agent entries in population order.
    :rtype: list[dict]
    :raises ValueError: If ``agents_cfg`` is a mapping without
        ``pairs_from_axes: true``, if ``n_axes < 1``, or if the value is
        neither a list nor a mapping.
    """
    if isinstance(agents_cfg, list):
        return list(agents_cfg)
    if isinstance(agents_cfg, dict):
        if not agents_cfg.get("pairs_from_axes", False):
            raise ValueError(
                "an agents mapping must set pairs_from_axes: true; got keys "
                f"{sorted(agents_cfg)}"
            )
        if int(n_axes) < 1:
            raise ValueError(f"n_axes must be >= 1, got {n_axes}")
        prefix = str(agents_cfg.get("name_prefix", default_prefix))
        return [{"name": f"{prefix}-{i}"} for i in range(2 * int(n_axes))]
    raise ValueError(
        "config agents must be a list of entries or a mapping with "
        f"pairs_from_axes: true, got {type(agents_cfg).__name__}"
    )


def init_agents_mean_noise(
    cfg: dict[str, Any],
    space: TraitSpace,
    *,
    seed: int,
    init_noise: float,
) -> list[RouterAgent]:
    """IID Gaussian perturbations around the resource mean (default init).

    :param cfg: Training config.
    :type cfg: dict
    :param space: Trait space.
    :type space: TraitSpace
    :param seed: RNG seed.
    :type seed: int
    :param init_noise: Perturbation std.
    :type init_noise: float
    :returns: Initialized agents.
    :rtype: list[RouterAgent]
    """
    rng = np.random.default_rng(seed)
    x0 = space.mean.copy()
    agents: list[RouterAgent] = []
    for entry in cfg.get("agents", []):
        if init_noise > 0.0:
            pos = x0 + init_noise * rng.standard_normal(space.L)
        else:
            pos = x0.copy()
        agents.append(RouterAgent(name=entry["name"], position=pos))
    return agents


def init_agents_theory_gradient(
    cfg: dict[str, Any],
    space: TraitSpace,
    *,
    sigma: float,
    seed: int,
    init_noise: float = 0.0,
    theory_cfg: Optional[dict[str, Any]] = None,
) -> tuple[list[RouterAgent], dict[str, Any]]:
    """Initialize SFT agents at gradient-ascent theory positions.

    Runs grid Nash gradient ascent from a random **separated** state, then
    places each closed-loop agent at the corresponding theoretical endpoint
    (plus optional ``init_noise`` jitter).

    :param cfg: Training config.
    :type cfg: dict
    :param space: Trait space.
    :type space: TraitSpace
    :param sigma: Competitive reach for theory solve.
    :type sigma: float
    :param seed: Per-run seed (separated start + jitter).
    :type seed: int
    :param init_noise: Post-theory jitter std.
    :type init_noise: float
    :param theory_cfg: Optional ``closed_loop.theory_gradient`` block.
    :type theory_cfg: dict | None
    :returns: Agents and theory metadata (for logging / history).
    :rtype: tuple[list[RouterAgent], dict]
    """
    tc = theory_cfg or {}
    names = [a["name"] for a in cfg.get("agents", [])]
    meta = run_theory_gradient_positions(
        space,
        names,
        sigma=sigma,
        seed=seed,
        learning_rate=float(tc.get("learning_rate", 5e-3)),
        n_steps=int(tc.get("n_steps", 5000)),
        tol=float(tc.get("tol", 1e-8)),
        min_pairwise=float(tc.get("min_pairwise", 0.2)),
    )
    agents = init_agents_at_positions(
        cfg, meta["theory_end"], space, seed=seed, init_noise=init_noise,
    )
    return agents, meta


def co_locate_theory_pairs(
    theory_end: np.ndarray,
    agent_names: Sequence[str],
    *,
    pairing: str = "harm_adjacent",
) -> np.ndarray:
    """Place each pair of agents at its theory-endpoint centroid.

    :param theory_end: ``(N, L)`` gradient-ascent endpoints.
    :type theory_end: numpy.ndarray
    :param agent_names: Names in row order of ``theory_end``.
    :type agent_names: Sequence[str]
    :param pairing: Pairing rule, ``'harm_adjacent'`` (default) or
        ``'nearest'``; see :func:`pair_indices_for_method`.
    :type pairing: str
    :returns: ``(N, L)`` positions, bit-identical within each pair.
    :rtype: numpy.ndarray
    :raises ValueError: If the row count does not match ``agent_names``, or
        ``pairing`` is unknown.
    """
    if theory_end.shape[0] != len(agent_names):
        raise ValueError("theory_end rows must match agent_names length")
    pairs = pair_indices_for_method(theory_end, pairing)
    out = np.empty_like(theory_end)
    for pair in pairs:
        centroid = theory_end[pair].mean(axis=0)
        for i in pair:
            out[int(i)] = centroid
    return out


_MERGE_AXIS_ALIASES: dict[str, str] = {
    "policy": "policy_violation",
    "halluc": "hallucination",
    "overrefusal": "overrefusal",
    "privacy": "privacy",
    "harm": "harm",
}


def init_agents_theory_gradient_paired(
    cfg: dict[str, Any],
    space: TraitSpace,
    *,
    sigma: float,
    seed: int,
    init_noise: float = 0.0,
    theory_cfg: Optional[dict[str, Any]] = None,
    skip_initial_theory: bool = False,
) -> tuple[list[RouterAgent], dict[str, Any]]:
    """Theory init with co-located harm pairs, then a paired theory refinement step.

    Runs gradient ascent from a separated start, co-locates each harm pair at
    the pair centroid, re-runs gradient ascent from that paired state, then
    places SFT agents at the co-located paired endpoints (optional jitter).

    When ``skip_initial_theory`` is ``True``, the first gradient-ascent pass
    is omitted and co-located harm pairs are built directly from a random
    separated draw. This is intended for runs that immediately apply
    ``agent_start_overrides`` before the final Nash solve.

    :param cfg: Training config.
    :type cfg: dict
    :param space: Trait space.
    :type space: TraitSpace
    :param sigma: Competitive reach.
    :type sigma: float
    :param seed: Per-run seed.
    :type seed: int
    :param init_noise: Post-theory jitter (keep small so pairs stay merged).
    :type init_noise: float
    :param theory_cfg: Optional ``closed_loop.theory_gradient`` block.
    :type theory_cfg: dict | None
    :param skip_initial_theory: If ``True``, skip the first theory pass.
    :type skip_initial_theory: bool
    :returns: Agents and theory metadata.
    :rtype: tuple[list[RouterAgent], dict]
    """
    from infl_ens.training.pool_dynamics import (
        classify_layout,
        pairwise_spread,
        run_gradient_ascent_theory,
    )

    tc = theory_cfg or {}
    names = [a["name"] for a in cfg.get("agents", [])]
    lr = float(tc.get("learning_rate", 5e-3))
    n_steps = int(tc.get("n_steps", 5000))
    tol = float(tc.get("tol", 1e-8))
    min_pairwise = float(tc.get("min_pairwise", 0.2))
    pairing = str(tc.get("pairing", "harm_adjacent"))

    if skip_initial_theory:
        p0 = random_separated_initial_positions(
            space,
            len(names),
            seed,
            min_pairwise=min_pairwise,
        )
        paired_start = co_locate_theory_pairs(p0, names, pairing=pairing)
        meta0 = {
            "initial": p0,
            "theory_end": paired_start.copy(),
            "layout": classify_layout(paired_start),
            "converged": True,
            "n_steps": 0,
            "final_spread": pairwise_spread(paired_start),
            "skipped_initial_theory": True,
        }
    else:
        meta0 = run_theory_gradient_positions(
            space,
            names,
            sigma=sigma,
            seed=seed,
            learning_rate=lr,
            n_steps=n_steps,
            tol=tol,
            min_pairwise=min_pairwise,
        )
        paired_start = co_locate_theory_pairs(
            meta0["theory_end"], names, pairing=pairing,
        )
    grad1 = run_gradient_ascent_theory(
        space,
        paired_start,
        list(names),
        sigma=sigma,
        learning_rate=lr,
        n_steps=n_steps,
        tol=tol,
        seed=seed,
    )
    second_pass_end = np.asarray(grad1["positions"][-1], dtype=float)
    theory_end = co_locate_theory_pairs(
        second_pass_end, names, pairing=pairing,
    )
    pair_indices = pair_indices_for_method(theory_end, pairing)
    pair_names = [
        [names[int(i)] for i in pair]
        for pair in pair_indices
    ]
    pair_keys = [
        "pair_" + "_".join(names[int(i)] for i in pair)
        for pair in pair_indices
    ]

    def _within(positions: np.ndarray) -> dict[str, float]:
        """Partner L2 distance under the resolved pairing."""
        return {
            key: float(
                np.linalg.norm(
                    positions[int(pair[0])] - positions[int(pair[1])]
                )
            )
            for key, pair in zip(pair_keys, pair_indices)
        }

    # After co-location partners are identical by construction, so
    # ``within_pair_distance_after_pair`` is always zero and says nothing.
    # The informative diagnostics are the distances measured *before* each
    # co-location: the first pass shows how tightly the free Nash solve
    # clustered the partners, the second whether the paired refinement
    # pulled them apart again.
    within_pair_distances = _within(theory_end)
    meta = {
        **meta0,
        "init_mode": "theory_gradient_paired",
        "pairing_method": pairing,
        "theory_layout_initial": meta0["layout"],
        "theory_layout_paired_refine": grad1["layout"],
        "theory_converged_paired_refine": grad1["converged"],
        "theory_n_steps_paired_refine": grad1["n_steps"],
        "theory_end": theory_end,
        "paired_harm_order": pair_names,
        "within_pair_distance_after_pair": within_pair_distances,
        "within_pair_distance_second_pass": _within(second_pass_end),
        "pair_positions": {
            key: theory_end[int(pair[0])].tolist()
            for key, pair in zip(pair_keys, pair_indices)
        },
        "pair_dominant_axis": {
            key: int(np.argmax(theory_end[int(pair[0])]))
            for key, pair in zip(pair_keys, pair_indices)
        },
    }
    if not skip_initial_theory:
        meta["within_pair_distance_first_pass"] = _within(
            np.asarray(meta0["theory_end"], dtype=float),
        )
    agents = init_agents_at_positions(
        cfg, theory_end, space, seed=seed, init_noise=init_noise,
    )
    return agents, meta


