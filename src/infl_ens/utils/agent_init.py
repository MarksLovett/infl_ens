"""Agent position initialization helpers for closed-loop and simulators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from infl_ens.data.trait_space import TraitSpace
from infl_ens.inflgame.router import RouterAgent


def load_reference_from_history(path: Path) -> np.ndarray:
    """Load final-round positions from a ``history.json`` file.

    :param path: Path to history JSON.
    :type path: pathlib.Path
    :returns: Positions ``(N, L)`` in sorted agent-name order.
    :rtype: numpy.ndarray
    """
    records = json.loads(path.read_text(encoding="utf-8"))
    names = sorted(records[-1]["positions"].keys())
    return np.stack([
        np.asarray(records[-1]["positions"][n], dtype=float) for n in names
    ])


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


def underrepresented_grid_point(
    space: TraitSpace,
    *,
    avoid: Sequence[np.ndarray] = (),
    min_avoid_distance: float = 0.1,
    weight_quantile: float = 0.05,
) -> np.ndarray:
    """Pick a low-resource grid point in :math:`\\mathbb{B}`.

    Selects among the lowest ``weight_quantile`` fraction of KDE grid
    weights, preferring candidates at least ``min_avoid_distance`` away
    from any point in ``avoid``. Falls back to the global minimum-weight
    grid point when no candidate satisfies the separation constraint.

    :param space: Trait space carrying ``grid`` and ``weights``.
    :type space: TraitSpace
    :param avoid: Optional positions to stay away from (other agents).
    :type avoid: Sequence[numpy.ndarray]
    :param min_avoid_distance: Minimum L2 distance from ``avoid`` points.
    :type min_avoid_distance: float
    :param weight_quantile: Upper quantile for the low-mass candidate pool.
    :type weight_quantile: float
    :returns: Selected grid coordinate, shape ``(L,)``.
    :rtype: numpy.ndarray
    :raises ValueError: If ``weight_quantile`` is not in ``(0, 1]``.
    """
    if not 0.0 < weight_quantile <= 1.0:
        raise ValueError(
            f"weight_quantile must lie in (0, 1], got {weight_quantile}"
        )
    weights = np.asarray(space.weights, dtype=float)
    grid = np.asarray(space.grid, dtype=float)
    cutoff = float(np.quantile(weights, weight_quantile))
    candidate_idx = np.flatnonzero(weights <= cutoff + 1e-15)
    if candidate_idx.size == 0:
        candidate_idx = np.array([int(np.argmin(weights))])

    avoid_arr = [np.asarray(p, dtype=float) for p in avoid]
    separated_idx = candidate_idx
    if avoid_arr and min_avoid_distance > 0.0:
        keep: list[int] = []
        for idx in candidate_idx:
            point = grid[idx]
            if all(
                float(np.linalg.norm(point - ref)) >= min_avoid_distance
                for ref in avoid_arr
            ):
                keep.append(int(idx))
        if keep:
            separated_idx = np.asarray(keep, dtype=int)

    best_idx = int(separated_idx[np.argmin(weights[separated_idx])])
    if avoid_arr and min_avoid_distance > 0.0:
        dists = [
            min(float(np.linalg.norm(grid[i] - ref)) for ref in avoid_arr)
            for i in separated_idx
        ]
        weights_pool = weights[separated_idx]
        rank = np.lexsort((-np.asarray(dists), weights_pool))
        best_idx = int(separated_idx[rank[0]])
    return grid[best_idx].copy()


def apply_agent_start_overrides(
    agents: list[RouterAgent],
    overrides: dict[str, Any],
    space: TraitSpace,
    *,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Move named agents to configured start overrides in place.

    Override values may be:

    - ``\"underrepresented\"``: snap to a low-mass grid point, shared by
      all agents using the same token in one call.
    - a length-``L`` numeric list: explicit coordinates in ``[0, 1]^L``.

    :param agents: Router agents to mutate.
    :type agents: list[RouterAgent]
    :param overrides: Map from agent name to override spec.
    :type overrides: dict
    :param space: Trait space used to resolve ``underrepresented``.
    :type space: TraitSpace
    :param seed: Unused today; reserved for stochastic tie breaks.
    :type seed: int
    :returns: Resolved override coordinates keyed by agent name.
    :rtype: dict[str, list[float]]
    :raises ValueError: If an agent name or coordinate spec is invalid.
    """
    del seed
    if not overrides:
        return {}
    by_name = {a.name: a for a in agents}
    missing = sorted(set(overrides) - set(by_name))
    if missing:
        raise ValueError(
            f"agent_start_overrides unknown agents: {', '.join(missing)}"
        )

    resolved: dict[str, np.ndarray] = {}
    token_groups: dict[str, list[str]] = {}
    for name, spec in overrides.items():
        if isinstance(spec, str):
            token_groups.setdefault(spec, []).append(name)
        else:
            pos = np.asarray(spec, dtype=float)
            if pos.shape != (space.L,):
                raise ValueError(
                    f"override for {name!r} must have shape ({space.L},), "
                    f"got {pos.shape}"
                )
            resolved[name] = np.clip(pos, 0.0, 1.0)

    avoid_names = set(overrides)
    avoid_points = [
        a.position.copy()
        for a in agents
        if a.name not in avoid_names
    ]
    for token, names in token_groups.items():
        if token == "underrepresented":
            point = underrepresented_grid_point(
                space,
                avoid=avoid_points,
                min_avoid_distance=0.1,
            )
        else:
            raise ValueError(
                f"unknown agent_start_overrides token {token!r}; "
                "use 'underrepresented' or an explicit coordinate list"
            )
        for name in names:
            resolved[name] = point.copy()

    for name, pos in resolved.items():
        by_name[name].position = pos.copy()
    return {name: pos.tolist() for name, pos in resolved.items()}


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


def init_agents_pairs_near_reference(
    cfg: dict[str, Any],
    space: TraitSpace,
    reference: np.ndarray,
    *,
    seed: int,
    init_noise: float = 0.01,
) -> list[RouterAgent]:
    """Place clones in pairs near a (2,2) reference layout.

    Agents ``0,1`` start near the two low-harm reference clones; ``2,3`` near
    the two high-harm reference clones. Each position is the reference point
    plus ``init_noise * N(0, I)``.

    :param cfg: Training config with ``agents`` list.
    :type cfg: dict
    :param space: Trait space (for ``L``).
    :type space: TraitSpace
    :param reference: ``(N, L)`` template in the same agent-name order as
        ``cfg['agents']`` (typically sorted names from a reference run).
    :type reference: numpy.ndarray
    :param seed: RNG seed for pair jitter.
    :type seed: int
    :param init_noise: Gaussian jitter scale.
    :type init_noise: float
    :returns: Initialized router agents.
    :rtype: list[RouterAgent]
    """
    names = sorted(a["name"] for a in cfg.get("agents", []))
    if reference.shape[0] != len(names):
        raise ValueError(
            f"reference has {reference.shape[0]} agents, config has {len(names)}",
        )
    pairs = adjacent_harm_pair_indices(reference)
    anchors = [reference[int(i)] for pair in pairs for i in pair]
    rng = np.random.default_rng(seed)
    agents: list[RouterAgent] = []
    for entry, base in zip(cfg.get("agents", []), anchors):
        pos = np.asarray(base, dtype=float).copy()
        if init_noise > 0.0:
            pos = pos + init_noise * rng.standard_normal(space.L)
        agents.append(RouterAgent(name=entry["name"], position=pos))
    return agents


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


def axis_index_from_merge_label(
    train_as: str,
    axis_labels: Sequence[str],
    *,
    group_index: int = 0,
) -> int:
    """Map a merge group label to a trait-axis index.

    :param train_as: Merge training name (e.g. ``merge-harm``).
    :type train_as: str
    :param axis_labels: Trait-space axis labels.
    :type axis_labels: Sequence[str]
    :param group_index: Fallback index when no label match is found.
    :type group_index: int
    :returns: Axis index in ``[0, L)``.
    :rtype: int
    """
    slug = train_as.removeprefix("merge-").lower()
    slug = _MERGE_AXIS_ALIASES.get(slug, slug)
    labels = [lab.lower() for lab in axis_labels]
    if slug in labels:
        return labels.index(slug)
    for i, lab in enumerate(labels):
        if slug in lab or lab.replace("_", "") in slug.replace("_", ""):
            return i
    return int(group_index) % max(len(labels), 1)


def axis_corner_unit(axis_idx: int, L: int) -> np.ndarray:
    """Return axis-aligned hypercube vertex ``e_i`` in ``[0, 1]^L``.

    :param axis_idx: Active coordinate index.
    :type axis_idx: int
    :param L: Trait dimensionality.
    :type L: int
    :returns: Corner vector with a one in coordinate ``axis_idx``.
    :rtype: numpy.ndarray
    """
    corner = np.zeros(L, dtype=float)
    corner[int(axis_idx)] = 1.0
    return corner


def merge_pairs_hypercube_edge_positions(
    merge_groups: Sequence[tuple[str, Sequence[str]]],
    agent_names: Sequence[str],
    L: int,
    *,
    seed: int,
    pair_separation: float = 0.04,
    randomize_corners: bool = True,
    corner_jitter: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Place merge pairs near randomly assigned hypercube vertices ``e_i``.

    Each merge group is assigned a distinct axis-aligned corner such as
    ``(1, 0, 0, 0, 0)``. Partners start at ``corner ± pair_separation / 2``
    along that axis (clipped to ``[0, 1]^L``).

    :param merge_groups: ``(train_as, member_names)`` pairs of size two.
    :type merge_groups: Sequence[tuple[str, Sequence[str]]]
    :param agent_names: Full router agent name list (config order).
    :type agent_names: Sequence[str]
    :param L: Trait dimensionality.
    :type L: int
    :param seed: RNG seed for corner permutation and jitter.
    :type seed: int
    :param pair_separation: Target L2 distance between merge partners.
    :type pair_separation: float
    :param randomize_corners: If ``True``, shuffle which corner each group gets.
    :type randomize_corners: bool
    :param corner_jitter: Uniform jitter added to all coordinates before
        partner offsets (optional).
    :type corner_jitter: float
    :returns: Positions ``(N, L)`` and placement metadata.
    :rtype: tuple[numpy.ndarray, dict]
    :raises ValueError: If groups are not pairs or exceed ``L`` corners.
    """
    if pair_separation <= 0.0:
        raise ValueError(f"pair_separation must be positive, got {pair_separation}")
    k = len(merge_groups)
    if k > L:
        raise ValueError(
            f"hypercube-edge init supports at most {L} merge groups, got {k}",
        )
    rng = np.random.default_rng(seed)
    axis_indices = list(range(L))
    if randomize_corners:
        rng.shuffle(axis_indices)
    axis_for_group = axis_indices[:k]

    name_to_idx = {n: i for i, n in enumerate(agent_names)}
    positions = np.zeros((len(agent_names), L), dtype=float)
    corner_assignments: dict[str, dict[str, Any]] = {}
    within_pair_distances: dict[str, float] = {}
    half = 0.5 * pair_separation

    for g_idx, (train_as, members) in enumerate(merge_groups):
        members = list(members)
        if len(members) != 2:
            raise ValueError(
                f"hypercube-edge init requires pairs of size 2; "
                f"{train_as!r} has {len(members)} members",
            )
        ax = int(axis_for_group[g_idx])
        corner = axis_corner_unit(ax, L)
        if corner_jitter > 0.0:
            corner = np.clip(
                corner + rng.uniform(-corner_jitter, corner_jitter, size=L),
                0.0,
                1.0,
            )
        direction = axis_corner_unit(ax, L)
        positions[name_to_idx[members[0]]] = np.clip(
            corner - half * direction, 0.0, 1.0,
        )
        positions[name_to_idx[members[1]]] = np.clip(
            corner + half * direction, 0.0, 1.0,
        )
        i, j = name_to_idx[members[0]], name_to_idx[members[1]]
        within_pair_distances[train_as] = float(
            np.linalg.norm(positions[i] - positions[j]),
        )
        corner_assignments[train_as] = {
            "axis_index": ax,
            "corner": corner.tolist(),
            "members": members,
        }

    return positions, {
        "init_mode": "merge_hypercube_edges",
        "randomize_corners": randomize_corners,
        "corner_jitter": corner_jitter,
        "pair_separation_target": pair_separation,
        "corner_assignments": corner_assignments,
        "within_pair_distance_at_init": within_pair_distances,
    }


def init_agents_merge_hypercube_edges(
    cfg: dict[str, Any],
    space: TraitSpace,
    merge_groups: Sequence[tuple[str, Sequence[str]]],
    *,
    seed: int,
    init_noise: float = 0.0,
    edge_cfg: Optional[dict[str, Any]] = None,
) -> tuple[list[RouterAgent], dict[str, Any]]:
    """Initialize merge partners on randomly assigned hypercube vertices.

    :param cfg: Training config with ``agents`` list.
    :type cfg: dict
    :param space: Trait space.
    :type space: TraitSpace
    :param merge_groups: Parsed ``sft_merge_groups`` entries.
    :type merge_groups: Sequence[tuple[str, Sequence[str]]]
    :param seed: Per-run seed.
    :type seed: int
    :param init_noise: Optional post-placement jitter (keep small).
    :type init_noise: float
    :param edge_cfg: Optional ``closed_loop.merge_hypercube`` block.
    :type edge_cfg: dict | None
    :returns: Agents and placement metadata.
    :rtype: tuple[list[RouterAgent], dict]
    """
    ec = edge_cfg or {}
    names = [a["name"] for a in cfg.get("agents", [])]
    positions, placement_meta = merge_pairs_hypercube_edge_positions(
        merge_groups,
        names,
        space.L,
        seed=seed,
        pair_separation=float(ec.get("pair_separation", 0.04)),
        randomize_corners=bool(ec.get("randomize_corners", True)),
        corner_jitter=float(ec.get("corner_jitter", 0.0)),
    )
    agents = init_agents_at_positions(
        cfg, positions, space, seed=seed, init_noise=init_noise,
    )
    meta = {
        "merge_groups": [[t, list(m)] for t, m in merge_groups],
        "theory_initial": positions.tolist(),
        **placement_meta,
    }
    return agents, meta


def merge_near_initial_positions(
    space: TraitSpace,
    merge_groups: Sequence[tuple[str, Sequence[str]]],
    agent_names: Sequence[str],
    *,
    sigma: float,
    seed: int,
    pair_separation: float = 0.04,
    theory_cfg: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Place merge partners near a shared anchor from a reduced Nash solve.

    Runs gradient-ascent theory on one virtual agent per merge group to
    obtain separated axis anchors, then offsets each pair by
    ``± pair_separation / 2`` along the group's trait axis.

    :param space: Trait space.
    :type space: TraitSpace
    :param merge_groups: ``(train_as, member_names)`` pairs of size two.
    :type merge_groups: Sequence[tuple[str, Sequence[str]]]
    :param agent_names: Full router agent name list (config order).
    :type agent_names: Sequence[str]
    :param sigma: Competitive reach for anchor theory solve.
    :type sigma: float
    :param seed: RNG seed for anchor theory.
    :type seed: int
    :param pair_separation: Target L2 distance between merge partners.
    :type pair_separation: float
    :param theory_cfg: Optional ``closed_loop.theory_gradient`` block.
    :type theory_cfg: dict | None
    :returns: Positions ``(N, L)`` and placement metadata.
    :rtype: tuple[numpy.ndarray, dict]
    :raises ValueError: If merge groups are not disjoint pairs of size two.
    """
    if pair_separation <= 0.0:
        raise ValueError(f"pair_separation must be positive, got {pair_separation}")
    tc = theory_cfg or {}
    k = len(merge_groups)
    if k < 1:
        raise ValueError("merge_near_initial_positions requires at least one merge group")
    anchor_names = [f"_anchor-{i}" for i in range(k)]
    anchor_meta = run_theory_gradient_positions(
        space,
        anchor_names,
        sigma=sigma,
        seed=seed,
        learning_rate=float(tc.get("learning_rate", 5e-3)),
        n_steps=int(tc.get("n_steps", 5000)),
        tol=float(tc.get("tol", 1e-8)),
        min_pairwise=float(tc.get("min_pairwise", 0.2)),
    )
    anchors = anchor_meta["theory_end"]
    axis_labels = space.axis_labels or tuple(f"axis-{i}" for i in range(space.L))
    name_to_idx = {n: i for i, n in enumerate(agent_names)}
    positions = np.zeros((len(agent_names), space.L), dtype=float)
    within_pair_distances: dict[str, float] = {}
    half = 0.5 * pair_separation
    for g_idx, (train_as, members) in enumerate(merge_groups):
        members = list(members)
        if len(members) != 2:
            raise ValueError(
                f"merge_near_theory requires pairs of size 2; "
                f"{train_as!r} has {len(members)} members",
            )
        ax_idx = axis_index_from_merge_label(
            train_as, axis_labels, group_index=g_idx,
        )
        direction = np.zeros(space.L, dtype=float)
        direction[ax_idx] = 1.0
        anchor = anchors[g_idx]
        positions[name_to_idx[members[0]]] = np.clip(anchor - half * direction, 0.0, 1.0)
        positions[name_to_idx[members[1]]] = np.clip(anchor + half * direction, 0.0, 1.0)
        within_pair_distances[train_as] = float(
            np.linalg.norm(
                positions[name_to_idx[members[0]]]
                - positions[name_to_idx[members[1]]],
            ),
        )
    return positions, {
        "anchor_theory_layout": anchor_meta["layout"],
        "anchor_theory_converged": anchor_meta["converged"],
        "anchor_theory_n_steps": anchor_meta["n_steps"],
        "anchor_positions": anchors.tolist(),
        "within_pair_distance_at_init": within_pair_distances,
        "pair_separation_target": pair_separation,
    }


def init_agents_merge_near_theory(
    cfg: dict[str, Any],
    space: TraitSpace,
    merge_groups: Sequence[tuple[str, Sequence[str]]],
    *,
    sigma: float,
    seed: int,
    init_noise: float = 0.0,
    near_cfg: Optional[dict[str, Any]] = None,
    theory_cfg: Optional[dict[str, Any]] = None,
) -> tuple[list[RouterAgent], dict[str, Any]]:
    """Initialize merge partners near shared anchors from reduced theory.

    :param cfg: Training config with ``agents`` list.
    :type cfg: dict
    :param space: Trait space.
    :type space: TraitSpace
    :param merge_groups: Parsed ``sft_merge_groups`` entries.
    :type merge_groups: Sequence[tuple[str, Sequence[str]]]
    :param sigma: Competitive reach.
    :type sigma: float
    :param seed: Per-run seed.
    :type seed: int
    :param init_noise: Optional post-placement jitter (keep small).
    :type init_noise: float
    :param near_cfg: Optional ``closed_loop.merge_near`` block.
    :type near_cfg: dict | None
    :param theory_cfg: Optional ``closed_loop.theory_gradient`` block.
    :type theory_cfg: dict | None
    :returns: Agents and theory metadata for logging.
    :rtype: tuple[list[RouterAgent], dict]
    """
    nc = near_cfg or {}
    names = [a["name"] for a in cfg.get("agents", [])]
    pair_sep = float(nc.get("pair_separation", 0.04))
    positions, placement_meta = merge_near_initial_positions(
        space,
        merge_groups,
        names,
        sigma=sigma,
        seed=seed,
        pair_separation=pair_sep,
        theory_cfg=theory_cfg,
    )
    agents = init_agents_at_positions(
        cfg, positions, space, seed=seed, init_noise=init_noise,
    )
    meta = {
        "init_mode": "merge_near_theory",
        "merge_groups": [[t, list(m)] for t, m in merge_groups],
        "theory_initial": positions.tolist(),
        **placement_meta,
    }
    return agents, meta


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


def _reference_is_22(pos: np.ndarray) -> bool:
    """True if positions classify as a (2,2) layout."""
    from infl_ens.training.pool_dynamics import classify_layout

    return classify_layout(pos) == "2,2"


def _sigma_sort_key(path: Path) -> tuple[int, float]:
    """Sort ``sigma0.25``-style directory names by numeric σ."""
    name = path.parent.parent.name if path.name == "history.json" else path.name
    if name.startswith("sigma"):
        try:
            return (0, float(name[5:]))
        except ValueError:
            pass
    return (1, 0.0)


def _discover_reference_histories(
    repo_root: Path,
    *,
    sigma_fraction: Optional[float] = None,
) -> list[Path]:
    """Collect candidate ``history.json`` paths, per-σ first then others."""
    ordered: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            ordered.append(path)

    if sigma_fraction is not None:
        s = sigma_fraction
        for root in (
            "results/theory_match_fixes/baseline_blend05",
            "results/pool_and_noise_10seeds",
            "results/pairs_near_eq_sigma_sweep",
            "results/pairs_near_theory_sigma_sweep",
        ):
            add(repo_root / root / f"sigma{s}" / "seed0" / "history.json")

    search_roots = [
        repo_root / "results/theory_match_fixes/baseline_blend05",
        repo_root / "results/pool_and_noise_10seeds",
        repo_root / "results/pairs_near_eq_sigma_sweep",
        repo_root / "results/pairs_near_theory_sigma_sweep",
    ]
    discovered: list[Path] = []
    for base in search_roots:
        if not base.is_dir():
            continue
        for sigma_dir in sorted(base.glob("sigma*"), key=_sigma_sort_key):
            hist = sigma_dir / "seed0" / "history.json"
            if hist.is_file():
                discovered.append(hist)
    for hist in sorted(discovered, key=_sigma_sort_key):
        add(hist)
    return ordered


def resolve_theory_22_reference(
    space: TraitSpace,
    agent_names: Sequence[str],
    *,
    sigma: float,
    ref_path: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    sigma_fraction: Optional[float] = None,
) -> np.ndarray:
    """Obtain a (2,2) reference layout for ``pairs_near_theory`` init.

    Loads ``ref_path`` when given; otherwise scans known result trees (matching
    σ first, then any σ with a (2,2) final layout). If nothing on disk qualifies,
  runs gradient ascent from a **separated** random start (same as
    ``theory_gradient`` init). At high σ where theory collapses, falls back to the
    nearest available (2,2) reference from a lower-σ run.

    :param space: Trait space.
    :type space: TraitSpace
    :param agent_names: Agent names (for gradient fallback).
    :type agent_names: Sequence[str]
    :param sigma: Absolute competitive reach.
    :type sigma: float
    :param ref_path: Optional explicit history path.
    :type ref_path: pathlib.Path | None
    :param repo_root: Repository root for default paths.
    :type repo_root: pathlib.Path | None
    :param sigma_fraction: σ fraction (for default path lookup).
    :type sigma_fraction: float | None
    :returns: Reference positions ``(N, L)`` sorted by agent name.
    :rtype: numpy.ndarray
    """
    candidates: list[Path] = []
    if ref_path is not None:
        candidates.append(Path(ref_path))
    if repo_root is not None:
        candidates.extend(
            _discover_reference_histories(
                repo_root, sigma_fraction=sigma_fraction,
            ),
        )

    fallback_22: Optional[np.ndarray] = None
    for path in candidates:
        if not path.is_file():
            continue
        ref = load_reference_from_history(path)
        if _reference_is_22(ref):
            if sigma_fraction is not None and f"sigma{sigma_fraction}" in str(path):
                return ref
            if fallback_22 is None:
                fallback_22 = ref

    if fallback_22 is not None:
        return fallback_22

    meta = run_theory_gradient_positions(
        space, list(agent_names), sigma=sigma, seed=0,
    )
    if _reference_is_22(meta["theory_end"]):
        return meta["theory_end"]

    raise RuntimeError(
        "Could not resolve a (2,2) theory reference: no qualifying history on "
        f"disk and theory_gradient layout={meta['layout']!r} at sigma={sigma}",
    )
