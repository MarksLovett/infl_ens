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
    low, high = harm_pair_indices(reference)
    anchors = [reference[low[0]], reference[low[1]], reference[high[0]], reference[high[1]]]
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
