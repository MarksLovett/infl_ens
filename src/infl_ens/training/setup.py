"""Shared setup for the training tasks: data, trait space, agents, sigma.

Every task in :mod:`infl_ens.training.tasks` starts the same way: load the
benchmark splits named by the config, build (or reload from cache) the
trait space, and resolve the competitive reach :math:`\\sigma`.  Those steps
live here so the closed loop, the baseline replay and the evaluation
stages cannot disagree about them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from infl_ens.data.benchmarks import BenchmarkSplit
from infl_ens.data.benchmarks.loading import load_benchmark_splits
from infl_ens.data.trait_space import TraitSpace
from infl_ens.data.trait_space_cache import build_or_load_safety_trait_space
from infl_ens.inflgame.router import RouterAgent
from infl_ens.utils.resource import gaussian_stability_threshold


def load_splits(cfg: dict[str, Any]) -> list[BenchmarkSplit]:
    """Load all benchmark splits referenced by ``cfg["benchmarks"]``.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :returns: Loaded splits in config order.
    :rtype: list[BenchmarkSplit]
    """
    return load_benchmark_splits(cfg.get("benchmarks", []))


def make_trait_space(cfg: dict[str, Any], splits: list[BenchmarkSplit]) -> TraitSpace:
    """Build a :class:`TraitSpace` from benchmark splits and config.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :param splits: Already-loaded benchmark splits.
    :type splits: list[BenchmarkSplit]
    :returns: The constructed trait space.
    :rtype: TraitSpace
    """
    return build_or_load_safety_trait_space(cfg, splits)


def sigma_from_config(
    cfg: dict[str, Any], n_agents: int, space: TraitSpace,
) -> float:
    """Resolve the competitive reach :math:`\\sigma` from the config.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :param n_agents: Number of agents.
    :type n_agents: int
    :param space: Trait space (for the stability threshold).
    :type space: TraitSpace
    :returns: Scalar :math:`\\sigma`.
    :rtype: float
    """
    mode = cfg.get("sigma_mode", "absolute")
    if mode == "absolute":
        return float(cfg["sigma"])
    if mode == "stability_fraction":
        frac = float(cfg.get("sigma_fraction", 0.8))
        s0 = gaussian_stability_threshold(n_agents, space.grid, space.weights)
        return frac * max(s0, 1e-3)
    raise ValueError(f"unknown sigma_mode {mode!r}")


def init_agents(
    cfg: dict[str, Any],
    space: TraitSpace,
    splits: list[BenchmarkSplit],
    *,
    init_noise: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> list[RouterAgent]:
    """Initialise router agents from the config's ``agents`` list.

    When no per-agent ``calibration`` split is given, each agent's
    position defaults to the resource-weighted mean :math:`\\mathbb{E}_B[b]`
    plus optional i.i.d. Gaussian noise
    ``init_noise * \\mathcal{N}(0, I_L)`` to break exact clone symmetry.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :param space: Trait space defining the projector.
    :type space: TraitSpace
    :param splits: Already-loaded benchmark splits.
    :type splits: list[BenchmarkSplit]
    :param init_noise: Standard deviation of the initial position
        perturbation around ``space.mean``. Ignored for calibrated agents.
    :type init_noise: float
    :param rng: RNG used for the perturbation. Defaults to
        ``numpy.random.default_rng(cfg['seed'])`` when ``init_noise > 0``.
    :type rng: numpy.random.Generator | None
    :returns: List of :class:`RouterAgent` ready for routing/training.
    :rtype: list[RouterAgent]
    :raises ValueError: If ``init_noise < 0``.
    """
    if init_noise < 0.0:
        raise ValueError(f"init_noise must be >= 0, got {init_noise}")
    by_name = {s.name: s for s in splits}
    agents: list[RouterAgent] = []
    x0 = space.mean.copy()
    if init_noise > 0.0 and rng is None:
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
    for entry in cfg.get("agents", []):
        name = entry["name"]
        cal_name = entry.get("calibration")
        if cal_name and cal_name in by_name:
            agents.append(RouterAgent.from_calibration(
                name=name,
                calibration_queries=by_name[cal_name].prompts[:200],
                project=space.project,
            ))
        else:
            if init_noise > 0.0:
                assert rng is not None
                pos = x0 + init_noise * rng.standard_normal(space.L)
            else:
                pos = x0.copy()
            agents.append(RouterAgent(name=name, position=pos))
    return agents


def coords_for_prompts(
    prompts: Sequence[str],
    coord_by_text: dict[str, np.ndarray],
    project: Any,
) -> np.ndarray:
    """Look up projected coordinates, encoding only what is missing.

    The trait-space projector re-encodes with the sentence encoder on every
    call, and the closed loop projects the same prompt pool every round.
    Caching the pool projection once and reusing rows here removes one full
    corpus encode per round.

    :param prompts: Prompts to locate in trait space.
    :type prompts: Sequence[str]
    :param coord_by_text: Prompt text to coordinate row, from the pool
        projection.
    :type coord_by_text: dict[str, numpy.ndarray]
    :param project: Trait-space projector for prompts absent from the cache.
    :type project: Callable[[Sequence[str]], numpy.ndarray]
    :returns: Coordinates, shape ``(len(prompts), L)``.
    :rtype: numpy.ndarray
    """
    prompts = list(prompts)
    missing = [q for q in prompts if q not in coord_by_text]
    if missing:
        extra = np.asarray(project(missing), dtype=float)
        for q, row in zip(missing, extra):
            coord_by_text[q] = row
    return np.stack([coord_by_text[q] for q in prompts], axis=0)


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    """Write the closed-loop history to disk.

    Called after every round so a long run is inspectable mid-flight and a
    crash does not lose the trajectory.

    :param path: Destination ``history.json`` path.
    :type path: pathlib.Path
    :param history: Per-round records accumulated so far.
    :type history: list[dict]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)


def write_resolved_config(cfg: dict[str, Any], path: Path) -> Optional[Path]:
    """Dump the config with every generated block resolved to a literal.

    ``agents`` may be written as a mapping and ``sft_merge_groups`` as the
    ``from_init`` sentinel; both are expanded in-process before the run
    starts. The evaluation, routing and figure stages read those keys
    literally, so the resolved copy is what they are pointed at.

    :param cfg: Fully resolved configuration dictionary.
    :type cfg: dict
    :param path: Destination path (``resolved_config.yaml``).
    :type path: pathlib.Path
    :returns: The written path, or ``None`` if PyYAML is unavailable.
    :rtype: pathlib.Path | None
    """
    try:
        import yaml
    except ImportError:  # pragma: no cover - environment-level
        print("warning: PyYAML unavailable; skipping resolved_config.yaml")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
    return path


__all__ = [
    "coords_for_prompts",
    "init_agents",
    "load_splits",
    "make_trait_space",
    "sigma_from_config",
    "write_history",
    "write_resolved_config",
]
