"""Compare strategic gradient-ascent equilibria with SFT closed-loop positions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from infl_ens.data.trait_space_cache import build_or_load_safety_trait_space
from infl_ens.evaluation.benchmarks import load_benchmark_splits
from infl_ens.training.router_training import RouterTrainingConfig, train_router_positions
from infl_ens.utils.resource import gaussian_stability_threshold


def build_theory_trait_space(cfg: dict):
    """Rebuild the trait space from a closed-loop run config.

    Uses :func:`infl_ens.evaluation.benchmarks.load_benchmark_splits` and
    :func:`infl_ens.data.trait_space_cache.build_or_load_safety_trait_space`.

    :param cfg: Parsed YAML config of the SFT run.
    :type cfg: dict
    :returns: Tuple ``(space, splits)``.
    :rtype: tuple
    """
    splits = load_benchmark_splits(cfg.get("benchmarks", []))
    space = build_or_load_safety_trait_space(cfg, splits)
    return space, splits


def _load_history_records(history_path: Path) -> list[dict]:
    """Load all rounds from ``history.json``.

    :param history_path: Path to the history file.
    :type history_path: pathlib.Path
    :returns: Per-round records.
    :rtype: list[dict]
    :raises ValueError: If the file is empty.
    """
    with history_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    if not records:
        raise ValueError(f"{history_path} contains no rounds")
    return records


def initial_positions_from_history(history_path: Path) -> tuple[list[str], np.ndarray]:
    """Extract round-0 positions from a closed-loop ``history.json``.

    :param history_path: Path to the history file.
    :type history_path: pathlib.Path
    :returns: Tuple of ``(agent_names, positions)`` with shape ``(N, L)``.
    :rtype: tuple[list[str], numpy.ndarray]
    """
    records = _load_history_records(history_path)
    r0 = records[0]
    names = list(r0["positions"].keys())
    pos = np.stack([np.asarray(r0["positions"][n]) for n in names], axis=0)
    return names, pos


def theory_gradient_starts_from_history(
    history_path: Path,
) -> Optional[np.ndarray]:
    """Pre-GA separated positions from ``theory_init`` metadata, if present.

    :param history_path: Path to ``history.json``.
    :type history_path: pathlib.Path
    :returns: ``(N, L)`` array or ``None``.
    :rtype: numpy.ndarray | None
    """
    r0 = _load_history_records(history_path)[0]
    meta = r0.get("theory_init")
    if not meta or "theory_initial" not in meta:
        return None
    return np.stack([np.asarray(row, dtype=float) for row in meta["theory_initial"]])


def sft_trajectory_from_history(history_path: Path, names: Sequence[str]) -> np.ndarray:
    """Stack per-round SFT positions into a ``(T, N, L)`` tensor.

    :param history_path: Path to the history file.
    :type history_path: pathlib.Path
    :param names: Agent name order.
    :type names: Sequence[str]
    :returns: Positions tensor.
    :rtype: numpy.ndarray
    """
    records = _load_history_records(history_path)
    return np.stack(
        [
            np.stack([np.asarray(r["positions"][n]) for n in names], axis=0)
            for r in records
        ],
        axis=0,
    )


def sigma_from_cfg(cfg: dict, n_agents: int, space) -> float:
    """Compute the absolute sigma value from a config.

    Supports ``sigma_mode='absolute'`` or ``sigma_mode='stability_fraction'``.

    :param cfg: Parsed config.
    :type cfg: dict
    :param n_agents: Number of agents.
    :type n_agents: int
    :param space: Trait space (for the stability threshold).
    :type space: infl_ens.data.trait_space.TraitSpace
    :returns: Absolute scalar sigma.
    :rtype: float
    :raises ValueError: If ``sigma_mode`` is unknown.
    """
    mode = cfg.get("sigma_mode", "stability_fraction")
    if mode == "absolute":
        return float(cfg["sigma"])
    if mode == "stability_fraction":
        sigma_star = gaussian_stability_threshold(n_agents, space.grid, space.weights)
        return float(cfg.get("sigma_fraction", 0.5)) * max(sigma_star, 0.05)
    raise ValueError(f"unknown sigma_mode: {mode!r}")


def run_strategic_ascent(
    cfg: dict,
    history_path: Path,
    *,
    learning_rate: float = 5e-3,
    n_steps: int = 5000,
    tol: float = 1e-8,
    seed: int = 0,
) -> dict[str, Any]:
    """Run strategic gradient-ascent from the SFT run's initial state.

    :param cfg: Parsed YAML config of the SFT run.
    :type cfg: dict
    :param history_path: Path to that run's ``history.json``.
    :type history_path: pathlib.Path
    :param learning_rate: Step size for gradient ascent.
    :type learning_rate: float
    :param n_steps: Maximum gradient steps.
    :type n_steps: int
    :param tol: Convergence tolerance on max coordinate change.
    :type tol: float
    :param seed: RNG seed forwarded to the trainer.
    :type seed: int
    :returns: Dictionary with ``space``, ``agents``, ``positions``,
        ``utilities``, ``converged``, ``sigma``, ``sigma_star``, etc.
    :rtype: dict
    """
    space, splits = build_theory_trait_space(cfg)
    names, sft_start = initial_positions_from_history(history_path)
    theory_start = theory_gradient_starts_from_history(history_path)
    ga_start = theory_start if theory_start is not None else sft_start

    agents = [
        RouterAgent(name=n, position=ga_start[i].copy())
        for i, n in enumerate(names)
    ]
    sigma = sigma_from_cfg(cfg, len(agents), space)
    sigma_star = gaussian_stability_threshold(
        len(agents),
        space.grid,
        space.weights,
    )

    rt_cfg = RouterTrainingConfig(
        sigma=sigma,
        learning_rate=learning_rate,
        n_steps=n_steps,
        tol=tol,
        clip_to_box=True,
    )
    info = train_router_positions(space, agents, rt_cfg, seed=seed)
    info.update({
        "space": space,
        "agents": agents,
        "sigma": sigma,
        "sigma_star": float(sigma_star),
        "sft_start_positions": sft_start,
        "theory_start_positions": ga_start if theory_start is not None else sft_start,
        "theory_gradient_init": theory_start is not None,
        "initial_positions": sft_start,
        "names": names,
        "splits": splits,
    })
    return info


def build_theory_summary(
    info: dict[str, Any],
    sft_end: np.ndarray,
    theo_end: np.ndarray,
    *,
    config_path: Path,
    history_path: Path,
    pool_end: Optional[np.ndarray] = None,
    u_pool_sft: Optional[np.ndarray] = None,
    u_pool_theo: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable theory-vs-SFT endpoint summary.

    :param info: Output of :func:`run_strategic_ascent`.
    :type info: dict
    :param sft_end: Final SFT positions, shape ``(N, L)``.
    :type sft_end: numpy.ndarray
    :param theo_end: Final theory positions used for comparison.
    :type theo_end: numpy.ndarray
    :param config_path: Source YAML config path.
    :type config_path: pathlib.Path
    :param history_path: Source ``history.json`` path.
    :type history_path: pathlib.Path
    :param pool_end: Optional matched-pool theory endpoints.
    :type pool_end: numpy.ndarray | None
    :param u_pool_sft: Optional pool utilities at SFT endpoints.
    :type u_pool_sft: numpy.ndarray | None
    :param u_pool_theo: Optional pool utilities at theory endpoints.
    :type u_pool_theo: numpy.ndarray | None
    :returns: Summary dict suitable for ``json.dump``.
    :rtype: dict
    """
    grad_end = info["positions"][-1]
    names = info["names"]
    return {
        "config": str(config_path),
        "history": str(history_path),
        "sigma": float(info["sigma"]),
        "sigma_star": float(info["sigma_star"]),
        "theory_converged": bool(info["converged"]),
        "theory_n_steps": int(info["n_steps"]),
        "agents": [
            {
                "name": name,
                "initial": info["initial_positions"][i].tolist(),
                "theory_start": info["theory_start_positions"][i].tolist(),
                "sft_start": info["sft_start_positions"][i].tolist(),
                "sft_end": sft_end[i].tolist(),
                "theory_end": theo_end[i].tolist(),
                "theory_grad_end": grad_end[i].tolist(),
                "theory_pool_end": (
                    pool_end[i].tolist() if pool_end is not None else None
                ),
                "gap": float(np.linalg.norm(sft_end[i] - theo_end[i])),
                "gap_pool": (
                    float(np.linalg.norm(sft_end[i] - pool_end[i]))
                    if pool_end is not None
                    else None
                ),
                "u_pool_sft": (
                    float(u_pool_sft[i]) if u_pool_sft is not None else None
                ),
                "u_pool_theory": (
                    float(u_pool_theo[i]) if u_pool_theo is not None else None
                ),
            }
            for i, name in enumerate(names)
        ],
    }
