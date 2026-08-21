"""CLI: overlay two closed-loop trajectories on a single trait-space plot.

Run with::

    python scripts/compare_runs.py \\
        --config configs/benchmark/router/<config>.yaml \\
        --history-a results/test_position_only_r5/history.json \\
        --label-a "post-fix r5" \\
        --history-b results/position_only_long_round_sweep/r20/history.json \\
        --label-b "pre-fix r20" \\
        --axis-labels harm hallucination \\
        --title "position_only bug fix smoke test" \\
        --output-stem scripts/figures/test_position_only_r5/compare_runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from infl_ens.data.benchmarks import (  # noqa: E402
    BenchmarkSplit,
    build_safety_trait_space,
    load_beavertails,
    load_halueval,
)
from infl_ens.data.encoders import SentenceTransformerEncoder  # noqa: E402
from infl_ens.data.trait_space import TraitSpace  # noqa: E402
from infl_ens.inflgame.router import RouterAgent  # noqa: E402
from infl_ens.training.router_training import (  # noqa: E402
    RouterTrainingConfig,
    train_router_positions,
)
from infl_ens.utils.resource import gaussian_stability_threshold  # noqa: E402
from infl_ens.vis.closed_loop import plot_trajectory_overlay  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file via PyYAML.

    :param path: Path to the config file.
    :type path: pathlib.Path
    :returns: Parsed configuration mapping.
    :rtype: dict
    """
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_splits(cfg: dict[str, Any]) -> list[BenchmarkSplit]:
    """Load all benchmark splits referenced by ``cfg['benchmarks']``.

    :param cfg: Parsed YAML config.
    :type cfg: dict
    :returns: List of loaded benchmark splits.
    :rtype: list[BenchmarkSplit]
    """
    splits: list[BenchmarkSplit] = []
    for entry in cfg.get("benchmarks", []):
        kind = entry["kind"]
        path = entry["path"]
        max_records = entry.get("max_records")
        if kind == "beavertails":
            splits.append(load_beavertails(path, max_records=max_records))
        elif kind == "halueval":
            splits.append(load_halueval(
                path, tasks=entry.get("tasks"), max_records=max_records,
            ))
        else:  # pragma: no cover
            raise ValueError(f"unknown benchmark kind {kind!r}")
    return splits


def _build_trait_space(cfg: dict[str, Any], splits: list[BenchmarkSplit]) -> TraitSpace:
    """Reconstruct the closed-loop's trait space from its YAML config.

    :param cfg: Parsed YAML config.
    :type cfg: dict
    :param splits: Loaded benchmark splits.
    :type splits: list[BenchmarkSplit]
    :returns: Trait space matching the config.
    :rtype: TraitSpace
    """
    ts_cfg = cfg.get("trait_space", {})
    encoder = SentenceTransformerEncoder(
        model_name=ts_cfg.get(
            "encoder", "sentence-transformers/all-MiniLM-L6-v2",
        ),
    )
    return build_safety_trait_space(
        splits, encoder,
        n_grid=int(ts_cfg.get("n_grid", 32)),
        kde_bandwidth=ts_cfg.get("kde_bandwidth"),
        threshold=float(ts_cfg.get("threshold", 0.5)),
    )


def _resolve_sigma(
    cfg: dict[str, Any], n_agents: int, space: TraitSpace,
) -> float:
    """Resolve :math:`\\sigma` from the config.

    :param cfg: Parsed YAML config.
    :type cfg: dict
    :param n_agents: Number of agents.
    :type n_agents: int
    :param space: Trait space.
    :type space: TraitSpace
    :returns: Competitive reach.
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


def _trajectories(history: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Extract per-agent position trajectories from ``history.json``.

    :param history: Loaded ``history.json``.
    :type history: list[dict]
    :returns: Mapping ``{agent_name: positions}`` where ``positions``
        has shape ``(R, L)``.
    :rtype: dict[str, numpy.ndarray]
    """
    names = list(history[0]["positions"].keys())
    out: dict[str, np.ndarray] = {}
    for name in names:
        out[name] = np.array(
            [h["positions"][name] for h in history], dtype=float,
        )
    return out


def _theory_ne(
    history: list[dict[str, Any]], space: TraitSpace,
    sigma: float, seed: int, n_steps: int, lr: float,
) -> dict[str, np.ndarray]:
    """Re-run ``train_router_positions`` from the history's round-0 positions.

    :param history: Loaded ``history.json``.
    :type history: list[dict]
    :param space: Trait space.
    :type space: TraitSpace
    :param sigma: Competitive reach.
    :type sigma: float
    :param seed: RNG seed.
    :type seed: int
    :param n_steps: Gradient-ascent steps.
    :type n_steps: int
    :param lr: Learning rate.
    :type lr: float
    :returns: Mapping ``{agent_name: theory_ne_position}``.
    :rtype: dict[str, numpy.ndarray]
    """
    agents = [
        RouterAgent(name=name, position=np.asarray(pos, dtype=float))
        for name, pos in history[0]["positions"].items()
    ]
    rt_cfg = RouterTrainingConfig(
        sigma=sigma, learning_rate=lr, n_steps=n_steps,
        tol=1e-8, clip_to_box=True,
    )
    train_router_positions(space, agents, rt_cfg, seed=seed)
    return {a.name: a.position.copy() for a in agents}


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Optional argument vector; defaults to ``sys.argv[1:]``.
    :type argv: list[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description=(
            "Overlay two closed-loop trajectories on a single trait-space "
            "plot, with theory NE markers, to verify whether a code change "
            "between the two runs affected the position dynamics."
        ),
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="YAML config; both runs assumed to share it.")
    parser.add_argument("--history-a", required=True, type=Path,
                        help="First history.json (e.g. the new / post-fix run).")
    parser.add_argument("--label-a", default="run A",
                        help="Legend label for history A.")
    parser.add_argument("--history-b", required=True, type=Path,
                        help="Second history.json (e.g. the old / pre-fix run).")
    parser.add_argument("--label-b", default="run B",
                        help="Legend label for history B.")
    parser.add_argument("--output-stem", required=True,
                        help="Output stem; <stem>.{pdf,png} are written.")
    parser.add_argument("--axis-labels", nargs=2, default=["dim 0", "dim 1"])
    parser.add_argument("--title", default="")
    parser.add_argument("--theory-steps", type=int, default=5000)
    parser.add_argument("--theory-lr", type=float, default=5e-3)
    args = parser.parse_args(argv)

    cfg = _load_yaml(args.config)
    with args.history_a.open("r", encoding="utf-8") as fh:
        hist_a = json.load(fh)
    with args.history_b.open("r", encoding="utf-8") as fh:
        hist_b = json.load(fh)

    splits = _load_splits(cfg)
    space = _build_trait_space(cfg, splits)
    n_agents = len(hist_a[0]["positions"])
    sigma = _resolve_sigma(cfg, n_agents, space)
    s0 = gaussian_stability_threshold(n_agents, space.grid, space.weights)
    print(f"sigma = {sigma:.4f}  (σ₀* = {s0:.4f})")

    print("extracting trajectories ...")
    traj_a = _trajectories(hist_a)
    traj_b = _trajectories(hist_b)
    print(f"  A: {len(hist_a)} rounds, {len(traj_a)} agents")
    print(f"  B: {len(hist_b)} rounds, {len(traj_b)} agents")

    print(f"computing theory NE (sigma = {sigma:.4f}, "
          f"{args.theory_steps} steps) ...")
    theory_a = _theory_ne(
        hist_a, space, sigma,
        seed=int(cfg.get("seed", 0)),
        n_steps=args.theory_steps, lr=args.theory_lr,
    )

    summary: dict[str, dict[str, float]] = {}
    for name in traj_a:
        a_end = traj_a[name][-1]
        b_end = traj_b[name][-1] if name in traj_b else None
        ne = theory_a[name]
        summary[name] = {
            "a_end_l2_to_theory": float(np.linalg.norm(a_end - ne)),
            "b_end_l2_to_theory": (
                float(np.linalg.norm(b_end - ne))
                if b_end is not None else float("nan")
            ),
            "a_b_l2": (
                float(np.linalg.norm(a_end - b_end))
                if b_end is not None else float("nan")
            ),
        }

    plot_trajectory_overlay(
        resource_mean=space.mean,
        traj_a=traj_a,
        traj_b=traj_b,
        theory_a=theory_a,
        label_a=args.label_a,
        label_b=args.label_b,
        axis_labels=(args.axis_labels[0], args.axis_labels[1]),
        sigma=sigma,
        sigma_star=s0,
        title=args.title,
        output_stem=Path(args.output_stem),
    )
    print(f"wrote {args.output_stem}.{{pdf,png}}")

    print()
    print(f"{'agent':<10} "
          f"{args.label_a + ' end → NE':>22} "
          f"{args.label_b + ' end → NE':>22} "
          f"{'A end ↔ B end':>17}")
    for name, s in summary.items():
        print(
            f"{name:<10} "
            f"{s['a_end_l2_to_theory']:>22.4f} "
            f"{s['b_end_l2_to_theory']:>22.4f} "
            f"{s['a_b_l2']:>17.4f}"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
