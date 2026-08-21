"""CLI: pairwise trait-space resource heatmaps from a router YAML config.

Run with::

    python scripts/plot_benchmark_space_heatmaps.py \\
        --config configs/benchmark/router/seven_axis_safety.yaml \\
        --positions-json results/seven_axis_theory_n14/positions.json \\
        --output-stem scripts/figures/seven_axis_safety_resource
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
FIGS_DIR = ROOT / "scripts" / "figures"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infl_ens.data.splits import load_split_manifest  # noqa: E402
from infl_ens.data.trait_space_cache import build_or_load_safety_trait_space  # noqa: E402
from infl_ens.evaluation.benchmarks import load_benchmark_splits  # noqa: E402
from infl_ens.vis.benchmark_space import plot_pairwise_heatmaps  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _apply_overrides(cfg: dict[str, Any], overrides: Sequence[str]) -> None:
    """Apply dotted ``KEY=VAL`` overrides to a parsed config in place."""
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        path = key.split(".")
        node = cfg
        for p in path[:-1]:
            node = node.setdefault(p, {})
        try:
            node[path[-1]] = json.loads(val)
        except json.JSONDecodeError:
            node[path[-1]] = val


def _load_positions(path: Optional[Path]) -> dict[str, np.ndarray]:
    """Load trained router positions from ``positions.json``."""
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: np.asarray(pos, dtype=float)
        for name, pos in payload.get("positions", {}).items()
    }


def _load_trajectories(path: Optional[Path]) -> dict[str, np.ndarray]:
    """Load per-agent position trajectories from a history file."""
    if path is None:
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"{path} is empty")
    if "positions" not in records[0]:
        raise ValueError(f"{path} is missing positions")
    names = list(records[0]["positions"].keys())
    return {
        name: np.stack(
            [np.asarray(record["positions"][name], dtype=float) for record in records],
            axis=0,
        )
        for name in names
    }


def _build_partition_labels(
    splits: Sequence[Any],
    manifest_path: Path,
) -> np.ndarray:
    """Build per-prompt partition ids (0=train, 1=val, 2=test)."""
    manifest = load_split_manifest(manifest_path)
    labels: list[int] = []
    for split in splits:
        part = manifest.partition_for(split.name)
        arr = np.full(split.n, -1, dtype=int)
        for part_id, idx in ((0, part.train), (1, part.val), (2, part.test)):
            for i in idx:
                arr[int(i)] = part_id
        if np.any(arr < 0):
            raise ValueError(f"partition labels incomplete for {split.name}")
        labels.extend(arr.tolist())
    return np.asarray(labels, dtype=int)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot pairwise heatmaps of a benchmark trait-space resource distribution.",
    )
    p.add_argument("--config", type=Path, required=True, help="Router YAML config.")
    p.add_argument("--positions-json", type=Path, default=None)
    p.add_argument("--history-json", type=Path, default=None)
    p.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Color prompt scatter by train/val/test partition slices.",
    )
    p.add_argument(
        "--output-stem",
        type=Path,
        default=FIGS_DIR / "benchmark_space_heatmaps",
    )
    p.add_argument("--max-scatter", type=int, default=1500)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--config-override", action="append", default=[])
    p.add_argument("--density-mode", choices=("kde", "empirical"), default="kde")
    p.add_argument("--pairwise-bins", type=int, default=32)
    p.add_argument(
        "--mass-norm",
        choices=(
            "global", "per_panel", "per_panel_sqrt",
            "per_panel_log", "per_panel_power",
        ),
        default="per_panel_power",
    )
    p.add_argument("--scatter-alpha", type=float, default=0.10)
    p.add_argument("--smooth-sigma", type=float, default=2.0)
    p.add_argument("--vmax-percentile", type=float, default=92.0)
    p.add_argument("--dpi", type=int, default=220)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = _load_yaml(args.config)
    _apply_overrides(cfg, args.config_override)
    splits = load_benchmark_splits(cfg.get("benchmarks", []))
    space = build_or_load_safety_trait_space(cfg, splits)
    labels = tuple(space.axis_labels or tuple(f"axis {i}" for i in range(space.L)))
    prompts = [p for split in splits for p in split.prompts]
    prompt_coords = space.project(prompts) if prompts else None
    partition_labels = None
    if args.split_manifest is not None:
        partition_labels = _build_partition_labels(splits, args.split_manifest)
        if prompt_coords is not None and partition_labels.shape[0] != prompt_coords.shape[0]:
            raise ValueError("partition label count does not match projected prompts")

    fig = plot_pairwise_heatmaps(
        space.grid,
        space.weights,
        axis_labels=labels,
        prompt_coords=prompt_coords,
        positions=_load_positions(args.positions_json),
        trajectories=_load_trajectories(args.history_json),
        partition_labels=partition_labels,
        max_scatter=int(args.max_scatter),
        title=args.title or "Benchmark-space resource distribution",
        density_mode=str(args.density_mode),
        pairwise_bins=int(args.pairwise_bins),
        mass_norm=str(args.mass_norm),
        scatter_alpha=float(args.scatter_alpha),
        smooth_sigma=float(args.smooth_sigma),
        vmax_percentile=float(args.vmax_percentile),
        output_stem=args.output_stem,
        save_dpi=int(args.dpi),
    )
    for ext in ("png", "pdf"):
        print(f"wrote {args.output_stem.with_suffix(f'.{ext}')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
