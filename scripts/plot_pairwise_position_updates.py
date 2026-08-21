"""CLI: pairwise position trajectories and update magnitudes for L-D runs.

Run with::

    python scripts/plot_pairwise_position_updates.py \\
        --history results/seven_axis_pair_merge_r40/seed0/history.json \\
        --axis-labels harm hallucination jailbreak privacy overrefusal injection policy_violation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = ROOT / "scripts" / "figures"

from infl_ens.vis.closed_loop import plot_pairwise_position_updates  # noqa: E402


def _load_history(path: Path) -> list[dict]:
    """Load a closed-loop history file."""
    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"{path} is empty")
    if "positions" not in records[0]:
        raise ValueError(f"{path} is missing positions")
    return records


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot pairwise position trajectories and update magnitudes.",
    )
    p.add_argument("--history", type=Path, required=True)
    p.add_argument(
        "--axis-labels",
        nargs="+",
        default=["harm", "hallucination", "privacy"],
    )
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--output-stem", type=Path, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    records = _load_history(args.history)
    stem = args.output_stem or (FIGS_DIR / "pairwise_position_updates")
    plot_pairwise_position_updates(
        records,
        axis_labels=args.axis_labels,
        title=args.title,
        output_stem=stem,
    )
    print(f"wrote {stem.with_suffix('.pdf')}")
    print(f"wrote {stem.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
