"""CLI: aggregate seed × sigma closed-loop sweeps into mean ± std figures.

Run after :file:`scripts/run_position_only_seed_sigma_sweep.sh` or any
compatible launcher that uses the same directory naming.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from infl_ens.training.sweep_aggregate import aggregate_group_seed_sweep  # noqa: E402
from infl_ens.utils.sweep_discovery import discover_group_seed_runs  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    """CLI parser.

    :returns: Configured parser.
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="Aggregate seed×sigma sweep runs into mean±std figures.",
    )
    p.add_argument(
        "--root", type=Path, required=True,
        help="Sweep results root (sigma*/seed* or r*/seed*).",
    )
    p.add_argument(
        "--layout", choices=("auto", "sigma_seed", "round_seed"), default="auto",
        help="Directory layout under --root.",
    )
    p.add_argument(
        "--figure-root", type=Path, required=True,
        help="Figure tree for this sweep (writes aggregate/ subfolder).",
    )
    p.add_argument("--config", type=Path, default=None, help="Unused; reserved.")
    p.add_argument(
        "--axis-labels", nargs=2, default=["harm", "hallucination"],
        help="Trait-space axis labels.",
    )
    p.add_argument("--title", type=str, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional CLI args.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    args = _build_parser().parse_args(argv)
    cells = discover_group_seed_runs(args.root, layout=args.layout)
    if not cells:
        print(f"no runs found under {args.root}", file=sys.stderr)
        return 1

    kind = cells[0].group_kind
    n_group = len({c.group_slug for c in cells})
    n_seed = len({c.seed for c in cells})
    print(f"found {len(cells)} cells ({n_group} {kind} groups × up to {n_seed} seeds)")

    aggregate_group_seed_sweep(
        cells,
        args.figure_root,
        axis_labels=(args.axis_labels[0], args.axis_labels[1]),
        title=args.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
