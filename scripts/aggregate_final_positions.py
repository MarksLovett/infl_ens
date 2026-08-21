"""CLI: mean and variance of final trait positions across seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from infl_ens.training.sweep_aggregate import (
    aggregate_final_positions,
    print_final_positions_report,
)
from infl_ens.utils.sweep_discovery import discover_sigma_seed_history_paths


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument(
        "--axis-labels", nargs=2, default=["harm", "hallucination"],
    )
    args = p.parse_args(argv)

    entries = discover_sigma_seed_history_paths(args.root)
    if not entries:
        print(f"no histories under {args.root}", file=sys.stderr)
        return 1

    stats = aggregate_final_positions(entries)
    print_final_positions_report(stats, (args.axis_labels[0], args.axis_labels[1]))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
