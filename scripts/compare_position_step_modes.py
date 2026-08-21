"""CLI: compare position-step policies on stability-test sweep results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from infl_ens.training.position_stability import run_position_step_modes_comparison


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--figure-root", type=Path, required=True)
    args = p.parse_args(argv)

    rows = run_position_step_modes_comparison(args.root, args.figure_root)
    if not rows:
        print(f"no runs under {args.root}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
