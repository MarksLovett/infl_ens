"""CLI: compare large-batch / static pool centroids vs small-batch simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from infl_ens.training.position_stability import run_batch_size_static_comparison


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--figure-root", type=Path, required=True)
    p.add_argument(
        "--reference-root", type=Path, default=None,
        help="Optional prior sweep, e.g. position_step_stability_test/mode_static",
    )
    args = p.parse_args(argv)

    rows = run_batch_size_static_comparison(
        args.root,
        args.figure_root,
        reference_root=args.reference_root,
    )
    if not rows:
        print(f"no results under {args.root}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
