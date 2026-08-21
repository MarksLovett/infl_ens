"""CLI: summarize pairs_near_theory sweep layouts per sigma."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from infl_ens.training.sweep_aggregate import summarize_pairs_near_theory_sweep

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    """Print per-sigma layout counts and compare to mean_noise baseline."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT / "results/pairs_near_theory_10seeds",
    )
    p.add_argument(
        "--baseline-root",
        type=Path,
        default=_REPO_ROOT / "results/pool_and_noise_10seeds",
        help="Optional mean_noise baseline for side-by-side counts.",
    )
    args = p.parse_args()

    if not args.root.is_dir():
        print(f"no sweep root at {args.root}", file=sys.stderr)
        return 1

    baseline = args.baseline_root if args.baseline_root.is_dir() else None
    summarize_pairs_near_theory_sweep(args.root, baseline_root=baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
