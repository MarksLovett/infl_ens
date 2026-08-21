"""CLI: aggregate a flat sweep of closed-loop runs into one figure + CSV.

Run with::

    python scripts/plot_sweep.py \\
        --root results/sweep_seeds \\
        --mode seeds \\
        --output-stem scripts/figures/sweep_seeds \\
        --with-theory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = ROOT / "scripts" / "figures"

from infl_ens.training.sweep_aggregate import (  # noqa: E402
    summarise_flat_sweep_run,
    write_flat_sweep_csv,
)
from infl_ens.utils.sweep_discovery import discover_flat_sweep_runs  # noqa: E402
from infl_ens.vis.sweeps import plot_sweep_grid  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser.

    :returns: Configured parser.
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="Aggregate a sweep of closed-loop runs into one figure + CSV.",
    )
    p.add_argument("--root", type=Path, required=True,
                   help="Sweep root directory (e.g. results/sweep_seeds).")
    p.add_argument("--mode", choices=["seeds", "sigma", "kde"], required=True)
    p.add_argument("--cluster-threshold", type=float, default=0.1,
                   help="L2 distance below which two positions are considered "
                        "in the same cluster.")
    p.add_argument("--axis-labels", nargs=2, default=["harm", "hallucination"])
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--with-theory", action="store_true",
                   help="Overlay theoretical NE endpoints from theory_vs_sft.json.")
    p.add_argument("--output-stem", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None,
                   help="Optional CSV summary path. Defaults to "
                        "<output-stem>.csv.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional argument vector.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    args = _build_parser().parse_args(argv)
    runs = discover_flat_sweep_runs(args.root, args.mode)
    if not runs:
        print(f"no runs found under {args.root} matching mode={args.mode}",
              file=sys.stderr)
        return 1
    print(f"discovered {len(runs)} runs in {args.root}:")
    for r in runs:
        print(f"  {r['slug']:<14}  value={r['value']:<8}  "
              f"theory={'yes' if r['theory_path'] else 'no'}")

    summaries = [
        summarise_flat_sweep_run(r, cluster_threshold=args.cluster_threshold)
        for r in runs
    ]

    print(f"\n{'slug':<14} {'value':>8} {'SFT eq':<14} {'theory eq':<14}")
    print("-" * 60)
    for s in summaries:
        sft_eq = "(" + ", ".join(str(x) for x in s["equilibrium_type"]) + ")"
        theo_eq = (("(" + ", ".join(str(x) for x in s["theory_eq_type"]) + ")")
                   if s["theory_eq_type"] is not None else "—")
        print(f"{s['slug']:<14} {s['value']:>8} {sft_eq:<14} {theo_eq:<14}")

    stem = args.output_stem or (FIGS_DIR / f"sweep_{args.mode}")
    plot_sweep_grid(
        summaries,
        mode=args.mode,
        axis_labels=(args.axis_labels[0], args.axis_labels[1]),
        title=args.title,
        with_theory=args.with_theory,
        output_stem=stem,
    )
    print(f"\nwrote {stem.with_suffix('.pdf')}")
    print(f"wrote {stem.with_suffix('.png')}")

    csv_path = args.csv or stem.with_suffix(".csv")
    write_flat_sweep_csv(summaries, csv_path)
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
