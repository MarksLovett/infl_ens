"""Aggregate per-seed ``eval_results.json`` files into mean ± std over seeds.

Example (pairs_near_eq r40 final-round evals)::

    python scripts/aggregate_eval_across_seeds.py \\
        --glob 'results/eval_pairs_near_eq_r40_seed*_final/eval_results.json' \\
        --output-dir results/eval_pairs_near_eq_r40_final_aggregate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from infl_ens.evaluation.aggregate import (
    aggregate_eval_across_seeds,
    build_eval_matrix,
    load_eval_report,
    write_aggregated_eval_report,
    write_eval_matrix_outputs,
)

_SEED_IN_PATH = re.compile(r"seed(\d+)", re.IGNORECASE)


def _seed_from_path(path: Path) -> int | None:
    """Extract seed index from a path containing ``seed<N>``.

    :param path: Report path.
    :type path: pathlib.Path
    :returns: Seed number or ``None``.
    :rtype: int | None
    """
    m = _SEED_IN_PATH.search(str(path))
    return int(m.group(1)) if m else None


def _discover_from_glob(pattern: str) -> dict[int, Path]:
    """Build ``seed -> report`` from a glob pattern.

    :param pattern: Glob passed to :func:`pathlib.Path.glob`.
    :type pattern: str
    :returns: Discovered reports.
    :rtype: dict[int, pathlib.Path]
    """
    root = Path(".")
    found: dict[int, Path] = {}
    for p in sorted(root.glob(pattern)):
        if not p.is_file():
            continue
        seed = _seed_from_path(p)
        if seed is None:
            print(f"warning: cannot parse seed from {p}", file=sys.stderr)
            continue
        found[seed] = p.resolve()
    return found


def _print_table(metrics: list) -> None:
    """Print a human-readable comparison table.

    :param metrics: Aggregated metric records.
    :type metrics: list
    """
    benches = sorted({m.benchmark for m in metrics})
    agents = sorted({m.agent for m in metrics})
    for bench in benches:
        print(f"\n=== {bench} ===")
        print(f"{'agent':<10} {'mean_nll':>10} {'std':>8} {'n':>4}")
        for agent in agents:
            cell = next(
                (m for m in metrics if m.agent == agent and m.benchmark == bench),
                None,
            )
            if cell is None:
                continue
            print(
                f"{cell.agent:<10} {cell.mean_nll:10.4f} "
                f"{cell.std_nll:8.4f} {cell.n_seeds:4d}"
            )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    :param argv: Optional argument vector.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Average eval_results.json across seeds.",
    )
    parser.add_argument(
        "--glob",
        default="results/eval_pairs_near_eq_r40_seed*_final/eval_results.json",
        help="Glob for per-seed eval_results.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/eval_pairs_near_eq_r40_final_aggregate",
        help="Where to write eval_aggregate.json.",
    )
    parser.add_argument(
        "--write-matrix",
        action="store_true",
        help="Also write eval_matrix.{csv,md,tex,json} under output-dir.",
    )
    args = parser.parse_args(argv)

    reports = _discover_from_glob(args.glob)
    if not reports:
        print(f"error: no reports matched {args.glob!r}", file=sys.stderr)
        return 1

    print(f"found {len(reports)} seeds: {sorted(reports)}")
    metrics = aggregate_eval_across_seeds(reports)
    if not metrics:
        print("error: no complete (agent, benchmark) cells across all seeds",
              file=sys.stderr)
        return 1

    meta = {
        "n_seeds": len(reports),
        "seeds": sorted(reports),
        "source_glob": args.glob,
    }
    out_path = write_aggregated_eval_report(
        metrics, args.output_dir, meta=meta,
    )
    _print_table(metrics)
    print(f"\nwrote {out_path}")
    if args.write_matrix:
        matrix = build_eval_matrix(metrics)
        mpaths = write_eval_matrix_outputs(matrix, args.output_dir)
        for fmt, mpath in sorted(mpaths.items()):
            print(f"wrote {mpath}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
