"""Export an aggregated eval report as copy-pasteable matrices.

Reads ``eval_aggregate.json`` (from
:func:`infl_ens.evaluation.aggregate.write_aggregated_eval_report`) and
writes CSV, Markdown, LaTeX, and JSON matrix files.

Example::

    python scripts/export_eval_matrix.py \\
        --input results/eval_pairs_near_eq_r40_final_aggregate/eval_aggregate.json \\
        --output-dir results/eval_pairs_near_eq_r40_final_aggregate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from infl_ens.evaluation.aggregate import (
    build_eval_matrix,
    format_eval_matrix_markdown,
    load_aggregated_report,
    write_eval_matrix_outputs,
)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    :param argv: Optional argument vector.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Export eval_aggregate.json as agent×benchmark matrices.",
    )
    parser.add_argument(
        "--input",
        default="results/eval_pairs_near_eq_r40_final_aggregate/eval_aggregate.json",
        help="Path to eval_aggregate.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for matrix files (default: same as input parent).",
    )
    parser.add_argument(
        "--stem",
        default="eval_matrix",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--mean-only",
        action="store_true",
        help="Omit ± std in formatted cells (means-only matrix).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_md",
        help="Also print the Markdown table to stdout.",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"error: {in_path} not found", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else in_path.parent
    _, metrics = load_aggregated_report(in_path)
    matrix = build_eval_matrix(metrics)
    include_std = not args.mean_only

    paths = write_eval_matrix_outputs(
        matrix,
        out_dir,
        stem=args.stem,
        include_std=include_std,
    )

    if args.print_md:
        print(format_eval_matrix_markdown(matrix, include_std=include_std))

    for fmt, path in sorted(paths.items()):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
