"""Single CLI entry point for the data submodule.

Per AGENTS.md §4 rule 1, every submodule has one CLI. For data, the CLI
dispatches between benchmark-specific preparation tasks based on the
config's ``task`` field:

- ``task: build_safety_trait_space`` — build the combined 2-D BeaverTails +
  HaluEval trait space and pickle it under
  ``results/<run_id>/trait_space.pkl``.
- ``task: preview`` — load a benchmark split and print summary stats. Used
  by the smoke tests in ``tests/test_benchmark_loaders.py``.

Launch with::

    python -m infl_ens.data benchmark=safety_truth

The actual benchmark-specific code lives in
:mod:`infl_ens.data.benchmarks`. This file is dispatch only.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from infl_ens.data.benchmarks import (
    BenchmarkSplit,
    build_safety_trait_space,
    load_beavertails,
    load_halueval,
)
from infl_ens.data.encoders import SentenceTransformerEncoder


def _print_split(split: BenchmarkSplit) -> None:
    """Print a one-line summary of a :class:`BenchmarkSplit`.

    :param split: The split to summarise.
    :type split: BenchmarkSplit
    """
    s = split.scores
    print(
        f"[{split.name:>14}] axis={split.axis_name:<14} "
        f"n={split.n:<6d} "
        f"score mean={s.mean():.3f} std={s.std():.3f} "
        f"pos={int((s >= 0.5).sum())} neg={int((s < 0.5).sum())}"
    )


def _cmd_preview(args: argparse.Namespace) -> int:
    """Print loader summaries without writing any artifacts.

    :param args: Parsed CLI arguments.
    :type args: argparse.Namespace
    :returns: Process exit code (0 on success).
    :rtype: int
    """
    if args.beavertails:
        _print_split(load_beavertails(args.beavertails, max_records=args.max_records))
    if args.halueval:
        _print_split(load_halueval(args.halueval, max_records=args.max_records))
    return 0


def _cmd_build_safety_trait_space(args: argparse.Namespace) -> int:
    """Build and pickle the 2-D BeaverTails + HaluEval trait space.

    :param args: Parsed CLI arguments.
    :type args: argparse.Namespace
    :returns: Process exit code.
    :rtype: int
    """
    encoder = SentenceTransformerEncoder(model_name=args.encoder)
    splits: list[BenchmarkSplit] = []
    if args.beavertails:
        splits.append(load_beavertails(args.beavertails, max_records=args.max_records))
    if args.halueval:
        splits.append(load_halueval(args.halueval, max_records=args.max_records))
    if not splits:
        print("error: must provide at least one of --beavertails / --halueval",
              file=sys.stderr)
        return 2

    space = build_safety_trait_space(
        splits, encoder, n_grid=args.n_grid, kde_bandwidth=args.kde_bandwidth,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(
            {
                "grid": space.grid,
                "weights": space.weights,
                "axis_labels": space.axis_labels,
                "encoder_model_name": args.encoder,
            },
            fh,
        )
    print(f"wrote trait space to {out}  L={space.L} K={space.K}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser.

    :returns: Configured argument parser.
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        prog="python -m infl_ens.data",
        description="Data subpackage CLI (preview / build trait space).",
    )
    sub = p.add_subparsers(dest="task", required=True)

    pv = sub.add_parser("preview", help="Print summary stats for benchmark splits.")
    pv.add_argument("--beavertails", type=str, default=None)
    pv.add_argument("--halueval", type=str, default=None)
    pv.add_argument("--max-records", type=int, default=None)
    pv.set_defaults(func=_cmd_preview)

    bld = sub.add_parser(
        "build-safety-trait-space",
        help="Build the combined BeaverTails + HaluEval trait space.",
    )
    bld.add_argument("--beavertails", type=str, default=None,
                     help="Path to BeaverTails JSONL file or directory.")
    bld.add_argument("--halueval", type=str, default=None,
                     help="Path to HaluEval task file or directory.")
    bld.add_argument("--encoder", type=str,
                     default="sentence-transformers/all-MiniLM-L6-v2")
    bld.add_argument("--n-grid", type=int, default=32)
    bld.add_argument("--kde-bandwidth", type=float, default=None)
    bld.add_argument("--max-records", type=int, default=5000)
    bld.add_argument("--output", type=str, required=True,
                     help="Output .pkl path (e.g. results/<run>/trait_space.pkl).")
    bld.set_defaults(func=_cmd_build_safety_trait_space)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m infl_ens.data``.

    :param argv: Argument vector (defaults to ``sys.argv[1:]``).
    :type argv: Sequence[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
