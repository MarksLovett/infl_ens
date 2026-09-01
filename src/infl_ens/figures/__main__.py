"""Render the figures of an experiment.

Run with::

    python -m infl_ens.figures --config configs/experiments/seven_axis_3arm.yaml
    python -m infl_ens.figures --config ... --only pair_positions,within_pair
    python -m infl_ens.figures --config ... --list
    python -m infl_ens.figures --config ... --only trait_representation --gpu

Figures are written under the experiment's ``figures_dir``
(``figures/<experiment>/`` by default).  A figure whose inputs have not
been produced yet is skipped with a warning.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from infl_ens.config import ConfigError
from infl_ens.experiment import load_experiment
from infl_ens.figures.render import FIGURES, render_all


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m infl_ens.figures``.

    :param argv: Argument vector (defaults to ``sys.argv[1:]``).
    :type argv: Sequence[str] | None
    :returns: Exit code (0 on success, 2 on a config error).
    :rtype: int
    """
    parser = argparse.ArgumentParser(description="Render the figures of an experiment.")
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument("--only", default=None, help="Comma-separated figure names.")
    parser.add_argument("--figures-dir", type=Path, default=None, help="Override figures_dir.")
    parser.add_argument("--gpu", action="store_true", help="Also render encoder-backed figures.")
    parser.add_argument("--list", action="store_true", help="List figure names and exit.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.list:
        for name, spec in FIGURES.items():
            flag = " (gpu)" if spec.requires_gpu else ""
            print(f"{name:<22} {spec.description}{flag}")
        return 0
    try:
        exp = load_experiment(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    only = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else None
    try:
        written = render_all(exp, only=only, figures_dir=args.figures_dir, include_gpu=args.gpu)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    n = sum(len(v) for v in written.values())
    print(f"rendered {n} file(s) under {args.figures_dir or exp.figures_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
