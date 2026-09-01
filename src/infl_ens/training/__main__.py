"""Single CLI entry point for all training in ``infl_ens``.

Per AGENTS.md rule 1 every training run goes through this one command,
driven by a YAML config (with ``includes:`` composition and ``KEY=VAL``
overrides after ``--``; see :mod:`infl_ens.config`)::

    python -m infl_ens.training --config configs/arms/soft_topk3_pairs.yaml
    python -m infl_ens.training --config configs/arms/hard_pairs_matched.yaml -- \\
        closed_loop.n_rounds=2 data_split=null

The ``task`` field selects the runner from
:data:`infl_ens.training.tasks.TASKS` (``closed_loop`` or
``baseline_replay``).  A closed-loop config with an ``eval`` block also
scores the trained adapters when training finishes.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from infl_ens.config import ConfigError, load_config
from infl_ens.training.tasks import TASKS


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m infl_ens.training``.

    :param argv: Argument vector (defaults to ``sys.argv[1:]``).
    :type argv: Sequence[str] | None
    :returns: Process exit code: 0 on success, 2 on a config error or an
        unknown task.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Run one infl_ens training task from a YAML config.",
    )
    parser.add_argument("--config", required=True, help="Path to the run config.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional KEY=VAL overrides (after --).",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, args.overrides)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    task = cfg.get("task")
    if task not in TASKS:
        print(
            f"error: unknown task {task!r}; expected one of {sorted(TASKS)}",
            file=sys.stderr,
        )
        return 2
    return int(TASKS[task](cfg))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
