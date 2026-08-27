"""Single CLI entry point for adapter evaluation on benchmarks.

Run with::

    python -m infl_ens.evaluation --config configs/arms/soft_topk3_pairs.yaml
    python -m infl_ens.evaluation --config <run>/resolved_config.yaml -- \\
        eval.partitions='["val"]' eval.rounds='[4,5,6]'

A closed-loop **training** YAML that carries a top-level ``eval`` block is
accepted as-is: the run directory, base model, benchmarks and split
manifest are read from the training blocks and each ``eval.partitions``
entry is scored into ``<output_dir>/eval_<partition>/``.  See
:func:`infl_ens.evaluation.evaluate.run_unified_eval`.

A standalone job (``task: adapter_eval`` or ``task: run_eval``) is still
accepted for ad-hoc scoring of one adapter directory; see
:class:`infl_ens.evaluation.evaluate.EvalJobConfig`.

Config loading (``includes:`` composition and ``KEY=VAL`` overrides after
``--``) is shared with every other CLI through :mod:`infl_ens.config`.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from infl_ens.config import ConfigError, load_config
from infl_ens.evaluation.evaluate import (
    EvalJobConfig,
    is_unified_config,
    run_eval_job,
    run_unified_eval,
)

_TASKS = frozenset({"adapter_eval", "run_eval"})


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the configured evaluation.

    :param argv: Optional argument vector (defaults to ``sys.argv[1:]``).
    :type argv: Sequence[str] | None
    :returns: Process exit code (0 on success, 2 on a config error).
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Score saved LoRA adapters on the configured benchmarks.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Training YAML with an eval block, or a standalone eval job.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional KEY=VAL config overrides.",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, args.overrides, validate=False)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if is_unified_config(cfg):
        reports = run_unified_eval(cfg)
        for path in reports:
            print(f"wrote {path}")
        return 0
    job = EvalJobConfig.from_mapping(cfg)
    if job.task not in _TASKS:
        print(
            f"error: unknown task {job.task!r}; expected one of {sorted(_TASKS)}",
            file=sys.stderr,
        )
        return 2

    results = run_eval_job(job)
    print(f"wrote {len(results)} result(s) to {job.output_dir}/eval_results.json")
    for r in results:
        label = r.benchmark
        if r.agent:
            label += f" [{r.agent}"
            if r.round is not None:
                label += f" r{r.round}"
            label += "]"
        print(
            f"  {label}: mean_nll={r.mean_nll:.4f} "
            f"({r.n_examples} examples, {r.n_tokens} tokens)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
