"""Run an experiment end to end.

Run from the repository root::

    python -m infl_ens.pipeline --config configs/experiments/seven_axis_3arm.yaml
    python -m infl_ens.pipeline --config ... --stages routing,figures
    python -m infl_ens.pipeline --config ... --only-arm soft --stages train
    python -m infl_ens.pipeline --config ... --smoke
    python -m infl_ens.pipeline --config ... --dry-run

``--dry-run`` resolves and validates every config, prints the stage plan
with each arm's task, output directory and trait-space cache fingerprint,
and exits without touching the GPU.  Progress is logged to stdout and to
``<results_dir>/pipeline.log``; ``<results_dir>/stage_status.json`` records
where a run got to.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from infl_ens.config import ConfigError
from infl_ens.experiment import ALL_STAGES, ExperimentConfig, load_experiment
from infl_ens.pipeline.stages import PipelineContext, run_pipeline, run_smoke


def describe(exp: ExperimentConfig, stages: Sequence[str]) -> str:
    """Human-readable plan for ``--dry-run``.

    :param exp: Parsed experiment.
    :type exp: ExperimentConfig
    :param stages: Stages that would run.
    :type stages: Sequence[str]
    :returns: Multi-line description.
    :rtype: str
    """
    from infl_ens.data.trait_space_cache import trait_space_fingerprint

    lines = [
        f"experiment {exp.name} ({exp.path})",
        f"  results_dir: {exp.results_dir}",
        f"  figures_dir: {exp.figures_dir}",
        f"  stages:      {', '.join(stages)}",
        f"  eval:        rounds={list(exp.eval.perround_rounds)} "
        f"perround={exp.eval.perround_partition} routing={exp.eval.routing_partition} "
        f"max_eval_records={exp.eval.max_eval_records}",
        f"  figures:     {', '.join(exp.figures.include) or '(all cpu figures)'}",
        "  arms:",
    ]
    for arm in exp.arms:
        cfg = arm.load()
        lines.append(
            f"    - {arm.name:<12} {arm.role:<10} task={cfg.get('task'):<15} "
            f"fingerprint={trait_space_fingerprint(cfg)}  -> {arm.output_dir}"
        )
    return "\n".join(lines)


def _configure_logging(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(results_dir / "pipeline.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m infl_ens.pipeline``.

    :param argv: Argument vector (defaults to ``sys.argv[1:]``).
    :type argv: Sequence[str] | None
    :returns: Exit code: 0 on success, 1 on a stage failure, 2 on a
        config error.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description="Run an infl_ens experiment end to end.")
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument(
        "--stages", default=None,
        help=f"Comma-separated subset of {','.join(ALL_STAGES)} (default: the experiment's list).",
    )
    parser.add_argument("--only-arm", action="append", default=[], help="Restrict per-arm stages.")
    parser.add_argument("--force", action="store_true", help="Re-run stages whose outputs exist.")
    parser.add_argument("--smoke", action="store_true", help="Run the smoke gate instead of the stages.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs and print the plan.")
    args = parser.parse_args(argv)

    try:
        exp = load_experiment(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    stages = (
        [s.strip() for s in args.stages.split(",") if s.strip()]
        if args.stages else list(exp.stages)
    )
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        print(f"error: unknown stage(s) {unknown}; known: {list(ALL_STAGES)}", file=sys.stderr)
        return 2
    for name in args.only_arm:
        try:
            exp.arm(name)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.dry_run:
        print(describe(exp, stages))
        return 0

    _configure_logging(Path(exp.results_dir))
    ctx = PipelineContext(exp=exp, force=args.force, only_arms=tuple(args.only_arm))
    try:
        if args.smoke:
            run_smoke(ctx)
        else:
            run_pipeline(ctx, stages)
    except Exception as exc:  # noqa: BLE001 - reported, then non-zero exit
        logging.getLogger("infl_ens.pipeline").error("failed: %r", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
