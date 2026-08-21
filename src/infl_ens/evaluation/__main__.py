"""Single CLI entry point for adapter evaluation on benchmarks.

Run with::

    python -m infl_ens.evaluation --config configs/evaluation/adapter_on_benchmarks.yaml

Supported tasks (``task`` field in the YAML):

- ``adapter_eval``: score one LoRA adapter on every benchmark listed
  under ``benchmarks``.
- ``run_eval``: discover adapters under ``run_dir/agents/`` and score
  each on every benchmark.

Optional ``KEY=VAL`` overrides after ``--`` mirror
:mod:`infl_ens.training.__main__`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from infl_ens.evaluation.evaluate import EvalJobConfig, run_eval_job

_TASKS = frozenset({"adapter_eval", "run_eval"})


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file using PyYAML if available, else a tiny fallback.

    :param path: Path to the YAML file.
    :type path: pathlib.Path
    :returns: Parsed mapping.
    :rtype: dict
    """
    try:
        import yaml
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:  # pragma: no cover
        out: dict[str, Any] = {}
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
        return out


def _apply_overrides(cfg: dict[str, Any], overrides: Sequence[str]) -> None:
    """Apply ``key.subkey=value`` overrides in place.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :param overrides: Sequence of dotted overrides.
    :type overrides: Sequence[str]
    """
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        path = key.split(".")
        node = cfg
        for p in path[:-1]:
            node = node.setdefault(p, {})
        try:
            node[path[-1]] = json.loads(val)
        except json.JSONDecodeError:
            node[path[-1]] = val


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the configured evaluation task.

    :param argv: Optional argument vector (defaults to ``sys.argv[1:]``).
    :type argv: Sequence[str] | None
    :returns: Process exit code (0 on success).
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Evaluate saved LoRA adapters on BeaverTails and HaluEval.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML evaluation config.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional KEY=VAL config overrides.",
    )
    args = parser.parse_args(argv)

    cfg = _load_yaml(Path(args.config))
    _apply_overrides(cfg, args.overrides)
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
