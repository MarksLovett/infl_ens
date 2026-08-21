#!/usr/bin/env python3
"""Run variance / ICA / mid-mass niche gates on seven-axis benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infl_ens.evaluation.axis_niche import (  # noqa: E402
    format_niche_markdown,
    niche_results_to_dict,
    run_axis_niche_diagnostic,
)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--router-config",
        default="configs/benchmark/router/seven_axis_pair_merge_split.yaml",
    )
    parser.add_argument(
        "--history",
        default="results/seven_axis_pair_merge_split/seed0/history.json",
    )
    parser.add_argument(
        "--merge-run-dir",
        default="results/seven_axis_pair_merge_split/seed0",
    )
    parser.add_argument("--partition", default="test")
    parser.add_argument("--max-eval-records", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--no-routing",
        action="store_true",
        help="Skip mid-mass routing term (trait-space only).",
    )
    args = parser.parse_args()

    history = None if args.no_routing else Path(args.history)
    merge_dir = None if args.no_routing else Path(args.merge_run_dir)
    results = run_axis_niche_diagnostic(
        router_config=Path(args.router_config),
        repo_root=ROOT,
        history_path=history,
        merge_run_dir=merge_dir,
        partition=args.partition,
        max_eval_records=args.max_eval_records,
        seed=args.seed,
    )
    print(format_niche_markdown(results))
    payload = niche_results_to_dict(results)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output_json}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
