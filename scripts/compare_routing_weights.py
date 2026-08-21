#!/usr/bin/env python3
"""Compare naive-G vs G(1−G) expected routing on the flat test pool.

Adapter-free reweighting of pre-scored merge NLLs. Success signature:
expected routing under G(1−G) moves toward oracle vs naive-G expected
routing.

Example::

    python scripts/compare_routing_weights.py \\
        --router-config configs/benchmark/router/seven_axis_pair_merge_split.yaml \\
        --history results/seven_axis_pair_merge_split/seed0/history.json \\
        --merge-run-dir results/seven_axis_pair_merge_split/seed0 \\
        --baseline-run-dir results/seven_axis_baseline_replay_split/seed0 \\
        --merge-nll-cache results/seven_axis_pair_merge_split/seed0/merge_nll_test.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infl_ens.evaluation.routing_eval import (  # noqa: E402
    format_headline_markdown,
    report_to_dict,
    run_flat_routing_eval,
)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-config", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--merge-run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--max-eval-records", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--merge-nll-cache", type=Path, default=None)
    parser.add_argument("--save-merge-nll-cache", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    cache = args.merge_nll_cache
    save_cache = args.save_merge_nll_cache or cache
    score = cache is None or not cache.is_file()

    report = run_flat_routing_eval(
        router_config=args.router_config,
        history_path=args.history,
        merge_run_dir=args.merge_run_dir,
        baseline_run_dir=args.baseline_run_dir,
        repo_root=ROOT,
        partition=args.partition,
        max_eval_records=args.max_eval_records,
        seed=args.seed,
        score_adapters=score,
        merge_nll_cache=cache,
        save_merge_nll_cache=save_cache if score else None,
    )
    f = report.flat
    print(format_headline_markdown(report))
    print("=== G vs G(1−G) comparison ===")
    print(f"naive-G expected:     {f.learned_expected_nll:.4f}  "
          f"(Δ pooled {f.learned_expected_nll - f.pooled_nll:+.4f}, "
          f"Δ oracle {f.learned_expected_nll - f.oracle_nll:+.4f})")
    print(f"G(1−G) expected:      {f.strategic_expected_nll:.4f}  "
          f"(Δ pooled {f.strategic_expected_nll - f.pooled_nll:+.4f}, "
          f"Δ oracle {f.strategic_expected_nll - f.oracle_nll:+.4f})")
    print(f"G(1−G) − naive-G:     {f.strategic_expected_nll - f.learned_expected_nll:+.4f}")
    print(f"oracle − G(1−G):      {f.oracle_nll - f.strategic_expected_nll:+.4f}")

    payload = report_to_dict(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
