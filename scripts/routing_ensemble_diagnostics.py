"""Route-then-score diagnostics on a flat test pool.

Computes headline flat-pool NLL under pooled baseline, learned routing
(expected proportional :math:`G`, sampled :math:`G`, and argmax :math:`G`),
and oracle merge routing. Per-benchmark tables are diagnostic only.

Example (on doob)::

    python scripts/routing_ensemble_diagnostics.py \\
        --router-config configs/benchmark/router/seven_axis_pair_merge_split.yaml \\
        --history results/seven_axis_pair_merge_split/seed0/history.json \\
        --merge-run-dir results/seven_axis_pair_merge_split/seed0 \\
        --baseline-run-dir results/seven_axis_baseline_replay_split/seed0 \\
        --partition test --max-eval-records 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from infl_ens.evaluation.routing_eval import (
    format_headline_markdown,
    report_to_dict,
    run_flat_routing_eval,
)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-config", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--merge-run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--max-eval-records", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--forward-batch-size", type=int, default=8)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    report = run_flat_routing_eval(
        router_config=args.router_config,
        history_path=args.history,
        merge_run_dir=args.merge_run_dir,
        baseline_run_dir=args.baseline_run_dir,
        repo_root=repo_root,
        partition=args.partition,
        max_eval_records=args.max_eval_records,
        seed=args.seed,
        round_idx=args.round,
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        forward_batch_size=args.forward_batch_size,
    )

    if report.merge_name_map != {
        k: k for k in report.merge_name_map
    }:
        remapped = {
            k: v for k, v in report.merge_name_map.items() if k != v
        }
        print(f"merge adapter aliases: {remapped}")

    print(format_headline_markdown(report))

    f = report.flat
    print(
        f"argmax vs expected learned: {f.learned_argmax_nll - f.learned_expected_nll:+.4f}"
    )
    print(
        f"oracle vs expected learned: {f.oracle_nll - f.learned_expected_nll:+.4f}"
    )

    dead = [
        name
        for name, row in report.clone_support_argmax.items()
        if row["n_wins"] == 0
        and report.clone_support_expected[name]["mean_g"] < 0.01
    ]
    if dead:
        print(f"\nDead clones (0 argmax wins, mean G < 0.01): {', '.join(dead)}")

    print("\n=== PER-BENCHMARK (diagnosis only) ===")
    print(
        f"{'benchmark':<18} {'n':>5} {'pooled':>8} {'expect':>8} "
        f"{'argmax':>8} {'oracle':>8}"
    )
    for bench, row in sorted(report.per_benchmark.items()):
        print(
            f"{bench:<18} {row['n']:5d} {row['pooled_nll']:8.4f} "
            f"{row['learned_expected_nll']:8.4f} "
            f"{row['learned_argmax_nll']:8.4f} {row['oracle_nll']:8.4f}"
        )

    print("\n=== CLONE SUPPORT (expected G vs argmax wins) ===")
    for name in sorted(report.clone_support_expected):
        eg = report.clone_support_expected[name]["mean_g"]
        nw = report.clone_support_argmax[name]["n_wins"]
        print(f"  {name}: mean_G={eg:.4f}  argmax_wins={nw}")

    summary = report_to_dict(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
