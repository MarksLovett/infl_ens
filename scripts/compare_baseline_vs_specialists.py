"""Compare pooled baseline vs specialist adapters at a fixed round.

Example::

    python scripts/compare_baseline_vs_specialists.py \\
        --baseline-dir results/baseline_replay_r40/seed0 \\
        --specialist-dir results/pairs_near_eq_round_sweep/r40/seed0 \\
        --round 39
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from infl_ens.evaluation.compare import ModelScore, compare_baseline_vs_specialists


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    :param argv: Optional argument vector.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Compare pooled baseline vs specialist round-N adapters.",
    )
    parser.add_argument("--baseline-dir", required=True, help="Baseline replay run dir.")
    parser.add_argument("--specialist-dir", required=True, help="Closed-loop run dir.")
    parser.add_argument("--round", type=int, default=39, help="Adapter round index.")
    parser.add_argument("--baseline-name", default="pooled-baseline")
    parser.add_argument(
        "--specialists",
        default="clone-0,clone-1,clone-2,clone-3",
        help="Comma-separated specialist names.",
    )
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-eval-records", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--forward-batch-size", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write comparison JSON (default: baseline-dir).",
    )
    args = parser.parse_args(argv)

    baseline_dir = Path(args.baseline_dir)
    specialist_dir = Path(args.specialist_dir)
    out_dir = Path(args.output_dir or baseline_dir)
    specialists = [s.strip() for s in args.specialists.split(",") if s.strip()]

    payload = compare_baseline_vs_specialists(
        baseline_dir,
        specialist_dir,
        round_idx=args.round,
        baseline_name=args.baseline_name,
        specialists=specialists,
        base_model=args.base_model,
        max_eval_records=args.max_eval_records,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        forward_batch_size=args.forward_batch_size,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_round{args.round:02d}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    all_scores = [ModelScore(**row) for row in payload["benchmark_scores"]]
    benches = sorted({s.benchmark for s in all_scores})
    labels = [args.baseline_name] + specialists
    print(f"\n=== benchmark NLL (round {args.round}) ===")
    print(f"{'model':<16}" + "".join(f"{b:>14}" for b in benches))
    for lab in labels:
        row = f"{lab:<16}"
        for b in benches:
            cell = next((s for s in all_scores if s.label == lab and s.benchmark == b), None)
            row += f"{cell.mean_nll:14.4f}" if cell else f"{'—':>14}"
        print(row)

    cross_rows = payload["cross_nll_on_pooled_round_batch"]
    if cross_rows:
        print(f"\n=== cross-NLL on pooled round-{args.round} batch ===")
        for row in cross_rows:
            print(
                f"{row['label']:<16} {row['mean_nll_on_pooled_batch']:.4f} "
                f"({row['n_examples']} ex)",
            )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
