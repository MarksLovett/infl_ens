"""Four-way comparison: base, pooled generalist, 4 specialists, 2 merge specialists.

Example::

    python scripts/compare_all_r40_models.py \\
        --specialist-dir results/pairs_near_eq_round_sweep/r40/seed0 \\
        --baseline-dir results/baseline_replay_r40/seed0 \\
        --merge-dir results/pair_merge_round_sweep/r40/seed0 \\
        --round 39
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from infl_ens.evaluation.compare import ModelScore, compare_all_models


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    :param argv: Optional argument vector.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Compare base, pooled, specialists, and pair-merge adapters.",
    )
    parser.add_argument("--specialist-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--merge-dir", required=True)
    parser.add_argument(
        "--base-eval-json",
        default="results/base_model_eval_qwen2_5_1_5b/base_eval.json",
    )
    parser.add_argument("--round", type=int, default=39)
    parser.add_argument("--baseline-name", default="pooled-baseline")
    parser.add_argument(
        "--merge-names",
        default=None,
        help="Comma-separated merge trainer names (default: discover merge-*).",
    )
    parser.add_argument(
        "--specialists",
        default="clone-0,clone-1,clone-2,clone-3",
    )
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-eval-records", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--forward-batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    specialist_dir = Path(args.specialist_dir)
    baseline_dir = Path(args.baseline_dir)
    merge_dir = Path(args.merge_dir)
    out_dir = Path(args.output_dir or merge_dir)
    specialists = [s.strip() for s in args.specialists.split(",") if s.strip()]
    merge_names = None
    if args.merge_names:
        merge_names = [s.strip() for s in args.merge_names.split(",") if s.strip()]

    base_eval_path = Path(args.base_eval_json)
    payload = compare_all_models(
        specialist_dir,
        baseline_dir,
        merge_dir,
        base_eval_json=base_eval_path,
        round_idx=args.round,
        baseline_name=args.baseline_name,
        merge_names=merge_names,
        specialists=specialists,
        base_model=args.base_model,
        max_eval_records=args.max_eval_records,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        forward_batch_size=args.forward_batch_size,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_all_round{args.round:02d}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    all_scores = [ModelScore(**row) for row in payload["benchmark_scores"]]
    if merge_names is None:
        agents_dir = merge_dir / "agents"
        merge_names = sorted(
            p.name for p in agents_dir.iterdir()
            if p.is_dir() and p.name.startswith("merge-")
        )

    benches = sorted({s.benchmark for s in all_scores})
    labels = ["base"] if base_eval_path.is_file() else []
    labels += [args.baseline_name] + specialists + list(merge_names)
    print(f"\n=== benchmark NLL (round {args.round}) ===")
    print(f"{'model':<16}" + "".join(f"{b:>14}" for b in benches))
    for lab in labels:
        row = f"{lab:<16}"
        for b in benches:
            cell = next(
                (s for s in all_scores if s.label == lab and s.benchmark == b),
                None,
            )
            row += f"{cell.mean_nll:14.4f}" if cell else f"{'—':>14}"
        print(row)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
