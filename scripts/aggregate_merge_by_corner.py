"""Aggregate proximity-merge benchmark NLL by shared trait-space corner.

Example::

    python scripts/aggregate_merge_by_corner.py \\
        --run-root results/proximity_merge_round_sweep/r40 \\
        --round 39 \\
        --eval \\
        --output results/proximity_merge_corner_aggregate.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from infl_ens.evaluation.compare import aggregate_merge_by_corner, process_merge_seed


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    :param argv: Optional argument vector.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Aggregate merge benchmark NLL by shared corner (harm role).",
    )
    parser.add_argument(
        "--run-root",
        default="results/proximity_merge_round_sweep/r40",
        help="Directory containing seed0, seed1, ...",
    )
    parser.add_argument("--round", type=int, default=39)
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run benchmark eval if compare_all JSON missing.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-eval-records", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--forward-batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    root = Path(args.run_root)
    seed_dirs = sorted(root.glob("seed*"))
    if not seed_dirs:
        raise SystemExit(f"no seed* under {root}")

    all_records = []
    for sd in seed_dirs:
        all_records.extend(
            process_merge_seed(
                sd,
                round_idx=args.round,
                do_eval=args.eval,
                base_model=args.base_model,
                max_eval_records=args.max_eval_records,
                eval_seed=args.eval_seed,
                max_seq_length=args.max_seq_length,
                forward_batch_size=args.forward_batch_size,
            ),
        )

    report = aggregate_merge_by_corner(all_records)
    report["run_root"] = str(root.resolve())
    report["round"] = args.round
    report["n_seeds"] = len(seed_dirs)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    benches = sorted({r["benchmark"] for r in report["aggregate_by_corner"]})
    roles = sorted({r["corner_role"] for r in report["aggregate_by_corner"]})
    print(f"wrote {out} ({len(seed_dirs)} seeds)")
    print(f"\n=== mean NLL by corner role (round {args.round}) ===")
    print(f"{'corner':<22}" + "".join(f"{b:>16}" for b in benches))
    for role in roles:
        line = f"{role:<22}"
        for b in benches:
            cell = next(
                (
                    r for r in report["aggregate_by_corner"]
                    if r["corner_role"] == role and r["benchmark"] == b
                ),
                None,
            )
            if cell is None:
                line += f"{'—':>16}"
            else:
                line += (
                    f"{cell['mean_nll']:8.4f}±{cell['std_nll']:<5.3f}"
                    f" n={cell['n_seeds']}"
                )
        print(line)

    print("\n=== per-seed corner assignment ===")
    for row in report["per_seed"]:
        c = row["centroid"]
        print(
            f"seed{row['seed']:>2} {row['corner_role']:<20} "
            f"members={row['members']} "
            f"harm={c[0]:+.3f} halluc={c[1]:+.3f} "
            f"adapter={row['train_name']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
