#!/usr/bin/env python3
"""Build train/test specialist-vs-pooled-baseline tables from split eval JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infl_ens.evaluation.specialist_tables import (  # noqa: E402
    write_specialist_comparison_tables,
)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="results/six_axis_pair_merge_split/seed0",
    )
    parser.add_argument(
        "--baseline-run-dir",
        default="results/six_axis_baseline_replay_split/seed0",
    )
    args = parser.parse_args()

    run = ROOT / args.run_dir
    baseline = ROOT / args.baseline_run_dir
    summary = write_specialist_comparison_tables(
        run / "eval_train/eval_results.json",
        run / "eval_test/eval_results.json",
        baseline / "eval_train/eval_results.json",
        baseline / "eval_test/eval_results.json",
        run / "tables",
    )
    print(json.dumps(
        {
            "baseline_agent": summary["baseline_agent"],
            "train_wins": summary["train_wins"],
            "test_wins": summary["test_wins"],
            "n_axes": summary["n_axes"],
            "tables_dir": str(run / "tables"),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
