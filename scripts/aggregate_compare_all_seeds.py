"""CLI: aggregate compare_all_round*.json across seeds (mean ± std)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infl_ens.evaluation.compare import (
    aggregate_compare_reports,
    print_aggregate_compare_table,
)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = sorted(Path().glob(args.glob))
    if not paths:
        raise SystemExit(f"no files for glob {args.glob!r}")

    report = aggregate_compare_reports(paths)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"aggregated {report['n_files']} seeds -> {out}")
    print_aggregate_compare_table(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
