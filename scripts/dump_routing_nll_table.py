#!/usr/bin/env python3
"""Dump pooled / learned / oracle NLL per run."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _row(cell: str, path: Path) -> dict | None:
    if not path.is_file():
        return None
    flat = json.loads(path.read_text(encoding="utf-8"))["flat"]
    return {
        "cell": cell,
        "pooled": flat["pooled_nll"],
        "learned": flat["learned_routing_expected_nll"],
        "oracle": flat["oracle_routing_nll"],
        "delta_vs_pooled": flat["learned_routing_expected_nll"] - flat["pooled_nll"],
        "oracle_minus_learned": (
            flat["oracle_routing_nll"] - flat["learned_routing_expected_nll"]
        ),
    }


def main() -> int:
    rows: list[dict] = []
    spread = ROOT / "results/attribution_spread_rerun"
    for run_dir in sorted(spread.iterdir()) if spread.is_dir() else []:
        if not run_dir.is_dir():
            continue
        row = _row(run_dir.name, run_dir / "routing_weight_comparison.json")
        if row:
            rows.append(row)
    for label, rel in [
        ("attribution_2x2/ga_theory_pre_s0", "results/attribution_2x2/ga_theory_pre/seed0/routing_weight_comparison.json"),
        ("attribution_2x2/ga_no_theory_pre_s0", "results/attribution_2x2/ga_no_theory_pre/seed0/routing_weight_comparison.json"),
        ("hypercube_ga_s0", "results/seven_axis_collapse_hypercube_ga/seed0/routing_weight_comparison.json"),
        ("pooled_baseline_s0", "results/seven_axis_collapse_hypercube_ga_baseline/seed0/routing_weight_comparison.json"),
    ]:
        row = _row(label, ROOT / rel)
        if row:
            rows.append(row)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
