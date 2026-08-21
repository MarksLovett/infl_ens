#!/usr/bin/env python3
"""Summarize seed-0 split isolation runs (fixed split, varied training seed)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REFERENCE = {
    "attribution_2x2_ga_no_theory_pre_train0": {
        "pooled_nll": 1.9457460591664868,
        "learned_nll": 1.9550691510177387,
        "delta_vs_pooled": 0.009323091851251908,
    },
}


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/seed0_isolation")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    rows: list[dict] = []
    for run_dir in sorted(root.iterdir()) if root.is_dir() else []:
        if not run_dir.is_dir() or not run_dir.name.startswith("ga_no_theory_pre_train"):
            continue
        routing_path = run_dir / "routing_weight_comparison.json"
        if not routing_path.is_file():
            rows.append({"cell": run_dir.name, "status": "missing_routing"})
            continue
        flat = json.loads(routing_path.read_text(encoding="utf-8"))["flat"]
        pooled = flat["pooled_nll"]
        learned = flat["learned_routing_expected_nll"]
        oracle = flat["oracle_routing_nll"]
        delta = learned - pooled
        rows.append({
            "cell": run_dir.name,
            "pooled_nll": pooled,
            "learned_nll": learned,
            "oracle_nll": oracle,
            "delta_vs_pooled": delta,
            "oracle_minus_learned": oracle - learned,
            "delta_minus_ref": delta - REFERENCE[
                "attribution_2x2_ga_no_theory_pre_train0"
            ]["delta_vs_pooled"],
        })

    print("=== seed-0 isolation (fixed five_axis_seed0.json split) ===")
    ref = REFERENCE["attribution_2x2_ga_no_theory_pre_train0"]
    print(
        f"reference (train0): pooled={ref['pooled_nll']:.4f} "
        f"learned={ref['learned_nll']:.4f} Δ={ref['delta_vs_pooled']:+.4f}",
    )
    for row in rows:
        if row.get("status") == "missing_routing":
            print(f"{row['cell']}: (no routing results)")
            continue
        print(
            f"{row['cell']}: pooled={row['pooled_nll']:.4f} "
            f"learned={row['learned_nll']:.4f} "
            f"Δ={row['delta_vs_pooled']:+.4f} "
            f"(Δ−ref {row['delta_minus_ref']:+.4f}) "
            f"oracle−learned {row['oracle_minus_learned']:+.4f}",
        )

    payload = {"reference": REFERENCE, "cells": rows}
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
