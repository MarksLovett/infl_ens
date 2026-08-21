#!/usr/bin/env python3
"""Summarize within-merge spread runs (oracle gap + geometry)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REF_ORACLE_MINUS_LEARNED = -0.0379


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/within_merge_spread")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    rows: list[dict] = []
    for run_dir in sorted(root.iterdir()) if root.is_dir() else []:
        if not run_dir.is_dir():
            continue
        if not run_dir.name.startswith("oracle_k2_"):
            continue
        routing_path = run_dir / "routing_weight_comparison.json"
        hist_path = run_dir / "history.json"
        row: dict = {"cell": run_dir.name}
        if routing_path.is_file():
            flat = json.loads(routing_path.read_text(encoding="utf-8"))["flat"]
            learned = flat["learned_routing_expected_nll"]
            oracle = flat["oracle_routing_nll"]
            row.update({
                "learned_expected_nll": learned,
                "oracle_nll": oracle,
                "oracle_minus_learned": oracle - learned,
                "delta_vs_ref": (oracle - learned) - REF_ORACLE_MINUS_LEARNED,
                "agreement_argmax": flat.get("routing_agreement_argmax"),
            })
        if hist_path.is_file():
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            within_init = hist[0].get("agent_geometry", {}).get("within_merge_l2", {})
            within_final = hist[-1].get("agent_geometry", {}).get("within_merge_l2", {})
            row["within_merge_round0"] = within_init
            row["within_merge_final"] = within_final
        rows.append(row)

    print("=== within-merge spread (seed-0 split) ===")
    print(f"reference GA colocated oracle−learned: {REF_ORACLE_MINUS_LEARNED:+.4f}")
    for row in rows:
        if "oracle_minus_learned" not in row:
            print(f"{row['cell']}: (incomplete)")
            continue
        print(
            f"{row['cell']}: oracle−learned {row['oracle_minus_learned']:+.4f} "
            f"(Δ ref {row['delta_vs_ref']:+.4f}) "
            f"agree {row.get('agreement_argmax', float('nan')):.3f}",
        )
        if "within_merge_round0" in row:
            print(f"  within_merge r0: {row['within_merge_round0']}")
            print(f"  within_merge final: {row['within_merge_final']}")

    payload = {"reference_oracle_minus_learned": REF_ORACLE_MINUS_LEARNED, "cells": rows}
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
