#!/usr/bin/env python3
"""Summarize spread-calibrated attribution re-run (NLL + geometry per cell/seed)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from infl_ens.training.pool_dynamics import agent_pairwise_geometry

MERGE_GROUPS: list[tuple[str, list[str]]] = [
    ("merge-harm", ["clone-0", "clone-1"]),
    ("merge-hallucination", ["clone-2", "clone-3"]),
    ("merge-privacy", ["clone-4", "clone-5"]),
    ("merge-overrefusal", ["clone-6", "clone-7"]),
    ("merge-policy", ["clone-8", "clone-9"]),
]
AGENT_NAMES = [f"clone-{i}" for i in range(10)]


def _geometry_from_history(hist: list[dict]) -> dict | None:
    """Extract or backfill post-pre / post-init geometry."""
    ti = hist[0].get("theory_init", {})
    geom = ti.get("theory_pre", {}).get("agent_geometry")
    if geom is not None:
        return geom
    geom = ti.get("agent_geometry")
    if geom is not None:
        return geom
    pre = ti.get("theory_pre", {})
    end = pre.get("theory_pre_end")
    if end is not None:
        positions = np.asarray(end, dtype=float)
        phase = "post_theory_pre"
    else:
        end_pos = hist[0].get("positions")
        if end_pos is None:
            return None
        positions = np.stack(
            [np.asarray(end_pos[name], dtype=float) for name in AGENT_NAMES],
            axis=0,
        )
        phase = "post_init"
    geom = agent_pairwise_geometry(
        positions, AGENT_NAMES, merge_groups=MERGE_GROUPS,
    )
    geom["geometry_phase"] = phase
    return geom


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/attribution_spread_rerun")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    rows: list[dict] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        routing_path = run_dir / "routing_weight_comparison.json"
        hist_path = run_dir / "history.json"
        row: dict = {"cell": run_dir.name}
        if routing_path.is_file():
            flat = json.loads(routing_path.read_text(encoding="utf-8"))["flat"]
            row.update({
                "pooled_nll": flat["pooled_nll"],
                "learned_expected_nll": flat["learned_routing_expected_nll"],
                "oracle_nll": flat["oracle_routing_nll"],
                "delta_vs_pooled": (
                    flat["learned_routing_expected_nll"] - flat["pooled_nll"]
                ),
            })
        if hist_path.is_file():
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            geom = _geometry_from_history(hist)
            if geom:
                row["mean_pairwise_l2"] = geom.get("mean_pairwise_l2")
                row["within_merge_l2"] = geom.get("within_merge_l2", {})
                row["geometry_phase"] = geom.get("geometry_phase")
        rows.append(row)

    print("=== attribution spread re-run summary ===")
    for row in rows:
        if "learned_expected_nll" in row:
            print(
                f"{row['cell']}: learned={row['learned_expected_nll']:.4f} "
                f"(Δ pooled {row['delta_vs_pooled']:+.4f}) "
                f"geom mean_pairwise={row.get('mean_pairwise_l2', float('nan')):.4f}",
            )
    payload = {"cells": rows}
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
