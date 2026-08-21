#!/usr/bin/env python3
"""Correlate bimodal oracle geometry with gap change (spread vs colocated)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BENCH_TO_MERGE = {
    "beavertails": "merge-harm",
    "halueval": "merge-hallucination",
    "ai4privacy": "merge-privacy",
    "orbench": "merge-overrefusal",
    "do_not_answer": "merge-policy",
}


def _per_bench_gap(routing_json: Path) -> dict[str, float]:
    """Mean expected NLL gap (learned - oracle) per benchmark."""
    flat = json.loads(routing_json.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for bench, row in flat.get("per_benchmark", {}).items():
        out[bench] = float(row["learned_expected_nll"] - row["oracle_nll"])
    out["_flat"] = float(
        flat["flat"]["learned_routing_expected_nll"]
        - flat["flat"]["oracle_routing_nll"],
    )
    return out


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-routing",
        default="results/attribution_2x2/ga_theory_pre/seed0/routing_weight_comparison.json",
    )
    parser.add_argument(
        "--spread-routing",
        default="results/within_merge_spread/oracle_k2_aligned/routing_weight_comparison.json",
    )
    parser.add_argument(
        "--geometry",
        default="results/attribution_2x2/ga_theory_pre/seed0/merge_oracle_geometry.json",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    base = _per_bench_gap(Path(args.baseline_routing))
    spread = _per_bench_gap(Path(args.spread_routing))
    geom = json.loads(Path(args.geometry).read_text(encoding="utf-8"))

    rows: list[dict] = []
    for bench in sorted(BENCH_TO_MERGE):
        merge = BENCH_TO_MERGE[bench]
        g = geom["per_merge"][merge]
        delta = spread[bench] - base[bench]
        rows.append({
            "benchmark": bench,
            "merge": merge,
            "gap_baseline": base[bench],
            "gap_spread": spread[bench],
            "gap_worsening": delta,
            "pca_elongation": g.get("pca_elongation"),
            "gmm_bic_delta": g.get("gmm_bic1_minus_bic2"),
            "kmeans2_silhouette": g.get("kmeans2_silhouette"),
        })

    elong = np.array([r["pca_elongation"] for r in rows], dtype=float)
    worsen = np.array([r["gap_worsening"] for r in rows], dtype=float)
    corr_elong = float(np.corrcoef(elong, worsen)[0, 1]) if len(rows) > 2 else float("nan")

    print("=== gap worsening (spread aligned − GA colocated) vs bimodality ===")
    print(
        f"{'bench':<16} {'merge':<22} {'Δgap':>8} {'elong':>6} {'ΔBIC':>8}",
    )
    for r in sorted(rows, key=lambda x: x["gap_worsening"], reverse=True):
        print(
            f"{r['benchmark']:<16} {r['merge']:<22} "
            f"{r['gap_worsening']:+8.4f} {r['pca_elongation']:6.2f} "
            f"{r['gmm_bic_delta']:8.0f}",
        )
    print(f"\nflat gap: baseline {base['_flat']:+.4f}  spread {spread['_flat']:+.4f}  "
          f"worsening {spread['_flat'] - base['_flat']:+.4f}")
    print(f"corr(pca_elongation, gap_worsening) = {corr_elong:+.3f}")

    payload = {"rows": rows, "corr_elongation_worsening": corr_elong, "flat": {
        "baseline": base["_flat"], "spread": spread["_flat"],
        "worsening": spread["_flat"] - base["_flat"],
    }}
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
