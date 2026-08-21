#!/usr/bin/env python3
"""Summarize oracle-centroid shift vs ga_theory_pre reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REF = {
    "g_argmax_agreement": 0.742,
    "oracle_minus_learned": -0.0379,
    "per_benchmark_agreement": {
        "ai4privacy": 0.635,
        "beavertails": 0.655,
        "do_not_answer": 0.839,
        "halueval": 0.772,
        "orbench": 0.849,
    },
    "per_benchmark_gap": {
        "ai4privacy": 0.0388,
        "beavertails": 0.0249,
        "do_not_answer": 0.0383,
        "halueval": 0.0358,
        "orbench": 0.0599,
    },
}


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="results/oracle_centroid_shift/ga_theory_pre/seed0",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    routing_path = run_dir / "routing_weight_comparison.json"
    decomp_path = run_dir / "routing_gap_decomposition.json"
    theory_path = run_dir / "theory_g_vs_oracle.json"
    hist_path = run_dir / "history.json"

    row: dict = {"run_dir": str(run_dir)}
    if routing_path.is_file():
        flat = json.loads(routing_path.read_text(encoding="utf-8"))["flat"]
        learned = flat["learned_routing_expected_nll"]
        oracle = flat["oracle_routing_nll"]
        agree = flat.get("routing_agreement_argmax", flat.get("agreement_argmax"))
        row.update({
            "learned_expected_nll": learned,
            "oracle_nll": oracle,
            "oracle_minus_learned": oracle - learned,
            "delta_vs_ref": (oracle - learned) - REF["oracle_minus_learned"],
            "g_argmax_agreement": agree,
            "agreement_delta_vs_ref": (
                None if agree is None else agree - REF["g_argmax_agreement"]
            ),
            "per_benchmark": flat.get("per_benchmark", {}),
        })

    if decomp_path.is_file():
        decomp = json.loads(decomp_path.read_text(encoding="utf-8"))
        row["per_axis_decomposition"] = decomp.get("per_axis", {})

    if theory_path.is_file():
        row["theory_g_vs_oracle"] = json.loads(theory_path.read_text(encoding="utf-8"))

    if hist_path.is_file():
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
        row["within_merge_round0"] = hist[0].get("agent_geometry", {}).get(
            "within_merge_l2", {},
        )
        row["within_merge_final"] = hist[-1].get("agent_geometry", {}).get(
            "within_merge_l2", {},
        )

    persist_path = run_dir / "centroid_persistence.json"
    if persist_path.is_file():
        persist = json.loads(persist_path.read_text(encoding="utf-8"))
        row["centroid_persistence"] = {
            "verdict": persist.get("verdict"),
            "ok": persist.get("ok"),
            "mean_final_dist_to_oracle": persist["summary"]["mean_final_dist_to_oracle"],
            "mean_final_dist_to_ref_ga": persist["summary"]["mean_final_dist_to_ref_ga"],
        }

    print("=== oracle-centroid shift vs reference ===")
    print(
        f"reference: agree={REF['g_argmax_agreement']:.3f}  "
        f"oracle−learned={REF['oracle_minus_learned']:+.4f}",
    )
    if "oracle_minus_learned" in row:
        print(
            f"experiment: agree={row.get('g_argmax_agreement', float('nan')):.3f}  "
            f"oracle−learned={row['oracle_minus_learned']:+.4f}  "
            f"(Δ ref {row['delta_vs_ref']:+.4f})",
        )
    print("\n--- per benchmark (argmax agreement, Δ_exp) ---")
    print(f"{'bench':<16} {'agree':>8} {'Δagree':>8} {'Δ_exp':>8} {'Δgap':>8}")
    per_bench = row.get("per_benchmark", {})
    for bench in sorted(per_bench.keys()):
        b = per_bench[bench]
        agree = b.get("agreement_argmax", float("nan"))
        gap = b.get("learned_expected_nll", 0) - b.get("oracle_nll", 0)
        ref_ag = REF["per_benchmark_agreement"].get(bench, float("nan"))
        ref_gap = REF["per_benchmark_gap"].get(bench, float("nan"))
        print(
            f"{bench:<16} {agree:8.3f} {agree - ref_ag:+8.3f} "
            f"{gap:+8.4f} {gap - ref_gap:+8.4f}",
        )

    if "within_merge_final" in row:
        print(f"\nwithin_merge final: {row['within_merge_final']}")
    if "centroid_persistence" in row:
        cp = row["centroid_persistence"]
        print(
            f"centroid persistence: {cp['verdict']} ok={cp['ok']} "
            f"(L2→oracle {cp['mean_final_dist_to_oracle']:.4f}, "
            f"L2→refGA {cp['mean_final_dist_to_ref_ga']:.4f})",
        )

    payload = {"reference": REF, "experiment": row}
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
