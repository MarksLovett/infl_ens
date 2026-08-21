#!/usr/bin/env python3
"""Analyze merge-pair occupancy from a flat routing report.

Reads routing diagnostics JSON and recommends how many clone/merge agents
to keep based on per-clone G mass and argmax wins.

Example::

    python scripts/analyze_pair_occupancy.py \\
        --routing-json results/seven_axis_collapse_dead_axes/seed0/routing_weight_comparison.json \\
        --router-config configs/benchmark/router/seven_axis_collapse_dead_axes.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infl_ens.evaluation.routing_eval import parse_merge_groups  # noqa: E402
from infl_ens.training.__main__ import _load_yaml  # noqa: E402

MEAN_G_ACTIVE = 0.05
ARGMAX_SHARE_ACTIVE = 0.01


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-json", type=Path, required=True)
    parser.add_argument("--router-config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.routing_json.read_text(encoding="utf-8"))
    flat = payload["flat"]
    clone_argmax = payload.get("clone_support_argmax", {})
    clone_g = payload.get("clone_support_expected", {})

    cfg = _load_yaml(args.router_config)
    clone_to_merge, merge_names = parse_merge_groups(cfg.get("closed_loop", {}))

    pairs: list[dict[str, Any]] = []
    n_active_clones = 0
    n_singleton_merges = 0
    for merge in merge_names:
        members = [c for c, m in clone_to_merge.items() if m == merge]
        row_clones: list[dict[str, Any]] = []
        total_g = 0.0
        total_wins = 0
        for name in members:
            g = float(clone_g.get(name, {}).get("mean_g", 0.0))
            wins = int(clone_argmax.get(name, {}).get("n_wins", 0))
            share = float(clone_argmax.get(name, {}).get("share", 0.0))
            active = g >= MEAN_G_ACTIVE or share >= ARGMAX_SHARE_ACTIVE
            if active:
                n_active_clones += 1
            total_g += g
            total_wins += wins
            row_clones.append({
                "clone": name,
                "mean_g": g,
                "argmax_wins": wins,
                "argmax_share": share,
                "active": active,
            })
        g_vals = [c["mean_g"] for c in row_clones]
        dominant = max(g_vals) if g_vals else 0.0
        partner_share = (
            min(g_vals) / max(dominant, 1e-12) if len(g_vals) == 2 else 1.0
        )
        is_singleton = len(row_clones) == 2 and partner_share < 0.15
        if is_singleton:
            n_singleton_merges += 1
        pairs.append({
            "merge": merge,
            "clones": row_clones,
            "pair_mean_g": total_g,
            "pair_argmax_wins": total_wins,
            "partner_g_ratio": partner_share,
            "recommendation": (
                "singleton" if is_singleton else "keep_pair"
            ),
        })

    n_merges_keep = len(merge_names) - n_singleton_merges
    n_clones_keep = n_active_clones

    print("=== Pair occupancy (collapse run) ===")
    print(f"{'merge':<22} {'clone':<10} {'mean_G':>8} {'argmax':>8} {'active':>7}")
    for p in pairs:
        for i, c in enumerate(p["clones"]):
            merge_label = p["merge"] if i == 0 else ""
            act = "yes" if c["active"] else "no"
            print(
                f"{merge_label:<22} {c['clone']:<10} {c['mean_g']:8.4f} "
                f"{c['argmax_wins']:8d} {act:>7}",
            )
        print(
            f"{'':22}  -> {p['recommendation']} "
            f"(partner G ratio {p['partner_g_ratio']:.2f})",
        )

    print()
    print("=== Headline routing (flat test pool) ===")
    for key in (
        "pooled_nll",
        "learned_routing_expected_nll",
        "learned_routing_argmax_nll",
        "oracle_routing_nll",
    ):
        if key in flat:
            print(f"  {key}: {flat[key]:.4f}")
    gap = flat["oracle_routing_nll"] - flat.get(
        "learned_routing_expected_nll",
        flat.get("learned_routing_nll", 0),
    )
    print(f"  oracle − learned_expected: {gap:+.4f}")

    print()
    print("=== Agent-count recommendation ===")
    print(f"  active clones (mean G≥{MEAN_G_ACTIVE} or argmax wins): {n_active_clones}")
    print(f"  singleton merge pairs (partner G ratio < 0.15): {n_singleton_merges}")
    print(f"  suggest keeping: {n_merges_keep} merges / ~{max(n_active_clones, n_merges_keep * 2)} clones")
    if n_singleton_merges > 0:
        print(
            "  pairs flagged singleton → candidate for merge-to-one-adapter "
            "in the next right-sizing step",
        )

    summary = {
        "pairs": pairs,
        "n_active_clones": n_active_clones,
        "n_singleton_merges": n_singleton_merges,
        "recommended_merges": n_merges_keep,
        "recommended_clones": max(n_active_clones, n_merges_keep * 2),
        "flat": flat,
        "oracle_minus_learned_expected": gap,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
