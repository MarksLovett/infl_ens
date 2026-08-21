#!/usr/bin/env python3
"""Count distinct final-layout equilibrium types under a sigma×seed results tree."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from functools import partial
from pathlib import Path

from infl_ens.training.pool_dynamics import classify_layout, pairwise_spread
from infl_ens.utils.sweep_discovery import collect_final_layout_labels


def main() -> int:
    """Print equilibrium-type counts and write summary JSON."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--spread-thresh", type=float, default=0.45)
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write summary JSON (default: <root>/equilibrium_types.json).",
    )
    args = p.parse_args()

    classify_fn = partial(classify_layout, spread_thresh=args.spread_thresh)
    rows = collect_final_layout_labels(
        args.root,
        classify_fn=classify_fn,
        spread_fn=pairwise_spread,
    )
    if not rows:
        print(f"no completed runs under {args.root}", file=sys.stderr)
        return 1

    counts = Counter(r["label"] for r in rows)
    n_types = len(counts)
    n_runs = len(rows)

    print(f"\n=== equilibrium types @ {args.root} ===")
    print(f"  completed runs : {n_runs}")
    print(f"  distinct types : {n_types}")
    print(f"  types present  : {sorted(counts.keys())}")
    for label in sorted(counts.keys()):
        print(f"    {label:12s} {counts[label]:4d}  ({100.0 * counts[label] / n_runs:.1f}%)")

    print("\n  per sigma:")
    for sigma in sorted({r["sigma"] for r in rows}):
        sub = [r for r in rows if r["sigma"] == sigma]
        sub_counts = Counter(r["label"] for r in sub)
        parts = ", ".join(f"{k}={sub_counts[k]}" for k in sorted(sub_counts))
        print(f"    σ={sigma:g}  n={len(sub)}  {parts}  types={len(sub_counts)}")

    out = args.json_out or (args.root / "equilibrium_types.json")
    payload = {
        "root": str(args.root),
        "n_runs": n_runs,
        "n_types": n_types,
        "counts": dict(counts),
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
