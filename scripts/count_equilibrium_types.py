#!/usr/bin/env python3
"""Count distinct final-layout equilibrium types under a sigma×seed results tree."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from classify_equilibrium import classify_22, pairwise_spread  # noqa: E402

_SIGMA_RE = re.compile(r"^sigma(?P<val>[0-9.]+)$", re.IGNORECASE)
_SEED_RE = re.compile(r"^seed(?P<val>\d+)$", re.IGNORECASE)


def collect_labels(root: Path, *, spread_thresh: float) -> list[dict]:
    """Scan ``root/sigma*/seed*/history.json`` and classify finals."""
    rows: list[dict] = []
    if not root.is_dir():
        return rows
    for sigma_dir in sorted(root.iterdir()):
        if not sigma_dir.is_dir():
            continue
        sm = _SIGMA_RE.match(sigma_dir.name)
        if not sm:
            continue
        sigma = float(sm.group("val"))
        for seed_dir in sorted(sigma_dir.iterdir()):
            sd = _SEED_RE.match(seed_dir.name)
            hist_path = seed_dir / "history.json"
            if not sd or not hist_path.is_file():
                continue
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            names = sorted(hist[-1]["positions"].keys())
            pos = np.stack(
                [np.asarray(hist[-1]["positions"][n], dtype=float) for n in names],
            )
            label = classify_22(pos, spread_thresh=spread_thresh)
            rows.append({
                "sigma": sigma,
                "seed": int(sd.group("val")),
                "spread": pairwise_spread(pos),
                "label": label,
            })
    return rows


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

    rows = collect_labels(args.root, spread_thresh=args.spread_thresh)
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
