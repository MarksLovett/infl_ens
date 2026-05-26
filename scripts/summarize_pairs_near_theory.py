#!/usr/bin/env python3
"""Summarize pairs_near_theory sweep: (2,2) vs collapsed per sigma."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infl_ens.training.pool_dynamics import classify_layout, pairwise_spread  # noqa: E402

_SIGMA_RE = re.compile(r"^sigma(?P<val>[0-9.]+)$")
_SEED_RE = re.compile(r"^seed(?P<val>\d+)$")


def main() -> int:
    """Print per-sigma layout counts and compare to mean_noise baseline."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "results/pairs_near_theory_10seeds")
    p.add_argument(
        "--baseline-root",
        type=Path,
        default=ROOT / "results/pool_and_noise_10seeds",
        help="Optional mean_noise baseline for side-by-side counts.",
    )
    args = p.parse_args()

    rows: list[dict] = []
    for sigma_dir in sorted(args.root.iterdir()):
        sm = _SIGMA_RE.match(sigma_dir.name)
        if not sm:
            continue
        sigma = float(sm.group("val"))
        for seed_dir in sorted(sigma_dir.iterdir()):
            m = _SEED_RE.match(seed_dir.name)
            if not m or not (seed_dir / "history.json").is_file():
                continue
            hist = json.loads((seed_dir / "history.json").read_text(encoding="utf-8"))
            names = sorted(hist[0]["positions"].keys())
            p0 = np.stack([np.asarray(hist[0]["positions"][n]) for n in names])
            pf = np.stack([np.asarray(hist[-1]["positions"][n]) for n in names])
            rows.append({
                "sigma": sigma,
                "seed": int(m.group("val")),
                "spread0": pairwise_spread(p0),
                "spreadf": pairwise_spread(pf),
                "layout": classify_layout(pf),
                "init_mode": hist[0].get("init_mode", "?"),
            })

    print(f"\n=== pairs_near_theory @ {args.root} ===")
    for sigma in sorted({r["sigma"] for r in rows}):
        sub = [r for r in rows if r["sigma"] == sigma]
        n22 = sum(1 for r in sub if r["layout"] == "2,2")
        ncol = sum(1 for r in sub if r["layout"] == "collapsed")
        other = len(sub) - n22 - ncol
        print(f"\n  sigma={sigma}  (2,2)={n22}/{len(sub)}  collapsed={ncol}/{len(sub)}  other={other}")
        print(f"    (2,2) seeds:     {sorted(r['seed'] for r in sub if r['layout'] == '2,2')}")
        print(f"    collapsed seeds: {sorted(r['seed'] for r in sub if r['layout'] == 'collapsed')}")
        print(f"  {'seed':>4} {'s0':>8} {'sf':>8} {'layout':>10}")
        for r in sorted(sub, key=lambda x: x["seed"]):
            print(f"  {r['seed']:4d} {r['spread0']:8.3f} {r['spreadf']:8.3f} {r['layout']:>10}")

    if args.baseline_root.is_dir():
        print(f"\n=== baseline mean_noise @ {args.baseline_root} ===")
        for sigma_dir in sorted(args.baseline_root.iterdir()):
            sm = _SIGMA_RE.match(sigma_dir.name)
            if not sm:
                continue
            sigma = float(sm.group("val"))
            layouts = []
            for seed_dir in sorted(sigma_dir.iterdir()):
                hp = seed_dir / "history.json"
                if not hp.is_file():
                    continue
                hist = json.loads(hp.read_text(encoding="utf-8"))
                names = sorted(hist[-1]["positions"].keys())
                pf = np.stack([np.asarray(hist[-1]["positions"][n]) for n in names])
                layouts.append(classify_layout(pf))
            n22 = sum(1 for x in layouts if x == "2,2")
            print(f"  sigma={sigma}  (2,2)={n22}/{len(layouts)}  collapsed={len(layouts)-n22}/{len(layouts)}")

    out = args.root / "summary.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
