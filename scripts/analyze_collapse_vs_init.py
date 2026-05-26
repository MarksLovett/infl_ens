"""Relate seed collapse to round-0 positions in pool_and_noise runs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/pool_and_noise_10seeds")
spread_thresh = 0.45

sigma_re = re.compile(r"^sigma([0-9.]+)$")
seed_re = re.compile(r"^seed(\d+)$")


def spread(pos: np.ndarray) -> float:
    n = len(pos)
    return float(np.mean([
        np.linalg.norm(pos[i] - pos[j])
        for i in range(n) for j in range(i + 1, n)
    ]))


def classify(pos: np.ndarray) -> str:
    return "2,2" if spread(pos) >= spread_thresh else "collapsed"


rows = []
for sd in sorted(root.iterdir()):
    sm = sigma_re.match(sd.name)
    if not sm:
        continue
    sigma = float(sm.group(1))
    for seed_dir in sorted(sd.iterdir()):
        m = seed_re.match(seed_dir.name)
        if not m:
            continue
        hist = json.loads((seed_dir / "history.json").read_text())
        names = sorted(hist[0]["positions"].keys())
        p0 = np.stack([np.asarray(hist[0]["positions"][n]) for n in names])
        pf = np.stack([np.asarray(hist[-1]["positions"][n]) for n in names])
        rows.append({
            "sigma": sigma,
            "seed": int(m.group(1)),
            "spread0": spread(p0),
            "spreadf": spread(pf),
            "label": classify(pf),
            "harm0": p0[:, 0].tolist(),
            "harmf": pf[:, 0].tolist(),
        })

for sigma in sorted({r["sigma"] for r in rows}):
    sub = [r for r in rows if r["sigma"] == sigma]
    ok = [r for r in sub if r["label"] == "2,2"]
    bad = [r for r in sub if r["label"] == "collapsed"]
    print(f"\n=== sigma {sigma} ===")
    print(f"  (2,2) seeds:     {[r['seed'] for r in ok]}")
    print(f"  collapsed seeds: {[r['seed'] for r in bad]}")
    if ok:
        print(f"  init spread  (2,2):  {np.mean([r['spread0'] for r in ok]):.5f} ± {np.std([r['spread0'] for r in ok]):.5f}")
    if bad:
        print(f"  init spread  (col):  {np.mean([r['spread0'] for r in bad]):.5f} ± {np.std([r['spread0'] for r in bad]):.5f}")
    print(f"  {'seed':>4} {'s0':>8} {'sf':>8} {'label':>10}  harm0 (4 clones)")
    for r in sorted(sub, key=lambda x: x["seed"]):
        h = ", ".join(f"{x:.3f}" for x in r["harm0"])
        print(f"  {r['seed']:4d} {r['spread0']:8.5f} {r['spreadf']:8.3f} {r['label']:10s}  [{h}]")
