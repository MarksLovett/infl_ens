"""Classify final 4-agent trait layouts as (2,2), collapsed, or other."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_SIGMA_RE = re.compile(r"^sigma(?P<val>[0-9.]+)$", re.IGNORECASE)
_SEED_RE = re.compile(r"^seed(?P<val>\d+)$", re.IGNORECASE)


def pairwise_spread(pos: np.ndarray) -> float:
    """Mean pairwise L2 distance among rows of *pos*.

    :param pos: ``(N, L)`` positions.
    :type pos: numpy.ndarray
    :returns: Mean off-diagonal distance.
    :rtype: float
    """
    n = pos.shape[0]
    return float(np.mean([
        np.linalg.norm(pos[i] - pos[j])
        for i in range(n) for j in range(i + 1, n)
    ]))


def classify_22(pos: np.ndarray, *, spread_thresh: float = 0.45) -> str:
    """Label layout: ``2,2``, ``collapsed``, or ``other``.

    (2,2) means two clusters of two agents each (harm axis split with
    matching pair structure typical of symmetric bifurcation).

    :param pos: ``(4, 2)`` positions ordered by clone index.
    :type pos: numpy.ndarray
    :param spread_thresh: Below this, call ``collapsed``.
    :type spread_thresh: float
    :returns: Classification label.
    :rtype: str
    """
    spread = pairwise_spread(pos)
    if spread < spread_thresh:
        return "collapsed"

    # Split on harm (axis 0): low vs high half
    harm = pos[:, 0]
    med = float(np.median(harm))
    low = set(np.where(harm <= med)[0].tolist())
    high = set(np.where(harm > med)[0].tolist())
    # Tie-break: if med splits unevenly, use k-means style 2 clusters on harm
    if len(low) != 2 or len(high) != 2:
        order = np.argsort(harm)
        low = set(order[:2].tolist())
        high = set(order[2:].tolist())

    if len(low) != 2 or len(high) != 2:
        return "other"

    # Within each harm-half, hallucination spread should be modest (paired niche)
    hal = pos[:, 1]
    low_hal_spread = float(np.std(hal[list(low)]))
    high_hal_spread = float(np.std(hal[list(high)]))

    # (2,2) niches: two low-harm clones + two high-harm clones, spread between halves
    harm_sep = float(np.mean(pos[list(high), 0]) - np.mean(pos[list(low), 0]))
    if harm_sep < 0.35:
        return "other"

    return "2,2"


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional CLI args.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--spread-thresh", type=float, default=0.45)
    args = p.parse_args(argv)

    rows: list[dict] = []
    for sigma_dir in sorted(args.root.iterdir()):
        if not sigma_dir.is_dir():
            continue
        sm = _SIGMA_RE.match(sigma_dir.name)
        if not sm:
            continue
        sigma = float(sm.group("val"))
        for seed_dir in sorted(sigma_dir.iterdir()):
            sd = _SEED_RE.match(seed_dir.name)
            if not sd:
                continue
            hist = seed_dir / "history.json"
            if not hist.is_file():
                continue
            with hist.open(encoding="utf-8") as fh:
                last = json.load(fh)[-1]
            names = sorted(last["positions"].keys())
            pos = np.stack(
                [np.asarray(last["positions"][n], dtype=float) for n in names],
            )
            spread = pairwise_spread(pos)
            label = classify_22(pos, spread_thresh=args.spread_thresh)
            rows.append({
                "sigma": sigma,
                "seed": int(sd.group("val")),
                "spread": spread,
                "label": label,
                "positions": {n: last["positions"][n] for n in names},
            })

    if not rows:
        print(f"no runs under {args.root}", file=sys.stderr)
        return 1

    for sigma in sorted({r["sigma"] for r in rows}):
        sub = [r for r in rows if r["sigma"] == sigma]
        n22 = sum(1 for r in sub if r["label"] == "2,2")
        nc = sum(1 for r in sub if r["label"] == "collapsed")
        no = sum(1 for r in sub if r["label"] == "other")
        print(f"\n=== sigma_fraction = {sigma:g}  (n={len(sub)}) ===")
        print(f"  (2,2): {n22}/{len(sub)}   collapsed: {nc}/{len(sub)}   other: {no}/{len(sub)}")
        print(f"  {'seed':>4} {'spread':>8} {'label':>10}")
        for r in sorted(sub, key=lambda x: x["seed"]):
            print(f"  {r['seed']:4d} {r['spread']:8.3f} {r['label']:10s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
