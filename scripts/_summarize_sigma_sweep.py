"""Quick summary of position_only sigma sweep theory vs SFT gaps."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "results" / "position_only_cum_sigma_sweep"
REWEEP = ROOT / "results" / "loss_reweight_cum_sigma_sweep"


def _sigma_frac_from_slug(slug: str) -> float | None:
    m = re.match(r"sigma(.+)", slug)
    return float(m.group(1)) if m else None


def summarize_sweep(root: Path, label: str) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    if not root.exists():
        print("  (sweep dir missing)")
        return
    for slug in sorted(p.name for p in root.iterdir() if p.is_dir()):
        hist = root / slug / "history.json"
        theo = root / slug / "theory_vs_sft.json"
        print(f"\n--- {slug} ---")
        if not hist.exists():
            print("  history: missing")
            continue
        h = json.load(hist.open())
        names = sorted(h[-1]["positions"])
        P = np.array([h[-1]["positions"][n] for n in names])
        spread = float(np.linalg.norm(P[:, None] - P[None, :], axis=-1).max())
        print(f"  rounds={len(h)}  max_pairwise_spread={spread:.3f}")
        print(f"  SFT ends: " + ", ".join(
            f"{n}=[{P[i,0]:.2f},{P[i,1]:.2f}]" for i, n in enumerate(names)
        ))
        sf = _sigma_frac_from_slug(slug)
        if sf is not None:
            regime = "unstable" if sf < 1.0 else "stable (symmetric NE)"
            print(f"  intended sigma_fraction={sf}  ({regime})")
        if theo.exists():
            d = json.load(theo.open())
            ratio = d["sigma"] / d["sigma_star"]
            gaps = [a["gap"] for a in d["agents"]]
            print(f"  theory_vs_sft sigma/sigma*={ratio:.3f}  max_gap={max(gaps):.3f}")
            if sf is not None and abs(ratio - sf) > 0.05:
                print(f"  ** WARNING: theory computed at wrong sigma "
                      f"(json says {ratio:.2f}*, run used {sf})")


if __name__ == "__main__":
    summarize_sweep(SWEEP, "position_only_cum_sigma_sweep")
    summarize_sweep(REWEEP, "loss_reweight_cum_sigma_sweep (reference)")
