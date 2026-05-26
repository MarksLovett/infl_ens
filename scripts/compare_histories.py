"""Compare position trajectories between two history.json files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def compare(label: str, path_a: Path, path_b: Path) -> None:
    if not path_a.exists() or not path_b.exists():
        print(f"{label}: MISSING a={path_a.exists()} b={path_b.exists()}")
        return
    with path_a.open() as fa, path_b.open() as fb:
        ha, hb = json.load(fa), json.load(fb)
    names = list(ha[0]["positions"].keys())
    print(f"=== {label} ===")
    print(f"  loss_reweight: {ha[0].get('loss_reweight')} vs {hb[0].get('loss_reweight')}")
    for r in range(min(len(ha), len(hb))):
        max_gap = max(
            np.linalg.norm(
                np.asarray(ha[r]["positions"][n])
                - np.asarray(hb[r]["positions"][n])
            )
            for n in names
        )
        print(f"  round {r}: max L2 gap = {max_gap:.6f}")
    print()


def main() -> None:
    pairs = [
        (
            "position_only_cum vs loss_reweight_cum (matched config)",
            ROOT / "results/safety_truth_n4_r10_position_only_cum/history.json",
            ROOT / "results/safety_truth_n4_r10_loss_reweight_cum/history.json",
        ),
        (
            "position_only_long r10 vs loss_reweight r10",
            ROOT / "results/position_only_long_round_sweep/r10/history.json",
            ROOT / "results/loss_reweight_cum_round_sweep/r10/history.json",
        ),
    ]
    for label, a, b in pairs:
        compare(label, a, b)


if __name__ == "__main__":
    main()
