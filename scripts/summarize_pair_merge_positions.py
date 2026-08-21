"""Summarize final router layout and implied merge-agent positions from history.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from infl_ens.training.pool_dynamics import classify_layout
from infl_ens.utils.agent_init import harm_pair_indices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="e.g. results/pair_merge_round_sweep/r40/seed0")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--root", default="results/pair_merge_round_sweep/r40")
    args = parser.parse_args()

    names = ["clone-0", "clone-1", "clone-2", "clone-3"]
    merge_groups = [
        ("merge-low", ["clone-0", "clone-1"]),
        ("merge-high", ["clone-2", "clone-3"]),
    ]

    seeds = args.seeds if args.seeds is not None else list(range(10))
    root = Path(args.root)

    for seed in seeds:
        hist_path = root / f"seed{seed}" / "history.json"
        if not hist_path.is_file():
            print(f"seed{seed}: missing {hist_path}")
            continue
        last = json.loads(hist_path.read_text())[-1]
        pos = {n: np.asarray(last["positions"][n], dtype=float) for n in names}
        layout = classify_layout(np.stack([pos[n] for n in names]))
        low_idx, high_idx = harm_pair_indices(np.stack([pos[n] for n in names]))
        low_names = [names[i] for i in low_idx]
        high_names = [names[i] for i in high_idx]
        canonical = low_names == ["clone-0", "clone-1"] and high_names == [
            "clone-2",
            "clone-3",
        ]
        print(
            f"seed{seed}: layout={layout} canonical_01_vs_23={canonical} "
            f"low_harm={low_names} high_harm={high_names}",
        )

    run_dir = Path(args.run_dir)
    history = json.loads((run_dir / "history.json").read_text())
    first, last = history[0], history[-1]
    for tag, rec in [("after theory (round 0)", first), (f"final (round {last['round']})", last)]:
        pos_r = {n: np.asarray(rec["positions"][n], dtype=float) for n in names}
        print(f"\n=== {run_dir} {tag} [harm, halluc] ===")
        for n in names:
            p = pos_r[n]
            print(f"  {n}: [{p[0]:+.4f}, {p[1]:+.4f}]")
        if rec.get("theory_init"):
            ti = rec["theory_init"]
            print(f"  theory_init: layout={ti.get('theory_layout')} converged={ti.get('theory_converged')}")

    last = history[-1]
    pos = {n: np.asarray(last["positions"][n], dtype=float) for n in names}
    print(f"\n=== {run_dir} round {last['round']} router positions [harm, halluc] ===")
    for n in names:
        p = pos[n]
        print(f"  {n}: [{p[0]:+.4f}, {p[1]:+.4f}]")
    print("\n=== implied merge trainer positions (member mean; not logged in history) ===")
    for train_name, members in merge_groups:
        m = np.mean([pos[x] for x in members], axis=0)
        print(f"  {train_name}: [{m[0]:+.4f}, {m[1]:+.4f}]  (mean of {members})")

    P = np.stack([pos[n] for n in names])
    print("\n=== seed0 pairwise L2 distances ===")
    for i in range(4):
        for j in range(i + 1, 4):
            d = float(np.linalg.norm(P[i] - P[j]))
            print(f"  {names[i]} — {names[j]}: {d:.4f}")

    # Cross-seed mean positions
    all_p = []
    for seed in seeds:
        hp = root / f"seed{seed}" / "history.json"
        if hp.is_file():
            rec = json.loads(hp.read_text())[-1]
            all_p.append([rec["positions"][n] for n in names])
    if all_p:
        M = np.mean(np.asarray(all_p), axis=0)
        print("\n=== mean final positions across seeds ===")
        for i, n in enumerate(names):
            print(f"  {n}: [{M[i, 0]:+.4f}, {M[i, 1]:+.4f}]")
        ml = (M[0] + M[1]) / 2
        mh = (M[2] + M[3]) / 2
        print(f"  merge-low (0+1): [{ml[0]:+.4f}, {ml[1]:+.4f}]")
        print(f"  merge-high (2+3): [{mh[0]:+.4f}, {mh[1]:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
