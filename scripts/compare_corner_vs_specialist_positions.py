"""Compare final router positions: proximity merge vs pairs_near_eq specialists."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from infl_ens.utils.agent_init import harm_pair_indices


def _load_final_positions(run_dir: Path) -> dict[str, np.ndarray]:
    hist = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    final = hist[-1]
    return {n: np.asarray(v, dtype=float) for n, v in final["positions"].items()}


def _corner_roles(positions: dict[str, np.ndarray], names: list[str]) -> dict[str, str]:
    entries = []
    P = np.stack([positions[n] for n in names])
    low_idx, high_idx = harm_pair_indices(P)
    low_names = sorted(names[int(i)] for i in low_idx)
    high_names = sorted(names[int(i)] for i in high_idx)
    for n in low_names:
        entries.append((f"role-{n}", low_names, positions[n]))
    for n in high_names:
        entries.append((f"role-{n}", high_names, positions[n]))
    # use pair centroids for role assignment
    merge_like = [
        ("low", low_names, np.mean([positions[m] for m in low_names], axis=0)),
        ("high", high_names, np.mean([positions[m] for m in high_names], axis=0)),
    ]
    return {
        "corner-low-harm": low_names,
        "corner-high-harm": high_names,
        "centroid_low": merge_like[0][2],
        "centroid_high": merge_like[1][2],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proximity-root", default="results/proximity_merge_round_sweep/r40")
    parser.add_argument("--specialist-root", default="results/pairs_near_eq_round_sweep/r40")
    parser.add_argument("--corner-agg", default="results/proximity_merge_corner_aggregate.json")
    args = parser.parse_args()

    prox_root = Path(args.proximity_root)
    spec_root = Path(args.specialist_root)
    names = ["clone-0", "clone-1", "clone-2", "clone-3"]

    corner_nll: dict[str, dict[str, float]] = {}
    if Path(args.corner_agg).is_file():
        agg = json.loads(Path(args.corner_agg).read_text(encoding="utf-8"))
        for row in agg["per_seed"]:
            role = row["corner_role"]
            corner_nll.setdefault(role, {"beavertails": [], "halueval": []})
            for sc in row["benchmark_scores"]:
                corner_nll[role][sc["benchmark"]].append(sc["mean_nll"])

    print("=== Final positions: proximity vs specialists (per seed) ===\n")
    dist_low, dist_high = [], []
    spec_clone_at_low: dict[str, list[float]] = {n: [] for n in names}

    for seed in range(10):
        pd = prox_root / f"seed{seed}"
        sd = spec_root / f"seed{seed}"
        if not pd.is_dir() or not sd.is_dir():
            continue
        ppos = _load_final_positions(pd)
        spos = _load_final_positions(sd)
        proles = _corner_roles(ppos, names)
        sroles = _corner_roles(spos, names)

        cl_p, ch_p = proles["centroid_low"], proles["centroid_high"]
        cl_s, ch_s = sroles["centroid_low"], sroles["centroid_high"]
        d_low = float(np.linalg.norm(cl_p - cl_s))
        d_high = float(np.linalg.norm(ch_p - ch_s))
        dist_low.append(d_low)
        dist_high.append(d_high)

        print(f"seed{seed}:")
        print(f"  proximity  low harm centroid: [{cl_p[0]:+.4f}, {cl_p[1]:+.4f}]  members {proles['corner-low-harm']}")
        print(f"  specialist low harm centroid: [{cl_s[0]:+.4f}, {cl_s[1]:+.4f}]  members {sroles['corner-low-harm']}")
        print(f"  proximity  high harm cent.:  [{ch_p[0]:+.4f}, {ch_p[1]:+.4f}]  members {proles['corner-high-harm']}")
        print(f"  specialist high harm cent.:  [{ch_s[0]:+.4f}, {ch_s[1]:+.4f}]  members {sroles['corner-high-harm']}")
        print(f"  centroid L2 gap  low={d_low:.4f}  high={d_high:.4f}")

        # specialist NLL at low corner: which clone indices sit on low harm?
        for n in sroles["corner-low-harm"]:
            spec_clone_at_low[n].append(seed)

    print("\n=== Mean centroid gap (proximity vs specialist), 10 seeds ===")
    print(f"  low-harm  corner: {statistics.mean(dist_low):.4f} ± {statistics.stdev(dist_low):.4f}")
    print(f"  high-harm corner: {statistics.mean(dist_high):.4f} ± {statistics.stdev(dist_high):.4f}")

    print("\n=== Which specialists usually sit on low-harm corner (pairs_near_eq)? ===")
    for n in names:
        print(f"  {n}: low-harm in {len(spec_clone_at_low[n])}/10 seeds")

    if corner_nll:
        print("\n=== Proximity merge NLL by corner (10-seed mean) ===")
        for role in ("corner-low-harm", "corner-high-harm"):
            bt = statistics.mean(corner_nll[role]["beavertails"])
            he = statistics.mean(corner_nll[role]["halueval"])
            print(f"  {role}: beaver {bt:.4f}  halu {he:.4f}")

    # training volume from last round merge counts
    print("\n=== Routed prompts (round 39) — proximity merge, seed0 ===")
    h = json.loads((prox_root / "seed0/history.json").read_text(encoding="utf-8"))
    r39 = h[-1]
    ap = r39.get("agent_prompts", {})
    for n in names:
        print(f"  {n}: {len(ap.get(n, []))}")
    mpc = r39.get("merge_prompt_counts", {})
    for k, v in sorted(mpc.items()):
        print(f"  {k}: {v}")

    print("\n=== Routed prompts (round 39) — specialists, seed0 ===")
    h2 = json.loads((spec_root / "seed0/history.json").read_text(encoding="utf-8"))
    ap2 = h2[-1].get("agent_prompts", {})
    for n in names:
        print(f"  {n}: {len(ap2.get(n, []))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
