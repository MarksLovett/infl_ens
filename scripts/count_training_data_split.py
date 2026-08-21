"""Count pooled vs per-specialist training examples from closed-loop history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze_history(path: Path) -> dict:
    """Summarize training example counts.

    :param path: ``history.json`` path.
    :type path: pathlib.Path
    :returns: Summary statistics.
    :rtype: dict
    """
    records = json.loads(path.read_text(encoding="utf-8"))
    agents = sorted(records[0]["agent_prompts"].keys())
    specialist_totals = {a: 0 for a in agents}
    pooled_per_round: list[int] = []
    per_round_specialist: list[dict[str, int]] = []

    for rec in records:
        ap = rec["agent_prompts"]
        counts = {a: len(ap.get(a, [])) for a in agents}
        pooled = sum(counts.values())
        pooled_per_round.append(pooled)
        per_round_specialist.append(counts)
        for a in agents:
            specialist_totals[a] += counts[a]

    pooled_total = sum(pooled_per_round)
    mean_spec = float(np.mean(list(specialist_totals.values())))
    return {
        "n_rounds": len(records),
        "agents": agents,
        "pooled_total": pooled_total,
        "pooled_per_round_mean": float(np.mean(pooled_per_round)),
        "specialist_totals": specialist_totals,
        "mean_specialist_total": mean_spec,
        "ratio_pooled_to_mean_specialist": pooled_total / mean_spec if mean_spec else float("nan"),
        "per_round_specialist": per_round_specialist,
        "pooled_per_round": pooled_per_round,
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="results/pairs_near_eq_round_sweep/r40",
        help="Directory with seed*/history.json",
    )
    args = parser.parse_args()
    root = Path(args.root)
    seeds = sorted(root.glob("seed*/history.json"))
    if not seeds:
        print(f"no histories under {root}")
        return 1

    rows = []
    for hp in seeds:
        s = analyze_history(hp)
        s["seed"] = hp.parent.name
        rows.append(s)

    agents = rows[0]["agents"]
    print("=== Total training examples over all rounds ===")
    print("(pooled = union of all routed examples each round; specialist = routed subset only)\n")
    hdr = f"{'seed':<8} {'pooled':>8}" + "".join(f"{a:>9}" for a in agents) + f"{'ratio':>8}"
    print(hdr)
    for s in rows:
        t = s["specialist_totals"]
        line = (
            f"{s['seed']:<8} {s['pooled_total']:8d}"
            + "".join(f"{t[a]:9d}" for a in agents)
            + f"{s['ratio_pooled_to_mean_specialist']:7.2f}x"
        )
        print(line)

    pooled = [s["pooled_total"] for s in rows]
    print(f"\nMean over seeds: pooled={np.mean(pooled):.0f} +/- {np.std(pooled, ddof=1):.0f}")
    for a in agents:
        v = [s["specialist_totals"][a] for s in rows]
        print(f"  {a}: {np.mean(v):.0f} +/- {np.std(v, ddof=1):.0f}")
    mean_spec = np.mean([s["mean_specialist_total"] for s in rows])
    print(f"  mean specialist: {mean_spec:.0f}")
    print(f"  pooled / mean specialist: {np.mean(pooled)/mean_spec:.2f}x")

    s0 = rows[0]
    print(f"\n=== Per round (example: {s0['seed']}) ===")
    print(f"batch size (pooled per round) mean = {s0['pooled_per_round_mean']:.1f}")
    for a in agents:
        per_r = [pr[a] for pr in s0["per_round_specialist"]]
        print(f"  {a}: mean {np.mean(per_r):.1f}/round, total {sum(per_r)}")
    print(f"  ratio pooled/mean specialist per round: {s0['pooled_per_round_mean'] / np.mean([np.mean([pr[a] for pr in s0['per_round_specialist']]) for a in agents]):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
