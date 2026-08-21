#!/usr/bin/env python3
"""Verify oracle-centroid shift persists through training (not cosmetic init).

The experiment is invalid if theory-pre or closed-loop dynamics pull merge
centers back toward the reference GA positions. This script traces per-merge
effective centers (colocated pair position) across:

* intended oracle centroid (target)
* reference GA final positions (revert baseline)
* post-init / post-theory-pre (from ``history[0].theory_init``)
* every closed-loop round in ``history.json``

Fails the gate when final centers are closer to reference GA than to oracle
centroids, or when merge centers drift far from the oracle target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infl_ens.evaluation.routing_eval import load_final_positions  # noqa: E402

MERGE_GROUPS: list[tuple[str, str, str]] = [
    ("merge-harm", "clone-0", "clone-1"),
    ("merge-hallucination", "clone-2", "clone-3"),
    ("merge-privacy", "clone-4", "clone-5"),
    ("merge-overrefusal", "clone-6", "clone-7"),
    ("merge-policy", "clone-8", "clone-9"),
]


def _center(pos_map: dict[str, list[float]], clone: str) -> np.ndarray:
    """Effective merge center (colocated pair uses clone position)."""
    return np.asarray(pos_map[clone], dtype=float)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def analyze_centroid_persistence(
    *,
    history_path: Path,
    centroid_metadata_path: Path,
    reference_history_path: Path,
    max_final_drift_l2: float = 0.25,
    max_within_merge_l2: float = 1e-6,
) -> dict[str, Any]:
    """Build per-round center trajectory and persistence verdict."""
    history = json.loads(history_path.read_text(encoding="utf-8"))
    meta = json.loads(centroid_metadata_path.read_text(encoding="utf-8"))
    oracle_targets: dict[str, np.ndarray] = {}
    for merge, _c0, _c1 in MERGE_GROUPS:
        for key, row in meta.get("per_merge", {}).items():
            if row.get("config_merge") == merge:
                oracle_targets[merge] = np.asarray(row["centroid"], dtype=float)
                break

    ref_agent_names = [c for _, c0, c1 in MERGE_GROUPS for c in (c0, c1)]
    ref_pos = load_final_positions(reference_history_path, ref_agent_names)
    ref_by_name = {name: ref_pos[i] for i, name in enumerate(ref_agent_names)}
    ref_ga_final: dict[str, np.ndarray] = {
        merge: _center(ref_by_name, c0) for merge, c0, _c1 in MERGE_GROUPS
    }

    phases: list[dict[str, Any]] = []

    if history and history[0].get("theory_init"):
        ti = history[0]["theory_init"]
        if "theory_end" in ti:
            phases.append({
                "phase": "post_theory_pre",
                "positions": ti["theory_end"],
            })
        geom = ti.get("agent_geometry") or ti.get("theory_pre", {}).get(
            "agent_geometry",
        )
        if geom and "post_init" in str(geom.get("geometry_phase", "")):
            pass

    for entry in history:
        r = entry.get("round")
        if r is None:
            continue
        phases.append({
            "phase": f"round_{r}",
            "positions": entry["positions"],
            "within_merge_l2": (
                entry.get("agent_geometry", {}).get("within_merge_l2", {})
            ),
        })

    per_merge_traj: dict[str, list[dict]] = {m: [] for m, _, _ in MERGE_GROUPS}
    within_violations: list[str] = []

    for phase_row in phases:
        pos_map = phase_row["positions"]
        phase = phase_row["phase"]
        w_within = phase_row.get("within_merge_l2", {})
        for merge, c0, c1 in MERGE_GROUPS:
            center = _center(pos_map, c0)
            p1 = _center(pos_map, c1)
            within = _dist(center, p1)
            d_oracle = _dist(center, oracle_targets[merge])
            d_ref = _dist(center, ref_ga_final[merge])
            per_merge_traj[merge].append({
                "phase": phase,
                "center": center.tolist(),
                "dist_to_oracle_centroid": d_oracle,
                "dist_to_reference_ga_final": d_ref,
                "closer_to_oracle_than_ref": d_oracle < d_ref,
                "within_merge_l2": within,
            })
            if within > max_within_merge_l2:
                within_violations.append(f"{merge}@{phase}:{within:.2e}")

    final_phase = phases[-1]["phase"] if phases else "none"
    per_merge_final: dict[str, dict] = {}
    n_closer_oracle = 0
    n_merges = len(MERGE_GROUPS)
    mean_d_oracle = 0.0
    mean_d_ref = 0.0
    for merge, traj in per_merge_traj.items():
        if not traj:
            continue
        fin = traj[-1]
        per_merge_final[merge] = fin
        if fin["closer_to_oracle_than_ref"]:
            n_closer_oracle += 1
        mean_d_oracle += fin["dist_to_oracle_centroid"]
        mean_d_ref += fin["dist_to_reference_ga_final"]

    mean_d_oracle /= max(n_merges, 1)
    mean_d_ref /= max(n_merges, 1)

    post_theory = next(
        (p for p in phases if p["phase"] == "post_theory_pre"), None,
    )
    post_theory_drift: dict[str, float] | None = None
    if post_theory is not None:
        post_theory_drift = {}
        for merge, c0, _c1 in MERGE_GROUPS:
            center = _center(post_theory["positions"], c0)
            post_theory_drift[merge] = _dist(center, oracle_targets[merge])

    persisted = (
        n_closer_oracle >= n_merges - 1
        and mean_d_oracle < mean_d_ref
        and mean_d_oracle <= max_final_drift_l2
    )
    cosmetic = (
        mean_d_ref < mean_d_oracle
        or n_closer_oracle < n_merges // 2
    )

    if cosmetic:
        verdict = "cosmetic_shift_reverted"
    elif persisted:
        verdict = "centroids_persisted"
    else:
        verdict = "ambiguous_partial_drift"

    return {
        "history": str(history_path),
        "reference_history": str(reference_history_path),
        "centroid_metadata": str(centroid_metadata_path),
        "n_phases": len(phases),
        "final_phase": final_phase,
        "max_final_drift_l2": max_final_drift_l2,
        "per_merge_trajectory": per_merge_traj,
        "per_merge_final": per_merge_final,
        "post_theory_pre_drift_l2": post_theory_drift,
        "summary": {
            "mean_final_dist_to_oracle": mean_d_oracle,
            "mean_final_dist_to_ref_ga": mean_d_ref,
            "n_merges_closer_to_oracle": n_closer_oracle,
            "within_merge_violations": within_violations,
        },
        "verdict": verdict,
        "ok": persisted and not cosmetic,
    }


def _print_summary(result: dict) -> None:
    """Human-readable persistence report."""
    s = result["summary"]
    print("=== oracle-centroid persistence (effective G centers) ===")
    print(f"phases={result['n_phases']}  final={result['final_phase']}")
    print(
        f"mean final L2→oracle: {s['mean_final_dist_to_oracle']:.4f}  "
        f"mean final L2→ref GA: {s['mean_final_dist_to_ref_ga']:.4f}  "
        f"merges closer to oracle: {s['n_merges_closer_to_oracle']}/5",
    )
    if result.get("post_theory_pre_drift_l2"):
        print("post_theory_pre drift L2→oracle:")
        for merge, d in result["post_theory_pre_drift_l2"].items():
            print(f"  {merge}: {d:.4f}")
    print(f"\n{'merge':<22} {'L2→oracle':>10} {'L2→refGA':>10} {'oracle?':>8}")
    for merge, fin in result["per_merge_final"].items():
        print(
            f"{merge:<22} {fin['dist_to_oracle_centroid']:10.4f} "
            f"{fin['dist_to_reference_ga_final']:10.4f} "
            f"{'yes' if fin['closer_to_oracle_than_ref'] else 'no':>8}",
        )
    if s["within_merge_violations"]:
        print("\nwithin_merge violations:")
        for v in s["within_merge_violations"][:10]:
            print(f"  {v}")
    print(f"\nverdict: {result['verdict']}  ok={result['ok']}")
    if result["verdict"] == "cosmetic_shift_reverted":
        print(
            "FAIL: centers reverted toward reference GA — "
            "routing results would not test oracle-centroid placement.",
        )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        default="results/oracle_centroid_shift/ga_theory_pre/seed0/history.json",
    )
    parser.add_argument(
        "--centroid-metadata",
        default="results/oracle_centroid_shift_init/centroid_metadata.json",
    )
    parser.add_argument(
        "--reference-history",
        default="results/attribution_2x2/ga_theory_pre/seed0/history.json",
    )
    parser.add_argument("--max-final-drift-l2", type=float, default=0.25)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = analyze_centroid_persistence(
        history_path=ROOT / args.history,
        centroid_metadata_path=ROOT / args.centroid_metadata,
        reference_history_path=ROOT / args.reference_history,
        max_final_drift_l2=args.max_final_drift_l2,
    )
    _print_summary(result)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output_json}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
