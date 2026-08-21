#!/usr/bin/env python3
"""Evaluate trait-whitening arms vs reference baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from infl_ens.evaluation.routing_eval import (
    aggregate_clone_g_to_merge,
    final_round,
    load_final_positions,
    load_flat_partition_pool,
    parse_merge_groups,
    resolve_merge_adapters,
)
from infl_ens.inflgame.router.allocation import allocation_weights
from infl_ens.training.__main__ import _load_splits, _load_yaml, _make_trait_space, _sigma_from_cfg

BENCH_TO_AXIS = {
    "beavertails": "harm",
    "halueval": "hallucination",
    "ai4privacy": "privacy",
    "orbench": "overrefusal",
    "do_not_answer": "policy",
}

MERGE_GROUPS = [
    ("merge-harm", "clone-0", "clone-1"),
    ("merge-hallucination", "clone-2", "clone-3"),
    ("merge-privacy", "clone-4", "clone-5"),
    ("merge-overrefusal", "clone-6", "clone-7"),
    ("merge-policy", "clone-8", "clone-9"),
]

REF = {
    "g_argmax_agreement": 0.742,
    "oracle_minus_learned": -0.0379,
    "theory_oracle_centroid_l2": 0.176,
}


def _alignment_l2(
    history_path: Path,
    merge_nll: np.ndarray,
    space,
    cfg: dict,
    merge_run_dir: Path,
) -> dict:
    """Mean L2 from final merge centers to oracle centroids in arm space."""
    cl = cfg.get("closed_loop", {})
    agent_names = [a["name"] for a in cfg["agents"]]
    _ctm, config_merge_names = parse_merge_groups(cl)
    rnd = final_round(history_path)
    merge_names, merge_name_map = resolve_merge_adapters(
        merge_run_dir, rnd, config_merge_names,
    )
    prompts, _r, _b = load_flat_partition_pool(
        cfg, repo_root=Path("."), partition="test", max_eval_records=1000, seed=0,
    )
    coords = np.asarray(space.project(prompts), dtype=float)
    oracle_idx = np.argmin(merge_nll[: len(prompts)], axis=1)
    final_pos = load_final_positions(history_path, agent_names)
    final_by_name = {n: final_pos[i] for i, n in enumerate(agent_names)}

    per_merge: dict[str, dict] = {}
    dists: list[float] = []
    for merge_cfg, c0, _c1 in MERGE_GROUPS:
        resolved = merge_name_map[merge_cfg]
        j = merge_names.index(resolved)
        mask = oracle_idx == j
        if not mask.any():
            continue
        oracle_centroid = coords[mask].mean(axis=0)
        theory_center = final_by_name[c0]
        d = float(np.linalg.norm(theory_center - oracle_centroid))
        per_merge[resolved] = {
            "l2_theory_to_oracle_centroid": d,
            "n_oracle_prompts": int(mask.sum()),
        }
        dists.append(d)
    return {
        "mean_l2_theory_to_oracle_centroid": float(np.mean(dists)) if dists else float("nan"),
        "per_merge": per_merge,
    }


def evaluate_arm(
    *,
    arm: str,
    config_path: Path,
    run_dir: Path,
    merge_nll_cache: Path,
    baseline_run_dir: Path,
) -> dict:
    """Metrics for one whitening arm."""
    cfg = _load_yaml(config_path)
    hist = run_dir / "history.json"
    routing_path = run_dir / "routing_weight_comparison.json"

    row: dict = {"arm": arm, "config": str(config_path), "run_dir": str(run_dir)}
    if routing_path.is_file():
        flat = json.loads(routing_path.read_text(encoding="utf-8"))["flat"]
        learned = flat["learned_routing_expected_nll"]
        oracle = flat["oracle_routing_nll"]
        row.update({
            "learned_expected_nll": learned,
            "oracle_nll": oracle,
            "oracle_minus_learned": oracle - learned,
            "g_argmax_agreement": flat.get("routing_agreement_argmax"),
            "per_benchmark": flat.get("per_benchmark", {}),
        })

    if hist.is_file():
        history = json.loads(hist.read_text(encoding="utf-8"))
        row["within_merge_round0"] = history[0].get("agent_geometry", {}).get(
            "within_merge_l2", {},
        )
        row["within_merge_final"] = history[-1].get("agent_geometry", {}).get(
            "within_merge_l2", {},
        )
        merge_nll_path = run_dir / "merge_nll_test.npy"
        if not merge_nll_path.is_file():
            merge_nll_path = merge_nll_cache
        if merge_nll_path.is_file():
            merge_nll = np.load(merge_nll_path)
            splits = _load_splits(cfg)
            space = _make_trait_space(cfg, splits)
            row["alignment"] = _alignment_l2(
                hist, merge_nll, space, cfg, run_dir,
            )

    return row


def _decisive_read(rows: list[dict]) -> str:
    """Heuristic verdict string."""
    by_arm = {r["arm"]: r for r in rows}
    base = by_arm.get("baseline", {})
    std = by_arm.get("standardize", {})
    wh = by_arm.get("whiten", {})
    if not base:
        return "incomplete"
    base_gap = base.get("oracle_minus_learned", 0)
    std_gap = std.get("oracle_minus_learned", base_gap)
    wh_gap = wh.get("oracle_minus_learned", base_gap)
    base_align = base.get("alignment", {}).get("mean_l2_theory_to_oracle_centroid", 0.176)
    wh_align = wh.get("alignment", {}).get("mean_l2_theory_to_oracle_centroid", base_align)

    gap_help = min(std_gap, wh_gap) < base_gap - 0.005
    align_help = wh_align < base_align - 0.02

    if gap_help and align_help:
        if std_gap <= wh_gap + 0.003 and std_gap < base_gap - 0.005:
            return "axis_scale_distortion (standardize captures gain)"
        if wh_gap < std_gap - 0.003:
            return "correlation_distortion (whiten needed beyond standardize)"
        return "isotropy_distortion_fixed (whitening helps)"
    if not gap_help and not align_help:
        guard = (
            "clean_null only valid if sigma was data-relative or rescaled "
            "(see sigma_case in summary)"
        )
        return f"clean_null (linear coordinate distortion not the cause); {guard}"
    return "mixed_partial"


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/trait_whitening")
    parser.add_argument(
        "--merge-nll-cache",
        default="results/attribution_2x2/ga_theory_pre/seed0/merge_nll_test.npy",
    )
    parser.add_argument(
        "--baseline-run-dir",
        default="results/seven_axis_collapse_hypercube_ga_baseline/seed0",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    arms = ["baseline", "standardize", "whiten"]
    rows = []
    for arm in arms:
        cfg = Path("configs/benchmark/router/trait_whitening") / f"{arm}.yaml"
        run_dir = root / arm / "seed0"
        if not run_dir.is_dir():
            continue
        rows.append(evaluate_arm(
            arm=arm,
            config_path=cfg,
            run_dir=run_dir,
            merge_nll_cache=Path(args.merge_nll_cache),
            baseline_run_dir=Path(args.baseline_run_dir),
        ))

    verdict = _decisive_read(rows)
    print("=== trait whitening evaluation ===")
    print(f"reference: agree={REF['g_argmax_agreement']:.3f}  "
          f"oracle−learned={REF['oracle_minus_learned']:+.4f}  "
          f"align L2≈{REF['theory_oracle_centroid_l2']:.3f}")
    for row in rows:
        print(
            f"{row['arm']}: oracle−learned {row.get('oracle_minus_learned', float('nan')):+.4f}  "
            f"agree {row.get('g_argmax_agreement', float('nan')):.3f}  "
            f"align L2 {row.get('alignment', {}).get('mean_l2_theory_to_oracle_centroid', float('nan')):.4f}",
        )
    print(f"\ndecisive_read: {verdict}")
    print("\n--- per axis (agree, Δ_exp) ---")
    print(f"{'arm':<14} {'bench':<16} {'agree':>7} {'Δ_exp':>8}")
    for row in rows:
        for bench, b in sorted(row.get("per_benchmark", {}).items()):
            gap = b.get("learned_expected_nll", 0) - b.get("oracle_nll", 0)
            print(
                f"{row['arm']:<14} {bench:<16} "
                f"{b.get('agreement_argmax', float('nan')):7.3f} {gap:+8.4f}",
            )

    payload = {"reference": REF, "arms": rows, "decisive_read": verdict}
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
