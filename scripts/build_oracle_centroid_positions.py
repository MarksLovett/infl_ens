#!/usr/bin/env python3
"""Build colocated fixed_positions at 1-component oracle-prompt centroids.

Uses oracle merge assignments from cached merge NLL on the reference run
(ga_theory_pre seed-0). Each merge's two clones share the mean trait
coordinate of prompts for which that merge is oracle-optimal — not the k=2
sub-agent centers used in the falsified spread experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from infl_ens.evaluation.routing_eval import (
    final_round,
    load_final_positions,
    load_flat_partition_pool,
    parse_merge_groups,
    resolve_merge_adapters,
)
from infl_ens.training.__main__ import _load_splits, _load_yaml, _make_trait_space

MERGE_ORDER = [
    "merge-harm",
    "merge-hallucination",
    "merge-privacy",
    "merge-overrefusal",
    "merge-policy",
]
CLONE_GROUPS: list[tuple[str, str]] = [
    ("clone-0", "clone-1"),
    ("clone-2", "clone-3"),
    ("clone-4", "clone-5"),
    ("clone-6", "clone-7"),
    ("clone-8", "clone-9"),
]

PREDICTION = (
    "Shift should help most on low-elongation (unimodal) merges and plateau "
    "on strongly bimodal ones (harm, elongation ~2.33); uniform or null "
    "improvement implies placement is not the G-direction bottleneck."
)


def _pca_elongation(x: np.ndarray) -> float:
    """Ratio of largest to second PCA singular value."""
    if x.shape[0] < 3:
        return 1.0
    xc = x - x.mean(axis=0)
    _, s, _ = np.linalg.svd(xc, full_matrices=False)
    if len(s) < 2 or s[1] < 1e-12:
        return float("inf") if s[0] > 1e-12 else 1.0
    return float(s[0] / s[1])


def compute_oracle_centroids(
    *,
    router_config: Path,
    history_path: Path,
    merge_run_dir: Path,
    merge_nll_cache: Path,
    repo_root: Path,
    reference_init_path: Path | None = None,
    partition: str = "test",
    max_eval_records: int | None = 1000,
    seed: int = 0,
) -> dict:
    """Compute per-merge oracle centroids and colocated clone positions."""
    cfg = _load_yaml(router_config)
    cl = cfg.get("closed_loop", {})
    _clone_to_merge, config_merge_names = parse_merge_groups(cl)
    rnd = final_round(history_path)
    merge_names, merge_name_map = resolve_merge_adapters(
        merge_run_dir, rnd, config_merge_names,
    )

    prompts, _responses, bench_labels = load_flat_partition_pool(
        cfg,
        repo_root=repo_root,
        partition=partition,
        max_eval_records=max_eval_records,
        seed=seed,
    )
    merge_nll = np.load(merge_nll_cache)
    if merge_nll.shape[0] != len(prompts):
        raise ValueError("merge_nll row count mismatch")

    full_splits = _load_splits(cfg)
    space = _make_trait_space(cfg, full_splits)
    coords = np.asarray(space.project(prompts), dtype=float)
    oracle_idx = np.argmin(merge_nll, axis=1)

    ref_init: dict[str, list[float]] = {}
    if reference_init_path and reference_init_path.is_file():
        payload = json.loads(reference_init_path.read_text(encoding="utf-8"))
        ref_init = payload.get("positions", payload.get("final_positions", payload))

    agent_names = [a["name"] for a in cfg["agents"]]
    ga_final = load_final_positions(history_path, agent_names)
    ga_final_by_name = {
        name: ga_final[i] for i, name in enumerate(agent_names)
    }

    per_merge: dict[str, dict] = {}
    centroids: dict[str, list[float]] = {}
    positions: dict[str, list[float]] = {}

    for merge_cfg, (c0, c1) in zip(MERGE_ORDER, CLONE_GROUPS, strict=True):
        resolved = merge_name_map[merge_cfg]
        j = merge_names.index(resolved)
        mask = oracle_idx == j
        pts = coords[mask]
        n = int(pts.shape[0])
        if n == 0:
            raise ValueError(f"No oracle prompts for {resolved}")
        centroid = pts.mean(axis=0)
        centroids[resolved] = centroid.tolist()
        positions[c0] = centroid.tolist()
        positions[c1] = centroid.tolist()

        ref_pos = None
        shift_l2 = None
        if c0 in ref_init:
            ref_pos = np.asarray(ref_init[c0], dtype=float)
            shift_l2 = float(np.linalg.norm(centroid - ref_pos))

        ga_pos = np.asarray(ga_final_by_name[c0], dtype=float)
        ga_shift_l2 = float(np.linalg.norm(centroid - ga_pos))

        per_merge[resolved] = {
            "config_merge": merge_cfg,
            "n_oracle_prompts": n,
            "oracle_share": float(n / len(prompts)),
            "centroid": centroid.tolist(),
            "pca_elongation": _pca_elongation(pts),
            "mean_pairwise_l2": float(
                np.mean([
                    float(np.linalg.norm(pts[i] - pts[j]))
                    for i in range(n)
                    for j in range(i + 1, n)
                ]) if n > 1 else 0.0,
            ),
            "shift_l2_from_reference_init": shift_l2,
            "shift_l2_from_ga_final": ga_shift_l2,
            "reference_init_position": ref_pos.tolist() if ref_pos is not None else None,
            "ga_final_position": ga_pos.tolist(),
        }

    return {
        "source": {
            "router_config": str(router_config),
            "history": str(history_path),
            "merge_nll_cache": str(merge_nll_cache),
            "n_prompts": len(prompts),
            "trait_mean": np.asarray(space.mean, dtype=float).tolist(),
            "merge_names": merge_names,
        },
        "prediction": PREDICTION,
        "per_merge": per_merge,
        "positions": positions,
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--router-config",
        type=Path,
        default=Path("configs/benchmark/router/attribution_2x2/ga_theory_pre.yaml"),
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("results/attribution_2x2/ga_theory_pre/seed0/history.json"),
    )
    parser.add_argument(
        "--merge-run-dir",
        type=Path,
        default=Path("results/attribution_2x2/ga_theory_pre/seed0"),
    )
    parser.add_argument(
        "--merge-nll-cache",
        type=Path,
        default=Path("results/attribution_2x2/ga_theory_pre/seed0/merge_nll_test.npy"),
    )
    parser.add_argument(
        "--reference-init",
        type=Path,
        default=Path("results/hypercube_edge_gradient_ascent/fixed_positions.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/oracle_centroid_shift_init/fixed_positions.json"),
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=Path("results/oracle_centroid_shift_init/centroid_metadata.json"),
    )
    args = parser.parse_args()

    result = compute_oracle_centroids(
        router_config=args.router_config,
        history_path=args.history,
        merge_run_dir=args.merge_run_dir,
        merge_nll_cache=args.merge_nll_cache,
        repo_root=Path("."),
        reference_init_path=args.reference_init,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "init_mode": "oracle_centroid_colocated",
        "component_count": 1,
        "prediction": PREDICTION,
        "source": result["source"],
        "per_merge": result["per_merge"],
        "positions": result["positions"],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== oracle 1-component centroids (colocated init) ===")
    print(f"prediction: {PREDICTION}")
    print(
        f"{'merge':<22} {'n':>5} {'elong':>6} {'shift_init':>11} "
        f"{'shift_ga_fin':>12}",
    )
    for merge, row in result["per_merge"].items():
        print(
            f"{merge:<22} {row['n_oracle_prompts']:5d} "
            f"{row['pca_elongation']:6.2f} "
            f"{row['shift_l2_from_reference_init']:11.4f} "
            f"{row['shift_l2_from_ga_final']:12.4f}",
        )
    print(f"\nwrote {args.output}")
    print(f"wrote {args.metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
