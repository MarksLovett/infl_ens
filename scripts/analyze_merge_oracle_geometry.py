#!/usr/bin/env python3
"""Per-merge trait geometry of oracle-winning prompts.

Uses cached merge NLL + trait projections to test whether each merge's
oracle-assigned prompts are unimodal or multimodal in trait space.
Informs where to place spread pair sub-agents for the gap experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from infl_ens.evaluation.routing_eval import (
    final_round,
    load_flat_partition_pool,
    parse_merge_groups,
    resolve_merge_adapters,
)
from infl_ens.training.__main__ import _load_splits, _load_yaml, _make_trait_space


def _pairwise_mean_dist(x: np.ndarray) -> float:
    """Mean pairwise L2 among rows."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    dsum = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            dsum += float(np.linalg.norm(x[i] - x[j]))
            cnt += 1
    return dsum / cnt


def _pca_elongation(x: np.ndarray) -> float:
    """Ratio of largest to second PCA eigenvalue (spread shape)."""
    if x.shape[0] < 3:
        return 1.0
    xc = x - x.mean(axis=0)
    _, s, _ = np.linalg.svd(xc, full_matrices=False)
    if len(s) < 2 or s[1] < 1e-12:
        return float("inf") if s[0] > 1e-12 else 1.0
    return float(s[0] / s[1])


def _gmm_bic_delta(x: np.ndarray) -> tuple[float, float, float]:
    """BIC(1) − BIC(2); positive favors two components."""
    if x.shape[0] < 8:
        return 0.0, float("nan"), float("nan")
    g1 = GaussianMixture(n_components=1, random_state=0).fit(x)
    g2 = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(x)
    bic1 = float(g1.bic(x))
    bic2 = float(g2.bic(x))
    return bic1 - bic2, bic1, bic2


def analyze_merge_oracle_geometry(
    *,
    router_config: Path,
    history_path: Path,
    merge_run_dir: Path,
    repo_root: Path,
    merge_nll_cache: Path,
    partition: str = "test",
    max_eval_records: int | None = 1000,
    seed: int = 0,
    bic_threshold: float = 10.0,
) -> dict:
    """Summarize oracle-prompt trait geometry per merge."""
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

    per_merge: dict[str, dict] = {}
    for j, merge in enumerate(merge_names):
        mask = oracle_idx == j
        pts = coords[mask]
        n = int(pts.shape[0])
        row: dict = {
            "merge": merge,
            "n_oracle_prompts": n,
            "oracle_share": float(n / len(prompts)),
        }
        if n == 0:
            row["geometry"] = "empty"
            per_merge[merge] = row
            continue

        bic_delta, bic1, bic2 = _gmm_bic_delta(pts)
        sil = float("nan")
        sub_agent_centers: list[list[float]] = []
        if n >= 8:
            km = KMeans(n_clusters=min(2, n), random_state=0, n_init=10)
            labels = km.fit_predict(pts)
            if len(set(labels)) > 1:
                sil = float(silhouette_score(pts, labels))
            sub_agent_centers = [
                c.tolist() for c in km.cluster_centers_
            ]

        mean_pairwise = _pairwise_mean_dist(pts)
        elongation = _pca_elongation(pts)
        bimodal = bic_delta > bic_threshold and n >= 20
        row.update({
            "mean_pairwise_l2": mean_pairwise,
            "pca_elongation": elongation,
            "gmm_bic1_minus_bic2": bic_delta,
            "gmm_bic_1comp": bic1,
            "gmm_bic_2comp": bic2,
            "kmeans2_silhouette": sil,
            "suggested_sub_agent_positions": sub_agent_centers,
            "bimodal_candidate": bimodal,
            "spread_recommended": bimodal,
        })
        bench_counts: dict[str, int] = {}
        for b, m in zip(bench_labels, mask, strict=True):
            if m:
                bench_counts[b] = bench_counts.get(b, 0) + 1
        row["oracle_by_benchmark"] = bench_counts
        per_merge[merge] = row

    return {
        "run": {
            "router_config": str(router_config),
            "history": str(history_path),
            "n_prompts": len(prompts),
            "merge_names": merge_names,
        },
        "bic_threshold": bic_threshold,
        "per_merge": per_merge,
    }


def _print_summary(result: dict) -> None:
    """Stdout table."""
    print("=== oracle-prompt geometry per merge ===")
    print(
        f"{'merge':<22} {'n':>5} {'share':>6} {'mpw':>6} "
        f"{'elong':>6} {'ΔBIC':>8} {'sil':>6} {'spread?':>8}",
    )
    for merge, row in result["per_merge"].items():
        if row.get("geometry") == "empty":
            print(f"{merge:<22} {0:5d} {'—':>6}")
            continue
        sil = row["kmeans2_silhouette"]
        sil_s = f"{sil:6.3f}" if np.isfinite(sil) else "   nan"
        flag = "yes" if row["spread_recommended"] else "no"
        print(
            f"{merge:<22} {row['n_oracle_prompts']:5d} "
            f"{row['oracle_share']:6.3f} "
            f"{row['mean_pairwise_l2']:6.3f} "
            f"{row['pca_elongation']:6.2f} "
            f"{row['gmm_bic1_minus_bic2']:8.1f} {sil_s} {flag:>8}",
        )
    rec = [
        m for m, r in result["per_merge"].items()
        if r.get("spread_recommended")
    ]
    print(f"\nspread candidates (ΔBIC > {result['bic_threshold']}): {rec or 'none'}")


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
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--bic-threshold", type=float, default=10.0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = analyze_merge_oracle_geometry(
        router_config=args.router_config,
        history_path=args.history,
        merge_run_dir=args.merge_run_dir,
        repo_root=args.repo_root,
        merge_nll_cache=args.merge_nll_cache,
        bic_threshold=args.bic_threshold,
    )
    _print_summary(result)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
