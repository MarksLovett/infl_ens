#!/usr/bin/env python3
"""Terminal diagnostic: is the expensive routing tail representation-limited?

Analysis-only on colocated GA seed-0. Defines the expensive argmax-wrong
tail ``T``, compares each merge's tail prompts to well-routed same-merge
prompts in trait space (collision vs separable), and cross-checks with
restricted confusability / topic-vs-skill metrics on ``T`` only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import NearestNeighbors

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


def _load_prompt_records(
    *,
    router_config: Path,
    history_path: Path,
    merge_run_dir: Path,
    merge_nll_cache: Path,
    repo_root: Path,
    partition: str = "test",
    max_eval_records: int | None = 1000,
    seed: int = 0,
) -> dict:
    """Per-prompt routing records with trait coordinates."""
    cfg = _load_yaml(router_config)
    cl = cfg.get("closed_loop", {})
    clone_to_merge, config_merge_names = parse_merge_groups(cl)
    agent_names = [a["name"] for a in cfg["agents"]]
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
    m = len(prompts)
    if merge_nll.shape[0] != m:
        raise ValueError("merge_nll row count mismatch")

    full_splits = _load_splits(cfg)
    space = _make_trait_space(cfg, full_splits)
    sigma = _sigma_from_cfg(cfg, len(agent_names), space)
    positions = load_final_positions(history_path, agent_names)
    coords = np.asarray(space.project(prompts), dtype=float)
    cov = float(sigma) ** 2 * np.eye(space.L)
    g_clone = allocation_weights(positions, coords, cov)

    clone_win = np.argmax(g_clone, axis=0)
    argmax_merge_idx = np.array(
        [
            merge_names.index(
                merge_name_map[clone_to_merge[agent_names[i]]],
            )
            for i in clone_win
        ],
        dtype=int,
    )
    oracle_merge_idx = np.argmin(merge_nll, axis=1)
    agree = argmax_merge_idx == oracle_merge_idx
    argmax_nll = merge_nll[np.arange(m), argmax_merge_idx]
    oracle_nll = merge_nll[np.arange(m), oracle_merge_idx]
    argmax_misroute_cost = np.where(agree, 0.0, argmax_nll - oracle_nll)

    records = []
    for i in range(m):
        records.append({
            "prompt_idx": i,
            "benchmark": bench_labels[i],
            "axis": BENCH_TO_AXIS.get(bench_labels[i], bench_labels[i]),
            "trait": coords[i].tolist(),
            "oracle_merge": merge_names[int(oracle_merge_idx[i])],
            "oracle_merge_idx": int(oracle_merge_idx[i]),
            "routed_merge": merge_names[int(argmax_merge_idx[i])],
            "routed_merge_idx": int(argmax_merge_idx[i]),
            "argmax_agrees_oracle": bool(agree[i]),
            "argmax_misroute_cost": float(argmax_misroute_cost[i]),
        })

    return {
        "merge_names": merge_names,
        "records": records,
        "coords": coords,
        "argmax_misroute_cost": argmax_misroute_cost,
        "agree": agree,
        "oracle_merge_idx": oracle_merge_idx,
        "argmax_merge_idx": argmax_merge_idx,
        "bench_labels": bench_labels,
    }


def _define_tail(
    agree: np.ndarray,
    cost: np.ndarray,
    *,
    tail_quantile: float = 0.75,
    tail_top_frac: float | None = 0.15,
) -> np.ndarray:
    """Boolean mask for expensive-tail set ``T``."""
    wrong = ~agree
    if not wrong.any():
        return np.zeros_like(agree, dtype=bool)
    thr_q = float(np.quantile(cost[wrong], tail_quantile))
    by_quartile = wrong & (cost >= thr_q)
    if tail_top_frac is not None:
        n_top = max(1, int(np.ceil(tail_top_frac * len(cost))))
        order = np.argsort(cost)[::-1]
        top_mask = np.zeros(len(cost), dtype=bool)
        top_mask[order[:n_top]] = True
        return wrong & top_mask
    return by_quartile


def _held_out_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    """5-fold CV accuracy for binary separability."""
    if len(np.unique(y)) < 2 or x.shape[0] < 10:
        return float("nan")
    n_splits = min(5, int(np.bincount(y.astype(int)).min()))
    if n_splits < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    try:
        scores = cross_val_score(clf, x, y, cv=n_splits, scoring="accuracy")
        return float(scores.mean())
    except ValueError:
        return float("nan")


def _collision_fraction(
    tail_coords: np.ndarray,
    well_coords: np.ndarray,
    *,
    collision_threshold: float,
) -> tuple[float, np.ndarray]:
    """Fraction of tail points with a well-routed neighbor within threshold."""
    if tail_coords.shape[0] == 0 or well_coords.shape[0] == 0:
        return float("nan"), np.array([])
    nn = NearestNeighbors(n_neighbors=1).fit(well_coords)
    dists, _ = nn.kneighbors(tail_coords)
    d = dists[:, 0]
    return float((d <= collision_threshold).mean()), d


def _adaptive_collision_threshold(well_coords: np.ndarray) -> float:
    """Threshold from within-well-routed 10th-percentile pairwise scale."""
    if well_coords.shape[0] < 3:
        return 0.05
    subs = well_coords
    if subs.shape[0] > 500:
        rng = np.random.default_rng(0)
        subs = subs[rng.choice(subs.shape[0], 500, replace=False)]
    dsum = 0.0
    cnt = 0
    for i in range(subs.shape[0]):
        for j in range(i + 1, subs.shape[0]):
            dsum += float(np.linalg.norm(subs[i] - subs[j]))
            cnt += 1
    mean_pw = dsum / max(cnt, 1)
    return max(0.02, 0.15 * mean_pw)


def analyze_tail_separability(
    data: dict,
    *,
    tail_mask: np.ndarray,
    collision_threshold: float | None = None,
) -> dict:
    """Collision + separability per merge and global verdict."""
    records = data["records"]
    coords = data["coords"]
    merge_names = data["merge_names"]
    agree = data["agree"]
    cost = data["argmax_misroute_cost"]
    oracle_idx = data["oracle_merge_idx"]

    total_wrong_cost = float(cost[~agree].sum()) if (~agree).any() else 0.0
    tail_cost_share = (
        float(cost[tail_mask].sum() / total_wrong_cost)
        if total_wrong_cost > 0 else 0.0
    )

    per_merge: dict[str, dict] = {}
    collision_fracs: list[float] = []
    separable_accs: list[float] = []

    for j, merge in enumerate(merge_names):
        well_mask = agree & (oracle_idx == j)
        tail_m = tail_mask & (oracle_idx == j)
        well_coords = coords[well_mask]
        tail_coords = coords[tail_m]
        n_well = int(well_mask.sum())
        n_tail = int(tail_m.sum())

        thr = (
            collision_threshold
            if collision_threshold is not None
            else _adaptive_collision_threshold(well_coords)
        )
        coll_frac, nn_dists = _collision_fraction(
            tail_coords, well_coords, collision_threshold=thr,
        )

        centroid_l2 = float("nan")
        sil = float("nan")
        acc = float("nan")
        if n_tail > 0 and n_well > 0:
            centroid_l2 = float(
                np.linalg.norm(tail_coords.mean(axis=0) - well_coords.mean(axis=0)),
            )
        if n_tail + n_well >= 4 and n_tail > 0 and n_well > 0:
            x = np.vstack([well_coords, tail_coords])
            y = np.array([0] * n_well + [1] * n_tail)
            if len(np.unique(y)) == 2:
                try:
                    sil = float(silhouette_score(x, y))
                except ValueError:
                    sil = float("nan")
            acc = _held_out_accuracy(x, y)

        n_coll = int((nn_dists <= thr).sum()) if nn_dists.size else 0
        n_sep = n_tail - n_coll
        mode = (
            "collision" if n_tail > 0 and coll_frac >= 0.5
            else ("separable" if n_tail > 0 and acc >= 0.65 and coll_frac < 0.5
                  else "mixed_or_insufficient")
        )

        per_merge[merge] = {
            "n_well_routed": n_well,
            "n_tail": n_tail,
            "centroid_l2": centroid_l2,
            "silhouette_tail_vs_well": sil,
            "cv_accuracy_tail_vs_well": acc,
            "collision_threshold_l2": thr,
            "collision_fraction": coll_frac,
            "n_collision": n_coll,
            "n_separable": n_sep,
            "tail_mode": mode,
            "mean_nn_dist_to_well": float(nn_dists.mean()) if nn_dists.size else None,
        }
        if n_tail > 0 and np.isfinite(coll_frac):
            collision_fracs.append(coll_frac)
        if n_tail > 0 and np.isfinite(acc):
            separable_accs.append(acc)

    # Restricted confusability on T: benchmark probe
    tail_idx = np.where(tail_mask)[0]
    bench_arr = np.array(data["bench_labels"])
    confusability_tail: dict = {}
    if tail_idx.size >= 20:
        x_t = coords[tail_idx]
        y_bench = bench_arr[tail_idx]
        classes = np.unique(y_bench)
        if classes.size >= 2:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            try:
                pred = cross_val_predict(
                    clf, x_t, y_bench,
                    cv=min(5, int(np.min(np.unique(y_bench, return_counts=True)[1]))),
                    method="predict",
                )
                confusability_tail = {
                    "n_prompts": int(tail_idx.size),
                    "cv_balanced_accuracy": float(
                        np.mean([accuracy_score(y_bench == c, pred == c)
                                 for c in classes]),
                    ),
                    "verdict": (
                        "traits_encode_benchmark_on_tail"
                        if float(np.mean(pred == y_bench)) > 0.35
                        else "weak_benchmark_signal_on_tail"
                    ),
                }
            except ValueError:
                confusability_tail = {"error": "cv_failed"}

    # Restricted topic-vs-skill on T ∪ well_routed (per merge oracle skill)
    topic_tail: dict = {}
    subset_mask = tail_mask | agree
    if subset_mask.sum() >= 30:
        x_sub = coords[subset_mask]
        y_oracle = oracle_idx[subset_mask]
        k = min(len(merge_names), max(2, len(merge_names)))
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        topics = km.fit_predict(x_sub)
        topic_tail = {
            "n_prompts": int(subset_mask.sum()),
            "nmi_topic_oracle_merge": float(
                normalized_mutual_info_score(y_oracle, topics),
            ),
            "verdict": (
                "oracle_merge_clusterable_in_trait_space"
                if normalized_mutual_info_score(y_oracle, topics) > 0.15
                else "oracle_merge_not_clusterable"
            ),
        }

    global_collision = float(np.mean(collision_fracs)) if collision_fracs else float("nan")
    global_acc = float(np.mean(separable_accs)) if separable_accs else float("nan")

    if global_collision >= 0.55 and (not np.isfinite(global_acc) or global_acc < 0.65):
        verdict = "irreducible_under_representation"
        sentence = (
            "Collisions dominate the expensive tail: trait vectors cannot "
            "distinguish mis-routed from well-routed same-merge prompts, "
            "so the gap is information-limited under this trait scorer."
        )
    elif np.isfinite(global_acc) and global_acc >= 0.65 and global_collision < 0.5:
        verdict = "routing_function_fixable"
        sentence = (
            "Tail prompts occupy separable trait regions from well-routed "
            "same-merge prompts, so the signal exists but the router "
            "mis-maps it — a routing-function fix may still help."
        )
    else:
        verdict = "mixed"
        sentence = (
            "Per-merge failure modes mix collision and separability; no "
            "single global lever is indicated without merge-specific fixes."
        )

    return {
        "tail_definition": {
            "n_tail": int(tail_mask.sum()),
            "frac_of_pool": float(tail_mask.mean()),
            "frac_of_wrong": float(tail_mask[~agree].sum() / max((~agree).sum(), 1)),
            "tail_cost_share_of_wrong": tail_cost_share,
            "mean_tail_cost": float(cost[tail_mask].mean()) if tail_mask.any() else 0.0,
        },
        "per_merge": per_merge,
        "cross_check": {
            "confusability_on_T": confusability_tail,
            "topic_vs_oracle_on_tail_plus_well": topic_tail,
        },
        "global": {
            "mean_collision_fraction": global_collision,
            "mean_cv_accuracy_tail_vs_well": global_acc,
            "verdict": verdict,
            "one_sentence": sentence,
        },
    }


def _print_table(result: dict) -> None:
    """Stdout summary table."""
    t = result["tail_definition"]
    print("=== tail separability diagnostic ===")
    print(
        f"T: n={t['n_tail']}  frac_pool={t['frac_of_pool']:.3f}  "
        f"wrong_share={t['frac_of_wrong']:.3f}  "
        f"cost_share_wrong={t['tail_cost_share_of_wrong']:.3f}",
    )
    print(
        f"\n{'merge':<22} {'n_tail':>6} {'coll%':>7} {'cv_acc':>7} "
        f"{'centroid':>8} {'mode':>12}",
    )
    for merge, row in result["per_merge"].items():
        if row["n_tail"] == 0:
            continue
        print(
            f"{merge:<22} {row['n_tail']:6d} "
            f"{row['collision_fraction']:7.3f} "
            f"{row['cv_accuracy_tail_vs_well']:7.3f} "
            f"{row['centroid_l2']:8.4f} {row['tail_mode']:>12}",
        )
    g = result["global"]
    print(f"\nverdict: {g['verdict']}")
    print(g["one_sentence"])


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
    parser.add_argument("--tail-top-frac", type=float, default=0.15)
    parser.add_argument("--collision-threshold", type=float, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    data = _load_prompt_records(
        router_config=args.router_config,
        history_path=args.history,
        merge_run_dir=args.merge_run_dir,
        merge_nll_cache=args.merge_nll_cache,
        repo_root=Path("."),
    )
    tail_mask = _define_tail(
        data["agree"],
        data["argmax_misroute_cost"],
        tail_top_frac=args.tail_top_frac,
    )
    result = analyze_tail_separability(
        data,
        tail_mask=tail_mask,
        collision_threshold=args.collision_threshold,
    )
    _print_table(result)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run": {
                "router_config": str(args.router_config),
                "history": str(args.history),
                "merge_nll_cache": str(args.merge_nll_cache),
            },
            **result,
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
