#!/usr/bin/env python3
"""Decompose oracle−learned routing gap by prompt and benchmark axis.

For each test prompt, computes argmax-vs-oracle agreement, per-prompt
misroute NLL cost, and expected-routing gap. Aggregates by source
benchmark to test whether the gap is diffuse or concentrated on weak axes.
"""

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

BENCH_TO_AXIS: dict[str, str] = {
    "beavertails": "harm",
    "halueval": "hallucination",
    "ai4privacy": "privacy",
    "orbench": "overrefusal",
    "do_not_answer": "policy_violation",
}

BENCH_TO_MERGE: dict[str, str] = {
    "beavertails": "merge-harm",
    "halueval": "merge-hallucination",
    "ai4privacy": "merge-privacy",
    "orbench": "merge-overrefusal",
    "do_not_answer": "merge-policy",
}


def _gini_coefficient(x: np.ndarray) -> float:
    """Gini coefficient for a non-negative vector (concentration measure)."""
    vals = np.asarray(x, dtype=float)
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    vals = np.sort(vals)
    n = vals.size
    cum = np.cumsum(vals)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def decompose_gap(
    *,
    router_config: Path,
    history_path: Path,
    merge_run_dir: Path,
    baseline_run_dir: Path,
    repo_root: Path,
    merge_nll_cache: Path | None,
    partition: str = "test",
    max_eval_records: int | None = 1000,
    seed: int = 0,
) -> dict:
    """Build per-prompt and per-axis gap decomposition."""
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
    m = len(prompts)

    merge_nll = np.load(merge_nll_cache)
    if merge_nll.shape[0] != m:
        raise ValueError(
            f"merge_nll rows {merge_nll.shape[0]} != prompts {m}",
        )

    full_splits = _load_splits(cfg)
    space = _make_trait_space(cfg, full_splits)
    sigma = _sigma_from_cfg(cfg, len(agent_names), space)
    positions = load_final_positions(history_path, agent_names)
    coords = np.asarray(space.project(prompts), dtype=float)
    cov = float(sigma) ** 2 * np.eye(space.L)
    g_clone = allocation_weights(positions, coords, cov)
    g_merge = aggregate_clone_g_to_merge(
        g_clone, agent_names, clone_to_merge, merge_names, merge_name_map,
    )

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
    expected_nll = (g_merge.T * merge_nll).sum(axis=1)
    argmax_nll = merge_nll[np.arange(m), argmax_merge_idx]
    oracle_nll = merge_nll[np.arange(m), oracle_merge_idx]

    agree = argmax_merge_idx == oracle_merge_idx
    argmax_misroute_cost = np.where(agree, 0.0, argmax_nll - oracle_nll)
    expected_gap = expected_nll - oracle_nll

    bench_arr = np.array(bench_labels)
    axis_labels = [BENCH_TO_AXIS.get(b, b) for b in bench_labels]

    per_prompt = []
    for i in range(m):
        per_prompt.append({
            "prompt_idx": i,
            "benchmark": bench_labels[i],
            "axis": axis_labels[i],
            "oracle_merge": merge_names[int(oracle_merge_idx[i])],
            "learned_argmax_merge": merge_names[int(argmax_merge_idx[i])],
            "argmax_agrees_oracle": bool(agree[i]),
            "oracle_nll": float(oracle_nll[i]),
            "learned_argmax_nll": float(argmax_nll[i]),
            "learned_expected_nll": float(expected_nll[i]),
            "argmax_misroute_cost": float(argmax_misroute_cost[i]),
            "expected_gap": float(expected_gap[i]),
            "on_axis_merge": (
                merge_names[int(argmax_merge_idx[i])]
                == merge_name_map.get(
                    BENCH_TO_MERGE.get(bench_labels[i], ""), "",
                )
            ),
        })

    # Per-benchmark / axis aggregates
    per_axis: dict[str, dict] = {}
    for bench in sorted(set(bench_labels)):
        mask = bench_arr == bench
        n = int(mask.sum())
        agree_rate = float(agree[mask].mean())
        axis = BENCH_TO_AXIS.get(bench, bench)
        correct_merge_cfg = BENCH_TO_MERGE.get(bench, "")
        resolved = merge_name_map.get(correct_merge_cfg, correct_merge_cfg)
        correct_idx = (
            merge_names.index(resolved)
            if resolved in merge_names
            else None
        )
        routed_to_own_axis = (
            float((argmax_merge_idx[mask] == correct_idx).mean())
            if correct_idx is not None
            else float("nan")
        )
        per_axis[bench] = {
            "axis": axis,
            "n": n,
            "argmax_agreement_rate": agree_rate,
            "routed_to_own_axis_rate": routed_to_own_axis,
            "mean_oracle_nll": float(oracle_nll[mask].mean()),
            "mean_learned_argmax_nll": float(argmax_nll[mask].mean()),
            "mean_learned_expected_nll": float(expected_nll[mask].mean()),
            "mean_argmax_misroute_cost": float(argmax_misroute_cost[mask].mean()),
            "total_argmax_misroute_cost": float(argmax_misroute_cost[mask].sum()),
            "mean_expected_gap": float(expected_gap[mask].mean()),
            "total_expected_gap": float(expected_gap[mask].sum()),
            "oracle_minus_expected": float(
                expected_nll[mask].mean() - oracle_nll[mask].mean(),
            ),
        }

    total_argmax_cost = float(argmax_misroute_cost.sum())
    total_expected_gap = float(expected_gap.sum())
    sorted_argmax = np.sort(argmax_misroute_cost)[::-1]
    sorted_expected = np.sort(expected_gap)[::-1]
    top10_n = max(1, m // 10)

    def _top_frac_share(sorted_costs: np.ndarray, total: float) -> float:
        if total <= 0:
            return 0.0
        return float(sorted_costs[:top10_n].sum() / total)

    misroute_only = argmax_misroute_cost[~agree]
    concentration = {
        "n_prompts": m,
        "n_misrouted_argmax": int((~agree).sum()),
        "argmax_agreement_rate": float(agree.mean()),
        "mean_argmax_misroute_cost_when_wrong": (
            float(misroute_only.mean()) if misroute_only.size else 0.0
        ),
        "median_argmax_misroute_cost_when_wrong": (
            float(np.median(misroute_only)) if misroute_only.size else 0.0
        ),
        "gini_argmax_misroute_cost": _gini_coefficient(argmax_misroute_cost),
        "gini_expected_gap": _gini_coefficient(np.maximum(expected_gap, 0)),
        "top10pct_share_of_argmax_misroute_cost": _top_frac_share(
            sorted_argmax, total_argmax_cost,
        ),
        "top10pct_share_of_expected_gap": _top_frac_share(
            sorted_expected, total_expected_gap,
        ),
    }

    flat_headline = {
        "learned_expected_nll": float(expected_nll.mean()),
        "learned_argmax_nll": float(argmax_nll.mean()),
        "oracle_nll": float(oracle_nll.mean()),
    }
    flat_headline["oracle_minus_learned_expected"] = (
        flat_headline["oracle_nll"] - flat_headline["learned_expected_nll"]
    )
    flat_headline["oracle_minus_learned_argmax"] = (
        flat_headline["oracle_nll"] - flat_headline["learned_argmax_nll"]
    )

    # Confusion matrix for argmax vs oracle
    n_merge = len(merge_names)
    confusion = np.zeros((n_merge, n_merge), dtype=int)
    for li, oi in zip(argmax_merge_idx, oracle_merge_idx):
        confusion[int(li), int(oi)] += 1
    routing_confusion = {
        "merge_names": merge_names,
        "counts": {
            merge_names[i]: {
                merge_names[j]: int(confusion[i, j]) for j in range(n_merge)
            }
            for i in range(n_merge)
        },
    }

    return {
        "run": {
            "router_config": str(router_config),
            "history": str(history_path),
            "merge_run_dir": str(merge_run_dir),
            "round": rnd,
            "n_prompts": m,
        },
        "headline": flat_headline,
        "concentration": concentration,
        "per_axis": per_axis,
        "per_prompt": per_prompt,
        "routing_confusion": routing_confusion,
    }


def _print_summary(result: dict) -> None:
    """Human-readable summary."""
    h = result["headline"]
    c = result["concentration"]
    print("=== routing gap decomposition ===")
    print(
        f"prompts={c['n_prompts']}  "
        f"argmax agreement={c['argmax_agreement_rate']:.3f}  "
        f"misrouted={c['n_misrouted_argmax']}",
    )
    print(
        f"oracle−learned_expected: {h['oracle_minus_learned_expected']:+.4f}  "
        f"oracle−learned_argmax: {h['oracle_minus_learned_argmax']:+.4f}",
    )
    print(
        f"when wrong: mean misroute cost={c['mean_argmax_misroute_cost_when_wrong']:.4f}  "
        f"median={c['median_argmax_misroute_cost_when_wrong']:.4f}",
    )
    print(
        f"concentration: gini(misroute)={c['gini_argmax_misroute_cost']:.3f}  "
        f"top10% share(misroute)={c['top10pct_share_of_argmax_misroute_cost']:.3f}  "
        f"top10% share(expected gap)={c['top10pct_share_of_expected_gap']:.3f}",
    )
    print("\n--- per axis (benchmark) ---")
    print(
        f"{'benchmark':<16} {'axis':<18} {'n':>5} {'agree':>7} "
        f"{'own_ax':>7} {'Δ_exp':>8} {'misrt':>8}",
    )
    rows = sorted(
        result["per_axis"].items(),
        key=lambda kv: kv[1]["mean_expected_gap"],
        reverse=True,
    )
    for bench, row in rows:
        print(
            f"{bench:<16} {row['axis']:<18} {row['n']:5d} "
            f"{row['argmax_agreement_rate']:7.3f} "
            f"{row['routed_to_own_axis_rate']:7.3f} "
            f"{row['mean_expected_gap']:+8.4f} "
            f"{row['mean_argmax_misroute_cost']:8.4f}",
        )
    print("\n(Δ_exp = mean expected_nll − oracle_nll; misrt = mean argmax misroute cost)")


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
        "--baseline-run-dir",
        type=Path,
        default=Path("results/seven_axis_collapse_hypercube_ga_baseline/seed0"),
    )
    parser.add_argument(
        "--merge-nll-cache",
        type=Path,
        default=Path("results/attribution_2x2/ga_theory_pre/seed0/merge_nll_test.npy"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-prompts-json", type=Path, default=None)
    args = parser.parse_args()

    if not args.merge_nll_cache.is_file():
        raise SystemExit(f"missing merge NLL cache: {args.merge_nll_cache}")

    result = decompose_gap(
        router_config=args.router_config,
        history_path=args.history,
        merge_run_dir=args.merge_run_dir,
        baseline_run_dir=args.baseline_run_dir,
        repo_root=args.repo_root,
        merge_nll_cache=args.merge_nll_cache,
    )
    _print_summary(result)

    if args.output_json:
        payload = {k: v for k, v in result.items() if k != "per_prompt"}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output_json}")

    if args.output_prompts_json:
        args.output_prompts_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_prompts_json.write_text(
            json.dumps(result["per_prompt"], indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.output_prompts_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
