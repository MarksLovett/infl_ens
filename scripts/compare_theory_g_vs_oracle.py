#!/usr/bin/env python3
"""Compare theory merge-level G rankings to oracle NLL on colocated GA."""

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

BENCH_TO_MERGE = {
    "beavertails": "merge-harm",
    "halueval": "merge-hallucination",
    "ai4privacy": "merge-privacy",
    "orbench": "merge-overrefusal",
    "do_not_answer": "merge-policy",
}


def compare_theory_g_oracle(
    *,
    router_config: Path,
    history_path: Path,
    merge_run_dir: Path,
    repo_root: Path,
    merge_nll_cache: Path,
    partition: str = "test",
    max_eval_records: int | None = 1000,
    seed: int = 0,
    g_soft_margin: float = 0.05,
    nll_tie_margin: float = 0.05,
) -> dict:
    """Theory G vs oracle diagnostics on a flat test pool."""
    cfg = _load_yaml(router_config)
    cl = cfg.get("closed_loop", {})
    clone_to_merge, config_merge_names = parse_merge_groups(cl)
    agent_names = [a["name"] for a in cfg["agents"]]
    rnd = final_round(history_path)
    merge_names, merge_name_map = resolve_merge_adapters(
        merge_run_dir, rnd, config_merge_names,
    )

    prompts, _responses, bench_labels = load_flat_partition_pool(
        cfg, repo_root=repo_root, partition=partition,
        max_eval_records=max_eval_records, seed=seed,
    )
    merge_nll = np.load(merge_nll_cache)
    m = len(prompts)

    full_splits = _load_splits(cfg)
    space = _make_trait_space(cfg, full_splits)
    sigma = _sigma_from_cfg(cfg, len(agent_names), space)
    positions = load_final_positions(history_path, agent_names)
    coords = np.asarray(space.project(prompts), dtype=float)
    cov = float(sigma) ** 2 * np.eye(space.L)
    g_clone = allocation_weights(positions, coords, cov)
    g_merge = aggregate_clone_g_to_merge(
        g_clone, agent_names, clone_to_merge, merge_names, merge_name_map,
    )  # (n_merge, M)

    g_merge_t = g_merge.T  # (M, n_merge)
    oracle_idx = np.argmin(merge_nll, axis=1)
    theory_idx = np.argmax(g_merge_t, axis=1)
    agree = theory_idx == oracle_idx

    g_oracle = g_merge_t[np.arange(m), oracle_idx]
    g_theory = g_merge_t[np.arange(m), theory_idx]
    g_top = g_merge_t.max(axis=1)
    g_margin = g_top - g_oracle  # 0 when oracle is G-top

    nll_oracle = merge_nll[np.arange(m), oracle_idx]
    nll_theory = merge_nll[np.arange(m), theory_idx]
    nll_wrong_cost = np.where(agree, 0.0, nll_theory - nll_oracle)

    # G soft: oracle not top but within margin of top G
    g_soft = (~agree) & (g_margin <= g_soft_margin)
    # NLL near-tie: wrong pick but all merges within tie margin of oracle
    nll_near_tie = (~agree) & (nll_wrong_cost <= nll_tie_margin)
    # G wrong: disagree and oracle G well below top
    g_wrong = (~agree) & (g_margin > g_soft_margin)
    # Clear NLL miss: disagree and meaningful NLL cost
    nll_clear_miss = (~agree) & (nll_wrong_cost > nll_tie_margin)

    per_bench: dict[str, dict] = {}
    bench_arr = np.array(bench_labels)
    for bench in sorted(set(bench_labels)):
        mask = bench_arr == bench
        n = int(mask.sum())
        if n == 0:
            continue
        per_bench[bench] = {
            "merge": BENCH_TO_MERGE.get(bench, bench),
            "n": n,
            "g_agreement": float(agree[mask].mean()),
            "mean_expected_gap": float(
                (g_merge_t[mask] * merge_nll[mask]).sum(axis=1).mean()
                - nll_oracle[mask].mean(),
            ),
            "mean_nll_wrong_cost": float(nll_wrong_cost[mask].mean()),
            "frac_g_soft": float(g_soft[mask].mean()),
            "frac_g_wrong": float(g_wrong[mask].mean()),
            "frac_nll_near_tie": float(nll_near_tie[mask].mean()),
            "frac_nll_clear_miss": float(nll_clear_miss[mask].mean()),
            "mean_g_margin_when_disagree": float(
                g_margin[mask & ~agree].mean()
                if (~agree & mask).any() else 0.0,
            ),
        }

    return {
        "sigma": sigma,
        "n_prompts": m,
        "g_argmax_agreement_oracle": float(agree.mean()),
        "mean_g_margin": float(g_margin.mean()),
        "mean_nll_wrong_cost": float(nll_wrong_cost.mean()),
        "disagree_breakdown": {
            "n_disagree": int((~agree).sum()),
            "frac_g_soft_among_disagree": float(
                g_soft[~agree].mean() if (~agree).any() else 0.0,
            ),
            "frac_g_wrong_among_disagree": float(
                g_wrong[~agree].mean() if (~agree).any() else 0.0,
            ),
            "frac_nll_near_tie_among_disagree": float(
                nll_near_tie[~agree].mean() if (~agree).any() else 0.0,
            ),
            "frac_nll_clear_miss_among_disagree": float(
                nll_clear_miss[~agree].mean() if (~agree).any() else 0.0,
            ),
        },
        "verdict": _verdict(g_soft, g_wrong, nll_near_tie, nll_clear_miss, agree),
        "per_benchmark": per_bench,
    }


def _verdict(
    g_soft: np.ndarray,
    g_wrong: np.ndarray,
    nll_near_tie: np.ndarray,
    nll_clear_miss: np.ndarray,
    agree: np.ndarray,
) -> dict[str, str]:
    """Heuristic read on σ vs G calibration."""
    n = int((~agree).sum())
    if n == 0:
        return {"primary": "perfect_g", "sigma_likely_helps": "no"}
    soft_share = float(g_soft[~agree].mean())
    wrong_share = float(g_wrong[~agree].mean())
    tie_share = float(nll_near_tie[~agree].mean())
    clear_share = float(nll_clear_miss[~agree].mean())
    if wrong_share > soft_share and clear_share > tie_share:
        primary = "g_miscalibrated"
        sigma_helps = "unlikely"
    elif soft_share >= wrong_share and tie_share >= clear_share:
        primary = "g_soft_near_ties"
        sigma_helps = "possible"
    else:
        primary = "mixed"
        sigma_helps = "uncertain"
    return {
        "primary": primary,
        "sigma_likely_helps": sigma_helps,
        "g_soft_share_disagree": f"{soft_share:.3f}",
        "g_wrong_share_disagree": f"{wrong_share:.3f}",
        "nll_near_tie_share_disagree": f"{tie_share:.3f}",
        "nll_clear_miss_share_disagree": f"{clear_share:.3f}",
    }


def _print_summary(result: dict) -> None:
    """Human-readable summary."""
    print("=== theory G vs oracle (colocated GA) ===")
    print(
        f"n={result['n_prompts']}  sigma={result['sigma']:.4f}  "
        f"G argmax agrees oracle: {result['g_argmax_agreement_oracle']:.3f}",
    )
    db = result["disagree_breakdown"]
    print(
        f"disagree={db['n_disagree']}  "
        f"g_soft={db['frac_g_soft_among_disagree']:.3f}  "
        f"g_wrong={db['frac_g_wrong_among_disagree']:.3f}  "
        f"nll_near_tie={db['frac_nll_near_tie_among_disagree']:.3f}  "
        f"nll_clear_miss={db['frac_nll_clear_miss_among_disagree']:.3f}",
    )
    v = result["verdict"]
    print(f"verdict: {v['primary']}  sigma_likely_helps={v['sigma_likely_helps']}")
    print("\n--- per benchmark (residual gap + G/oracle) ---")
    print(
        f"{'bench':<16} {'agree':>7} {'Δ_exp':>8} {'g_wrong':>8} "
        f"{'clear_miss':>10}",
    )
    rows = sorted(
        result["per_benchmark"].items(),
        key=lambda kv: kv[1]["mean_expected_gap"],
        reverse=True,
    )
    for bench, row in rows:
        print(
            f"{bench:<16} {row['g_agreement']:7.3f} "
            f"{row['mean_expected_gap']:+8.4f} {row['frac_g_wrong']:8.3f} "
            f"{row['frac_nll_clear_miss']:10.3f}",
        )


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
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = compare_theory_g_oracle(
        router_config=args.router_config,
        history_path=args.history,
        merge_run_dir=args.merge_run_dir,
        repo_root=Path("."),
        merge_nll_cache=args.merge_nll_cache,
    )
    _print_summary(result)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
