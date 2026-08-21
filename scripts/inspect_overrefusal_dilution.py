#!/usr/bin/env python3
"""Inspect overrefusal (orbench) routing gap: dilution vs argmax errors.

On colocated GA, orbench shows the highest expected-routing gap (+0.060)
despite the best G-argmax agreement (~85%). This script stratifies orbench
prompts, measures merge-weight diffuseness when argmax is correct, and
counterfactually sharpens weights (power transform or smaller kernel σ) to
test whether dilution — not wrong argmax — drives the residual gap.
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

ORACLE_MERGE_CFG = "merge-overrefusal"
BENCH = "orbench"


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy (nats) of probability vectors along ``axis``."""
    q = np.clip(p, 1e-12, 1.0)
    return -np.sum(q * np.log(q), axis=axis)


def _effective_n(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Inverse participation ratio :math:`1 / \\sum p_k^2`."""
    return 1.0 / np.sum(np.square(p), axis=axis)


def _sharpen_weights(g: np.ndarray, alpha: float) -> np.ndarray:
    """Power-sharpen merge weights; ``alpha=inf`` → one-hot argmax."""
    if alpha == float("inf"):
        out = np.zeros_like(g)
        out[np.arange(g.shape[0]), np.argmax(g, axis=1)] = 1.0
        return out
    powered = np.power(np.clip(g, 1e-12, None), alpha)
    return powered / powered.sum(axis=1, keepdims=True)


def _expected_nll(g_row: np.ndarray, merge_nll: np.ndarray) -> np.ndarray:
    """Per-prompt expected NLL under merge weights ``g_row`` (M, n_merge)."""
    return (g_row * merge_nll).sum(axis=1)


def inspect_overrefusal_dilution(
    *,
    router_config: Path,
    history_path: Path,
    merge_run_dir: Path,
    repo_root: Path,
    merge_nll_cache: Path,
    partition: str = "test",
    max_eval_records: int | None = 1000,
    seed: int = 0,
    high_gap_quantile: float = 0.75,
    sharpen_alphas: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, float("inf")),
    sigma_scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.35, 0.25, 0.1),
) -> dict:
    """Build orbench dilution diagnostics and sharpening counterfactuals."""
    cfg = _load_yaml(router_config)
    cl = cfg.get("closed_loop", {})
    clone_to_merge, config_merge_names = parse_merge_groups(cl)
    agent_names = [a["name"] for a in cfg["agents"]]
    rnd = final_round(history_path)
    merge_names, merge_name_map = resolve_merge_adapters(
        merge_run_dir, rnd, config_merge_names,
    )
    oracle_merge = merge_name_map[ORACLE_MERGE_CFG]
    oracle_j = merge_names.index(oracle_merge)

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
        raise ValueError(f"merge_nll rows {merge_nll.shape[0]} != prompts {m}")

    bench_arr = np.array(bench_labels)
    mask = bench_arr == BENCH
    if not mask.any():
        raise ValueError(f"No {BENCH} prompts in pool")

    full_splits = _load_splits(cfg)
    space = _make_trait_space(cfg, full_splits)
    sigma = _sigma_from_cfg(cfg, len(agent_names), space)
    positions = load_final_positions(history_path, agent_names)
    coords = np.asarray(space.project(prompts), dtype=float)

    def _g_merge_for_sigma(sig: float) -> np.ndarray:
        cov = float(sig) ** 2 * np.eye(space.L)
        g_clone = allocation_weights(positions, coords, cov)
        return aggregate_clone_g_to_merge(
            g_clone, agent_names, clone_to_merge, merge_names, merge_name_map,
        )

    g_merge = _g_merge_for_sigma(sigma)  # (n_merge, M)
    g_row = g_merge.T[mask]  # (n_orbench, n_merge)
    nll_row = merge_nll[mask]
    n = int(mask.sum())

    clone_win = np.argmax(
        allocation_weights(
            positions, coords, float(sigma) ** 2 * np.eye(space.L),
        ),
        axis=0,
    )
    argmax_merge_idx = np.array(
        [
            merge_names.index(
                merge_name_map[clone_to_merge[agent_names[i]]],
            )
            for i in clone_win
        ],
        dtype=int,
    )
    argmax_idx = argmax_merge_idx[mask]
    oracle_idx = np.argmin(nll_row, axis=1)
    agree = argmax_idx == oracle_idx

    oracle_nll = nll_row[np.arange(n), oracle_idx]
    argmax_nll = nll_row[np.arange(n), argmax_idx]
    expected_nll = _expected_nll(g_row, nll_row)
    expected_gap = expected_nll - oracle_nll
    argmax_gap = argmax_nll - oracle_nll

    w_oracle = g_row[np.arange(n), oracle_idx]
    w_argmax = g_row[np.arange(n), argmax_idx]
    w_off_oracle = 1.0 - w_oracle

    entropy = _entropy(g_row, axis=1)
    eff_n = _effective_n(g_row, axis=1)

    agree_gap = expected_gap[agree]
    gap_thr = float(np.quantile(agree_gap, high_gap_quantile)) if agree_gap.size else 0.0
    agree_high = agree & (expected_gap >= gap_thr)
    agree_low = agree & (expected_gap < gap_thr)
    wrong = ~agree

    def _stratum_stats(sel: np.ndarray, label: str) -> dict:
        if not sel.any():
            return {"label": label, "n": 0}
        dilution_cost = (
            (g_row[sel] * (nll_row[sel] - oracle_nll[sel, None])).sum(axis=1)
        )
        return {
            "label": label,
            "n": int(sel.sum()),
            "mean_expected_gap": float(expected_gap[sel].mean()),
            "mean_argmax_gap": float(argmax_gap[sel].mean()),
            "mean_oracle_weight": float(w_oracle[sel].mean()),
            "mean_argmax_weight": float(w_argmax[sel].mean()),
            "mean_off_oracle_weight": float(w_off_oracle[sel].mean()),
            "mean_entropy": float(entropy[sel].mean()),
            "mean_effective_n_merges": float(eff_n[sel].mean()),
            "mean_dilution_cost": float(dilution_cost.mean()),
            "frac_gap_from_dilution": float(
                dilution_cost.mean() / expected_gap[sel].mean()
                if expected_gap[sel].mean() > 1e-9 else 0.0,
            ),
        }

    strata = [
        _stratum_stats(agree_high, f"argmax_right_high_gap_q{high_gap_quantile:.2f}"),
        _stratum_stats(agree_low, f"argmax_right_low_gap"),
        _stratum_stats(wrong, "argmax_wrong"),
        _stratum_stats(np.ones(n, dtype=bool), "all_orbench"),
    ]

    power_sharpen: dict[str, dict] = {}
    for alpha in sharpen_alphas:
        key = "argmax" if alpha == float("inf") else f"alpha_{alpha:g}"
        g_sharp = _sharpen_weights(g_row, alpha)
        sharp_nll = _expected_nll(g_sharp, nll_row)
        sharp_gap = sharp_nll - oracle_nll
        power_sharpen[key] = {
            "mean_expected_nll": float(sharp_nll.mean()),
            "mean_expected_gap": float(sharp_gap.mean()),
            "gap_vs_baseline": float(sharp_gap.mean() - expected_gap.mean()),
            "mean_oracle_weight": float(
                g_sharp[np.arange(n), oracle_idx].mean(),
            ),
            "mean_entropy": float(_entropy(g_sharp, axis=1).mean()),
        }

    sigma_sharpen: dict[str, dict] = {}
    for scale in sigma_scales:
        sig = sigma * scale
        g_s = _g_merge_for_sigma(sig).T[mask]
        sharp_nll = _expected_nll(g_s, nll_row)
        sharp_gap = sharp_nll - oracle_nll
        sigma_sharpen[f"sigma_x{scale:g}"] = {
            "sigma": float(sig),
            "mean_expected_gap": float(sharp_gap.mean()),
            "gap_vs_baseline": float(sharp_gap.mean() - expected_gap.mean()),
            "mean_oracle_weight": float(
                g_s[np.arange(n), oracle_idx].mean(),
            ),
            "mean_entropy": float(_entropy(g_s, axis=1).mean()),
        }

    agree_high_n = int(agree_high.sum())
    gap_from_dilution_agree = float(
        (
            (g_row[agree] * (nll_row[agree] - oracle_nll[agree, None])).sum(axis=1)
        ).mean(),
    ) if agree.any() else 0.0
    gap_from_argmax = float(argmax_gap[wrong].mean()) if wrong.any() else 0.0
    weighted_dilution = (
        gap_from_dilution_agree * float(agree.mean())
        + gap_from_argmax * float(wrong.mean())
    )

    best_power = min(power_sharpen.items(), key=lambda kv: kv[1]["mean_expected_gap"])
    best_sigma = min(sigma_sharpen.items(), key=lambda kv: kv[1]["mean_expected_gap"])

    verdict = _verdict(
        baseline_gap=float(expected_gap.mean()),
        agree_rate=float(agree.mean()),
        strata=strata,
        gap_from_dilution_agree=gap_from_dilution_agree,
        best_power=best_power,
        best_sigma=best_sigma,
    )

    return {
        "benchmark": BENCH,
        "oracle_merge": oracle_merge,
        "n_orbench": n,
        "sigma": sigma,
        "headline": {
            "mean_oracle_nll": float(oracle_nll.mean()),
            "mean_expected_nll": float(expected_nll.mean()),
            "mean_expected_gap": float(expected_gap.mean()),
            "argmax_agreement": float(agree.mean()),
            "mean_oracle_weight": float(w_oracle.mean()),
            "mean_entropy": float(entropy.mean()),
            "mean_effective_n_merges": float(eff_n.mean()),
            "high_gap_threshold_among_agree": gap_thr,
        },
        "gap_decomposition": {
            "mean_gap_when_argmax_right": float(expected_gap[agree].mean())
            if agree.any() else 0.0,
            "mean_gap_when_argmax_wrong": float(expected_gap[wrong].mean())
            if wrong.any() else 0.0,
            "mean_dilution_cost_when_argmax_right": gap_from_dilution_agree,
            "mean_argmax_error_cost_when_wrong": gap_from_argmax,
            "approx_weighted_gap": weighted_dilution,
        },
        "strata": strata,
        "power_sharpening": power_sharpen,
        "sigma_sharpening": sigma_sharpen,
        "verdict": verdict,
    }


def _verdict(
    *,
    baseline_gap: float,
    agree_rate: float,
    strata: list[dict],
    gap_from_dilution_agree: float,
    best_power: tuple[str, dict],
    best_sigma: tuple[str, dict],
) -> dict[str, str | float]:
    """Heuristic read on whether local sharpening can recover orbench gap."""
    high = next(s for s in strata if "high_gap" in s["label"])
    low = next(s for s in strata if "low_gap" in s["label"])
    dilution_dominates = (
        agree_rate > 0.8
        and gap_from_dilution_agree > 0.5 * baseline_gap
    )
    diffuse_when_high = (
        high.get("n", 0) > 0
        and low.get("n", 0) > 0
        and high["mean_off_oracle_weight"] > low["mean_off_oracle_weight"] + 0.02
    )
    power_recover = baseline_gap - best_power[1]["mean_expected_gap"]
    sigma_recover = baseline_gap - best_sigma[1]["mean_expected_gap"]
    if dilution_dominates and (power_recover > 0.02 or sigma_recover > 0.02):
        primary = "dilution_dominant_sharpening_helps"
        action = "local_sharpen_or_sigma_down_on_orbench"
    elif dilution_dominates and not diffuse_when_high:
        primary = "dilution_without_diffuse_weights"
        action = "inspect_nll_spread_not_entropy"
    elif not dilution_dominates:
        primary = "argmax_errors_drive_gap"
        action = "fix_g_direction_not_sharpening"
    else:
        primary = "mixed"
        action = "uncertain"
    return {
        "primary": primary,
        "suggested_action": action,
        "dilution_dominates": str(dilution_dominates),
        "diffuse_when_argmax_right_high_gap": str(diffuse_when_high),
        "best_power": best_power[0],
        "power_gap_recovery": float(power_recover),
        "best_sigma_scale": best_sigma[0],
        "sigma_gap_recovery": float(sigma_recover),
    }


def _print_summary(result: dict) -> None:
    """Human-readable summary."""
    h = result["headline"]
    d = result["gap_decomposition"]
    print("=== overrefusal dilution inspection (orbench) ===")
    print(
        f"n={result['n_orbench']}  sigma={result['sigma']:.4f}  "
        f"argmax agree oracle: {h['argmax_agreement']:.3f}  "
        f"Δ_exp={h['mean_expected_gap']:+.4f}",
    )
    print(
        f"oracle_weight={h['mean_oracle_weight']:.3f}  "
        f"entropy={h['mean_entropy']:.3f}  "
        f"eff_n_merges={h['mean_effective_n_merges']:.2f}",
    )
    print(
        f"gap|argmax right: {d['mean_gap_when_argmax_right']:+.4f} "
        f"(dilution {d['mean_dilution_cost_when_argmax_right']:+.4f})  "
        f"gap|wrong: {d['mean_gap_when_argmax_wrong']:+.4f}",
    )
    print("\n--- strata (argmax-right high vs low gap vs wrong) ---")
    print(
        f"{'stratum':<36} {'n':>5} {'Δ_exp':>8} {'w_oracle':>9} "
        f"{'w_off':>7} {'entropy':>8}",
    )
    for row in result["strata"]:
        if row["n"] == 0:
            continue
        print(
            f"{row['label']:<36} {row['n']:5d} "
            f"{row['mean_expected_gap']:+8.4f} {row['mean_oracle_weight']:9.3f} "
            f"{row['mean_off_oracle_weight']:7.3f} {row['mean_entropy']:8.3f}",
        )
    print("\n--- power sharpening (merge weights) ---")
    for key, row in result["power_sharpening"].items():
        print(
            f"{key:<12} Δ_exp={row['mean_expected_gap']:+.4f} "
            f"({row['gap_vs_baseline']:+.4f} vs baseline)  "
            f"w_oracle={row['mean_oracle_weight']:.3f}",
        )
    print("\n--- sigma sharpening (recompute G) ---")
    for key, row in result["sigma_sharpening"].items():
        print(
            f"{key:<12} σ={row['sigma']:.4f}  Δ_exp={row['mean_expected_gap']:+.4f} "
            f"({row['gap_vs_baseline']:+.4f})  w_oracle={row['mean_oracle_weight']:.3f}",
        )
    v = result["verdict"]
    print(
        f"\nverdict: {v['primary']}  action={v['suggested_action']}\n"
        f"  power recovery ({v['best_power']}): {v['power_gap_recovery']:+.4f}\n"
        f"  sigma recovery ({v['best_sigma_scale']}): {v['sigma_gap_recovery']:+.4f}",
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

    result = inspect_overrefusal_dilution(
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
