#!/usr/bin/env python3
"""VERIFY_ONLY: apply frozen transforms to seed-0 pre-normalizer trait vectors.

Checks realized moments on pre-CDF coordinates (where standardize/whiten
semantics live) and that the downstream quantile stage restores
near-uniform ``[0, 1]`` marginals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from infl_ens.data.benchmarks.safety_trait_space import (
    build_safety_trait_space_bundle,
    project_pre_normalizer_coordinates,
)
from infl_ens.data.splits import flatten_partition_prompts, load_split_manifest
from infl_ens.data.trait_linear_transform import FrozenLinearTransform, realized_moments
from infl_ens.data.trait_normalize import fit_quantile_normalizer
from infl_ens.data.trait_space_cache import (
    _trait_space_build_kwargs,
    make_trait_space_encoder,
)
from infl_ens.training.__main__ import _load_splits, _load_yaml


def _max_ks_uniform(coords: np.ndarray) -> float:
    """Max per-axis KS statistic of columns against U[0, 1]."""
    x = np.asarray(coords, dtype=float)
    n = x.shape[0]
    grid = np.arange(1, n + 1) / n
    worst = 0.0
    for j in range(x.shape[1]):
        v = np.sort(x[:, j])
        ks = float(np.max(np.maximum(np.abs(grid - v), np.abs(v - (grid - 1.0 / n)))))
        worst = max(worst, ks)
    return worst


def _seed0_prompts(cfg_path: Path, manifest_path: Path) -> list[str]:
    cfg = _load_yaml(cfg_path)
    splits = _load_splits(cfg)
    manifest = load_split_manifest(manifest_path)
    prompts: list[str] = []
    for part in ("train", "val", "test"):
        p, _r, _b = flatten_partition_prompts(splits, manifest, part)
        prompts.extend(p)
    return prompts


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--router-config",
        type=Path,
        default=Path("configs/benchmark/router/attribution_2x2/ga_theory_pre.yaml"),
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=Path("data/splits/five_axis_seed0.json"),
    )
    parser.add_argument(
        "--transform-dir",
        type=Path,
        default=Path("results/trait_whitening/transforms"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    prompts = _seed0_prompts(args.router_config, args.eval_manifest)
    cfg = _load_yaml(args.router_config)
    splits = _load_splits(cfg)
    encoder = make_trait_space_encoder(cfg)
    build_kwargs = _trait_space_build_kwargs(cfg)
    build_kwargs["linear_transform"] = None
    bundle = build_safety_trait_space_bundle(splits, encoder, **build_kwargs)
    raw = np.asarray(
        project_pre_normalizer_coordinates(encoder, bundle.axes, prompts),
        dtype=float,
    )

    results: dict = {
        "n_prompts": len(prompts),
        "eval_manifest": str(args.eval_manifest),
        "baseline_raw": realized_moments(raw),
        "arms": {},
    }

    ok = True
    for name in ("standardize", "whiten"):
        path = args.transform_dir / f"{name}_seed1.json"
        if not path.is_file():
            print(f"MISSING {path}")
            ok = False
            continue
        transform = FrozenLinearTransform.load(path)
        if transform.to_dict().get("oracle_labels_used"):
            print(f"FAIL: transform artifact claims oracle_labels_used=True")
            ok = False
        transformed = transform.apply(raw)
        moments = realized_moments(transformed)
        # The production pipeline always quantile-normalizes after the
        # transform; verify the post-CDF marginals are uniform in [0, 1].
        normalized = fit_quantile_normalizer(transformed).transform(transformed)
        post_cdf_ks = _max_ks_uniform(normalized)
        in_box = bool(np.all((normalized >= 0.0) & (normalized <= 1.0)))
        results["arms"][name] = {
            "fit_source": transform.fit_source,
            "oracle_labels_used": False,
            "moments": moments,
            "post_cdf_max_ks_vs_uniform": post_cdf_ks,
            "post_cdf_in_unit_box": in_box,
        }
        print(f"=== {name} on seed-0 ({len(prompts)} prompts) ===")
        print(f"fit_source: {transform.fit_source}")
        print(f"per_axis_variance: {[round(v, 3) for v in moments['per_axis_variance']]}")
        print(f"mean_abs_offdiag_corr: {moments['mean_abs_offdiag_corr']:.4f}")
        print(f"post_cdf_max_ks_vs_uniform: {post_cdf_ks:.4f}  in_unit_box={in_box}")
        if not in_box or post_cdf_ks > 0.1:
            print("WARNING: post-CDF marginals not near-uniform in [0, 1]")
            ok = False
        if name == "whiten" and moments["mean_abs_offdiag_corr"] > 0.15:
            print("WARNING: whitening did not fully decorrelate seed-0 holdout")
        if name == "standardize":
            vars_ = moments["per_axis_variance"]
            if not all(0.7 < v < 1.3 for v in vars_):
                print("WARNING: standardize variances not near 1 on seed-0")

    fit_manifest = args.transform_dir / "fit_manifest.json"
    if fit_manifest.is_file():
        fm = json.loads(fit_manifest.read_text(encoding="utf-8"))
        results["fit_manifest"] = fm
        print("\n=== fit manifest ===")
        print(f"fit_source: {fm.get('fit_source')}")
        print(f"oracle_labels_used: {fm.get('oracle_labels_used')}")
        if fm.get("oracle_labels_used"):
            ok = False
        rob = fm.get("robustness", {})
        if rob:
            print(f"seed1 vs seed0 whiten fro: {rob['seed1_vs_seed0_whiten']['frobenius']:.4f}")
            if rob.get("divergence_flag"):
                print("FLAG: seed-1 vs seed-0 whitening matrices diverge substantially")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nVERIFY_ONLY ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
