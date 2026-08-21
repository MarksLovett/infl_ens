#!/usr/bin/env python3
"""Fit unsupervised trait-space transforms on held-out seed-1 prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from infl_ens.data.benchmarks.safety_trait_space import (
    build_safety_trait_space_bundle,
    project_pre_normalizer_coordinates,
)
from infl_ens.data.splits import load_split_manifest
from infl_ens.data.trait_linear_transform import (
    fit_standardize,
    fit_whiten,
    matrix_distance,
)
from infl_ens.data.trait_space_cache import (
    _trait_space_build_kwargs,
    make_trait_space_encoder,
)
from infl_ens.training.__main__ import _load_splits, _load_yaml

FIT_SOURCE = (
    "full-corpus pre-normalizer trait covariance (seed-1 manifest union), "
    "unsupervised, label-blind; no oracle labels used"
)


def _all_manifest_prompts(
    cfg_path: Path,
    manifest_path: Path,
    repo_root: Path,
) -> list[str]:
    """Union of train, val, and test prompts for a split manifest."""
    cfg = _load_yaml(cfg_path)
    splits = _load_splits(cfg)
    manifest = load_split_manifest(manifest_path)
    prompts: list[str] = []
    for part in ("train", "val", "test"):
        from infl_ens.data.splits import flatten_partition_prompts

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
        "--fit-manifest",
        type=Path,
        default=Path("data/splits/five_axis_seed1.json"),
    )
    parser.add_argument(
        "--robustness-manifest",
        type=Path,
        default=Path("data/splits/five_axis_seed0.json"),
        help="Used only to compare seed-1 vs seed-0 whitening matrices.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/trait_whitening/transforms"),
    )
    args = parser.parse_args()
    repo = Path(".")

    cfg = _load_yaml(args.router_config)
    splits = _load_splits(cfg)
    encoder = make_trait_space_encoder(cfg)
    build_kwargs = _trait_space_build_kwargs(cfg)
    # Transforms are fit on PRE-normalizer coordinates (before any frozen
    # linear transform and before the quantile stage), so force the
    # config's own transform off for the fit build.
    build_kwargs["linear_transform"] = None
    bundle = build_safety_trait_space_bundle(splits, encoder, **build_kwargs)

    fit_prompts = _all_manifest_prompts(args.router_config, args.fit_manifest, repo)
    fit_coords = np.asarray(
        project_pre_normalizer_coordinates(encoder, bundle.axes, fit_prompts),
        dtype=float,
    )

    standardize = fit_standardize(fit_coords, fit_source=FIT_SOURCE)
    whiten = fit_whiten(fit_coords, fit_source=FIT_SOURCE)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    std_path = args.output_dir / "standardize_seed1.json"
    whiten_path = args.output_dir / "whiten_seed1.json"
    standardize.save(std_path)
    whiten.save(whiten_path)

    robustness: dict = {}
    if args.robustness_manifest.is_file():
        rob_prompts = _all_manifest_prompts(
            args.router_config, args.robustness_manifest, repo,
        )
        rob_coords = np.asarray(
            project_pre_normalizer_coordinates(encoder, bundle.axes, rob_prompts),
            dtype=float,
        )
        whiten_s0 = fit_whiten(
            rob_coords,
            fit_source="seed-0 trait vectors (robustness only; not used in arms)",
        )
        robustness = {
            "seed1_vs_seed0_whiten": matrix_distance(whiten, whiten_s0),
            "divergence_flag": matrix_distance(whiten, whiten_s0)["frobenius"] > 1.0,
            "note": (
                "Identical matrices expected: both manifests union to same "
                "23,155-prompt corpus (not a held-out fit)."
            ),
        }

    manifest = {
        "fit_source": FIT_SOURCE,
        "oracle_labels_used": False,
        "held_out_fit": False,
        "fit_manifest": str(args.fit_manifest),
        "n_fit_prompts": len(fit_prompts),
        "transforms": {
            "standardize": str(std_path),
            "whiten": str(whiten_path),
        },
        "robustness": robustness,
    }
    manifest_path = args.output_dir / "fit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== fit whitening transforms (full-corpus, label-blind) ===")
    print(f"fit_source: {FIT_SOURCE}")
    print(f"n_fit_prompts={len(fit_prompts)}")
    print(f"wrote {std_path}")
    print(f"wrote {whiten_path}")
    if robustness:
        d = robustness["seed1_vs_seed0_whiten"]
        print(
            f"seed1 vs seed0 whiten matrix: fro={d['frobenius']:.4f} "
            f"op2={d['operator_2']:.4f}  flag={robustness['divergence_flag']}",
        )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
