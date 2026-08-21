#!/usr/bin/env python3
"""Report resolved sigma per whitening arm (stability_fraction from grid)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from infl_ens.training.__main__ import _load_splits, _load_yaml, _make_trait_space, _sigma_from_cfg
from infl_ens.utils.resource import gaussian_stability_threshold, weighted_covariance

ARMS = ("baseline", "standardize", "whiten")
REF_SIGMA = 0.2207


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-config",
        default="configs/benchmark/router/attribution_2x2/ga_theory_pre.yaml",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    ref_cfg = _load_yaml(Path(args.reference_config))
    splits = _load_splits(ref_cfg)
    sigma_mode = ref_cfg.get("sigma_mode", "absolute")
    sigma_fraction = float(ref_cfg.get("sigma_fraction", 0.8))
    n_agents = len(ref_cfg.get("agents", []))

    rows: dict[str, dict] = {}
    for arm in ARMS:
        if arm == "baseline":
            cfg_path = Path("configs/benchmark/router/trait_whitening/baseline.yaml")
        else:
            cfg_path = Path(f"configs/benchmark/router/trait_whitening/{arm}.yaml")
        cfg = _load_yaml(cfg_path)
        space = _make_trait_space(cfg, splits)
        s0 = gaussian_stability_threshold(n_agents, space.grid, space.weights)
        sigma = _sigma_from_cfg(cfg, n_agents, space)
        cov = weighted_covariance(space.grid, space.weights)
        lam_max = float(np.linalg.eigvalsh(cov)[-1])
        rows[arm] = {
            "sigma_mode": sigma_mode,
            "sigma_fraction": sigma_fraction,
            "sigma_0_star": s0,
            "sigma_resolved": sigma,
            "grid_cov_lambda_max": lam_max,
            "linear_transform": (cfg.get("trait_space") or {}).get("linear_transform"),
        }

    case = (
        "data_relative"
        if sigma_mode == "stability_fraction"
        else "absolute"
    )
    fair_as_is = case == "data_relative"

    print("=== sigma resolution per whitening arm ===")
    print(f"sigma_mode={sigma_mode!r}  sigma_fraction={sigma_fraction}")
    print(f"case: {case}")
    print(f"reference baseline sigma (prior run): {REF_SIGMA:.4f}")
    for arm, row in rows.items():
        print(
            f"{arm}: sigma_0*={row['sigma_0_star']:.4f}  "
            f"sigma={row['sigma_resolved']:.4f}  "
            f"lambda_max(grid)={row['grid_cov_lambda_max']:.4f}",
        )
    if fair_as_is:
        print(
            "\nσ is data-relative: sigma_fraction * gaussian_stability_threshold("
            "space.grid, space.weights), recomputed on the transformed grid. "
            "Whitening arms self-rescale σ; no manual σ fix required.",
        )
    else:
        print("\nσ is absolute — transformed arms need manual rescaling or sweep.")

    payload = {
        "sigma_case": case,
        "fair_without_manual_rescale": fair_as_is,
        "sigma_mode": sigma_mode,
        "sigma_fraction": sigma_fraction,
        "reference_baseline_sigma_prior": REF_SIGMA,
        "per_arm": rows,
        "code_path": (
            "_sigma_from_cfg → gaussian_stability_threshold(n, space.grid, "
            "space.weights) when sigma_mode=stability_fraction"
        ),
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
