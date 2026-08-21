#!/usr/bin/env python3
"""Generate trait-whitening experiment configs (baseline / standardize / whiten)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "configs/benchmark/router/attribution_2x2/ga_theory_pre.yaml"
OUT_DIR = ROOT / "configs/benchmark/router/trait_whitening"
TRANSFORM_DIR = "results/trait_whitening/transforms"

PREDICTION = (
    "Whitening should shrink theory-attractor vs oracle-centroid L2 from "
    "~0.176 and raise argmax agreement (especially privacy/harm). "
    "If standardize alone captures most gain → axis scale; if only full "
    "whiten helps → correlation. Clean null on both → stop (no nonlinear)."
)

ARMS = [
    ("baseline", None),
    ("standardize", f"{TRANSFORM_DIR}/standardize_seed1.json"),
    ("whiten", f"{TRANSFORM_DIR}/whiten_seed1.json"),
]


def _body(transform_path: str | None, arm: str) -> str:
    body = REFERENCE.read_text(encoding="utf-8")
    if body.startswith("# ATTRIBUTION_CELL"):
        body = body.split("\n\n", 1)[1]
    out = f"results/trait_whitening/{arm}/seed0"
    body = body.replace(
        "output_dir: results/attribution_2x2/ga_theory_pre/seed0",
        f"output_dir: {out}",
    )
    body = body.replace(
        "    output_dir: results/attribution_2x2/ga_theory_pre/seed0/agents",
        f"    output_dir: {out}/agents",
    )
    if transform_path:
        needle = "  coordinate_stretch_gammas:\n"
        insert = (
            f"  linear_transform: {transform_path}\n"
        )
        if "  linear_transform:" in body:
            raise ValueError("reference already has linear_transform")
        body = body.replace(
            "  cache_dir: data/trait_space_cache\n",
            f"  cache_dir: data/trait_space_cache\n{insert}",
        )
    return body


def main() -> int:
    """Write three arm YAMLs and manifest."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs: list[dict] = []
    for arm, transform in ARMS:
        header = (
            f"# TRAIT_WHITENING: {arm}\n"
            f"# Eval on seed-0; transform fit on seed-1 (if any).\n"
        )
        path = OUT_DIR / f"{arm}.yaml"
        path.write_text(header + "\n" + _body(transform, arm), encoding="utf-8")
        configs.append({
            "arm": arm,
            "config": str(path.relative_to(ROOT)),
            "output_dir": f"results/trait_whitening/{arm}/seed0",
            "linear_transform": transform,
        })

    manifest = {
        "description": "Trait-space whitening: baseline vs standardize vs whiten.",
        "reference": str(REFERENCE.relative_to(ROOT)),
        "eval_manifest": "data/splits/five_axis_seed0.json",
        "fit_manifest": "data/splits/five_axis_seed1.json",
        "prediction": PREDICTION,
        "validity": {
        "fit_on": "full-corpus trait covariance via seed-1 manifest (same 23,155 prompts as seed-0 union)",
        "eval_on": "seed-0 routing pool (4631 test prompts for gap metrics)",
        "oracle_labels_in_fit": False,
        "not_held_out": (
            "seed-1 train+val+test union equals full corpus; transform is not "
            "fit on a partition held out from seed-0 eval prompts, but uses "
            "only input second moments (label-blind)."
        ),
            "layers": "grid + project (dynamics and G routing)",
        },
        "arms": configs,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
