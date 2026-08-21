#!/usr/bin/env python3
"""Generate seed-0 split isolation configs (vary training seed, fixed manifest)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/benchmark/router/attribution_2x2/ga_no_theory_pre.yaml"
OUT_DIR = ROOT / "configs/benchmark/router/seed0_isolation"
TRAIN_SEEDS = (1, 2, 3)
MANIFEST = "data/splits/five_axis_seed0.json"
SPLIT_SEED = 0

FOOTER = (
    "# Policy A: fixed 5-merge SFT groups (no sft_merge_only_if_collapsed).\n"
    "# data_split pinned to five_axis_seed0.json; only training seed varies.\n"
)


def _body_for_train_seed(train_seed: int) -> str:
    """Patch template YAML for one training-seed arm."""
    body = TEMPLATE.read_text(encoding="utf-8")
    if body.startswith("# ATTRIBUTION_CELL"):
        body = body.split("\n\n", 1)[1]
    body = body.replace("seed: 0\n", f"seed: {train_seed}\n", 1)
    body = body.replace(
        "  seed: 0\n  train_frac:",
        f"  seed: {SPLIT_SEED}\n  train_frac:",
        1,
    )
    body = body.replace(
        f"manifest: {MANIFEST}",
        f"manifest: {MANIFEST}",
    )
    out_dir = f"results/seed0_isolation/ga_no_theory_pre_train{train_seed}"
    body = body.replace(
        "output_dir: results/attribution_2x2/ga_no_theory_pre/seed0",
        f"output_dir: {out_dir}",
    )
    body = body.replace(
        "    output_dir: results/attribution_2x2/ga_no_theory_pre/seed0/agents",
        f"    output_dir: {out_dir}/agents",
    )
    body = body.replace(
        "    seed: 0\n",
        f"    seed: {train_seed}\n",
        1,
    )
    return body


def main() -> int:
    """Write isolation YAML files and manifest."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs: list[dict] = []
    for train_seed in TRAIN_SEEDS:
        name = f"ga_no_theory_pre_train{train_seed}"
        header = (
            f"# SEED0_ISOLATION: {name}\n"
            f"# Fixed split: {MANIFEST} (data_split.seed={SPLIT_SEED})\n"
            f"# Training seed: {train_seed} (top-level seed + sft.seed)\n"
        )
        body = _body_for_train_seed(train_seed)
        path = OUT_DIR / f"{name}.yaml"
        path.write_text(header + FOOTER + "\n" + body, encoding="utf-8")
        configs.append({
            "name": name,
            "train_seed": train_seed,
            "split_seed": SPLIT_SEED,
            "manifest": MANIFEST,
            "config": str(path.relative_to(ROOT)),
            "output_dir": f"results/seed0_isolation/{name}",
        })

    manifest = {
        "description": (
            "GA no-theory-pre on fixed five_axis_seed0.json split; "
            "vary training seed to test +0.009 reproducibility."
        ),
        "reference": {
            "attribution_2x2_ga_no_theory_pre_seed0": {
                "pooled_nll": 1.9457460591664868,
                "learned_nll": 1.9550691510177387,
                "delta_vs_pooled": 0.009323091851251908,
            },
        },
        "configs": configs,
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
