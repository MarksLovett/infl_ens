#!/usr/bin/env python3
"""Generate within-merge spread experiment configs (seed-0 split)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/benchmark/router/attribution_2x2/ga_no_theory_pre.yaml"
OUT_DIR = ROOT / "configs/benchmark/router/within_merge_spread"
INIT_DIR = "results/within_merge_spread_init"
MANIFEST = "data/splits/five_axis_seed0.json"
SPLIT_SEED = 0
TRAIN_SEED = 0

FOOTER = (
    "# Policy A: fixed 5-merge SFT; snap_collapsed_pairs OFF (preserve spread).\n"
    "# Seed-0 split; theory_pre OFF.\n"
)


def _body(arm: str, fixed_rel: str, train_seed: int) -> str:
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
        "manifest: data/splits/five_axis_seed0.json",
        f"manifest: {MANIFEST}",
    )
    out = f"results/within_merge_spread/{arm}"
    body = body.replace(
        "output_dir: results/attribution_2x2/ga_no_theory_pre/seed0",
        f"output_dir: {out}",
    )
    body = body.replace(
        "    output_dir: results/attribution_2x2/ga_no_theory_pre/seed0/agents",
        f"    output_dir: {out}/agents",
    )
    body = body.replace(
        "  fixed_positions: results/hypercube_edge_gradient_ascent/fixed_positions.json\n",
        f"  fixed_positions: {fixed_rel}\n",
    )
    body = body.replace("  snap_collapsed_pairs: true\n", "  snap_collapsed_pairs: false\n")
    body = body.replace("    enabled: false", "    enabled: false", 1)
    body = body.replace("    seed: 0\n", f"    seed: {train_seed}\n", 1)
    return body


def main() -> int:
    """Write YAML configs and manifest."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arms = [
        ("oracle_k2_aligned", f"{INIT_DIR}/fixed_positions_aligned.json"),
        ("oracle_k2_misaligned", f"{INIT_DIR}/fixed_positions_misaligned.json"),
    ]
    configs: list[dict] = []
    for name, fixed_rel in arms:
        header = (
            f"# WITHIN_MERGE_SPREAD: {name}\n"
            f"# Sub-agents at oracle k=2 centers ({name.split('_', 2)[-1]}).\n"
        )
        path = OUT_DIR / f"{name}.yaml"
        path.write_text(
            header + FOOTER + "\n" + _body(name, fixed_rel, TRAIN_SEED),
            encoding="utf-8",
        )
        configs.append({
            "name": name,
            "config": str(path.relative_to(ROOT)),
            "output_dir": f"results/within_merge_spread/{name}",
            "fixed_positions": fixed_rel,
        })

    manifest = {
        "description": (
            "Within-merge spread: k=2 oracle centers vs cyclic-misaligned control. "
            "Fixed 5 merges, seed-0 split, snap OFF."
        ),
        "reference_ga_colocated": {
            "path": "results/attribution_2x2/ga_theory_pre/seed0",
            "oracle_minus_learned_expected": -0.0379,
        },
        "configs": configs,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
