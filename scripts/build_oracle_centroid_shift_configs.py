#!/usr/bin/env python3
"""Generate oracle-centroid shift config from ga_theory_pre reference."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "configs/benchmark/router/attribution_2x2/ga_theory_pre.yaml"
OUT_DIR = ROOT / "configs/benchmark/router/oracle_centroid_shift"
INIT_PATH = "results/oracle_centroid_shift_init/fixed_positions.json"
OUT_RUN = "results/oracle_centroid_shift/ga_theory_pre/seed0"

FOOTER = (
    "# Policy A: fixed 5-merge SFT; colocated oracle-centroid init.\n"
    "# Seed-0 split (five_axis_seed0.json); theory_pre ON (same as reference).\n"
)


def main() -> int:
    """Write experiment YAML and manifest."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body = REFERENCE.read_text(encoding="utf-8")
    if body.startswith("# ATTRIBUTION_CELL"):
        body = body.split("\n\n", 1)[1]

    body = body.replace(
        "output_dir: results/attribution_2x2/ga_theory_pre/seed0",
        f"output_dir: {OUT_RUN}",
    )
    body = body.replace(
        "    output_dir: results/attribution_2x2/ga_theory_pre/seed0/agents",
        f"    output_dir: {OUT_RUN}/agents",
    )
    body = body.replace(
        "  fixed_positions: results/hypercube_edge_gradient_ascent/fixed_positions.json\n",
        f"  fixed_positions: {INIT_PATH}\n",
    )

    header = (
        "# ORACLE_CENTROID_SHIFT: colocated init at 1-component oracle centroids\n"
        "# Only init centers + output paths differ from ga_theory_pre reference.\n"
    )
    cfg_path = OUT_DIR / "ga_theory_pre.yaml"
    cfg_path.write_text(header + FOOTER + "\n" + body, encoding="utf-8")

    meta_path = ROOT / "results/oracle_centroid_shift_init/centroid_metadata.json"
    prediction = None
    if meta_path.is_file():
        prediction = json.loads(meta_path.read_text(encoding="utf-8")).get("prediction")

    manifest = {
        "description": (
            "Oracle-centroid shift: colocated pairs init at per-merge mean of "
            "oracle-winning prompts (1-component, not k=2 spread)."
        ),
        "reference": {
            "config": str(REFERENCE.relative_to(ROOT)),
            "output_dir": "results/attribution_2x2/ga_theory_pre/seed0",
            "g_argmax_agreement": 0.742,
            "oracle_minus_learned_expected": -0.0379,
            "privacy_harm_agreement": 0.635,
        },
        "prediction": prediction,
        "validity_gate": (
            "Routing metrics are only interpretable if "
            "verify_oracle_centroid_persistence.py passes: final merge "
            "centers must stay nearer oracle centroids than reference GA "
            "final positions (theory-pre + closed-loop must not revert the shift)."
        ),
        "experiment": {
            "config": str(cfg_path.relative_to(ROOT)),
            "output_dir": OUT_RUN,
            "fixed_positions": INIT_PATH,
        },
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"wrote {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
