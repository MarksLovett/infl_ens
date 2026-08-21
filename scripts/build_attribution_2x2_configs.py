#!/usr/bin/env python3
"""Generate the four 2x2 attribution cell configs from the hypercube template."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/benchmark/router/seven_axis_collapse_hypercube_ga.yaml"
OUT_DIR = ROOT / "configs/benchmark/router/attribution_2x2"

CELLS = {
    "ga_theory_pre": {
        "header": (
            "# ATTRIBUTION_CELL: ga_theory_pre\n"
            "# 2x2 axes: init_mode=fixed_positions, theory_pre.enabled=true\n"
        ),
        "output_dir": "results/attribution_2x2/ga_theory_pre/seed0",
        "init_mode": "fixed_positions",
        "fixed_positions": (
            "  fixed_positions: "
            "results/hypercube_edge_gradient_ascent/fixed_positions.json\n"
        ),
        "theory_pre_enabled": "true",
    },
    "ga_no_theory_pre": {
        "header": (
            "# ATTRIBUTION_CELL: ga_no_theory_pre\n"
            "# 2x2 axes: init_mode=fixed_positions, theory_pre.enabled=false\n"
        ),
        "output_dir": "results/attribution_2x2/ga_no_theory_pre/seed0",
        "init_mode": "fixed_positions",
        "fixed_positions": (
            "  fixed_positions: "
            "results/hypercube_edge_gradient_ascent/fixed_positions.json\n"
        ),
        "theory_pre_enabled": "false",
    },
    "random_theory_pre": {
        "header": (
            "# ATTRIBUTION_CELL: random_theory_pre\n"
            "# 2x2 axes: init_mode=mean_noise, theory_pre.enabled=true\n"
        ),
        "output_dir": "results/attribution_2x2/random_theory_pre/seed0",
        "init_mode": "mean_noise",
        "fixed_positions": "",
        "theory_pre_enabled": "true",
    },
    "random_no_theory_pre": {
        "header": (
            "# ATTRIBUTION_CELL: random_no_theory_pre\n"
            "# 2x2 axes: init_mode=mean_noise, theory_pre.enabled=false\n"
        ),
        "output_dir": "results/attribution_2x2/random_no_theory_pre/seed0",
        "init_mode": "mean_noise",
        "fixed_positions": "",
        "theory_pre_enabled": "false",
    },
}

FOOTER = (
    "# Policy A: fixed 5-merge SFT groups (no sft_merge_only_if_collapsed).\n"
    "# Artifact paths (output_dir, sft.output_dir) differ per cell by design.\n"
)


def main() -> int:
    """Write four cell YAML files."""
    text = TEMPLATE.read_text(encoding="utf-8")
    text = text.replace(
        "  sft_merge_only_if_collapsed: true\n",
        "",
    )
    for name, spec in CELLS.items():
        body = text
        body = body.replace(
            "output_dir: results/seven_axis_collapse_hypercube_ga/seed0",
            f"output_dir: {spec['output_dir']}",
        )
        body = body.replace(
            "    output_dir: results/seven_axis_collapse_hypercube_ga/seed0/agents",
            f"    output_dir: {spec['output_dir']}/agents",
        )
        body = body.replace(
            "  init_mode: fixed_positions\n"
            "  fixed_positions: results/hypercube_edge_gradient_ascent/fixed_positions.json\n",
            f"  init_mode: {spec['init_mode']}\n{spec['fixed_positions']}",
        )
        body = body.replace(
            "    enabled: true",
            f"    enabled: {spec['theory_pre_enabled']}",
            1,
        )
        # Drop old header comments from template.
        if body.startswith("# Five-axis"):
            body = body.split("\n\n", 1)[1]
        out = spec["header"] + FOOTER + "\n" + body
        path = OUT_DIR / f"{name}.yaml"
        path.write_text(out, encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
