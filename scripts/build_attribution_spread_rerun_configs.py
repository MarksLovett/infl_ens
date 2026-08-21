#!/usr/bin/env python3
"""Generate spread-calibrated attribution re-run configs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from infl_ens.utils.init_noise_calibration import (
    mean_pairwise_spread,
    solve_init_noise,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/benchmark/router/attribution_2x2/ga_no_theory_pre.yaml"
OUT_DIR = ROOT / "configs/benchmark/router/attribution_spread_rerun"
BASE_CONFIG = "configs/benchmark/router/seven_axis_collapse_hypercube_ga.yaml"

N_AGENTS = 10
N_DIMS = 5
TARGET_MATCHED = 0.9
TARGET_MODERATE = 0.45
RANDOM_SEEDS = (0, 1, 2)
GA_NO_SEEDS = (1, 2, 3, 4)
GA_SPOT_CHECK_SEED = 1

INIT_NOISE_MATCHED = solve_init_noise(TARGET_MATCHED, n_agents=N_AGENTS, n_dims=N_DIMS)
INIT_NOISE_MODERATE = solve_init_noise(
    TARGET_MODERATE, n_agents=N_AGENTS, n_dims=N_DIMS,
)

# Optional: override with trait-space recalibration on doob via
# scripts/recalibrate_init_noise_on_trait_space.py and set env vars:
#   INIT_NOISE_MATCHED_OVERRIDE INIT_NOISE_MODERATE_OVERRIDE

FOOTER = (
    "# Policy A: fixed 5-merge SFT groups (no sft_merge_only_if_collapsed).\n"
    "# Artifact paths differ per cell/seed by design.\n"
)


def _replace_seed(body: str, seed: int) -> str:
    """Patch seed fields and manifest path."""
    body = body.replace("seed: 0\n", f"seed: {seed}\n", 1)
    body = body.replace(
        "  seed: 0\n  train_frac:",
        f"  seed: {seed}\n  train_frac:",
        1,
    )
    body = body.replace(
        "manifest: data/splits/five_axis_seed0.json",
        f"manifest: data/splits/five_axis_seed{seed}.json",
    )
    body = body.replace(
        "    seed: 0\n",
        f"    seed: {seed}\n",
        1,
    )
    return body


def _write_config(
    name: str,
    *,
    seed: int,
    header: str,
    output_dir: str,
    init_mode: str,
    init_noise: float,
    theory_pre_enabled: str,
    fixed_positions_line: str,
    target_spread: float | None,
) -> Path:
    """Write one YAML config from the attribution template."""
    body = TEMPLATE.read_text(encoding="utf-8")
    if body.startswith("# ATTRIBUTION_CELL"):
        body = body.split("\n\n", 1)[1]
    body = _replace_seed(body, seed)
    body = body.replace(
        "output_dir: results/attribution_2x2/ga_no_theory_pre/seed0",
        f"output_dir: {output_dir}",
    )
    body = body.replace(
        "    output_dir: results/attribution_2x2/ga_no_theory_pre/seed0/agents",
        f"    output_dir: {output_dir}/agents",
    )
    init_block = f"  init_mode: {init_mode}\n"
    if fixed_positions_line:
        init_block += fixed_positions_line
    body = body.replace(
        "  init_mode: fixed_positions\n"
        "  fixed_positions: results/hypercube_edge_gradient_ascent/fixed_positions.json\n",
        init_block,
    )
    body = body.replace("  init_noise: 0.0\n", f"  init_noise: {init_noise:.6f}\n", 1)
    body = body.replace(
        "    enabled: false",
        f"    enabled: {theory_pre_enabled}",
        1,
    )
    spread_note = ""
    if target_spread is not None:
        spread_note = (
            f"# target_mean_pairwise: {target_spread}\n"
            f"# calibrated_init_noise: {init_noise:.6f}\n"
        )
    out = header + spread_note + FOOTER + "\n" + body
    path = OUT_DIR / f"{name}.yaml"
    path.write_text(out, encoding="utf-8")
    return path


def main() -> int:
    """Write spread re-run YAML files and manifest."""
    global INIT_NOISE_MATCHED, INIT_NOISE_MODERATE
    if os.environ.get("INIT_NOISE_MATCHED_OVERRIDE"):
        INIT_NOISE_MATCHED = float(os.environ["INIT_NOISE_MATCHED_OVERRIDE"])
    if os.environ.get("INIT_NOISE_MODERATE_OVERRIDE"):
        INIT_NOISE_MODERATE = float(os.environ["INIT_NOISE_MODERATE_OVERRIDE"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "init_noise_matched": INIT_NOISE_MATCHED,
        "init_noise_moderate": INIT_NOISE_MODERATE,
        "target_mean_pairwise_matched": TARGET_MATCHED,
        "target_mean_pairwise_moderate": TARGET_MODERATE,
        "configs": [],
    }
    ga_fixed = (
        "  fixed_positions: "
        "results/hypercube_edge_gradient_ascent/fixed_positions.json\n"
    )

    for seed in GA_NO_SEEDS:
        name = f"ga_no_theory_pre_seed{seed}"
        path = _write_config(
            name,
            seed=seed,
            header=(
                f"# SPREAD_RERUN_CELL: {name}\n"
                "# GA reproducibility arm (no theory_pre).\n"
            ),
            output_dir=f"results/attribution_spread_rerun/{name}",
            init_mode="fixed_positions",
            init_noise=0.0,
            theory_pre_enabled="false",
            fixed_positions_line=ga_fixed,
            target_spread=None,
        )
        manifest["configs"].append(str(path.relative_to(ROOT)))

    spot_name = f"ga_theory_pre_seed{GA_SPOT_CHECK_SEED}"
    path = _write_config(
        spot_name,
        seed=GA_SPOT_CHECK_SEED,
        header=(
            f"# SPREAD_RERUN_CELL: {spot_name}\n"
            "# GA theory_pre spot-check (equivalence across seeds).\n"
        ),
        output_dir=f"results/attribution_spread_rerun/{spot_name}",
        init_mode="fixed_positions",
        init_noise=0.0,
        theory_pre_enabled="true",
        fixed_positions_line=ga_fixed,
        target_spread=None,
    )
    manifest["configs"].append(str(path.relative_to(ROOT)))

    random_cells = [
        (
            "random_s09_theory_pre",
            TARGET_MATCHED,
            INIT_NOISE_MATCHED,
            "true",
            "matched spread + theory_pre",
        ),
        (
            "random_s09_no_theory_pre",
            TARGET_MATCHED,
            INIT_NOISE_MATCHED,
            "false",
            "matched spread, no theory_pre",
        ),
        (
            "random_s045_theory_pre",
            TARGET_MODERATE,
            INIT_NOISE_MODERATE,
            "true",
            "moderate spread + theory_pre",
        ),
    ]
    for cell, target, noise, pre, desc in random_cells:
        for seed in RANDOM_SEEDS:
            name = f"{cell}_seed{seed}"
            path = _write_config(
                name,
                seed=seed,
                header=(
                    f"# SPREAD_RERUN_CELL: {name}\n"
                    f"# Random arm: {desc}.\n"
                ),
                output_dir=f"results/attribution_spread_rerun/{name}",
                init_mode="mean_noise",
                init_noise=noise,
                theory_pre_enabled=pre,
                fixed_positions_line="",
                target_spread=target,
            )
            manifest["configs"].append(str(path.relative_to(ROOT)))

    manifest["verified_spread"] = {
        "matched": mean_pairwise_spread(INIT_NOISE_MATCHED, n_agents=N_AGENTS, n_dims=N_DIMS),
        "moderate": mean_pairwise_spread(INIT_NOISE_MODERATE, n_agents=N_AGENTS, n_dims=N_DIMS),
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    _assert_init_noise(configs := {
        p.stem: __import__("yaml").safe_load(p.read_text())
        for p in sorted(OUT_DIR.glob("*.yaml"))
    })
    print(json.dumps(manifest, indent=2))
    return 0


def _assert_init_noise(configs: dict[str, dict]) -> None:
    """Ensure random cells have positive init_noise; GA cells keep zero."""
    for name, cfg in configs.items():
        cl = cfg["closed_loop"]
        noise = float(cl.get("init_noise", 0.0))
        mode = str(cl.get("init_mode"))
        if mode == "mean_noise" and noise <= 0.0:
            raise ValueError(f"{name}: init_noise must be > 0")
        if mode == "fixed_positions" and noise != 0.0:
            raise ValueError(f"{name}: GA init_noise must be 0.0")
    print(f"init_noise assertions passed ({len(configs)} configs)")


if __name__ == "__main__":
    raise SystemExit(main())
