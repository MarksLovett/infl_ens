#!/usr/bin/env python3
"""Smoke-test resolve_theory_22_reference for high sigma."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from infl_ens.data import load_benchmark_splits  # noqa: E402
from infl_ens.training.__main__ import _make_trait_space  # noqa: E402
from infl_ens.utils.agent_init import (  # noqa: E402
    _reference_is_22,
    resolve_theory_22_reference,
)


def main() -> int:
    cfg = yaml.safe_load(
        (ROOT / "configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml").read_text(),
    )
    splits = load_benchmark_splits(cfg["benchmark"])
    space = _make_trait_space(cfg, splits)
    names = sorted(a["name"] for a in cfg["agents"])
    sigma_star = float(cfg.get("sigma_star", space.sigma_star))
    for sf in (0.25, 1.0, 1.5):
        sigma = float(sf) * sigma_star
        ref = resolve_theory_22_reference(
            space, names, sigma=sigma, repo_root=ROOT, sigma_fraction=sf,
        )
        print(
            f"sigma_fraction={sf}  (2,2)={_reference_is_22(ref)}  "
            f"spread_ok  harm_sep={ref[:, 0].max() - ref[:, 0].min():.3f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
