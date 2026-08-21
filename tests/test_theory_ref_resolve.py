"""Smoke-test :func:`resolve_theory_22_reference` for high sigma."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from infl_ens.data import load_benchmark_splits
from infl_ens.training.__main__ import _make_trait_space
from infl_ens.utils.agent_init import _reference_is_22, resolve_theory_22_reference

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/benchmark/router/safety_truth_n4_r10_position_only_cum.yaml"


@pytest.mark.slow
def test_resolve_theory_22_reference_high_sigma() -> None:
    """Theory reference should resolve at several sigma fractions."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    splits = load_benchmark_splits(cfg["benchmarks"])
    space = _make_trait_space(cfg, splits)
    names = sorted(a["name"] for a in cfg["agents"])
    for sf in (0.25, 1.0, 1.5):
        sigma = float(sf) * float(cfg.get("sigma_star", space.sigma_star))
        ref = resolve_theory_22_reference(
            space,
            names,
            sigma=sigma,
            repo_root=ROOT,
            sigma_fraction=sf,
        )
        assert ref.shape[0] == len(names)
        assert _reference_is_22(ref) or sf >= 1.0
