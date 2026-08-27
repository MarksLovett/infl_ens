"""Tests for adaptive position-step blending."""

from __future__ import annotations

import numpy as np

from infl_ens.training.position_step import effective_blend, parse_position_step


def test_static_returns_blend_max() -> None:
    """Static mode uses the configured ceiling unchanged."""
    b = effective_blend(
        np.array([0.5, 0.5]),
        np.array([0.9, 0.1]),
        blend=0.5,
        mode="static",
    )
    assert abs(b - 0.5) < 1e-9


def test_cap_linf_scales_large_jump() -> None:
    """L∞ cap shrinks blend when centroid is far away."""
    current = np.array([0.5, 0.5])
    target = np.array([0.9, 0.1])
    b = effective_blend(
        current, target, blend=0.5, mode="cap_linf", step_cap=0.1,
    )
    # max |delta| = 0.4 -> beta = min(0.5, 0.1/0.4) = 0.25
    assert abs(b - 0.25) < 1e-9


def test_cap_l2_scales_by_euclidean_norm() -> None:
    """L2 cap uses Euclidean displacement."""
    current = np.array([0.0, 0.0])
    target = np.array([0.3, 0.4])  # norm 0.5
    b = effective_blend(
        current, target, blend=0.5, mode="cap_l2", step_cap=0.1,
    )
    assert abs(b - 0.2) < 1e-9


def test_trust_box_stays_inside_unit_square() -> None:
    """Trust-box mode keeps the one-step update inside [0, 1]^L."""
    current = np.array([0.02, 0.98])
    target = np.array([0.95, 0.05])
    b = effective_blend(
        current, target, blend=0.5, mode="trust_box", step_cap=0.5,
    )
    pos = (1.0 - b) * current + b * target
    assert np.all(pos >= 0.0) and np.all(pos <= 1.0)


def test_parse_position_step_defaults() -> None:
    """Missing config yields static mode."""
    ps = parse_position_step(None)
    assert ps["mode"] == "static"
