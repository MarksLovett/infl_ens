"""Offline tests for :mod:`infl_ens.data.trait_normalize`."""

from __future__ import annotations

import numpy as np

from infl_ens.data.trait_normalize import (
    QuantileNormalizer,
    fit_quantile_normalizer,
)


def test_monotone_deterministic_and_bounded() -> None:
    """CDF map is monotone, repeatable, in [0, 1], and clamps out-of-range."""
    rng = np.random.default_rng(0)
    coords = np.column_stack(
        [rng.standard_normal(500) * 3.0 + 1.0, rng.exponential(2.0, 500)],
    )
    qn = fit_quantile_normalizer(coords, n_knots=101)
    out = qn.transform(coords)
    assert out.shape == coords.shape
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    # Deterministic.
    assert np.array_equal(out, qn.transform(coords))
    # Monotone per axis.
    for j in range(2):
        order = np.argsort(coords[:, j])
        assert np.all(np.diff(out[order, j]) >= 0.0)
    # Out-of-range values map to exactly 0/1.
    extremes = np.asarray([[-100.0, -100.0], [100.0, 100.0]])
    ext_out = qn.transform(extremes)
    assert np.allclose(ext_out[0], 0.0)
    assert np.allclose(ext_out[1], 1.0)
    # 1-D input keeps rank.
    single = qn.transform(coords[0])
    assert single.shape == (2,)
    assert np.allclose(single, out[0])


def test_ties_and_constant_column() -> None:
    """Tied values share one midpoint image; constant columns map to 0.5."""
    col = np.concatenate([np.zeros(50), np.ones(50)])
    coords = np.column_stack([col, np.zeros(100)])
    qn = fit_quantile_normalizer(coords, n_knots=101)
    out = qn.transform(coords)
    assert np.all(np.isfinite(out))
    # All tied zeros map to a single value, all tied ones to another.
    assert np.unique(out[:50, 0]).size == 1
    assert np.unique(out[50:, 0]).size == 1
    assert out[0, 0] < out[-1, 0]
    # Constant column: everything maps to 0.5.
    assert np.allclose(out[:, 1], 0.5)


def test_near_uniform_marginals_on_fit_data() -> None:
    """The fitted CDF makes calibration marginals near-uniform."""
    rng = np.random.default_rng(1)
    coords = np.column_stack(
        [rng.standard_normal(2000), rng.lognormal(0.0, 1.0, 2000)],
    )
    qn = fit_quantile_normalizer(coords)
    out = qn.transform(coords)
    n = out.shape[0]
    ecdf_grid = np.arange(1, n + 1) / n
    for j in range(2):
        v = np.sort(out[:, j])
        ks = np.max(np.abs(ecdf_grid - v))
        assert ks < 0.05


def test_dict_roundtrip_preserves_transform() -> None:
    """to_dict/from_dict reproduces the transform exactly."""
    rng = np.random.default_rng(2)
    coords = rng.standard_normal((300, 3))
    qn = fit_quantile_normalizer(coords, n_knots=51)
    clone = QuantileNormalizer.from_dict(qn.to_dict())
    assert clone.L == 3
    probe = rng.standard_normal((40, 3)) * 5.0
    assert np.allclose(clone.transform(probe), qn.transform(probe))
