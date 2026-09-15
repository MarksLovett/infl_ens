"""Tests for the fitted prompt-level router and its routing-eval columns."""

from __future__ import annotations

import numpy as np
import pytest

from infl_ens.evaluation.fitted_router import FittedRouter, fit_argmin_router
from infl_ens.evaluation.routing_eval import (
    FlatRoutingReport,
    FlatRoutingScores,
    compute_flat_scores,
    format_headline_markdown,
    report_to_dict,
)


def _separable_problem(seed: int = 0, m: int = 300):
    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(m, 3))
    # Expert k is best when coordinate k is the largest: a linearly separable
    # argmin structure the softmax router must recover.
    best = np.argmax(coords, axis=1)
    nll = np.full((m, 3), 2.0)
    nll[np.arange(m), best] = 1.0
    return coords, nll, best


def test_fitted_router_recovers_separable_assignment() -> None:
    coords, nll, best = _separable_problem()
    router = fit_argmin_router(coords, nll)
    assert isinstance(router, FittedRouter)
    assert router.n_experts == 3
    assert router.train_accuracy > 0.95
    proba = router.predict_proba(coords)
    assert proba.shape == (300, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert (router.predict(coords) == best).mean() > 0.95


def test_fitted_router_is_deterministic() -> None:
    coords, nll, _ = _separable_problem(seed=1)
    a = fit_argmin_router(coords, nll)
    b = fit_argmin_router(coords, nll)
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.bias, b.bias)
    assert a.n_iter == b.n_iter


def test_fitted_router_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        fit_argmin_router(np.zeros((4, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        fit_argmin_router(np.zeros((4, 2)), np.zeros((4, 1)))
    with pytest.raises(ValueError):
        fit_argmin_router(np.zeros((0, 2)), np.zeros((0, 2)))


def test_compute_flat_scores_adds_universal_and_fitted_columns() -> None:
    merge_nll = np.array([[1.0, 3.0], [4.0, 2.0], [2.0, 2.5]])
    pooled = np.array([2.0, 3.0, 2.0])
    g_merge = np.array([[0.75, 0.25, 0.5], [0.25, 0.75, 0.5]])
    p_merge = g_merge.copy()
    argmax = np.array([0, 1, 0])
    labels = ["a", "b", "a"]
    universal = pooled + 1e-4
    fitted_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])

    flat, per_bench = compute_flat_scores(
        merge_nll=merge_nll, pooled_nll=pooled, g_merge=g_merge, p_merge=p_merge,
        argmax_merge_idx=argmax, strategic_merge_idx=argmax, sampled_merge_idx=argmax,
        bench_labels=labels, round_idx=7,
        universal_nll=universal, fitted_proba=fitted_proba, fitted_val_accuracy=0.8,
    )
    assert flat.round_idx == 7 and flat.n_prompts == 3
    assert flat.oracle_nll == pytest.approx((1.0 + 2.0 + 2.0) / 3)
    assert flat.universal_nll == pytest.approx(universal.mean())
    # Fitted argmax picks experts [0, 1, 1] -> NLL [1, 2, 2.5].
    assert flat.fitted_argmax_nll == pytest.approx((1.0 + 2.0 + 2.5) / 3)
    expected = (fitted_proba * merge_nll).sum(axis=1).mean()
    assert flat.fitted_expected_nll == pytest.approx(expected)
    assert flat.fitted_val_accuracy == 0.8
    assert per_bench["a"]["universal_nll"] == pytest.approx(universal[[0, 2]].mean())
    assert per_bench["b"]["fitted_agreement_argmax"] == 1.0
    assert per_bench["a"]["fitted_agreement_argmax"] == 0.5

    # Without the optional inputs the new columns are absent / None.
    flat0, per0 = compute_flat_scores(
        merge_nll=merge_nll, pooled_nll=pooled, g_merge=g_merge, p_merge=p_merge,
        argmax_merge_idx=argmax, strategic_merge_idx=argmax, sampled_merge_idx=argmax,
        bench_labels=labels, round_idx=7,
    )
    assert flat0.universal_nll is None and flat0.fitted_argmax_nll is None
    assert "universal_nll" not in per0["a"] and "fitted_argmax_nll" not in per0["a"]


def _report(flat: FlatRoutingScores) -> FlatRoutingReport:
    return FlatRoutingReport(
        flat=flat,
        merge_names=["x", "y"],
        merge_name_map={"x": "x", "y": "y"},
    )


def test_report_and_markdown_carry_new_keys() -> None:
    flat = FlatRoutingScores(
        pooled_nll=2.0, learned_argmax_nll=1.9, learned_expected_nll=1.95,
        learned_sampled_nll=1.97, strategic_expected_nll=1.96, strategic_argmax_nll=1.9,
        oracle_nll=1.5, n_prompts=10, round_idx=3,
        universal_nll=2.0002, fitted_argmax_nll=1.7, fitted_expected_nll=1.75,
        fitted_val_accuracy=0.6,
    )
    d = report_to_dict(_report(flat))
    assert d["flat"]["universal_nll"] == pytest.approx(2.0002)
    assert d["flat"]["fitted_routing_expected_nll"] == 1.75
    assert d["flat"]["fitted_routing_argmax_nll"] == 1.7
    assert d["flat"]["fitted_router_val_accuracy"] == 0.6
    md = format_headline_markdown(_report(flat))
    assert "Fitted router (expected)" in md and "Universal only" in md

    plain = FlatRoutingScores(
        pooled_nll=2.0, learned_argmax_nll=1.9, learned_expected_nll=1.95,
        learned_sampled_nll=1.97, strategic_expected_nll=1.96, strategic_argmax_nll=1.9,
        oracle_nll=1.5, n_prompts=10, round_idx=3,
    )
    d0 = report_to_dict(_report(plain))
    assert d0["flat"]["universal_nll"] is None
    md0 = format_headline_markdown(_report(plain))
    assert "Fitted router" not in md0 and "Universal only" not in md0
