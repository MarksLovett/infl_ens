"""The sequence-level mixture NLL and its uniform control.

``learned_expected_nll`` is the expected NLL of *sampling one* expert. The
ensemble number is the mixture, and Jensen guarantees ``oracle <= mixture <=
expected``. That ordering is a theorem, so a violation on a real run is a bug,
not a result -- these tests pin it down on cases where the answer is known
independently of the implementation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from infl_ens.evaluation.routing_eval import mixture_nll


def naive_mixture(nll: np.ndarray, n: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Reference in probability space: no logsumexp, no vectorisation.

    Underflows for long sequences, which is exactly why the shipped version
    uses logsumexp -- but for short ones it is an independent check.
    """
    out = np.empty(len(nll))
    for i in range(len(nll)):
        total = math.fsum(
            w[i, p] * math.exp(-n[i] * nll[i, p]) for p in range(nll.shape[1])
        )
        out[i] = -math.log(total) / n[i]
    return out


def test_two_experts_matches_a_hand_computed_value():
    """One token, P = [1/2, 1/4], equal weights -> -log(3/8)."""
    nll = np.array([[math.log(2.0), math.log(4.0)]])
    got = mixture_nll(nll, np.array([1.0]), np.array([0.5, 0.5]))
    assert got[0] == pytest.approx(-math.log(0.375))

    expected = float((nll * 0.5).sum())
    oracle = float(nll.min())
    assert oracle < got[0] < expected


def test_matches_the_probability_space_reference():
    rng = np.random.default_rng(0)
    nll = rng.uniform(0.5, 3.0, size=(64, 5))
    n = rng.integers(1, 12, size=64).astype(float)
    w = rng.dirichlet(np.ones(5), size=64)
    np.testing.assert_allclose(
        mixture_nll(nll, n, w), naive_mixture(nll, n, w), rtol=1e-12, atol=1e-12
    )


def test_jensen_sandwich_holds_for_arbitrary_weights():
    """oracle <= mixture <= expected, the ordering every re-scored run must show."""
    rng = np.random.default_rng(1)
    nll = rng.uniform(0.1, 4.0, size=(500, 7))
    n = rng.integers(1, 200, size=500).astype(float)
    w = rng.dirichlet(np.ones(7), size=500)

    mix = mixture_nll(nll, n, w)
    expected = (w * nll).sum(axis=1)
    oracle = nll.min(axis=1)

    assert np.all(mix <= expected + 1e-12)
    assert np.all(mix >= oracle - 1e-12)


def test_a_one_hot_weight_recovers_that_expert():
    rng = np.random.default_rng(2)
    nll = rng.uniform(0.1, 4.0, size=(32, 4))
    n = rng.integers(1, 50, size=32).astype(float)
    w = np.zeros((32, 4))
    w[:, 2] = 1.0
    np.testing.assert_allclose(mixture_nll(nll, n, w), nll[:, 2], rtol=1e-12)


def test_zero_weights_are_allowed_and_finite():
    """Top-k truncation zeroes entries; log(0) must not leak a nan."""
    nll = np.array([[1.0, 2.0, 3.0]])
    w = np.array([[0.5, 0.5, 0.0]])
    got = mixture_nll(nll, np.array([10.0]), w)
    assert np.isfinite(got).all()
    # The zeroed expert is the worst, so dropping it changes nothing much.
    assert got[0] == pytest.approx(
        mixture_nll(nll[:, :2], np.array([10.0]), w[:, :2])[0]
    )


def test_informative_weights_beat_the_uniform_control():
    """Weight comonotone with quality => mixture no worse than uniform.

    This is the cheapest check on the router's central claim. It is Chebyshev's
    sum inequality, so it holds exactly when the router ranks experts correctly;
    on real runs the gap is the router's contribution.
    """
    rng = np.random.default_rng(3)
    nll = rng.uniform(0.1, 4.0, size=(200, 6))
    n = rng.integers(1, 80, size=200).astype(float)

    # Give the lowest-NLL expert the largest share, per example.
    order = np.argsort(np.argsort(nll, axis=1), axis=1)          # rank, 0 = best
    good = np.exp(-order.astype(float))
    good /= good.sum(axis=1, keepdims=True)
    uniform = np.full(6, 1.0 / 6)

    assert np.all(mixture_nll(nll, n, good) <= mixture_nll(nll, n, uniform) + 1e-12)


def test_long_sequences_collapse_the_mixture_onto_the_oracle():
    """The weights enter as -log(w_best)/n, so they vanish as n grows.

    Stated in the artifact as the reason a uniform control landing near the
    G-weighted mixture is an artifact of sequence length, not proof the router
    is useless.
    """
    nll = np.array([[1.0, 1.5]])
    w = np.array([0.5, 0.5])
    short = mixture_nll(nll, np.array([2.0]), w)[0]
    long = mixture_nll(nll, np.array([2000.0]), w)[0]
    oracle = nll.min()

    assert long - oracle < short - oracle
    assert long == pytest.approx(oracle + math.log(2.0) / 2000.0, abs=1e-6)


def test_survives_sequence_lengths_that_underflow_in_probability_space():
    """exp(-n*nll) is 0 in float64 here; logsumexp must still be exact."""
    nll = np.array([[2.0, 2.5]])
    got = mixture_nll(nll, np.array([5000.0]), np.array([0.5, 0.5]))
    assert np.isfinite(got).all()
    assert got[0] == pytest.approx(2.0 + math.log(2.0) / 5000.0, abs=1e-9)


def test_unnormalised_weights_shift_the_result_by_log_c_over_n():
    """The caller owns normalisation: scaling every weight by c is not a no-op."""
    rng = np.random.default_rng(4)
    nll = rng.uniform(0.1, 3.0, size=(16, 3))
    n = rng.integers(1, 30, size=16).astype(float)
    w = rng.dirichlet(np.ones(3), size=16)

    np.testing.assert_allclose(
        mixture_nll(nll, n, 4.0 * w),
        mixture_nll(nll, n, w) - math.log(4.0) / n,
        rtol=1e-12,
    )


def test_token_counts_are_floored_at_one():
    """A zero-token example would divide by zero; it is clipped, not dropped."""
    nll = np.array([[1.0, 2.0]])
    got = mixture_nll(nll, np.array([0.0]), np.array([0.5, 0.5]))
    assert np.isfinite(got).all()
    assert got[0] == pytest.approx(mixture_nll(nll, np.array([1.0]), np.array([0.5, 0.5]))[0])
