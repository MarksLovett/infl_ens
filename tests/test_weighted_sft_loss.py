"""Offline tests for soft (dense) routing primitives.

Covers the two pure pieces the soft-routing path relies on:

- :func:`infl_ens.training.sft_training.weighted_causal_lm_loss` — the
  per-example weighted cross-entropy that the ``WeightedSFTTrainer`` uses.
- :func:`infl_ens.inflgame.router.allocation.top_k_allocation_weights` —
  the top-``k`` sparsified, per-query renormalised allocation weights, and
  the ``top_k=1`` hard-assignment equivalence.

Neither needs ``trl`` (which lives only on the training box); the loss test
uses ``torch`` directly and is skipped if it is unavailable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from infl_ens.inflgame.router.allocation import (
    allocation_weights,
    top_k_allocation_weights,
)


def test_weighted_loss_scales_linearly_with_weight() -> None:
    """loss == mean_b(w_b * per-example CE); doubling weights doubles loss."""
    torch = pytest.importorskip("torch")
    from infl_ens.training.sft_training import weighted_causal_lm_loss

    # Uniform logits ⇒ per-token CE = log(V) for every supervised token, so
    # each example's mean CE is exactly log(V) regardless of its labels.
    b, t, v = 2, 4, 5
    logits = torch.zeros(b, t, v)
    labels = torch.zeros(b, t, dtype=torch.long)
    log_v = math.log(v)

    # Equal weights → plain mean of log(V).
    loss_eq = weighted_causal_lm_loss(
        logits, labels, torch.tensor([1.0, 1.0]),
    )
    assert loss_eq.item() == pytest.approx(log_v, rel=1e-5)

    # Asymmetric weights → weighted mean (2*logV + 1*logV)/2 = 1.5*logV.
    loss_asym = weighted_causal_lm_loss(
        logits, labels, torch.tensor([2.0, 1.0]),
    )
    assert loss_asym.item() == pytest.approx(1.5 * log_v, rel=1e-5)

    # Doubling every weight doubles the loss (per-example LR semantics).
    loss_2x = weighted_causal_lm_loss(
        logits, labels, torch.tensor([2.0, 2.0]),
    )
    assert loss_2x.item() == pytest.approx(2.0 * log_v, rel=1e-5)

    # A zero-weight example contributes nothing.
    loss_zero = weighted_causal_lm_loss(
        logits, labels, torch.tensor([0.0, 1.0]),
    )
    assert loss_zero.item() == pytest.approx(0.5 * log_v, rel=1e-5)


def test_weighted_loss_respects_ignore_index() -> None:
    """Masked (-100) label positions are excluded from the per-example mean."""
    torch = pytest.importorskip("torch")
    from infl_ens.training.sft_training import weighted_causal_lm_loss

    b, t, v = 1, 4, 3
    logits = torch.zeros(b, t, v)
    # Mask the first two (shifted) target positions; the mean over the
    # remaining supervised tokens is still log(V).
    labels = torch.zeros(b, t, dtype=torch.long)
    labels[0, 1] = -100
    loss = weighted_causal_lm_loss(logits, labels, torch.tensor([1.0]))
    assert loss.item() == pytest.approx(math.log(v), rel=1e-5)


def test_top_k_one_is_hard_argmax_assignment() -> None:
    """soft_top_k=1 keeps exactly the argmax agent per query at weight 1."""
    rng = np.random.default_rng(0)
    positions = rng.random((3, 2))
    coords = rng.random((6, 2))
    cov = 0.2 ** 2 * np.eye(2)
    G = allocation_weights(positions, coords, cov)  # (3, 6), columns sum 1

    W = top_k_allocation_weights(G, 1)
    # Exactly one non-zero per column, all equal to 1.0.
    assert np.all((W > 0).sum(axis=0) == 1)
    assert np.allclose(W.sum(axis=0), 1.0)
    # The kept agent is the argmax of G.
    assert np.array_equal(np.argmax(W, axis=0), np.argmax(G, axis=0))


def test_top_k_renormalises_and_full_k_is_identity() -> None:
    """Top-k columns sum to one; top_k>=N reproduces column-normalised G."""
    G = np.array([
        [0.6, 0.1, 0.2],
        [0.3, 0.7, 0.3],
        [0.1, 0.2, 0.5],
    ])
    W2 = top_k_allocation_weights(G, 2)
    assert np.all((W2 > 0).sum(axis=0) == 2)
    assert np.allclose(W2.sum(axis=0), 1.0)
    # The dropped (zeroed) agent per column is the smallest G entry.
    assert np.array_equal(np.argmax(W2 == 0.0, axis=0),
                          np.argmin(G, axis=0))

    W_full = top_k_allocation_weights(G, 3)
    assert np.allclose(W_full, G / G.sum(axis=0, keepdims=True))


def test_top_k_validates_arguments() -> None:
    """top_k < 1 and non-2D inputs raise ValueError."""
    G = np.ones((2, 3)) * 0.5
    with pytest.raises(ValueError):
        top_k_allocation_weights(G, 0)
    with pytest.raises(ValueError):
        top_k_allocation_weights(np.ones(3), 1)
