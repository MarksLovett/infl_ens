"""Diagnose cross-benchmark confusability and the fit/project field mismatch.

A single axis can be perfectly separable (high AUC against its own label)
yet still fail to give the router a *distinct* direction if it collides
with another axis — i.e. if a prompt's coordinate on axis A is largely
predictable from its coordinate on axis B. When that happens the
benchmarks pile onto a shared sub-manifold and the closed-loop router
sees fewer effective dimensions than axes, so specialists cannot be
delineated even though each axis is individually well-calibrated.

This script answers two questions the per-axis separability diagnostic
(:mod:`scripts.diagnose_axis_separability`) cannot:

1. **4-way confusability.** Train a simple linear probe to predict *which
   benchmark* a prompt came from, using only the stacked trait
   coordinates. High balanced accuracy means the axes carry
   benchmark-distinct structure; chance-level accuracy on some classes
   (e.g. jailbreak vs. harm) localises *which* axes collapse together.
   The full confusion matrix shows the specific collisions.

2. **The fit/project field mismatch.** In
   :func:`infl_ens.data.benchmarks.safety_trait_space.build_safety_trait_space`
   each axis direction is fit on **response** embeddings (when responses
   exist) but the calibration corpus is projected on **prompt**
   embeddings. For a *prompt-level* property — jailbreak / adversarial
   framing lives in the user turn, not the answer — fitting on responses
   learns the wrong direction. This script re-fits each axis under a
   selectable field (``response`` / ``prompt`` / ``concat``) and reports
   how confusability and per-axis AUC change, so the field choice can be
   made per axis with evidence.

It also evaluates an optional **joint whitening** step: stacking the raw
axis directions and whitening them against the pooled calibration-corpus
covariance (ZCA) removes the common prompt-embedding mode that every axis
otherwise rides on, leaving each axis's *unique* contribution. The script
reports confusability before and after whitening.

Everything is computed locally and read-only with respect to the package
so the field/whitening choices can be measured before porting any change
into ``safety_trait_space.py``. ``--toy`` runs fully offline.

Example
-------

.. code-block:: console

   # Offline smoke test.
   $ python scripts/diagnose_axis_confusability.py --toy --whiten

   # Real run on doob over all four benchmarks, fitting each axis on the
   # field it actually scores, with a confusion-matrix figure.
   $ python scripts/diagnose_axis_confusability.py \\
         --config configs/benchmark/router/safety_truth.yaml \\
         --field-map harm=response,hallucination=response,jailbreak=prompt,privacy=response \\
         --whiten \\
         --output-stem scripts/figures/axis_confusability
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# Axis fitting (mirrors the package estimator, with a field control)
# -----------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    """Return ``v`` scaled to unit norm (no-op on near-zero input).

    :param v: Input vector.
    :type v: numpy.ndarray
    :returns: Unit-norm vector, or ``v`` unchanged if its norm underflows.
    :rtype: numpy.ndarray
    """
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _lda_direction(
    pos: np.ndarray, neg: np.ndarray, *, shrinkage: float = 0.1,
) -> np.ndarray:
    """Shrinkage Fisher direction :math:`\\Sigma_w^{-1}(\\mu_+-\\mu_-)`.

    :param pos: Positive-class embeddings, shape ``(P, D)``.
    :type pos: numpy.ndarray
    :param neg: Negative-class embeddings, shape ``(N, D)``.
    :type neg: numpy.ndarray
    :param shrinkage: Shrinkage intensity in ``[0, 1]``.
    :type shrinkage: float
    :returns: Unit-norm axis direction, shape ``(D,)``.
    :rtype: numpy.ndarray
    """
    mu_pos, mu_neg = pos.mean(axis=0), neg.mean(axis=0)
    d = pos.shape[1]
    cov_pos = np.cov(pos, rowvar=False) if len(pos) > 1 else np.zeros((d, d))
    cov_neg = np.cov(neg, rowvar=False) if len(neg) > 1 else np.zeros((d, d))
    sw = ((len(pos) - 1) * cov_pos + (len(neg) - 1) * cov_neg) / max(
        len(pos) + len(neg) - 2, 1
    )
    tau = float(np.trace(sw) / d) if d else 0.0
    sw_reg = (1.0 - shrinkage) * sw + shrinkage * tau * np.eye(d)
    try:
        direction = np.linalg.solve(sw_reg, mu_pos - mu_neg)
    except np.linalg.LinAlgError:
        direction = mu_pos - mu_neg
    return _unit(direction)


_VALID_FIELDS = ("response", "prompt", "concat")


def _fit_field_embeddings(
    split,
    encoder: Callable[[Sequence[str]], np.ndarray],
    field: str,
) -> np.ndarray:
    """Embed the chosen text field of a split for axis fitting.

    :param split: A :class:`BenchmarkSplit`.
    :type split: infl_ens.data.benchmarks.base.BenchmarkSplit
    :param encoder: Sentence encoder.
    :type encoder: Callable[[Sequence[str]], numpy.ndarray]
    :param field: One of ``'response'`` (answer text; falls back to prompt
        if the split has no responses), ``'prompt'`` (user turn), or
        ``'concat'`` (prompt + ``' '`` + response).
    :type field: str
    :returns: Embedding matrix aligned with ``split.scores``, shape
        ``(N, D)``.
    :rtype: numpy.ndarray
    :raises ValueError: If ``field`` is not recognised.
    """
    if field not in _VALID_FIELDS:
        raise ValueError(f"field must be one of {_VALID_FIELDS}, got {field!r}")
    if field == "prompt" or not split.responses:
        texts = list(split.prompts)
    elif field == "response":
        texts = list(split.responses)
    else:  # concat
        texts = [f"{p} {r}" for p, r in zip(split.prompts, split.responses)]
    return np.asarray(encoder(texts), dtype=float)


@dataclass(frozen=True)
class FittedAxis:
    """A fitted axis direction plus its self-separability AUC.

    :param name: Axis label.
    :type name: str
    :param field: Field the direction was fit on.
    :type field: str
    :param direction: Unit-norm direction in embedding space, shape ``(D,)``.
    :type direction: numpy.ndarray
    :param auc: ROC AUC of the raw projection vs. the axis's own label.
    :type auc: float
    """

    name: str
    field: str
    direction: np.ndarray
    auc: float


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the rank-sum identity (ties averaged).

    :param scores: 1-D score per example.
    :type scores: numpy.ndarray
    :param labels: Binary labels (``1`` / ``0``).
    :type labels: numpy.ndarray
    :returns: AUC in ``[0, 1]``; ``0.5`` is chance.
    :rtype: float
    """
    pos_n = int((labels == 1).sum())
    neg_n = int((labels == 0).sum())
    if pos_n == 0 or neg_n == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s_sorted = scores[order]
    i, n = 0, len(s_sorted)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - pos_n * (pos_n + 1) / 2.0) / (pos_n * neg_n))


# -----------------------------------------------------------------------------
# Linear confusability probe (multinomial logistic regression, NumPy-only)
# -----------------------------------------------------------------------------

def _softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise numerically-stable softmax.

    :param z: Logit matrix, shape ``(N, K)``.
    :type z: numpy.ndarray
    :returns: Probability matrix, shape ``(N, K)``.
    :rtype: numpy.ndarray
    """
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit_logreg(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    l2: float = 1e-3,
    lr: float = 0.5,
    n_steps: int = 800,
    seed: int = 0,
) -> np.ndarray:
    """Fit an L2-regularised multinomial logistic regression by GD.

    Kept dependency-free (no scikit-learn) so the diagnostic runs in the
    same minimal environment as the rest of ``scripts/``.

    :param x: Feature matrix with a bias column already appended, shape
        ``(N, F)``.
    :type x: numpy.ndarray
    :param y: Integer class labels in ``[0, n_classes)``, shape ``(N,)``.
    :type y: numpy.ndarray
    :param n_classes: Number of classes ``K``.
    :type n_classes: int
    :param l2: L2 penalty strength.
    :type l2: float
    :param lr: Learning rate.
    :type lr: float
    :param n_steps: Gradient-descent steps.
    :type n_steps: int
    :param seed: RNG seed for weight init.
    :type seed: int
    :returns: Weight matrix, shape ``(F, K)``.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    n, f = x.shape
    w = 0.01 * rng.standard_normal((f, n_classes))
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), y] = 1.0
    for _ in range(n_steps):
        probs = _softmax(x @ w)
        grad = x.T @ (probs - onehot) / n + l2 * w
        w -= lr * grad
    return w


def _balanced_accuracy_and_confusion(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    *,
    train_frac: float = 0.7,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Train/test a linear probe; return balanced accuracy + confusion.

    :param features: Trait coordinates, shape ``(N, L)``.
    :type features: numpy.ndarray
    :param labels: Benchmark-of-origin labels in ``[0, n_classes)``.
    :type labels: numpy.ndarray
    :param n_classes: Number of benchmarks ``K``.
    :type n_classes: int
    :param train_frac: Fraction held out for training.
    :type train_frac: float
    :param seed: RNG seed for the split and probe init.
    :type seed: int
    :returns: ``(balanced_accuracy, confusion)`` where ``confusion`` is a
        row-normalised ``(K, K)`` matrix (row = true class, col =
        predicted).
    :rtype: tuple[float, numpy.ndarray]
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    perm = rng.permutation(n)
    cut = int(train_frac * n)
    tr, te = perm[:cut], perm[cut:]

    # Standardise features on the training split.
    mu = features[tr].mean(axis=0)
    sd = features[tr].std(axis=0) + 1e-9
    xf = (features - mu) / sd
    xf = np.concatenate([xf, np.ones((n, 1))], axis=1)

    w = _fit_logreg(xf[tr], labels[tr], n_classes, seed=seed)
    pred = np.argmax(xf[te] @ w, axis=1)

    conf = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(labels[te], pred):
        conf[t, p] += 1.0
    per_class_recall = []
    for k in range(n_classes):
        row = conf[k].sum()
        if row > 0:
            per_class_recall.append(conf[k, k] / row)
        conf[k] = conf[k] / row if row > 0 else conf[k]
    bal_acc = float(np.mean(per_class_recall)) if per_class_recall else float("nan")
    return bal_acc, conf


# -----------------------------------------------------------------------------
# Joint whitening (ZCA on stacked axis directions via the corpus covariance)
# -----------------------------------------------------------------------------

def _project_axes(
    directions: np.ndarray, emb: np.ndarray,
) -> np.ndarray:
    """Project embeddings onto stacked axis directions.

    :param directions: Stacked unit directions, shape ``(L, D)``.
    :type directions: numpy.ndarray
    :param emb: Embeddings, shape ``(N, D)``.
    :type emb: numpy.ndarray
    :returns: Raw per-axis projections, shape ``(N, L)``.
    :rtype: numpy.ndarray
    """
    return emb @ directions.T


def _whiten_coords(coords: np.ndarray) -> np.ndarray:
    """ZCA-whiten the per-axis coordinates to unit covariance.

    Removes the shared mode that all axes ride on (the common
    prompt-embedding direction), so the whitened coordinates isolate each
    axis's unique contribution. Centering + whitening only; not rescaled
    to ``[0, 1]`` (this is a diagnostic of separability, not a calibrated
    coordinate).

    :param coords: Raw projected coordinates, shape ``(N, L)``.
    :type coords: numpy.ndarray
    :returns: Whitened coordinates, shape ``(N, L)``.
    :rtype: numpy.ndarray
    """
    c = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(c, rowvar=False)
    cov = np.atleast_2d(cov)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-8, None)
    whiten = vecs @ np.diag(vals**-0.5) @ vecs.T
    return c @ whiten


# -----------------------------------------------------------------------------
# Subspace-residual test: does a target axis live inside the span of others?
# -----------------------------------------------------------------------------

def _subspace_residual(
    target: np.ndarray, basis: np.ndarray,
) -> tuple[float, float]:
    """Decompose ``target`` into in-span and orthogonal parts wrt ``basis``.

    Projects the (unit-norm) ``target`` direction onto the subspace spanned
    by the rows of ``basis`` (the other axes' directions) via an
    orthonormal basis from the reduced QR factorisation, and returns how
    much of ``target`` lies *outside* that span.

    A residual norm near ``0`` means the target axis is (almost) a linear
    combination of the others — it carries no independent direction, so no
    amount of calibration or whitening can give the router a separate
    dimension for it. A residual near ``1`` means the axis is essentially
    orthogonal to the others and the collision is *not* a span problem.

    :param target: Unit-norm target axis direction, shape ``(D,)``.
    :type target: numpy.ndarray
    :param basis: Stacked directions of the *other* axes, shape ``(M, D)``.
    :type basis: numpy.ndarray
    :returns: ``(residual_norm, cos_to_span)`` where ``residual_norm`` is
        the L2 norm of the component of ``target`` orthogonal to the span
        (in ``[0, 1]``), and ``cos_to_span`` is the cosine between
        ``target`` and its in-span projection (``1`` = fully in span).
    :rtype: tuple[float, float]
    """
    if basis.shape[0] == 0:
        return 1.0, 0.0
    q, _ = np.linalg.qr(basis.T)  # columns: orthonormal basis of the span
    in_span = q @ (q.T @ target)
    resid = target - in_span
    resid_norm = float(np.linalg.norm(resid))
    in_norm = float(np.linalg.norm(in_span))
    cos = in_norm / max(float(np.linalg.norm(target)), 1e-12)
    return resid_norm, cos


# -----------------------------------------------------------------------------
# Nonlinear probe: RBF random-feature multinomial logistic regression
# -----------------------------------------------------------------------------

def _rbf_features(
    x: np.ndarray, n_features: int, gamma: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random Fourier features approximating an RBF kernel map.

    Maps ``x`` (shape ``(N, L)``) to ``cos(xW + b)`` features so a *linear*
    probe on the features behaves like a *nonlinear* probe on ``x``. The
    drawn ``W`` / ``b`` are returned so the same map can be applied to a
    held-out split.

    :param x: Input coordinates, shape ``(N, L)``.
    :type x: numpy.ndarray
    :param n_features: Number of random features ``F``.
    :type n_features: int
    :param gamma: RBF bandwidth (larger = more local / more nonlinear).
    :type gamma: float
    :param seed: RNG seed for the random projection.
    :type seed: int
    :returns: ``(features, W, b)`` with ``features`` shape ``(N, F)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    w = rng.standard_normal((d, n_features)) * np.sqrt(2.0 * gamma)
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    feats = np.sqrt(2.0 / n_features) * np.cos(x @ w + b)
    return feats, w, b


def _nonlinear_balanced_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    *,
    n_rff: int = 512,
    gamma: float = 1.0,
    train_frac: float = 0.7,
    seed: int = 0,
) -> float:
    """Balanced accuracy of an RBF-random-feature probe on origin recovery.

    Same train/test protocol as the linear probe, but on RBF random
    features so the decision boundary is nonlinear in the original trait
    coordinates. If this substantially exceeds the linear balanced
    accuracy, the benchmark-distinguishing signal *exists* but is nonlinear
    — which a stronger encoder (or a nonlinear axis) could expose. If it
    does not, the axes are inseparable even nonlinearly at this
    representation.

    :param features: Trait coordinates, shape ``(N, L)``.
    :type features: numpy.ndarray
    :param labels: Benchmark-of-origin labels in ``[0, n_classes)``.
    :type labels: numpy.ndarray
    :param n_classes: Number of benchmarks ``K``.
    :type n_classes: int
    :param n_rff: Number of random Fourier features.
    :type n_rff: int
    :param gamma: RBF bandwidth.
    :type gamma: float
    :param train_frac: Training fraction.
    :type train_frac: float
    :param seed: RNG seed.
    :type seed: int
    :returns: Balanced accuracy in ``[0, 1]``.
    :rtype: float
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    perm = rng.permutation(n)
    cut = int(train_frac * n)
    tr, te = perm[:cut], perm[cut:]

    mu = features[tr].mean(axis=0)
    sd = features[tr].std(axis=0) + 1e-9
    xs = (features - mu) / sd

    feats, w, b = _rbf_features(xs, n_rff, gamma, seed)
    feats = np.concatenate([feats, np.ones((n, 1))], axis=1)

    weights = _fit_logreg(feats[tr], labels[tr], n_classes, seed=seed, n_steps=600)
    pred = np.argmax(feats[te] @ weights, axis=1)

    recalls = []
    for k in range(n_classes):
        mask = labels[te] == k
        if mask.sum() > 0:
            recalls.append(float((pred[mask] == k).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


# -----------------------------------------------------------------------------
# Score-content cross-leak: does benchmark k's score predict axis l's coord?
# -----------------------------------------------------------------------------

def _score_coord_leak_matrix(
    splits,
    directions: np.ndarray,
    encoder: Callable[[Sequence[str]], np.ndarray],
    threshold: float,
) -> np.ndarray:
    """Per-benchmark score vs. per-axis coordinate separability matrix.

    For benchmark ``k`` (rows) and axis ``l`` (columns) reports the AUC of
    axis ``l``'s *prompt* coordinate at separating benchmark ``k``'s own
    positive/negative labels. The diagonal ``(k, k)`` is the axis doing its
    intended job (should be high). A high *off-diagonal* ``(k, l)`` means
    benchmark ``k``'s label is predictable from a *different* axis — i.e.
    the benchmark's scoring is entangled with another axis's content (a
    label/content cross-leak, not a geometry bug).

    :param splits: Benchmark splits, length ``L`` (same order as
        ``directions``).
    :type splits: Sequence[infl_ens.data.benchmarks.base.BenchmarkSplit]
    :param directions: Stacked unit axis directions, shape ``(L, D)``.
    :type directions: numpy.ndarray
    :param encoder: Sentence encoder.
    :type encoder: Callable[[Sequence[str]], numpy.ndarray]
    :param threshold: Positive/negative split threshold.
    :type threshold: float
    :returns: AUC matrix, shape ``(L, L)`` (row = benchmark, col = axis).
    :rtype: numpy.ndarray
    """
    n_axes = len(splits)
    leak = np.full((n_axes, n_axes), np.nan)
    for k, split in enumerate(splits):
        labels = (split.scores >= threshold).astype(int)
        if labels.min() == labels.max():
            continue  # single-class; AUC undefined
        prompt_emb = np.asarray(encoder(list(split.prompts)), dtype=float)
        coords = _project_axes(directions, prompt_emb)  # (N, L)
        for l in range(n_axes):
            leak[k, l] = _auc(coords[:, l], labels)
    return leak


# -----------------------------------------------------------------------------
# Prompt-cloud overlap: do two benchmarks' prompts co-locate in coord space?
# -----------------------------------------------------------------------------

def _prompt_overlap_matrices(
    coords: np.ndarray,
    origins: np.ndarray,
    n_axes: int,
    *,
    k_nn: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour origin-mixing and centroid-distance matrices.

    Two complementary views of whether benchmarks co-locate in the routing
    coordinate space (independent of any axis's *direction*):

    - **NN-mixing** ``(k, l)``: of the ``k_nn`` nearest neighbours (in
      coordinate space) of benchmark ``k``'s prompts, the fraction whose
      origin is benchmark ``l``. A high off-diagonal ``(k, l)`` means
      ``k``'s prompts sit *inside* ``l``'s cloud — direct evidence of
      content co-location (e.g. ToxicChat prompts among HaluEval prompts).
      Rows sum to ~1.
    - **Centroid distance** ``(k, l)``: Euclidean distance between the
      coordinate centroids of benchmarks ``k`` and ``l``, normalised by the
      mean within-benchmark RMS spread. Values << 1 mean the clouds are
      closer together than they are internally wide — i.e. not separated.

    :param coords: Stacked routing coordinates, shape ``(N, L)``.
    :type coords: numpy.ndarray
    :param origins: Benchmark-of-origin labels, shape ``(N,)``.
    :type origins: numpy.ndarray
    :param n_axes: Number of benchmarks ``L``.
    :type n_axes: int
    :param k_nn: Neighbours per query point for NN-mixing.
    :type k_nn: int
    :param seed: RNG seed (subsampling for the NN computation if large).
    :type seed: int
    :returns: ``(nn_mixing, centroid_dist)`` each shape ``(L, L)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    rng = np.random.default_rng(seed)
    n = len(origins)
    # Subsample for the O(n^2) NN step if the corpus is large.
    cap = 4000
    if n > cap:
        sel = rng.choice(n, size=cap, replace=False)
        c = coords[sel]
        o = origins[sel]
    else:
        c, o = coords, origins

    # NN-mixing.
    d2 = _pairwise_sq_dists(c)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k_nn]  # (m, k_nn)
    nn_origin = o[nn]  # (m, k_nn)
    nn_mixing = np.zeros((n_axes, n_axes))
    for k in range(n_axes):
        rows = nn_origin[o == k]
        if len(rows) == 0:
            continue
        for l in range(n_axes):
            nn_mixing[k, l] = float((rows == l).mean())

    # Centroid distances, normalised by within-benchmark RMS spread.
    centroids = np.stack(
        [coords[origins == k].mean(axis=0) for k in range(n_axes)], axis=0
    )
    spreads = []
    for k in range(n_axes):
        blk = coords[origins == k]
        spreads.append(
            float(np.sqrt(((blk - blk.mean(axis=0)) ** 2).sum(axis=1).mean()))
        )
    mean_spread = float(np.mean(spreads)) + 1e-12
    centroid_dist = np.zeros((n_axes, n_axes))
    for k in range(n_axes):
        for l in range(n_axes):
            centroid_dist[k, l] = (
                float(np.linalg.norm(centroids[k] - centroids[l])) / mean_spread
            )
    return nn_mixing, centroid_dist


def _pairwise_sq_dists(x: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix.

    :param x: Points, shape ``(m, d)``.
    :type x: numpy.ndarray
    :returns: ``(m, m)`` squared-distance matrix.
    :rtype: numpy.ndarray
    """
    sq = (x * x).sum(axis=1)
    return np.maximum(sq[:, None] + sq[None, :] - 2.0 * (x @ x.T), 0.0)


# -----------------------------------------------------------------------------
# Niche distinctness: collapse each benchmark's overlap row to one number
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class NicheSummary:
    """Per-benchmark niche-distinctness summary.

    :param name: Benchmark / axis label.
    :type name: str
    :param distinctness: Self NN-mixing minus the mean off-diagonal mixing,
        normalised so ``1`` is a perfectly isolated cloud and ``0`` means
        the benchmark's prompts are as likely to neighbour another
        benchmark as their own (no niche).
    :type distinctness: float
    :param self_mix: Diagonal NN-mixing (fraction of own neighbours that
        share origin).
    :type self_mix: float
    :param top_partner: Name of the benchmark with the largest off-diagonal
        mixing into this row.
    :type top_partner: str
    :param top_partner_mix: That largest off-diagonal mixing value.
    :type top_partner_mix: float
    :param spread_ratio: Top-partner mixing divided by the mean of the
        *other* off-diagonal mixings. Large (>~2) ⇒ selectively co-located
        with one benchmark; near ``1`` ⇒ diffuse across all others.
    :type spread_ratio: float
    """

    name: str
    distinctness: float
    self_mix: float
    top_partner: str
    top_partner_mix: float
    spread_ratio: float


def _niche_summary(
    nn_mixing: np.ndarray, names: list[str],
) -> list[NicheSummary]:
    """Collapse an NN-mixing matrix into per-benchmark niche summaries.

    Answers, per benchmark, two questions in one row each:

    1. *Does it have a niche at all?* — ``distinctness`` (high ⇒ its prompts
       cluster with their own kind; low ⇒ no distinct region, the
       benchmark is spread across the others).
    2. *If it collides, with whom and how?* — ``top_partner`` /
       ``spread_ratio`` separate **selective** co-location (one dominant
       partner; ``spread_ratio`` large) from **diffuse** co-location
       (overlaps everything roughly equally; ``spread_ratio`` near ``1``).
       This is the diffuse-vs-selective read that decides whether the fix
       is "separate two clouds" or "this benchmark has no niche".

    :param nn_mixing: Row-normalised NN-mixing matrix, shape ``(L, L)``.
    :type nn_mixing: numpy.ndarray
    :param names: Benchmark labels of length ``L``.
    :type names: list[str]
    :returns: One :class:`NicheSummary` per benchmark.
    :rtype: list[NicheSummary]
    """
    n = len(names)
    out: list[NicheSummary] = []
    for k in range(n):
        row = nn_mixing[k]
        self_mix = float(row[k])
        off = np.array([row[l] for l in range(n) if l != k], dtype=float)
        off_names = [names[l] for l in range(n) if l != k]
        mean_off = float(off.mean()) if off.size else 0.0
        # Distinctness: how much more do own neighbours dominate vs the
        # average other benchmark. Normalise by (1 - 1/n) so a one-hot self
        # row maps to 1.0 and a uniform row maps to 0.0.
        denom = max(1.0 - 1.0 / n, 1e-12)
        distinctness = (self_mix - mean_off) / denom
        if off.size:
            top_idx = int(np.argmax(off))
            top_partner = off_names[top_idx]
            top_val = float(off[top_idx])
            rest = np.delete(off, top_idx)
            rest_mean = float(rest.mean()) if rest.size else top_val
            spread_ratio = top_val / max(rest_mean, 1e-9)
        else:
            top_partner, top_val, spread_ratio = "-", 0.0, 1.0
        out.append(
            NicheSummary(
                name=names[k], distinctness=distinctness, self_mix=self_mix,
                top_partner=top_partner, top_partner_mix=top_val,
                spread_ratio=spread_ratio,
            )
        )
    return out


# -----------------------------------------------------------------------------
# Toy fixtures
# -----------------------------------------------------------------------------

def _toy_inputs(seed: int = 0):
    """Build four toy splits where two axes deliberately collide.

    Harm and jailbreak share most of their direction (the collision under
    test); hallucination and privacy are near-orthogonal. The encoder
    injects the signal into *prompt* text for jailbreak and into
    *response* text for the others, so the field control has an effect.

    :param seed: RNG seed.
    :type seed: int
    :returns: ``(splits, encoder)``.
    :rtype: tuple[list, Callable[[Sequence[str]], numpy.ndarray]]
    """
    from infl_ens.data.benchmarks.base import BenchmarkSplit

    rng = np.random.default_rng(seed)
    dim = 64
    harm_dir = _unit(rng.standard_normal(dim))
    # Real finding: jailbreak axis is ~orthogonal to the others (high
    # resid_norm) but its PROMPTS co-locate with hallucination's. So make
    # the direction independent here; the co-location is in the encoder's
    # origin offsets below, not in the axis geometry.
    jail_dir = _unit(rng.standard_normal(dim))
    hall_dir = _unit(rng.standard_normal(dim))
    priv_dir = _unit(rng.standard_normal(dim))

    specs = [
        ("beavertails", "harm", harm_dir, "response"),
        ("halueval", "hallucination", hall_dir, "response"),
        ("toxicchat", "jailbreak", jail_dir, "prompt"),
        ("ai4privacy", "privacy", priv_dir, "response"),
    ]
    dirs = {axis: d for _, axis, d, _ in specs}
    signal_field = {name: f for name, _, _, f in specs}

    splits = []
    for name, axis, _, _ in specs:
        n = 250
        labels = rng.integers(0, 2, size=n)
        prompts = [f"{name}|P|{i}|{int(l)}" for i, l in enumerate(labels)]
        responses = [f"{name}|R|{i}|{int(l)}" for i, l in enumerate(labels)]
        splits.append(
            BenchmarkSplit(
                name=name, prompts=prompts, scores=labels.astype(float),
                axis_name=axis, responses=responses,
            )
        )

    def encoder(texts: Sequence[str]) -> np.ndarray:
        out = np.empty((len(texts), dim), dtype=float)
        # Per-benchmark origin offsets; toxicchat is placed NEAR halueval so
        # the two clouds co-locate (mirrors the real ToxicChat/HaluEval
        # finding: orthogonal axis, overlapping prompts).
        origin_dirs = {
            name: _unit(
                np.random.default_rng(abs(hash(name)) % (2**32)).standard_normal(dim)
            )
            for name in ("beavertails", "halueval", "toxicchat", "ai4privacy")
        }
        origin_dirs["toxicchat"] = _unit(
            0.85 * origin_dirs["halueval"] + 0.15 * origin_dirs["toxicchat"]
        )
        axis_of = {
            "beavertails": "harm", "halueval": "hallucination",
            "toxicchat": "jailbreak", "ai4privacy": "privacy",
        }
        for i, t in enumerate(texts):
            parts = t.split("|")
            name = parts[0]
            field = "prompt" if parts[1] == "P" else "response"
            label = float(parts[3])
            h = abs(hash(t)) % (2**32)
            local = 0.35 * np.random.default_rng(h).standard_normal(dim)
            origin = 1.4 * origin_dirs[name]
            sig = 0.0
            if field == signal_field[name]:
                sig = 0.7 * (label - 0.5)
            out[i] = local + origin + sig * dirs[axis_of[name]]
        return out

    return splits, encoder


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _plot_confusion(
    conf_before: np.ndarray,
    conf_after: Optional[np.ndarray],
    class_names: list[str],
    output_stem: str,
) -> None:
    """Render the row-normalised confusion matrix (before / after whitening).

    :param conf_before: Confusion before whitening, shape ``(K, K)``.
    :type conf_before: numpy.ndarray
    :param conf_after: Confusion after whitening, or ``None``.
    :type conf_after: numpy.ndarray | None
    :param class_names: Benchmark/axis labels of length ``K``.
    :type class_names: list[str]
    :param output_stem: Output path stem (no extension).
    :type output_stem: str
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mats = [("raw coords", conf_before)]
    if conf_after is not None:
        mats.append(("whitened", conf_after))

    fig, axes = plt.subplots(1, len(mats), figsize=(5 * len(mats), 4.5))
    for ax, (title, m) in zip(np.atleast_1d(axes), mats):
        im = ax.imshow(m, vmin=0.0, vmax=1.0, cmap="magma")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("predicted benchmark")
        ax.set_ylabel("true benchmark")
        ax.set_title(f"origin-recovery confusion ({title})")
        for r in range(len(class_names)):
            for c in range(len(class_names)):
                ax.text(c, r, f"{m[r, c]:.2f}", ha="center", va="center",
                        color="white" if m[r, c] < 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = Path(output_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Config loading (real run)
# -----------------------------------------------------------------------------

def _load_from_config(config: Path, encoder_override: Optional[str] = None):
    """Load the four benchmark splits and the sentence encoder.

    Falls back gracefully if ToxicChat / AI4Privacy loaders are not yet in
    the package: only the splits that import successfully are used.

    :param config: Router YAML path.
    :type config: pathlib.Path
    :param encoder_override: If set, use this sentence-transformer model id
        instead of the one named in the YAML (for encoder A/B tests).
    :type encoder_override: str | None
    :returns: ``(splits, encoder)``.
    :rtype: tuple[list, Callable[[Sequence[str]], numpy.ndarray]]
    """
    import yaml

    from infl_ens.data import SentenceTransformerEncoder
    from infl_ens.data.benchmarks import load_beavertails, load_halueval

    cfg = yaml.safe_load(config.read_text())
    ts = cfg.get("trait_space", {})
    model_name = encoder_override or ts.get(
        "encoder", "sentence-transformers/all-MiniLM-L6-v2"
    )
    encoder = SentenceTransformerEncoder(model_name)

    splits = [load_beavertails("data/beavertails/30k_train.jsonl"),
              load_halueval("data/halueval")]
    # Optional axes — import only if present.
    try:
        from infl_ens.data.benchmarks import load_toxicchat  # type: ignore
        splits.append(load_toxicchat("data/toxicchat"))
    except Exception:
        pass
    try:
        from infl_ens.data.benchmarks import load_ai4privacy  # type: ignore
        splits.append(load_ai4privacy("data/ai4privacy"))
    except Exception:
        pass
    return splits, encoder


def _parse_field_map(spec: Optional[str], axis_names: list[str]) -> dict[str, str]:
    """Parse a ``axis=field,...`` spec into a per-axis field map.

    :param spec: Comma-separated ``axis=field`` pairs, or ``None`` for the
        package default (``response`` everywhere).
    :type spec: str | None
    :param axis_names: Axis names present, for validation / defaulting.
    :type axis_names: list[str]
    :returns: Mapping ``{axis_name: field}`` covering all axes.
    :rtype: dict[str, str]
    :raises ValueError: On unknown axis or field.
    """
    out = {name: "response" for name in axis_names}
    if not spec:
        return out
    for pair in spec.split(","):
        if not pair.strip():
            continue
        k, _, v = pair.partition("=")
        k, v = k.strip(), v.strip()
        if k not in out:
            raise ValueError(f"unknown axis {k!r}; have {axis_names}")
        if v not in _VALID_FIELDS:
            raise ValueError(f"field for {k!r} must be in {_VALID_FIELDS}, got {v!r}")
        out[k] = v
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _print_matrix(
    mat: np.ndarray, row_names: list[str], col_names: list[str], title: str,
) -> None:
    """Print a labelled square matrix with a header row.

    :param mat: Matrix to print, shape ``(R, C)``.
    :type mat: numpy.ndarray
    :param row_names: Row labels, length ``R``.
    :type row_names: list[str]
    :param col_names: Column labels, length ``C``.
    :type col_names: list[str]
    :param title: Heading printed above the matrix.
    :type title: str
    """
    print(title)
    head = " " * 16 + "".join(f"{c[:10]:>12}" for c in col_names)
    print(head)
    for i, rn in enumerate(row_names):
        cells = "".join(
            ("    nan    " if np.isnan(mat[i, j]) else f"{mat[i, j]:>12.3f}")
            for j in range(mat.shape[1])
        )
        print(f"  {rn:<14}{cells}")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Optional argument vector; defaults to ``sys.argv[1:]``.
    :type argv: list[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Router YAML.")
    parser.add_argument("--toy", action="store_true",
                        help="Use the built-in 4-benchmark toy fixtures.")
    parser.add_argument(
        "--field-map", default=None,
        help="Per-axis fit field, e.g. 'harm=response,jailbreak=prompt'. "
             "Unlisted axes default to 'response'.",
    )
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Positive/negative split threshold.")
    parser.add_argument("--shrinkage", type=float, default=0.1,
                        help="LDA covariance shrinkage in [0, 1].")
    parser.add_argument("--whiten", action="store_true",
                        help="Also report confusability after joint ZCA whitening.")
    parser.add_argument(
        "--encoder", default=None,
        help="Override the YAML sentence-transformer model id (encoder A/B).",
    )
    parser.add_argument(
        "--nonlinear-probe", action="store_true",
        help="Also run an RBF random-feature probe to test whether origin "
             "is nonlinearly recoverable where the linear probe failed.",
    )
    parser.add_argument(
        "--co-location", action="store_true",
        help="Report the score->coord cross-leak matrix and the prompt-cloud "
             "overlap matrices (NN-mixing + normalised centroid distance).",
    )
    parser.add_argument("--k-nn", type=int, default=10,
                        help="Neighbours per point for the NN-mixing overlap.")
    parser.add_argument("--rff", type=int, default=512,
                        help="Number of random Fourier features (nonlinear probe).")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="RBF bandwidth for the nonlinear probe.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the probe split.")
    parser.add_argument("--output-stem", default=None,
                        help="If set, write <stem>.{pdf,png} confusion figure.")
    args = parser.parse_args(argv)

    if not args.toy and args.config is None:
        parser.error("pass --toy or --config")

    if args.toy:
        splits, encoder = _toy_inputs(seed=args.seed)
    else:
        splits, encoder = _load_from_config(args.config, encoder_override=args.encoder)

    axis_names = [s.axis_name for s in splits]
    field_map = _parse_field_map(args.field_map, axis_names)

    # Fit each axis on its chosen field; record self-AUC.
    fitted: list[FittedAxis] = []
    for split in splits:
        field = field_map[split.axis_name]
        emb = _fit_field_embeddings(split, encoder, field)
        labels = (split.scores >= args.threshold).astype(int)
        pos, neg = emb[labels == 1], emb[labels == 0]
        if len(pos) < 2 or len(neg) < 2:
            raise ValueError(
                f"axis {split.axis_name!r}: need >=2 per class on field {field!r}"
            )
        direction = _lda_direction(pos, neg, shrinkage=args.shrinkage)
        fitted.append(
            FittedAxis(
                name=split.axis_name, field=field, direction=direction,
                auc=_auc(emb @ direction, labels),
            )
        )

    print("Per-axis fit field and self-separability:")
    print(f"  {'axis':<16}{'fit_field':<12}{'self_AUC':>10}")
    print("  " + "-" * 38)
    for fa in fitted:
        print(f"  {fa.name:<16}{fa.field:<12}{fa.auc:>10.3f}")

    # Build the routing-time coordinate cloud: project the *prompt* of every
    # split's records onto the stacked directions (this is what the router
    # sees), and label by benchmark of origin.
    directions = np.stack([fa.direction for fa in fitted], axis=0)
    coord_blocks, origin_blocks = [], []
    for k, split in enumerate(splits):
        prompt_emb = np.asarray(encoder(list(split.prompts)), dtype=float)
        coord_blocks.append(_project_axes(directions, prompt_emb))
        origin_blocks.append(np.full(len(split.prompts), k, dtype=int))
    coords = np.concatenate(coord_blocks, axis=0)
    origins = np.concatenate(origin_blocks, axis=0)

    bal_raw, conf_raw = _balanced_accuracy_and_confusion(
        coords, origins, len(splits), seed=args.seed,
    )
    chance = 1.0 / len(splits)
    print()
    print("4-way benchmark-of-origin recoverability from trait coords:")
    print(f"  chance balanced accuracy : {chance:.3f}")
    print(f"  raw coords balanced acc  : {bal_raw:.3f}")

    conf_white = None
    if args.whiten:
        coords_w = _whiten_coords(coords)
        bal_white, conf_white = _balanced_accuracy_and_confusion(
            coords_w, origins, len(splits), seed=args.seed,
        )
        print(f"  whitened coords bal acc  : {bal_white:.3f}")

    # Per-class recall (diagonal of the row-normalised confusion) localises
    # which benchmarks collapse together.
    print()
    print("Per-benchmark recall (raw coords; low = collides with others):")
    for k, name in enumerate(axis_names):
        print(f"  {name:<16}{conf_raw[k, k]:>8.3f}")

    # Subspace-residual test: for each axis, how much of its fitted
    # direction lies *outside* the span of the others? Near-zero residual
    # means the axis is a linear combination of the rest — no independent
    # dimension exists for the router to exploit, so calibration/whitening
    # cannot help and the fix must change the representation, not the axis.
    print()
    print("Subspace residual of each axis vs the span of the others:")
    print(f"  {'axis':<16}{'resid_norm':>12}{'cos_to_span':>14}")
    print("  " + "-" * 42)
    for i, fa in enumerate(fitted):
        others = np.stack(
            [f.direction for j, f in enumerate(fitted) if j != i], axis=0
        )
        resid, cos = _subspace_residual(fa.direction, others)
        print(f"  {fa.name:<16}{resid:>12.3f}{cos:>14.3f}")
    print("  (resid_norm -> 0 means the axis adds no independent direction)")

    # Nonlinear probe: does origin become recoverable with a nonlinear
    # boundary? Compare against the linear raw-coords balanced accuracy.
    if args.nonlinear_probe:
        bal_nl = _nonlinear_balanced_accuracy(
            coords, origins, len(splits),
            n_rff=args.rff, gamma=args.gamma, seed=args.seed,
        )
        print()
        print("Nonlinear (RBF random-feature) origin recoverability:")
        print(f"  linear raw bal acc    : {bal_raw:.3f}")
        print(f"  nonlinear bal acc     : {bal_nl:.3f}")
        lift = bal_nl - bal_raw
        verdict = (
            "signal is NONLINEAR (stronger/nonlinear axis could expose it)"
            if lift > 0.05
            else "no nonlinear lift: axes inseparable at this representation"
        )
        print(f"  nonlinear lift        : {lift:+.3f}  -> {verdict}")

    # Co-location analyses: distinguish a label/scoring cross-leak from raw
    # prompt-content overlap. Both are read off matrices over benchmarks.
    if args.co_location:
        print()
        leak = _score_coord_leak_matrix(
            splits, directions, encoder, args.threshold,
        )
        _print_matrix(
            leak, axis_names, axis_names,
            "Score->coord cross-leak AUC (row=benchmark label, col=axis coord):",
        )
        print("  (high diagonal = axis works; high OFF-diagonal = scoring "
              "entangled with another axis)")

        print()
        nn_mix, cdist = _prompt_overlap_matrices(
            coords, origins, len(splits), k_nn=args.k_nn, seed=args.seed,
        )
        _print_matrix(
            nn_mix, axis_names, axis_names,
            f"Prompt NN-mixing (row's {args.k_nn}-NN origin fractions; "
            "rows ~sum to 1):",
        )
        print("  (high OFF-diagonal = row benchmark's prompts sit inside the "
              "column benchmark's cloud)")
        print()
        _print_matrix(
            cdist, axis_names, axis_names,
            "Centroid distance / mean within-benchmark spread (<<1 = clouds "
            "overlap):",
        )

        # Collapse each NN-mixing row to a one-line niche summary so
        # diffuse-vs-selective co-location is read directly, not eyeballed.
        print()
        niches = _niche_summary(nn_mix, axis_names)
        print("Niche distinctness (per benchmark):")
        print(f"  {'benchmark':<14}{'distinct':>10}{'self_mix':>10}"
              f"{'top_partner':>16}{'partner_mix':>13}{'spread':>9}  read")
        print("  " + "-" * 84)
        for ns in niches:
            if ns.distinctness < 0.15:
                read = "NO niche (diffuse across regions)"
            elif ns.spread_ratio >= 2.0:
                read = f"selective: collides w/ {ns.top_partner}"
            else:
                read = "distinct, mild diffuse overlap"
            print(
                f"  {ns.name:<14}{ns.distinctness:>10.3f}{ns.self_mix:>10.3f}"
                f"{ns.top_partner:>16}{ns.top_partner_mix:>13.3f}"
                f"{ns.spread_ratio:>9.2f}  {read}"
            )
        print("  (distinct->1 = isolated cloud; distinct~0 = no niche. "
              "spread>=2 => one dominant partner; spread~1 => diffuse.)")

    if args.output_stem is not None:
        _plot_confusion(conf_raw, conf_white, axis_names, args.output_stem)
        print(f"\nwrote {args.output_stem}.{{pdf,png}}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
