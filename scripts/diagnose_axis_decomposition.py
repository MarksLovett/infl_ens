"""Cross-check supervised trait axes against PCA / ICA decompositions.

The trait-space builder learns each axis *supervised*: a shrinkage Fisher
direction :math:`\\Sigma_w^{-1}(\\mu_+-\\mu_-)` chosen because it separates
a known benchmark label. PCA and ICA are *unsupervised*: they look only at
the embedding cloud and never see labels. Comparing them answers a
question the supervised diagnostics cannot — *does the structure a label
picks out also exist as natural structure in the embeddings?*

- **PCA** finds orthogonal directions of maximum **variance**. It bounds
  the intrinsic dimensionality of the prompt cloud and asks "is there even
  room for :math:`L` separable axes here?" A discriminative axis can be
  real yet carry little variance, so PCA *missing* an axis is evidence the
  axis is low-energy, not that it is wrong.
- **ICA** finds directions of maximum statistical **independence**
  (non-Gaussianity) via FastICA. If a benchmark corresponds to a genuine
  independent generative factor, ICA should surface it without labels. ICA
  failing to recover a jailbreak-like source corroborates a "no distinct
  factor" finding; ICA recovering one the supervised axis missed would
  instead implicate the supervised direction.

This script reports, all read-only with respect to the package:

1. **PCA spectrum**: per-component explained-variance ratio and the
   cumulative count of components needed to reach a variance target
   (intrinsic dimensionality).
2. **Axis ↔ component alignment**: for each supervised LDA axis, the
   maximum absolute cosine with any PCA component and any ICA component,
   plus the index of the best match. Low alignment with *every*
   unsupervised component means the axis lives on a direction the
   unsupervised methods do not isolate.
3. **Origin recoverability from components**: balanced accuracy of a
   linear probe predicting benchmark-of-origin from the top-``k`` PCA
   components and from the ICA components, vs. from the raw supervised
   coordinates — does unsupervised structure separate the benchmarks any
   better than the learned axes?
4. **Per-axis label recoverability from components**: AUC of the
   best-aligned PCA/ICA component at separating each axis's own label —
   does any single unsupervised component recover the jailbreak label?

``--toy`` runs fully offline against a fixture in which jailbreak is an
orthogonal but **low-variance** label direction whose prompts co-locate
with hallucination — the regime PCA is expected to miss.

Example
-------

.. code-block:: console

   # Offline smoke test.
   $ python scripts/diagnose_axis_decomposition.py --toy --n-ica 6

   # Real run on doob.
   $ python scripts/diagnose_axis_decomposition.py \\
         --config configs/benchmark/router/safety_truth.yaml \\
         --variance-target 0.9 --n-ica 8 \\
         --output-stem scripts/figures/axis_decomposition
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# Supervised axis (mirrors the package estimator) — for the alignment check
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


# -----------------------------------------------------------------------------
# PCA (NumPy SVD)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PCAResult:
    """Result of a centered PCA.

    :param components: Principal directions as rows, shape ``(K, D)``,
        unit-norm and ordered by descending variance.
    :type components: numpy.ndarray
    :param explained_variance_ratio: Fraction of total variance per
        component, shape ``(K,)``.
    :type explained_variance_ratio: numpy.ndarray
    :param mean: Column mean removed before decomposition, shape ``(D,)``.
    :type mean: numpy.ndarray
    """

    components: np.ndarray
    explained_variance_ratio: np.ndarray
    mean: np.ndarray


def _pca(x: np.ndarray, n_components: Optional[int] = None) -> PCAResult:
    """Centered PCA via the thin SVD.

    :param x: Data matrix, shape ``(N, D)``.
    :type x: numpy.ndarray
    :param n_components: Number of components to keep; ``None`` keeps
        ``min(N, D)``.
    :type n_components: int | None
    :returns: The fitted :class:`PCAResult`.
    :rtype: PCAResult
    """
    mean = x.mean(axis=0)
    xc = x - mean
    # Economy SVD: rows of Vt are principal directions.
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = (s**2) / max(len(x) - 1, 1)
    ratio = var / var.sum() if var.sum() > 0 else var
    k = vt.shape[0] if n_components is None else min(n_components, vt.shape[0])
    return PCAResult(
        components=vt[:k], explained_variance_ratio=ratio[:k], mean=mean,
    )


def _intrinsic_dim(ratio: np.ndarray, target: float) -> int:
    """Number of leading components needed to reach a variance target.

    :param ratio: Per-component explained-variance ratios (descending).
    :type ratio: numpy.ndarray
    :param target: Cumulative variance target in ``(0, 1]``.
    :type target: float
    :returns: Count of components whose cumulative ratio first meets
        ``target``.
    :rtype: int
    """
    cum = np.cumsum(ratio)
    idx = int(np.searchsorted(cum, target) + 1)
    return min(idx, len(ratio))


# -----------------------------------------------------------------------------
# FastICA (self-contained; deflationary, log-cosh contrast)
# -----------------------------------------------------------------------------

def _fastica(
    x: np.ndarray,
    n_components: int,
    *,
    max_iter: int = 300,
    tol: float = 1e-5,
    seed: int = 0,
) -> np.ndarray:
    """Deflationary FastICA with the log-cosh (``tanh``) contrast.

    Whitens ``x`` (PCA to ``n_components`` + unit variance) then extracts
    independent directions one at a time with Gram-Schmidt deflation.
    Returned components are expressed in the *original* embedding space
    (unwhitened) and unit-normalised, so they are directly comparable to
    the supervised LDA axes by cosine.

    Self-contained (no scikit-learn) to match the minimal ``scripts/``
    environment.

    :param x: Data matrix, shape ``(N, D)``.
    :type x: numpy.ndarray
    :param n_components: Number of independent components ``C``.
    :type n_components: int
    :param max_iter: Max fixed-point iterations per component.
    :type max_iter: int
    :param tol: Convergence tolerance on the direction update.
    :type tol: float
    :param seed: RNG seed for initialisation.
    :type seed: int
    :returns: Independent directions as rows in original space, shape
        ``(C, D)``, unit-norm.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    mean = x.mean(axis=0)
    xc = (x - mean).T  # (D, N)
    d, n = xc.shape
    c = min(n_components, d)

    # Whitening via eigendecomposition of the covariance.
    cov = (xc @ xc.T) / n
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:c]
    vals = np.clip(vals[order], 1e-10, None)
    vecs = vecs[:, order]
    whiten = np.diag(vals**-0.5) @ vecs.T  # (c, d)
    dewhiten = vecs @ np.diag(vals**0.5)   # (d, c)
    xw = whiten @ xc  # (c, N)

    w_mat = np.zeros((c, c))
    for i in range(c):
        w = rng.standard_normal(c)
        w = _unit(w)
        for _ in range(max_iter):
            ws = w @ xw  # (N,)
            g = np.tanh(ws)
            g_prime = 1.0 - g**2
            w_new = (xw * g).mean(axis=1) - g_prime.mean() * w
            # Deflation: orthogonalise against earlier components.
            for j in range(i):
                w_new -= (w_new @ w_mat[j]) * w_mat[j]
            w_new = _unit(w_new)
            if np.abs(np.abs(w_new @ w) - 1.0) < tol:
                w = w_new
                break
            w = w_new
        w_mat[i] = w

    # Map whitened-space directions back to original space, unit-normalise.
    comps = (w_mat @ whiten)  # (c, d) acting on centered original data
    comps = np.stack([_unit(row) for row in comps], axis=0)
    return comps


# -----------------------------------------------------------------------------
# Alignment + recoverability metrics
# -----------------------------------------------------------------------------

def _max_abs_cosine(
    axis: np.ndarray, components: np.ndarray,
) -> tuple[float, int]:
    """Best absolute cosine alignment of an axis with any component.

    :param axis: Unit-norm supervised axis, shape ``(D,)``.
    :type axis: numpy.ndarray
    :param components: Unit-norm component rows, shape ``(K, D)``.
    :type components: numpy.ndarray
    :returns: ``(max_abs_cosine, argmax_index)``.
    :rtype: tuple[float, int]
    """
    cos = np.abs(components @ axis)
    idx = int(np.argmax(cos))
    return float(cos[idx]), idx


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
    i, m = 0, len(s_sorted)
    while i < m:
        j = i
        while j + 1 < m and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - pos_n * (pos_n + 1) / 2.0) / (pos_n * neg_n))


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


def _balanced_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    *,
    l2: float = 1e-3,
    lr: float = 0.5,
    n_steps: int = 800,
    train_frac: float = 0.7,
    seed: int = 0,
) -> float:
    """Balanced accuracy of an L2 multinomial logistic probe (NumPy-only).

    :param features: Feature matrix, shape ``(N, F)``.
    :type features: numpy.ndarray
    :param labels: Integer class labels in ``[0, n_classes)``.
    :type labels: numpy.ndarray
    :param n_classes: Number of classes ``K``.
    :type n_classes: int
    :param l2: L2 penalty strength.
    :type l2: float
    :param lr: Learning rate.
    :type lr: float
    :param n_steps: Gradient-descent steps.
    :type n_steps: int
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
    xf = (features - mu) / sd
    xf = np.concatenate([xf, np.ones((n, 1))], axis=1)

    w = 0.01 * rng.standard_normal((xf.shape[1], n_classes))
    onehot = np.zeros((cut, n_classes))
    onehot[np.arange(cut), labels[tr]] = 1.0
    for _ in range(n_steps):
        probs = _softmax(xf[tr] @ w)
        grad = xf[tr].T @ (probs - onehot) / cut + l2 * w
        w -= lr * grad

    pred = np.argmax(xf[te] @ w, axis=1)
    recalls = []
    for k in range(n_classes):
        mask = labels[te] == k
        if mask.sum() > 0:
            recalls.append(float((pred[mask] == k).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


# -----------------------------------------------------------------------------
# ICA-derived candidate axes + per-class origin recall
# -----------------------------------------------------------------------------

def _best_ica_axis_by_auc(
    emb: np.ndarray, labels: np.ndarray, ica: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Pick the single ICA component most predictive of a benchmark label.

    Implements the "best single ICA component by label AUC" derivation: of
    all independent components, choose the one whose projection best
    separates the benchmark's own positive/negative labels (AUC is
    sign-symmetric, so the component is flipped to point positive-up).

    :param emb: Benchmark block embeddings, shape ``(N, D)``.
    :type emb: numpy.ndarray
    :param labels: Binary labels for this benchmark, shape ``(N,)``.
    :type labels: numpy.ndarray
    :param ica: Independent component rows, shape ``(C, D)``, unit-norm.
    :type ica: numpy.ndarray
    :returns: ``(axis, component_index, auc)`` with ``axis`` unit-norm and
        oriented so positives project high.
    :rtype: tuple[numpy.ndarray, int, float]
    """
    best_idx, best_auc, best_flip = 0, 0.0, 1.0
    ec = emb - emb.mean(axis=0)
    for j in range(ica.shape[0]):
        proj = ec @ ica[j]
        a = _auc(proj, labels)
        flip = 1.0 if a >= 0.5 else -1.0
        a = max(a, 1.0 - a)
        if a > best_auc:
            best_idx, best_auc, best_flip = j, a, flip
    return _unit(best_flip * ica[best_idx]), best_idx, best_auc


def _ica_subspace_lda(
    emb: np.ndarray,
    labels: np.ndarray,
    ica: np.ndarray,
    *,
    shrinkage: float = 0.1,
) -> np.ndarray:
    """Re-fit a Fisher axis *inside* the span of the ICA components.

    Projects embeddings onto the ICA basis, runs shrinkage-LDA in that
    reduced space, then maps the direction back to the original embedding
    space. This is the "fairer" derivation: rather than forcing the axis to
    be a single independent component, it lets LDA combine the independent
    components — testing whether the discriminative jailbreak direction
    lives in the ICA *subspace* even if no single component captures it.

    :param emb: Benchmark block embeddings, shape ``(N, D)``.
    :type emb: numpy.ndarray
    :param labels: Binary labels, shape ``(N,)``.
    :type labels: numpy.ndarray
    :param ica: Independent component rows, shape ``(C, D)``.
    :type ica: numpy.ndarray
    :param shrinkage: LDA shrinkage intensity.
    :type shrinkage: float
    :returns: Unit-norm axis in the original embedding space, shape
        ``(D,)``.
    :rtype: numpy.ndarray
    """
    ec = emb - emb.mean(axis=0)
    z = ec @ ica.T  # (N, C) coordinates in the ICA basis
    pos, neg = z[labels == 1], z[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        return _unit(ica[0])
    w_reduced = _lda_direction(pos, neg, shrinkage=shrinkage)  # (C,)
    direction = ica.T @ w_reduced  # back to (D,)
    return _unit(direction)


def _per_class_recall(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    *,
    train_frac: float = 0.7,
    seed: int = 0,
) -> np.ndarray:
    """Per-class recall of a multinomial logistic origin probe.

    Same probe as :func:`_balanced_accuracy` but returns the per-class
    recall vector instead of its mean, so a single benchmark's origin
    recoverability (e.g. jailbreak / toxicchat) can be tracked across
    different axis sets.

    :param features: Coordinate matrix, shape ``(N, L)``.
    :type features: numpy.ndarray
    :param labels: Benchmark-of-origin labels in ``[0, n_classes)``.
    :type labels: numpy.ndarray
    :param n_classes: Number of benchmarks ``K``.
    :type n_classes: int
    :param train_frac: Training fraction.
    :type train_frac: float
    :param seed: RNG seed.
    :type seed: int
    :returns: Per-class recall vector, shape ``(K,)``.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    perm = rng.permutation(n)
    cut = int(train_frac * n)
    tr, te = perm[:cut], perm[cut:]

    mu = features[tr].mean(axis=0)
    sd = features[tr].std(axis=0) + 1e-9
    xf = (features - mu) / sd
    xf = np.concatenate([xf, np.ones((n, 1))], axis=1)

    w = 0.01 * rng.standard_normal((xf.shape[1], n_classes))
    onehot = np.zeros((cut, n_classes))
    onehot[np.arange(cut), labels[tr]] = 1.0
    for _ in range(800):
        probs = _softmax(xf[tr] @ w)
        grad = xf[tr].T @ (probs - onehot) / cut + 1e-3 * w
        w -= 0.5 * grad

    pred = np.argmax(xf[te] @ w, axis=1)
    recall = np.full(n_classes, np.nan)
    for k in range(n_classes):
        mask = labels[te] == k
        if mask.sum() > 0:
            recall[k] = float((pred[mask] == k).mean())
    return recall


# -----------------------------------------------------------------------------
# Toy fixtures
# -----------------------------------------------------------------------------

def _toy_inputs(seed: int = 0):
    """Build four toy splits where jailbreak is a low-variance label axis.

    Harm, hallucination and privacy get large origin offsets (high-variance
    benchmark structure). Jailbreak's axis is orthogonal to the others but
    its *label signal is weak* and its prompts are placed near the
    hallucination cloud — so PCA, ranking by variance, should rank the
    jailbreak direction low and fail to isolate it, while the supervised
    LDA axis still separates the label. This is the regime that makes the
    PCA-vs-supervised distinction visible.

    :param seed: RNG seed.
    :type seed: int
    :returns: ``(splits, encoder)``.
    :rtype: tuple[list, Callable[[Sequence[str]], numpy.ndarray]]
    """
    from infl_ens.data.benchmarks.base import BenchmarkSplit

    rng = np.random.default_rng(seed)
    dim = 64
    harm_dir = _unit(rng.standard_normal(dim))
    hall_dir = _unit(rng.standard_normal(dim))
    priv_dir = _unit(rng.standard_normal(dim))
    jail_dir = _unit(rng.standard_normal(dim))
    axis_dir = {
        "harm": harm_dir, "hallucination": hall_dir,
        "privacy": priv_dir, "jailbreak": jail_dir,
    }
    axis_of = {
        "beavertails": "harm", "halueval": "hallucination",
        "ai4privacy": "privacy", "toxicchat": "jailbreak",
    }
    # Per-axis label signal strength: jailbreak deliberately weak (low var).
    sig_strength = {
        "harm": 1.1, "hallucination": 1.1, "privacy": 1.1, "jailbreak": 0.5,
    }
    origin_dirs = {
        name: _unit(
            np.random.default_rng(abs(hash(name)) % (2**32)).standard_normal(dim)
        )
        for name in axis_of
    }
    # toxicchat prompts co-locate with halueval.
    origin_dirs["toxicchat"] = _unit(
        0.85 * origin_dirs["halueval"] + 0.15 * origin_dirs["toxicchat"]
    )

    splits = []
    for name, axis in axis_of.items():
        n = 300
        labels = rng.integers(0, 2, size=n)
        prompts = [f"{name}|{i}|{int(l)}" for i, l in enumerate(labels)]
        splits.append(
            BenchmarkSplit(
                name=name, prompts=prompts, scores=labels.astype(float),
                axis_name=axis, responses=list(prompts),
            )
        )

    def encoder(texts: Sequence[str]) -> np.ndarray:
        out = np.empty((len(texts), dim), dtype=float)
        for i, t in enumerate(texts):
            name, _, lab = t.split("|")
            label = float(lab)
            h = abs(hash(t)) % (2**32)
            local = 0.4 * np.random.default_rng(h).standard_normal(dim)
            origin = 1.3 * origin_dirs[name]
            axis = axis_of[name]
            sig = sig_strength[axis] * (label - 0.5) * axis_dir[axis]
            out[i] = local + origin + sig
        return out

    return splits, encoder


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _plot_spectrum(
    ratio: np.ndarray,
    axis_pca_cos: list[tuple[str, float]],
    output_stem: str,
) -> None:
    """Plot the PCA scree curve and per-axis best PCA alignment.

    :param ratio: Explained-variance ratios (descending).
    :type ratio: numpy.ndarray
    :param axis_pca_cos: ``(axis_name, max_abs_cosine)`` per supervised axis.
    :type axis_pca_cos: list[tuple[str, float]]
    :param output_stem: Output path stem (no extension).
    :type output_stem: str
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    k = min(len(ratio), 30)
    ax1.plot(np.arange(1, k + 1), ratio[:k], marker="o", ms=3)
    ax1.set_xlabel("PCA component")
    ax1.set_ylabel("explained variance ratio")
    ax1.set_title("PCA scree (top components)")
    ax1.grid(alpha=0.3)

    names = [n for n, _ in axis_pca_cos]
    vals = [v for _, v in axis_pca_cos]
    ax2.bar(names, vals, color="#4c72b0")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("max |cosine| with any PCA component")
    ax2.set_title("Supervised axis ↔ PCA alignment")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(alpha=0.3, axis="y")

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
    """Load benchmark splits and the sentence encoder (see sibling scripts).

    :param config: Router YAML path.
    :type config: pathlib.Path
    :param encoder_override: Optional sentence-transformer id override.
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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

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
    parser.add_argument("--encoder", default=None,
                        help="Override the YAML sentence-transformer id.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Positive/negative split threshold.")
    parser.add_argument("--shrinkage", type=float, default=0.1,
                        help="LDA covariance shrinkage in [0, 1].")
    parser.add_argument("--variance-target", type=float, default=0.9,
                        help="Cumulative variance target for intrinsic dim.")
    parser.add_argument("--n-ica", type=int, default=8,
                        help="Number of independent components for FastICA.")
    parser.add_argument("--top-k-pca", type=int, default=10,
                        help="Top PCA components fed to the origin probe.")
    parser.add_argument(
        "--rederive-from-ica", action="store_true",
        help="Re-derive candidate axes from ICA (best single component AND "
             "ICA-subspace LDA) for all benchmarks, and compare per-benchmark "
             "origin recall + label AUC against the supervised axes.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--output-stem", default=None,
                        help="If set, write <stem>.{pdf,png} scree+alignment.")
    args = parser.parse_args(argv)

    if not args.toy and args.config is None:
        parser.error("pass --toy or --config")

    if args.toy:
        splits, encoder = _toy_inputs(seed=args.seed)
    else:
        splits, encoder = _load_from_config(args.config, encoder_override=args.encoder)

    axis_names = [s.axis_name for s in splits]

    # Pooled prompt embeddings + benchmark-of-origin labels.
    blocks, origin_blocks, label_blocks = [], [], []
    for k, split in enumerate(splits):
        emb = np.asarray(encoder(list(split.prompts)), dtype=float)
        blocks.append(emb)
        origin_blocks.append(np.full(len(split.prompts), k, dtype=int))
        label_blocks.append((split.scores >= args.threshold).astype(int))
    emb_all = np.concatenate(blocks, axis=0)
    origins = np.concatenate(origin_blocks, axis=0)

    # Supervised LDA axes (fit per benchmark on its own block).
    axes = []
    for split, emb in zip(splits, blocks):
        labels = (split.scores >= args.threshold).astype(int)
        pos, neg = emb[labels == 1], emb[labels == 0]
        if len(pos) < 2 or len(neg) < 2:
            raise ValueError(f"axis {split.axis_name!r}: need >=2 per class")
        axes.append(_lda_direction(pos, neg, shrinkage=args.shrinkage))
    axes = np.stack(axes, axis=0)

    # --- PCA ---
    pca = _pca(emb_all)
    idim = _intrinsic_dim(pca.explained_variance_ratio, args.variance_target)
    print("PCA spectrum:")
    print(f"  components for {args.variance_target:.0%} variance : {idim}"
          f"  (of {len(pca.explained_variance_ratio)} dims)")
    head = pca.explained_variance_ratio[:6]
    print("  top-6 explained-variance ratios : "
          + ", ".join(f"{r:.3f}" for r in head))

    # --- ICA ---
    ica = _fastica(emb_all, args.n_ica, seed=args.seed)

    # --- Axis <-> component alignment ---
    print()
    print("Supervised axis vs unsupervised components (max |cosine|):")
    print(f"  {'axis':<16}{'PCA cos':>9}{'@k':>5}{'ICA cos':>10}{'@k':>5}")
    print("  " + "-" * 45)
    axis_pca_cos = []
    for name, ax in zip(axis_names, axes):
        pcos, pidx = _max_abs_cosine(ax, pca.components)
        icos, iidx = _max_abs_cosine(ax, ica)
        axis_pca_cos.append((name, pcos))
        print(f"  {name:<16}{pcos:>9.3f}{pidx:>5}{icos:>10.3f}{iidx:>5}")
    print("  (low cosine with EVERY component = axis not isolated by "
          "unsupervised structure)")

    # --- Origin recoverability: supervised coords vs PCA vs ICA ---
    sup_coords = (emb_all - emb_all.mean(axis=0)) @ axes.T
    k_pca = min(args.top_k_pca, pca.components.shape[0])
    pca_coords = (emb_all - pca.mean) @ pca.components[:k_pca].T
    ica_coords = (emb_all - emb_all.mean(axis=0)) @ ica.T
    n_cls = len(splits)
    bal_sup = _balanced_accuracy(sup_coords, origins, n_cls, seed=args.seed)
    bal_pca = _balanced_accuracy(pca_coords, origins, n_cls, seed=args.seed)
    bal_ica = _balanced_accuracy(ica_coords, origins, n_cls, seed=args.seed)
    print()
    print("Benchmark-origin recoverability (balanced accuracy):")
    print(f"  chance                   : {1.0 / n_cls:.3f}")
    print(f"  supervised LDA coords    : {bal_sup:.3f}")
    print(f"  top-{k_pca} PCA components{'':<3}: {bal_pca:.3f}")
    print(f"  {args.n_ica} ICA components{'':<5}: {bal_ica:.3f}")

    # --- Per-axis label recoverability from the best-aligned component ---
    print()
    print("Per-axis label AUC from the best-aligned unsupervised component:")
    print(f"  {'axis':<16}{'own LDA':>9}{'best PCA':>10}{'best ICA':>10}")
    print("  " + "-" * 45)
    for k, (name, ax, split, emb) in enumerate(
        zip(axis_names, axes, splits, blocks)
    ):
        labels = (split.scores >= args.threshold).astype(int)
        own = _auc(emb @ ax, labels)
        ec = emb - emb.mean(axis=0)
        pca_proj = ec @ pca.components.T  # (n_k, K)
        ica_proj = ec @ ica.T
        best_pca = max(
            _auc(pca_proj[:, j], labels) for j in range(pca_proj.shape[1])
        )
        best_ica = max(
            _auc(ica_proj[:, j], labels) for j in range(ica_proj.shape[1])
        )
        # AUC is symmetric under sign flip; report max(auc, 1-auc).
        best_pca = max(best_pca, 1.0 - best_pca)
        best_ica = max(best_ica, 1.0 - best_ica)
        print(f"  {name:<16}{own:>9.3f}{best_pca:>10.3f}{best_ica:>10.3f}")
    print("  (own LDA high but best PCA/ICA low => real but low-variance / "
          "entangled axis)")

    # --- ICA re-derivation: does a different axis recover origin? ---
    if args.rederive_from_ica:
        labels_per = [
            (s.scores >= args.threshold).astype(int) for s in splits
        ]
        # Build three axis sets.
        ica_single = []
        single_idx = []
        single_auc = []
        for emb, lab in zip(blocks, labels_per):
            ax, idx, a = _best_ica_axis_by_auc(emb, lab, ica)
            ica_single.append(ax)
            single_idx.append(idx)
            single_auc.append(a)
        ica_single = np.stack(ica_single, axis=0)
        ica_sub = np.stack(
            [_ica_subspace_lda(emb, lab, ica, shrinkage=args.shrinkage)
             for emb, lab in zip(blocks, labels_per)],
            axis=0,
        )

        def _coords(axis_set: np.ndarray) -> np.ndarray:
            return (emb_all - emb_all.mean(axis=0)) @ axis_set.T

        axis_sets = {
            "supervised": axes,
            "ica_single": ica_single,
            "ica_subspaceLDA": ica_sub,
        }

        # Per-benchmark origin recall under each axis set.
        print()
        print("ICA re-derivation — per-benchmark ORIGIN RECALL by axis set:")
        header = f"  {'benchmark':<16}" + "".join(
            f"{k:>18}" for k in axis_sets
        )
        print(header)
        print("  " + "-" * (16 + 18 * len(axis_sets)))
        recalls = {
            name: _per_class_recall(_coords(aset), origins, n_cls, seed=args.seed)
            for name, aset in axis_sets.items()
        }
        for k, bname in enumerate(axis_names):
            row = "".join(f"{recalls[name][k]:>18.3f}" for name in axis_sets)
            print(f"  {bname:<16}{row}")

        # Per-benchmark own-label AUC under each axis set (does the
        # re-derived axis keep within-benchmark separability?).
        print()
        print("ICA re-derivation — per-benchmark LABEL AUC by axis set:")
        print(header)
        print("  " + "-" * (16 + 18 * len(axis_sets)))
        for k, (bname, emb, lab) in enumerate(zip(axis_names, blocks, labels_per)):
            cells = []
            for name, aset in axis_sets.items():
                a = _auc(emb @ aset[k], lab)
                cells.append(max(a, 1.0 - a))
            print(f"  {bname:<16}" + "".join(f"{c:>18.3f}" for c in cells))

        # Alignment of the re-derived axes with the supervised axis and with
        # the high-index PCA component the supervised axis loaded on.
        print()
        print("ICA re-derivation — axis directions (cos with supervised):")
        print(f"  {'benchmark':<16}{'single_idx':>12}{'single_AUC':>12}"
              f"{'cos(single,sup)':>18}{'cos(sub,sup)':>16}")
        print("  " + "-" * 74)
        for k, bname in enumerate(axis_names):
            cs = abs(float(ica_single[k] @ axes[k]))
            cu = abs(float(ica_sub[k] @ axes[k]))
            print(f"  {bname:<16}{single_idx[k]:>12}{single_auc[k]:>12.3f}"
                  f"{cs:>18.3f}{cu:>16.3f}")
        print("  (origin recall UP only for jailbreak => real niche, wrong "
              "LDA direction; UP for all => generic ICA effect; FLAT for "
              "jailbreak => label-separable but no distinct niche)")

    if args.output_stem is not None:
        _plot_spectrum(pca.explained_variance_ratio, axis_pca_cos, args.output_stem)
        print(f"\nwrote {args.output_stem}.{{pdf,png}}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
