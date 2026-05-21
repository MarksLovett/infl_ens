"""Trait-space construction for the influencer-game router.

Builds an :math:`L`-dimensional trait space :math:`\\mathbb{B}` together with
an empirical resource distribution :math:`B(b)` from a corpus of queries.
The space is constructed automatically: queries are embedded by a user-supplied
encoder, projected onto a low-dimensional space (either via interpretable
*anchor* directions or via PCA), scaled to :math:`[0, 1]^L`, and the
resource density is estimated by Gaussian kernel-density estimation on a
uniform grid.

This module is data-prep only: no disk I/O, no global state.

References:
    Lovett & Fu (2024). "Learning Dynamics of the Influencer's Game in
    Resource Landscapes." See in particular §2 (Influencer's Game Model)
    and Fig. 1d–e (LinkedIn job-market construction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TraitSpace:
    """A discretised trait space :math:`\\mathbb{B}` with an empirical resource distribution.

    :param grid: Trait grid points, shape ``(K, L)``. ``K`` is the number of
        discretisation points and ``L`` is the trait dimensionality.
    :type grid: numpy.ndarray
    :param weights: Resource weights :math:`B(b_k)` at each grid point,
        shape ``(K,)``, normalised to sum to one.
    :type weights: numpy.ndarray
    :param project: Callable mapping a sequence of raw query strings to their
        coordinates in :math:`\\mathbb{B}`, returning shape ``(M, L)``. Used
        at routing time to embed live queries.
    :type project: Callable[[Sequence[str]], numpy.ndarray]
    :param axis_labels: Optional human-readable labels for the ``L`` axes
        (e.g. ``('math', 'code', 'creative')``).
    :type axis_labels: tuple[str, ...] | None
    """

    grid: np.ndarray
    weights: np.ndarray
    project: Callable[[Sequence[str]], np.ndarray]
    axis_labels: Optional[Tuple[str, ...]] = None

    @property
    def L(self) -> int:
        """Trait-space dimensionality :math:`L`.

        :returns: Number of trait dimensions.
        :rtype: int
        """
        return int(self.grid.shape[1])

    @property
    def K(self) -> int:
        """Number of grid points :math:`K`.

        :returns: Total grid size.
        :rtype: int
        """
        return int(self.grid.shape[0])

    @property
    def mean(self) -> np.ndarray:
        """Resource-weighted mean :math:`\\mathbb{E}_B[b]`.

        For the MV-Gaussian influencer's game, this equals the symmetric Nash
        equilibrium position (Lemma 7, Lovett & Fu 2024).

        :returns: Mean vector, shape ``(L,)``.
        :rtype: numpy.ndarray
        """
        return np.einsum("k,kl->l", self.weights, self.grid)

    @property
    def covariance(self) -> np.ndarray:
        """Resource-weighted covariance :math:`\\Sigma_B`.

        Drives the closed-form stability threshold

        .. math::

            \\sigma_0^* = \\sqrt{(N-2)/(N-1)}\\,\\sqrt{\\Lambda_{\\max}(\\Sigma_B)}.

        :returns: Covariance matrix, shape ``(L, L)``.
        :rtype: numpy.ndarray
        """
        c = self.grid - self.mean
        return np.einsum("k,kl,km->lm", self.weights, c, c)


def _make_anchor_projector(
    encoder: Callable[[Sequence[str]], np.ndarray],
    anchor_vecs: np.ndarray,
    scale_lo: np.ndarray,
    scale_hi: np.ndarray,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Build the runtime ``project`` callable used by :class:`TraitSpace`.

    Closes over the (unit-normalised) anchor matrix and the corpus-derived
    min/max scaling so live queries land in the same :math:`[0, 1]^L` box as
    the calibration corpus.

    :param encoder: Sentence-embedding callable.
    :type encoder: Callable[[Sequence[str]], numpy.ndarray]
    :param anchor_vecs: Unit-normalised anchor embeddings, shape ``(L, D)``.
    :type anchor_vecs: numpy.ndarray
    :param scale_lo: Per-axis cosine-similarity floor used for rescaling,
        shape ``(L,)``.
    :type scale_lo: numpy.ndarray
    :param scale_hi: Per-axis cosine-similarity ceiling, shape ``(L,)``.
    :type scale_hi: numpy.ndarray
    :returns: A ``project`` callable suitable for :class:`TraitSpace`.
    :rtype: Callable[[Sequence[str]], numpy.ndarray]
    """
    span = np.maximum(scale_hi - scale_lo, 1e-12)

    def project(queries: Sequence[str]) -> np.ndarray:
        emb = np.asarray(encoder(list(queries)), dtype=float)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        sims = emb @ anchor_vecs.T
        return np.clip((sims - scale_lo) / span, 0.0, 1.0)

    return project


def _make_pca_projector(
    encoder: Callable[[Sequence[str]], np.ndarray],
    components: np.ndarray,
    mean_emb: np.ndarray,
    scale_lo: np.ndarray,
    scale_hi: np.ndarray,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Build the runtime ``project`` callable for PCA mode.

    :param encoder: Sentence-embedding callable.
    :type encoder: Callable[[Sequence[str]], numpy.ndarray]
    :param components: Top-``L`` right singular vectors, shape ``(L, D)``.
    :type components: numpy.ndarray
    :param mean_emb: Corpus mean embedding, shape ``(D,)``.
    :type mean_emb: numpy.ndarray
    :param scale_lo: Per-axis floor of the PCA scores, shape ``(L,)``.
    :type scale_lo: numpy.ndarray
    :param scale_hi: Per-axis ceiling of the PCA scores, shape ``(L,)``.
    :type scale_hi: numpy.ndarray
    :returns: A ``project`` callable suitable for :class:`TraitSpace`.
    :rtype: Callable[[Sequence[str]], numpy.ndarray]
    """
    span = np.maximum(scale_hi - scale_lo, 1e-12)

    def project(queries: Sequence[str]) -> np.ndarray:
        emb = np.asarray(encoder(list(queries)), dtype=float)
        scores = (emb - mean_emb) @ components.T
        return np.clip((scores - scale_lo) / span, 0.0, 1.0)

    return project


def _kde_on_grid(
    coords: np.ndarray,
    grid: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Isotropic Gaussian KDE evaluated at a grid.

    Computed in log-space and normalised at the end.

    :param coords: Calibration-corpus coordinates, shape ``(N, L)``.
    :type coords: numpy.ndarray
    :param grid: Grid points, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param bandwidth: Isotropic Gaussian bandwidth in trait-space units.
    :type bandwidth: float
    :returns: Normalised density weights at each grid point, shape ``(K,)``.
    :rtype: numpy.ndarray
    """
    diffs = grid[:, None, :] - coords[None, :, :]
    sq = np.sum(diffs ** 2, axis=2)  # (K, N)
    log_w = -0.5 * sq / (bandwidth ** 2)
    log_w_max = log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w - log_w_max).sum(axis=1)
    return w / w.sum()


def build_trait_space(
    queries: Sequence[str],
    encoder: Callable[[Sequence[str]], np.ndarray],
    *,
    anchors: Optional[Sequence[str]] = None,
    L: int = 3,
    n_grid: int = 32,
    kde_bandwidth: Optional[float] = None,
) -> TraitSpace:
    """Construct a trait space and resource distribution from a query corpus.

    The pipeline is:

    1. Embed ``queries`` via ``encoder``.
    2. Project to ``L`` dimensions, either by cosine similarity to
       ``anchors`` (interpretable axes) or by PCA (data-driven axes).
    3. Min-max rescale to :math:`[0, 1]^L`.
    4. Lay an ``n_grid``-per-axis uniform grid.
    5. Estimate :math:`B(b)` by isotropic Gaussian KDE on the grid.

    :param queries: Calibration corpus. Should be representative of the live
        query distribution.
    :type queries: Sequence[str]
    :param encoder: Callable mapping a list of strings to an ``(N, D)``
        embedding matrix.
    :type encoder: Callable[[Sequence[str]], numpy.ndarray]
    :param anchors: Optional list of anchor concept strings. If supplied,
        its length determines ``L`` and produces interpretable axes.
    :type anchors: Sequence[str] | None
    :param L: Trait-space dimensionality. Ignored when ``anchors`` is given.
        Keep :math:`L \\in \\{2, 3, 4\\}` so the :math:`n_{\\text{grid}}^L`
        grid stays tractable; switch to a codebook for larger ``L``.
    :type L: int
    :param n_grid: Number of grid points per axis.
    :type n_grid: int
    :param kde_bandwidth: Optional KDE bandwidth in trait-space units. If
        ``None``, Scott's rule :math:`n^{-1/(L+4)}` is used.
    :type kde_bandwidth: float | None
    :returns: A :class:`TraitSpace` carrying the grid, normalised resource
        weights, and a ``project`` callable for runtime query embedding.
    :rtype: TraitSpace
    :raises ValueError: If ``len(queries) < 2 * L`` or ``L < 1``.
    """
    queries = list(queries)
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    if anchors is not None:
        L = len(anchors)
    if len(queries) < 2 * L:
        raise ValueError(
            f"need at least 2*L = {2 * L} calibration queries, got {len(queries)}"
        )

    emb = np.asarray(encoder(queries), dtype=float)

    if anchors is not None:
        anchor_vecs = np.asarray(encoder(list(anchors)), dtype=float)
        anchor_vecs = anchor_vecs / np.linalg.norm(anchor_vecs, axis=1, keepdims=True)
        emb_n = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        sims = emb_n @ anchor_vecs.T  # (N, L)
        scale_lo = sims.min(axis=0)
        scale_hi = sims.max(axis=0)
        span = np.maximum(scale_hi - scale_lo, 1e-12)
        coords = np.clip((sims - scale_lo) / span, 0.0, 1.0)
        project = _make_anchor_projector(encoder, anchor_vecs, scale_lo, scale_hi)
        axis_labels: Optional[Tuple[str, ...]] = tuple(anchors)
    else:
        mean_emb = emb.mean(axis=0)
        centered = emb - mean_emb
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:L]  # (L, D)
        scores = centered @ components.T
        scale_lo = scores.min(axis=0)
        scale_hi = scores.max(axis=0)
        span = np.maximum(scale_hi - scale_lo, 1e-12)
        coords = np.clip((scores - scale_lo) / span, 0.0, 1.0)
        project = _make_pca_projector(encoder, components, mean_emb, scale_lo, scale_hi)
        axis_labels = None

    axes = [np.linspace(0.0, 1.0, n_grid) for _ in range(L)]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([m.ravel() for m in mesh], axis=1)  # (n_grid^L, L)

    if kde_bandwidth is None:
        n = coords.shape[0]
        kde_bandwidth = float(n ** (-1.0 / (L + 4)))  # Scott's rule

    weights = _kde_on_grid(coords, grid, float(kde_bandwidth))

    return TraitSpace(
        grid=grid,
        weights=weights,
        project=project,
        axis_labels=axis_labels,
    )


def position_from_corpus(
    queries: Sequence[str],
    project: Callable[[Sequence[str]], np.ndarray],
    *,
    scores: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Estimate an agent's position in trait space from a corpus of queries.

    Returns the trait-space centroid of ``queries`` after projection,
    optionally weighted by ``scores`` (e.g. per-query benchmark accuracy or
    reward). This is the standard way to *observe* where a model lives in
    :math:`\\mathbb{B}`, in contrast to choosing positions strategically.

    Typical uses:

    - **Initial position from base-model behaviour**: pass queries the base
      model handles well; the centroid is the model's current "centre of
      mass" in trait space.
    - **Post-training update**: after a round of fine-tuning, pass the
      training corpus (or an eval set with per-query scores) to re-estimate
      where the model now sits.

    :param queries: Queries to embed and average. Must be non-empty.
    :type queries: Sequence[str]
    :param project: Trait-space projector from :class:`TraitSpace`.
    :type project: Callable[[Sequence[str]], numpy.ndarray]
    :param scores: Optional per-query non-negative weights of shape
        ``(len(queries),)``. If ``None``, a uniform mean is taken. If all
        scores are zero, falls back to the uniform mean.
    :type scores: Sequence[float] | None
    :returns: Position vector, shape ``(L,)``.
    :rtype: numpy.ndarray
    :raises ValueError: If ``queries`` is empty or ``scores`` has the wrong
        shape.
    """
    queries = list(queries)
    if not queries:
        raise ValueError("queries must be non-empty")
    coords = project(queries)
    if scores is None:
        return coords.mean(axis=0)
    w = np.asarray(scores, dtype=float)
    if w.shape != (len(queries),):
        raise ValueError(
            f"scores must have shape ({len(queries)},), got {w.shape}"
        )
    if (w < 0).any():
        raise ValueError("scores must be non-negative")
    total = float(w.sum())
    if total <= 0.0:
        return coords.mean(axis=0)
    return (w[:, None] * coords).sum(axis=0) / total
