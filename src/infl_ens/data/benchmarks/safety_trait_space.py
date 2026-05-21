"""Build a 2-D AI-safety trait space from BeaverTails and HaluEval.

The two benchmarks define orthogonal axes that the router uses as the
trait space :math:`\\mathbb{B} \\subset [0, 1]^2`:

- **Axis 0 (harm)**: learned from :mod:`infl_ens.data.benchmarks.beavertails`.
  The harm direction in embedding space is the difference of the centroids
  of harmful vs. non-harmful prompts.
- **Axis 1 (hallucination)**: learned analogously from
  :mod:`infl_ens.data.benchmarks.halueval`, using the centroid of
  hallucinated vs. faithful responses.

Each axis is a unit-norm direction :math:`\\mathbf{w}` in embedding space.
For any new query :math:`q`, we compute the embedding :math:`e_q` and the
projected coordinate

.. math::

    c \\;=\\; \\mathrm{clip}_{[0,1]}\\Big( \\frac{\\mathbf{w}^\\top e_q - c_-}{c_+ - c_-} \\Big),

where :math:`c_+, c_-` are the per-axis projections of the high- and
low-label centroids (so calibration positives land near 1 and negatives
near 0). The resource distribution :math:`B(b)` is then estimated by KDE
on the calibration corpus, as in
:func:`infl_ens.data.trait_space.build_trait_space`.

This is a *learned-anchor* variant of the anchor mode in
``build_trait_space``: we learn the axis directions from labelled
benchmark data instead of from anchor strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from infl_ens.data.benchmarks.base import BenchmarkSplit
from infl_ens.data.trait_space import TraitSpace, _kde_on_grid


@dataclass(frozen=True)
class LearnedAxis:
    """A unit-norm axis direction in embedding space.

    :param direction: Unit vector in :math:`\\mathbb{R}^D`, shape ``(D,)``.
    :type direction: numpy.ndarray
    :param lo: Calibration low-end projection (maps to ``0.0``).
    :type lo: float
    :param hi: Calibration high-end projection (maps to ``1.0``).
    :type hi: float
    :param name: Human-readable axis name.
    :type name: str
    """

    direction: np.ndarray
    lo: float
    hi: float
    name: str

    def project_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Project a batch of embeddings to ``[0, 1]`` along this axis.

        :param embeddings: Embedding matrix, shape ``(N, D)``.
        :type embeddings: numpy.ndarray
        :returns: Per-row coordinate in ``[0, 1]``, shape ``(N,)``.
        :rtype: numpy.ndarray
        """
        raw = embeddings @ self.direction
        span = max(self.hi - self.lo, 1e-12)
        return np.clip((raw - self.lo) / span, 0.0, 1.0)


def _learn_axis(
    embeddings: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
    name: str,
) -> LearnedAxis:
    """Fit a 1-D axis from labelled embeddings.

    The direction is the unit-normalised difference of the centroids of
    positive (``score >= threshold``) and negative (``score < threshold``)
    examples. Lo/hi are the *median* projections of negatives and
    positives respectively, which is robust to outliers in the embedding
    geometry.

    :param embeddings: Embedding matrix, shape ``(N, D)``.
    :type embeddings: numpy.ndarray
    :param scores: Per-row scores in ``[0, 1]``, shape ``(N,)``.
    :type scores: numpy.ndarray
    :param threshold: Score threshold for splitting positives / negatives.
    :type threshold: float
    :param name: Axis name.
    :type name: str
    :returns: A :class:`LearnedAxis` calibrated to map the negative-class
        median to ``0`` and the positive-class median to ``1``.
    :rtype: LearnedAxis
    :raises ValueError: If either class has fewer than two examples.
    """
    pos = embeddings[scores >= threshold]
    neg = embeddings[scores < threshold]
    if len(pos) < 2 or len(neg) < 2:
        raise ValueError(
            f"axis {name!r}: need at least 2 examples per class "
            f"(got {len(pos)} positive, {len(neg)} negative)"
        )
    direction = pos.mean(axis=0) - neg.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError(
            f"axis {name!r}: positive and negative centroids coincide"
        )
    direction = direction / norm
    proj_pos = pos @ direction
    proj_neg = neg @ direction
    return LearnedAxis(
        direction=direction,
        lo=float(np.median(proj_neg)),
        hi=float(np.median(proj_pos)),
        name=name,
    )


def _make_learned_projector(
    encoder: Callable[[Sequence[str]], np.ndarray],
    axes: list[LearnedAxis],
) -> Callable[[Sequence[str]], np.ndarray]:
    """Closure used as the trait-space ``project`` callable.

    :param encoder: Sentence encoder.
    :type encoder: Callable[[Sequence[str]], numpy.ndarray]
    :param axes: Per-dimension learned axes.
    :type axes: list[LearnedAxis]
    :returns: A function mapping a sequence of strings to coordinates in
        ``[0, 1]^L``.
    :rtype: Callable[[Sequence[str]], numpy.ndarray]
    """

    def project(queries: Sequence[str]) -> np.ndarray:
        emb = np.asarray(encoder(list(queries)), dtype=float)
        return np.stack(
            [axis.project_scores(emb) for axis in axes], axis=1,
        )

    return project


def build_safety_trait_space(
    splits: Sequence[BenchmarkSplit],
    encoder: Callable[[Sequence[str]], np.ndarray],
    *,
    n_grid: int = 32,
    kde_bandwidth: Optional[float] = None,
    threshold: float = 0.5,
    calibration_corpus: Optional[Sequence[str]] = None,
) -> TraitSpace:
    """Construct a multi-axis :class:`TraitSpace` from labelled benchmarks.

    One axis is learned per :class:`BenchmarkSplit`. After axis fitting,
    the resource distribution :math:`B(b)` is estimated by isotropic
    Gaussian KDE on a uniform grid over :math:`[0, 1]^L`, using either the
    concatenated benchmark prompts or a separately supplied calibration
    corpus.

    :param splits: One benchmark split per desired axis. The order
        determines axis ordering in the returned :class:`TraitSpace`.
    :type splits: Sequence[BenchmarkSplit]
    :param encoder: Sentence encoder used at fit *and* projection time.
    :type encoder: Callable[[Sequence[str]], numpy.ndarray]
    :param n_grid: Number of grid points per axis.
    :type n_grid: int
    :param kde_bandwidth: KDE bandwidth; ``None`` selects Scott's rule
        :math:`n^{-1/(L+4)}`.
    :type kde_bandwidth: float | None
    :param threshold: Score threshold for the binary split used to learn
        each axis direction.
    :type threshold: float
    :param calibration_corpus: Optional prompts used for the KDE step. If
        ``None``, the concatenation of all split prompts is used.
    :type calibration_corpus: Sequence[str] | None
    :returns: A :class:`TraitSpace` of dimension ``L = len(splits)``.
    :rtype: TraitSpace
    :raises ValueError: If ``splits`` is empty.
    """
    splits = list(splits)
    if not splits:
        raise ValueError("need at least one BenchmarkSplit")

    axes: list[LearnedAxis] = []
    for split in splits:
        emb = np.asarray(encoder(list(split.prompts + split.responses)), dtype=float)
        # Score the *response* axis where responses exist; otherwise use
        # the prompt embeddings with the prompt-level scores.
        if split.responses:
            n_p = len(split.prompts)
            emb = emb[n_p:]  # response embeddings
        ax = _learn_axis(emb, split.scores, threshold=threshold, name=split.axis_name)
        axes.append(ax)

    project = _make_learned_projector(encoder, axes)
    axis_labels = tuple(a.name for a in axes)
    L = len(axes)

    if calibration_corpus is None:
        calibration_corpus = [p for s in splits for p in s.prompts]
    coords = project(calibration_corpus)

    grid_axes = [np.linspace(0.0, 1.0, n_grid) for _ in range(L)]
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    grid = np.stack([m.ravel() for m in mesh], axis=1)

    if kde_bandwidth is None:
        n = coords.shape[0]
        kde_bandwidth = float(n ** (-1.0 / (L + 4)))

    weights = _kde_on_grid(coords, grid, float(kde_bandwidth))

    return TraitSpace(
        grid=grid,
        weights=weights,
        project=project,
        axis_labels=axis_labels,
    )
