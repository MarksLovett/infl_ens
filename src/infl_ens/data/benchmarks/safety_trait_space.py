"""Build an AI-safety trait space from labelled benchmark axes.

The safety benchmarks define axes that the router uses as the trait space
:math:`\\mathbb{B} \\subset [0, 1]^L`:

- **Axis 0 (harm)**: learned from :mod:`infl_ens.data.benchmarks.beavertails`.
  The harm direction in embedding space is a shrinkage Fisher discriminant
  separating harmful vs. non-harmful examples.
- **Axis 1 (hallucination)**: learned analogously from
  :mod:`infl_ens.data.benchmarks.halueval`, using the centroid of
  hallucinated vs. faithful responses.
- Additional prompt-level axes, such as ToxicChat jailbreak intent, are
  learned the same way from their labelled prompt embeddings.

Each axis is a unit-norm direction :math:`\\mathbf{w}` in embedding space.
For any new query :math:`q`, we compute the embedding :math:`e_q` and the
projected coordinate

.. math::

    c \\;=\\; \\mathrm{clip}_{[0,1]}\\Big( \\frac{\\mathbf{w}^\\top e_q - c_-}{c_+ - c_-} \\Big),

where :math:`c_-, c_+` are robust per-axis calibration percentiles (the
5th percentile of negatives and 95th percentile of positives). Learned
directions are Gram-Schmidt orthogonalised as they are added so overlapping
benchmark signals do not collapse the resource distribution onto a diagonal.
The resource distribution :math:`B(b)` is then estimated by KDE on the
calibration corpus, as in
:func:`infl_ens.data.trait_space.build_trait_space`.

This is a *learned-anchor* variant of the anchor mode in
``build_trait_space``: we learn the axis directions from labelled
benchmark data instead of from anchor strings.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence

import numpy as np

from infl_ens.data.benchmarks.base import BenchmarkSplit
from infl_ens.data.trait_space import TraitSpace, _kde_on_grid


@dataclass(frozen=True)
class SafetyTraitSpaceBundle:
    """Artifacts produced when fitting a multi-axis safety trait space.

    :param space: Discretised trait space with KDE weights and ``project``.
    :type space: TraitSpace
    :param axes: Learned axis directions and calibration metadata.
    :type axes: tuple[LearnedAxis, ...]
    :param coordinate_stretch_gamma: Default post-calibration stretch exponent.
    :type coordinate_stretch_gamma: float
    :param coordinate_stretch_gammas: Optional per-axis stretch overrides.
    :type coordinate_stretch_gammas: dict[str, float] | None
    """

    space: TraitSpace
    axes: tuple[LearnedAxis, ...]
    coordinate_stretch_gamma: float
    coordinate_stretch_gammas: Optional[dict[str, float]] = None


class _TextEmbeddingCache:
    """Memoise sentence-encoder outputs keyed by exact prompt text.

    :param encoder: Underlying encoder callable.
    :type encoder: Callable[[Sequence[str]], numpy.ndarray]
    """

    def __init__(self, encoder: Callable[[Sequence[str]], np.ndarray]) -> None:
        self._encoder = encoder
        self._store: dict[str, np.ndarray] = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return embeddings for ``texts``, encoding unseen strings once.

        :param texts: Prompt or response strings.
        :type texts: Sequence[str]
        :returns: Embedding matrix, shape ``(len(texts), D)``.
        :rtype: numpy.ndarray
        """
        ordered = list(texts)
        missing: list[str] = []
        seen: set[str] = set()
        for text in ordered:
            if text not in self._store and text not in seen:
                missing.append(text)
                seen.add(text)
        if missing:
            fresh = np.asarray(self._encoder(missing), dtype=float)
            for text, row in zip(missing, fresh):
                self._store[text] = np.asarray(row, dtype=float)
        return np.stack([self._store[text] for text in ordered], axis=0)


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
    :param residual_coef: Optional coefficients for subtracting a linear
        prediction from previous calibrated coordinates.
    :type residual_coef: numpy.ndarray | None
    :param residual_intercept: Intercept for coordinate residualization.
    :type residual_intercept: float
    :param residual_lo: Calibration low endpoint for residualized scores.
    :type residual_lo: float | None
    :param residual_hi: Calibration high endpoint for residualized scores.
    :type residual_hi: float | None
    """

    direction: np.ndarray
    lo: float
    hi: float
    name: str
    residual_coef: Optional[np.ndarray] = None
    residual_intercept: float = 0.0
    residual_lo: Optional[float] = None
    residual_hi: Optional[float] = None

    def project_base_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to the base calibrated coordinate.

        :param embeddings: Embedding matrix, shape ``(N, D)``.
        :type embeddings: numpy.ndarray
        :returns: Per-row coordinate in ``[0, 1]``, shape ``(N,)``.
        :rtype: numpy.ndarray
        """
        raw = embeddings @ self.direction
        span = max(self.hi - self.lo, 1e-12)
        return np.clip((raw - self.lo) / span, 0.0, 1.0)

    def project_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Project a batch of embeddings to ``[0, 1]`` along this axis.

        This method returns the independent base score. Projectors that use
        coordinate residualization apply the residual transform jointly
        across axes in :func:`_make_learned_projector`.

        :param embeddings: Embedding matrix, shape ``(N, D)``.
        :type embeddings: numpy.ndarray
        :returns: Per-row coordinate in ``[0, 1]``, shape ``(N,)``.
        :rtype: numpy.ndarray
        """
        return self.project_base_scores(embeddings)


def _learn_axis(
    embeddings: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
    name: str,
    prior_directions: Optional[Sequence[np.ndarray]] = None,
    shrinkage: float = 0.1,
    calibration_percentile: float = 5.0,
) -> LearnedAxis:
    """Fit a 1-D axis from labelled embeddings.

    The direction is a shrinkage Fisher linear-discriminant direction
    :math:`\\Sigma_w^{-1}(\\mu_+ - \\mu_-)` for positive
    (``score >= threshold``) and negative (``score < threshold``) examples.
    When prior directions are supplied, the direction is Gram-Schmidt
    orthogonalised against them before calibration. Lo/hi are the
    ``calibration_percentile``-th percentile of negatives and the
    ``100 - calibration_percentile``-th percentile of positives,
    respectively, which avoids median-induced coordinate saturation.

    :param embeddings: Embedding matrix, shape ``(N, D)``.
    :type embeddings: numpy.ndarray
    :param scores: Per-row scores in ``[0, 1]``, shape ``(N,)``.
    :type scores: numpy.ndarray
    :param threshold: Score threshold for splitting positives / negatives.
    :type threshold: float
    :param name: Axis name.
    :type name: str
    :param prior_directions: Previously learned unit directions to
        Gram-Schmidt out of the new axis.
    :type prior_directions: Sequence[numpy.ndarray] | None
    :param shrinkage: Within-class covariance shrinkage intensity in
        ``[0, 1]``. ``1`` collapses to a scaled centroid-difference
        direction.
    :type shrinkage: float
    :param calibration_percentile: Lower percentile for the negative class;
        the positive high endpoint uses ``100 - calibration_percentile``.
    :type calibration_percentile: float
    :returns: A :class:`LearnedAxis` calibrated to map the negative-class
        lower tail to ``0`` and the positive-class upper tail to ``1``.
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
    direction = _fit_shrinkage_fisher_direction(pos, neg, shrinkage=shrinkage)
    direction = _orthogonalize_direction(direction, prior_directions or [])
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError(
            f"axis {name!r}: positive and negative centroids coincide"
        )
    direction = direction / norm
    proj_pos = pos @ direction
    proj_neg = neg @ direction
    lo, hi = _calibration_bounds(
        proj_pos,
        proj_neg,
        calibration_percentile=calibration_percentile,
    )
    if hi <= lo:
        direction = -direction
        proj_pos = pos @ direction
        proj_neg = neg @ direction
        lo, hi = _calibration_bounds(
            proj_pos,
            proj_neg,
            calibration_percentile=calibration_percentile,
        )
    if hi <= lo:
        raise ValueError(
            f"axis {name!r}: calibration percentiles overlap "
            f"(lo={lo:.6g}, hi={hi:.6g})"
        )
    return LearnedAxis(
        direction=direction,
        lo=lo,
        hi=hi,
        name=name,
    )


def _blend_with_mode_direction(
    axis: LearnedAxis,
    own_prompt_embeddings: np.ndarray,
    other_prompt_embeddings: np.ndarray,
    *,
    mode_alignment_weight: float,
    prior_directions: Optional[Sequence[np.ndarray]] = None,
    shrinkage: float = 0.1,
    calibration_percentile: float = 5.0,
) -> LearnedAxis:
    """Blend a label axis with a one-vs-rest benchmark-mode direction.

    The label axis captures positive-vs-negative structure inside a
    benchmark. The mode direction captures benchmark-of-origin structure,
    encouraging the resource distribution to form one cloud per benchmark
    axis. ``mode_alignment_weight`` controls the blend, where ``0`` leaves
    the label axis unchanged and ``1`` uses only the benchmark-mode
    direction.

    :param axis: Label-separating learned axis.
    :type axis: LearnedAxis
    :param own_prompt_embeddings: Prompt embeddings for this benchmark.
    :type own_prompt_embeddings: numpy.ndarray
    :param other_prompt_embeddings: Prompt embeddings for all other
        benchmarks.
    :type other_prompt_embeddings: numpy.ndarray
    :param mode_alignment_weight: Blend coefficient in ``[0, 1]``.
    :type mode_alignment_weight: float
    :param prior_directions: Previously accepted directions for
        Gram-Schmidt decorrelation after blending.
    :type prior_directions: Sequence[numpy.ndarray] | None
    :param shrinkage: Shrinkage for the benchmark-mode Fisher direction.
    :type shrinkage: float
    :param calibration_percentile: Tail percentile for own-vs-rest
        calibration after blending.
    :type calibration_percentile: float
    :returns: Axis with blended direction and own-vs-rest calibration.
    :rtype: LearnedAxis
    :raises ValueError: If ``mode_alignment_weight`` is outside ``[0, 1]``.
    """
    if not 0.0 <= mode_alignment_weight <= 1.0:
        raise ValueError(
            "mode_alignment_weight must lie in [0, 1], "
            f"got {mode_alignment_weight}"
        )
    if mode_alignment_weight <= 0.0:
        return axis
    if len(own_prompt_embeddings) < 2 or len(other_prompt_embeddings) < 2:
        return axis

    mode_dir = _fit_shrinkage_fisher_direction(
        own_prompt_embeddings,
        other_prompt_embeddings,
        shrinkage=shrinkage,
    )
    mode_dir = mode_dir / max(float(np.linalg.norm(mode_dir)), 1e-12)
    direction = (
        (1.0 - mode_alignment_weight) * axis.direction
        + mode_alignment_weight * mode_dir
    )
    direction = _orthogonalize_direction(direction, prior_directions or [])
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return axis
    direction = direction / norm
    proj_own = own_prompt_embeddings @ direction
    proj_other = other_prompt_embeddings @ direction
    lo, hi = _calibration_bounds(
        proj_own,
        proj_other,
        calibration_percentile=calibration_percentile,
    )
    if hi <= lo:
        direction = -direction
        proj_own = own_prompt_embeddings @ direction
        proj_other = other_prompt_embeddings @ direction
        lo, hi = _calibration_bounds(
            proj_own,
            proj_other,
            calibration_percentile=calibration_percentile,
        )
    if hi <= lo:
        return axis
    return replace(axis, direction=direction, lo=lo, hi=hi)


def _fit_shrinkage_fisher_direction(
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    shrinkage: float = 0.1,
) -> np.ndarray:
    """Fit a shrinkage Fisher direction from labelled embeddings.

    :param pos: Positive-class embeddings, shape ``(P, D)``.
    :type pos: numpy.ndarray
    :param neg: Negative-class embeddings, shape ``(N, D)``.
    :type neg: numpy.ndarray
    :param shrinkage: Covariance shrinkage intensity in ``[0, 1]``.
    :type shrinkage: float
    :returns: Unnormalised discriminant direction.
    :rtype: numpy.ndarray
    :raises ValueError: If ``shrinkage`` is outside ``[0, 1]``.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage must lie in [0, 1], got {shrinkage}")
    mu_pos = pos.mean(axis=0)
    mu_neg = neg.mean(axis=0)
    d = pos.shape[1]
    cov_pos = np.cov(pos, rowvar=False) if len(pos) > 1 else np.zeros((d, d))
    cov_neg = np.cov(neg, rowvar=False) if len(neg) > 1 else np.zeros((d, d))
    sw = ((len(pos) - 1) * cov_pos + (len(neg) - 1) * cov_neg) / max(
        len(pos) + len(neg) - 2, 1
    )
    tau = float(np.trace(sw) / d) if d else 0.0
    sw_reg = (1.0 - shrinkage) * sw + shrinkage * tau * np.eye(d)
    mean_gap = mu_pos - mu_neg
    try:
        return np.linalg.solve(sw_reg, mean_gap)
    except np.linalg.LinAlgError:
        return mean_gap


def _orthogonalize_direction(
    direction: np.ndarray,
    prior_directions: Sequence[np.ndarray],
) -> np.ndarray:
    """Remove components along previously learned axis directions.

    :param direction: Candidate direction, shape ``(D,)``.
    :type direction: numpy.ndarray
    :param prior_directions: Unit or non-unit prior directions.
    :type prior_directions: Sequence[numpy.ndarray]
    :returns: Orthogonalised direction. If orthogonalisation fully removes
        the candidate, the original direction is returned so callers can
        raise their existing degenerate-axis error.
    :rtype: numpy.ndarray
    """
    original = np.asarray(direction, dtype=float)
    out = original.copy()
    for prior in prior_directions:
        basis = np.asarray(prior, dtype=float)
        denom = float(basis @ basis)
        if denom <= 1e-12:
            continue
        out = out - float(out @ basis) / denom * basis
    if float(np.linalg.norm(out)) <= 1e-12:
        return original
    return out


def _calibration_bounds(
    proj_pos: np.ndarray,
    proj_neg: np.ndarray,
    *,
    calibration_percentile: float = 5.0,
) -> tuple[float, float]:
    """Compute robust low/high calibration bounds for an axis.

    :param proj_pos: Positive-class raw projections.
    :type proj_pos: numpy.ndarray
    :param proj_neg: Negative-class raw projections.
    :type proj_neg: numpy.ndarray
    :param calibration_percentile: Lower percentile ``q`` for negatives;
        positives use ``100 - q``.
    :type calibration_percentile: float
    :returns: ``(lo, hi)`` calibration endpoints.
    :rtype: tuple[float, float]
    :raises ValueError: If ``calibration_percentile`` is not in
        ``[0, 50)``.
    """
    if not 0.0 <= calibration_percentile < 50.0:
        raise ValueError(
            "calibration_percentile must lie in [0, 50), "
            f"got {calibration_percentile}"
        )
    return (
        float(np.percentile(proj_neg, calibration_percentile)),
        float(np.percentile(proj_pos, 100.0 - calibration_percentile)),
    )


def _stretch_coordinates(coords: np.ndarray, gamma: float) -> np.ndarray:
    """Push calibrated coordinates farther from the origin.

    :param coords: Coordinate matrix in ``[0, 1]``, shape ``(N, L)``.
    :type coords: numpy.ndarray
    :param gamma: Concave stretch exponent. ``1`` leaves coordinates
        unchanged; values greater than ``1`` map moderate scores closer to
        the positive end of each axis.
    :type gamma: float
    :returns: Stretched coordinate matrix in ``[0, 1]``.
    :rtype: numpy.ndarray
    :raises ValueError: If ``gamma <= 0``.
    """
    if gamma <= 0.0:
        raise ValueError(f"coordinate_stretch_gamma must be positive, got {gamma}")
    if gamma == 1.0:
        return coords
    return 1.0 - np.power(1.0 - np.clip(coords, 0.0, 1.0), gamma)


def _coordinate_stretch_vector(
    axes: Sequence[LearnedAxis],
    *,
    coordinate_stretch_gamma: float = 1.0,
    coordinate_stretch_gammas: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """Build per-axis coordinate stretch exponents.

    :param axes: Learned axes whose names may be overridden.
    :type axes: Sequence[LearnedAxis]
    :param coordinate_stretch_gamma: Default stretch exponent for all axes.
    :type coordinate_stretch_gamma: float
    :param coordinate_stretch_gammas: Optional per-axis overrides keyed by
        axis name.
    :type coordinate_stretch_gammas: dict[str, float] | None
    :returns: Per-axis stretch exponents, shape ``(L,)``.
    :rtype: numpy.ndarray
    :raises ValueError: If any exponent is non-positive.
    """
    default_gamma = float(coordinate_stretch_gamma)
    gammas = np.asarray(
        [
            float(
                coordinate_stretch_gammas.get(axis.name, default_gamma)
                if coordinate_stretch_gammas is not None
                else default_gamma
            )
            for axis in axes
        ],
        dtype=float,
    )
    if np.any(gammas <= 0.0):
        raise ValueError(
            "coordinate stretch exponents must be positive, "
            f"got {gammas.tolist()}"
        )
    return gammas


def _make_learned_projector(
    encoder: Callable[[Sequence[str]], np.ndarray],
    axes: list[LearnedAxis],
    *,
    coordinate_stretch_gamma: float = 1.0,
    coordinate_stretch_gammas: Optional[dict[str, float]] = None,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Closure used as the trait-space ``project`` callable.

    :param encoder: Sentence encoder.
    :type encoder: Callable[[Sequence[str]], numpy.ndarray]
    :param axes: Per-dimension learned axes.
    :type axes: list[LearnedAxis]
    :param coordinate_stretch_gamma: Optional post-calibration stretch
        applied independently to each coordinate. Values greater than ``1``
        move resource mass farther along each axis.
    :type coordinate_stretch_gamma: float
    :param coordinate_stretch_gammas: Optional per-axis stretch overrides
        keyed by axis name.
    :type coordinate_stretch_gammas: dict[str, float] | None
    :returns: A function mapping a sequence of strings to coordinates in
        ``[0, 1]^L``.
    :rtype: Callable[[Sequence[str]], numpy.ndarray]
    """
    gammas = _coordinate_stretch_vector(
        axes,
        coordinate_stretch_gamma=coordinate_stretch_gamma,
        coordinate_stretch_gammas=coordinate_stretch_gammas,
    )

    def project(queries: Sequence[str]) -> np.ndarray:
        emb = np.asarray(encoder(list(queries)), dtype=float)
        base = np.stack(
            [axis.project_base_scores(emb) for axis in axes], axis=1,
        )
        coords = base.copy()
        for j, axis in enumerate(axes):
            if axis.residual_coef is None:
                continue
            if j == 0:
                continue
            pred = axis.residual_intercept + coords[:, :j] @ axis.residual_coef
            resid = base[:, j] - pred
            lo = axis.residual_lo if axis.residual_lo is not None else axis.lo
            hi = axis.residual_hi if axis.residual_hi is not None else axis.hi
            span = max(float(hi) - float(lo), 1e-12)
            coords[:, j] = np.clip((resid - float(lo)) / span, 0.0, 1.0)
        if np.all(gammas == 1.0):
            return coords
        return 1.0 - np.power(1.0 - np.clip(coords, 0.0, 1.0), gammas)

    return project


def _project_learned_coordinates(
    embeddings: np.ndarray,
    axes: Sequence[LearnedAxis],
    *,
    coordinate_stretch_gamma: float = 1.0,
    coordinate_stretch_gammas: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """Map precomputed embeddings to calibrated trait coordinates.

    :param embeddings: Sentence embeddings, shape ``(N, D)``.
    :type embeddings: numpy.ndarray
    :param axes: Learned axes including optional residualization metadata.
    :type axes: Sequence[LearnedAxis]
    :param coordinate_stretch_gamma: Default per-axis stretch exponent.
    :type coordinate_stretch_gamma: float
    :param coordinate_stretch_gammas: Optional per-axis stretch overrides.
    :type coordinate_stretch_gammas: dict[str, float] | None
    :returns: Coordinates in ``[0, 1]^L``, shape ``(N, L)``.
    :rtype: numpy.ndarray
    """
    gammas = _coordinate_stretch_vector(
        axes,
        coordinate_stretch_gamma=coordinate_stretch_gamma,
        coordinate_stretch_gammas=coordinate_stretch_gammas,
    )
    base = _base_coordinate_matrix(embeddings, axes)
    coords = base.copy()
    for j, axis in enumerate(axes):
        if axis.residual_coef is None or j == 0:
            continue
        pred = axis.residual_intercept + coords[:, :j] @ axis.residual_coef
        resid = base[:, j] - pred
        lo = axis.residual_lo if axis.residual_lo is not None else axis.lo
        hi = axis.residual_hi if axis.residual_hi is not None else axis.hi
        span = max(float(hi) - float(lo), 1e-12)
        coords[:, j] = np.clip((resid - float(lo)) / span, 0.0, 1.0)
    if np.all(gammas == 1.0):
        return coords
    return 1.0 - np.power(1.0 - np.clip(coords, 0.0, 1.0), gammas)


def _base_coordinate_matrix(embeddings: np.ndarray, axes: Sequence[LearnedAxis]) -> np.ndarray:
    """Project embeddings through each axis before coordinate residualization.

    :param embeddings: Embedding matrix, shape ``(N, D)``.
    :type embeddings: numpy.ndarray
    :param axes: Learned axes.
    :type axes: Sequence[LearnedAxis]
    :returns: Base calibrated coordinates, shape ``(N, L)``.
    :rtype: numpy.ndarray
    """
    return np.stack([axis.project_base_scores(embeddings) for axis in axes], axis=1)


def _apply_axis_residual(
    base_col: np.ndarray,
    prior_coords: np.ndarray,
    axis: LearnedAxis,
) -> np.ndarray:
    """Apply a fitted coordinate residualizer to one axis column.

    :param base_col: Base calibrated coordinate for this axis.
    :type base_col: numpy.ndarray
    :param prior_coords: Already-finalized prior coordinates, shape
        ``(N, j)``.
    :type prior_coords: numpy.ndarray
    :param axis: Axis carrying residualization coefficients.
    :type axis: LearnedAxis
    :returns: Residualized coordinate in ``[0, 1]``.
    :rtype: numpy.ndarray
    """
    if axis.residual_coef is None or prior_coords.shape[1] == 0:
        return base_col
    pred = axis.residual_intercept + prior_coords @ axis.residual_coef
    resid = base_col - pred
    lo = axis.residual_lo if axis.residual_lo is not None else axis.lo
    hi = axis.residual_hi if axis.residual_hi is not None else axis.hi
    span = max(float(hi) - float(lo), 1e-12)
    return np.clip((resid - float(lo)) / span, 0.0, 1.0)


def _fit_coordinate_residualizers(
    axes: list[LearnedAxis],
    splits: Sequence[BenchmarkSplit],
    *,
    cal_emb: np.ndarray,
    split_prompt_embeddings: Sequence[np.ndarray],
    threshold: float,
    calibration_percentile: float = 5.0,
) -> list[LearnedAxis]:
    """Decorrelate later coordinates from earlier coordinates.

    Each axis ``j > 0`` is linearly regressed against the already-finalized
    coordinates ``0..j-1`` on the calibration prompt corpus. Its coordinate
    becomes the residual, recalibrated with the axis's labelled split so the
    semantic positive class still maps high.

    :param axes: Learned axes before coordinate residualization.
    :type axes: list[LearnedAxis]
    :param splits: Benchmark splits aligned with ``axes``.
    :type splits: Sequence[BenchmarkSplit]
    :param cal_emb: Calibration-corpus embeddings, shape ``(N, D)``.
    :type cal_emb: numpy.ndarray
    :param split_prompt_embeddings: Prompt embeddings per split, aligned with
        ``splits``.
    :type split_prompt_embeddings: Sequence[numpy.ndarray]
    :param threshold: Label threshold for per-axis residual calibration.
    :type threshold: float
    :param calibration_percentile: Tail percentile used for residual lo/hi.
    :type calibration_percentile: float
    :returns: Axes with residualization metadata attached.
    :rtype: list[LearnedAxis]
    """
    if len(axes) <= 1:
        return axes

    cal_base = _base_coordinate_matrix(cal_emb, axes)
    final_cal = cal_base.copy()
    out: list[LearnedAxis] = [axes[0]]

    for j in range(1, len(axes)):
        design = np.column_stack([np.ones(final_cal.shape[0]), final_cal[:, :j]])
        coef_full, *_ = np.linalg.lstsq(design, cal_base[:, j], rcond=None)
        intercept = float(coef_full[0])
        coef = np.asarray(coef_full[1:], dtype=float)

        split = splits[j]
        split_emb = np.asarray(split_prompt_embeddings[j], dtype=float)
        split_base = _base_coordinate_matrix(split_emb, axes)
        split_prior = split_base.copy()
        for k in range(1, j):
            split_prior[:, k] = _apply_axis_residual(
                split_base[:, k],
                split_prior[:, :k],
                out[k],
            )
        pred = intercept + split_prior[:, :j] @ coef
        resid = split_base[:, j] - pred
        labels = split.scores >= threshold
        lo, hi = _calibration_bounds(
            resid[labels],
            resid[~labels],
            calibration_percentile=calibration_percentile,
        )
        if hi <= lo:
            # Keep the residual orientation aligned with the underlying axis.
            hi = lo + 1e-12
        ax = replace(
            axes[j],
            residual_coef=coef,
            residual_intercept=intercept,
            residual_lo=lo,
            residual_hi=hi,
        )
        out.append(ax)
        final_cal[:, j] = _apply_axis_residual(cal_base[:, j], final_cal[:, :j], ax)

    return out


def build_safety_trait_space_bundle(
    splits: Sequence[BenchmarkSplit],
    encoder: Callable[[Sequence[str]], np.ndarray],
    *,
    n_grid: int = 32,
    kde_bandwidth: Optional[float] = None,
    threshold: float = 0.5,
    calibration_corpus: Optional[Sequence[str]] = None,
    coordinate_residualize: bool = False,
    mode_alignment_weight: float = 0.0,
    mode_alignment_weights: Optional[dict[str, float]] = None,
    coordinate_stretch_gamma: float = 1.0,
    coordinate_stretch_gammas: Optional[dict[str, float]] = None,
) -> SafetyTraitSpaceBundle:
    """Construct a multi-axis :class:`TraitSpace` from labelled benchmarks.

    One axis is learned per :class:`BenchmarkSplit`. After axis fitting,
    the resource distribution :math:`B(b)` is estimated by isotropic
    Gaussian KDE on a uniform grid over :math:`[0, 1]^L`, using either the
    concatenated benchmark prompts or a separately supplied calibration
    corpus.

    Embeddings are memoised by exact text so each unique prompt or response
    is encoded at most once during the build.

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
    :param coordinate_residualize: If ``True``, regress each later
        coordinate against earlier coordinates on the calibration prompt
        corpus and recalibrate its residual. This is useful for keeping
        prompt-level axes such as privacy distinct from harm/hallucination.
    :type coordinate_residualize: bool
    :param mode_alignment_weight: Optional blend weight for adding a
        one-vs-rest benchmark-origin direction to each label direction.
        ``0`` preserves pure label axes; larger values encourage distinct
        benchmark-correlated resource modes.
    :type mode_alignment_weight: float
    :param mode_alignment_weights: Optional per-axis overrides keyed by
        axis name. Values override ``mode_alignment_weight`` for matching
        axes.
    :type mode_alignment_weights: dict[str, float] | None
    :param coordinate_stretch_gamma: Optional post-calibration stretch applied
        to each projected coordinate before KDE. ``1`` preserves the learned
        calibration; values greater than ``1`` push moderate scores farther
        toward the positive end of each axis.
    :type coordinate_stretch_gamma: float
    :param coordinate_stretch_gammas: Optional per-axis stretch overrides
        keyed by axis name. Values override ``coordinate_stretch_gamma`` for
        matching axes.
    :type coordinate_stretch_gammas: dict[str, float] | None
    :returns: Trait space plus learned-axis artifacts for caching.
    :rtype: SafetyTraitSpaceBundle
    :raises ValueError: If ``splits`` is empty.
    """
    splits = list(splits)
    if not splits:
        raise ValueError("need at least one BenchmarkSplit")

    embed_cache = _TextEmbeddingCache(encoder)
    axes: list[LearnedAxis] = []
    prior_directions: list[np.ndarray] = []
    prompt_embeddings_by_split = [
        embed_cache.encode(split.prompts) for split in splits
    ]
    label_embeddings_by_split: list[np.ndarray] = []
    for split in splits:
        score_target = str(split.metadata.get("score_target", "response"))
        if score_target not in ("prompt", "response"):
            raise ValueError(
                f"split {split.name!r}: metadata['score_target'] must be "
                f"'prompt' or 'response', got {score_target!r}"
            )
        if score_target == "response" and split.responses:
            label_embeddings_by_split.append(embed_cache.encode(split.responses))
        else:
            label_embeddings_by_split.append(
                prompt_embeddings_by_split[len(label_embeddings_by_split)],
            )

    for split_idx, split in enumerate(splits):
        emb = label_embeddings_by_split[split_idx]
        ax = _learn_axis(
            emb,
            split.scores,
            threshold=threshold,
            name=split.axis_name,
            prior_directions=prior_directions,
        )
        axis_mode_weight = (
            float(mode_alignment_weights.get(split.axis_name, mode_alignment_weight))
            if mode_alignment_weights is not None
            else mode_alignment_weight
        )
        if axis_mode_weight > 0.0:
            other_prompt_embeddings = np.concatenate(
                [
                    prompt_embeddings_by_split[i]
                    for i in range(len(splits))
                    if i != split_idx
                ],
                axis=0,
            )
            ax = _blend_with_mode_direction(
                ax,
                prompt_embeddings_by_split[split_idx],
                other_prompt_embeddings,
                mode_alignment_weight=axis_mode_weight,
                prior_directions=prior_directions,
            )
        axes.append(ax)
        prior_directions.append(ax.direction)

    if calibration_corpus is None:
        calibration_corpus = [p for s in splits for p in s.prompts]
    cal_emb = embed_cache.encode(calibration_corpus)
    if coordinate_residualize:
        axes = _fit_coordinate_residualizers(
            axes,
            splits,
            cal_emb=cal_emb,
            split_prompt_embeddings=prompt_embeddings_by_split,
            threshold=threshold,
        )

    stretch_gamma = float(coordinate_stretch_gamma)
    project = _make_learned_projector(
        encoder,
        axes,
        coordinate_stretch_gamma=stretch_gamma,
        coordinate_stretch_gammas=coordinate_stretch_gammas,
    )
    axis_labels = tuple(a.name for a in axes)
    L = len(axes)
    coords = _project_learned_coordinates(
        cal_emb,
        axes,
        coordinate_stretch_gamma=stretch_gamma,
        coordinate_stretch_gammas=coordinate_stretch_gammas,
    )

    grid_axes = [np.linspace(0.0, 1.0, n_grid) for _ in range(L)]
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    grid = np.stack([m.ravel() for m in mesh], axis=1)

    if kde_bandwidth is None:
        n = coords.shape[0]
        kde_bandwidth = float(n ** (-1.0 / (L + 4)))

    weights = _kde_on_grid(coords, grid, float(kde_bandwidth))

    space = TraitSpace(
        grid=grid,
        weights=weights,
        project=project,
        axis_labels=axis_labels,
    )
    return SafetyTraitSpaceBundle(
        space=space,
        axes=tuple(axes),
        coordinate_stretch_gamma=stretch_gamma,
        coordinate_stretch_gammas=coordinate_stretch_gammas,
    )


def build_safety_trait_space(
    splits: Sequence[BenchmarkSplit],
    encoder: Callable[[Sequence[str]], np.ndarray],
    *,
    n_grid: int = 32,
    kde_bandwidth: Optional[float] = None,
    threshold: float = 0.5,
    calibration_corpus: Optional[Sequence[str]] = None,
    coordinate_residualize: bool = False,
    mode_alignment_weight: float = 0.0,
    mode_alignment_weights: Optional[dict[str, float]] = None,
    coordinate_stretch_gamma: float = 1.0,
    coordinate_stretch_gammas: Optional[dict[str, float]] = None,
) -> TraitSpace:
    """Construct a multi-axis :class:`TraitSpace` from labelled benchmarks.

    Thin wrapper around :func:`build_safety_trait_space_bundle` that returns
    only the :class:`TraitSpace`.

    :param splits: One benchmark split per desired axis.
    :type splits: Sequence[BenchmarkSplit]
    :param encoder: Sentence encoder used at fit *and* projection time.
    :type encoder: Callable[[Sequence[str]], numpy.ndarray]
    :param n_grid: Number of grid points per axis.
    :type n_grid: int
    :param kde_bandwidth: KDE bandwidth; ``None`` selects Scott's rule.
    :type kde_bandwidth: float | None
    :param threshold: Score threshold for axis learning.
    :type threshold: float
    :param calibration_corpus: Optional KDE calibration corpus.
    :type calibration_corpus: Sequence[str] | None
    :param coordinate_residualize: Whether to decorrelate later axes.
    :type coordinate_residualize: bool
    :param mode_alignment_weight: Benchmark-mode blend weight.
    :type mode_alignment_weight: float
    :param mode_alignment_weights: Per-axis mode blend overrides.
    :type mode_alignment_weights: dict[str, float] | None
    :param coordinate_stretch_gamma: Default coordinate stretch exponent.
    :type coordinate_stretch_gamma: float
    :param coordinate_stretch_gammas: Per-axis stretch overrides.
    :type coordinate_stretch_gammas: dict[str, float] | None
    :returns: A :class:`TraitSpace` of dimension ``L = len(splits)``.
    :rtype: TraitSpace
    :raises ValueError: If ``splits`` is empty.
    """
    return build_safety_trait_space_bundle(
        splits,
        encoder,
        n_grid=n_grid,
        kde_bandwidth=kde_bandwidth,
        threshold=threshold,
        calibration_corpus=calibration_corpus,
        coordinate_residualize=coordinate_residualize,
        mode_alignment_weight=mode_alignment_weight,
        mode_alignment_weights=mode_alignment_weights,
        coordinate_stretch_gamma=coordinate_stretch_gamma,
        coordinate_stretch_gammas=coordinate_stretch_gammas,
    ).space
