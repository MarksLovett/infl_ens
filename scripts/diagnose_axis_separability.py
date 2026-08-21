"""Diagnose trait-axis separability and compare candidate axis estimators.

The closed-loop router can only push agents toward resource mass that
actually exists in the trait space. When the projected prompt
coordinates are diffuse — spread roughly uniformly over
:math:`[0, 1]^L` rather than clustered into well-separated basins — no
routing rule (canonical, strategic, or otherwise) can recover the
theoretical Nash structure, because the geometry it would converge to is
not present in :math:`B(b)`.

This script quantifies *how separable* the current axes are and how much
headroom three drop-in alternative estimators buy, **without changing the
trainer**. For each axis it fits and scores four estimators on labelled
benchmark embeddings:

- ``mean_diff`` — the current
  :func:`infl_ens.data.benchmarks.safety_trait_space._learn_axis`
  estimator: unit-normalised difference of class centroids
  (nearest-centroid / rank-1 LDA with isotropic covariance).
- ``lda`` — the Fisher linear-discriminant direction
  :math:`\\Sigma_w^{-1}(\\mu_+ - \\mu_-)` with a shrinkage-regularised
  pooled within-class covariance :math:`\\Sigma_w`.
- ``mean_diff_pct`` — ``mean_diff`` direction but calibrated with the 5th
  / 95th projected percentiles instead of class medians, so positives
  push toward ``1`` and negatives toward ``0`` rather than stacking on
  the median.
- ``lda_pct`` — ``lda`` direction with percentile calibration.

Separability metrics reported per axis / estimator:

- **AUC** of the 1-D projected score against the binary label (rank
  separability; invariant to the lo/hi calibration).
- **Cohen's d** between the projected positive and negative classes
  (standardised mean gap).
- **Calibrated-coordinate spread**: the standard deviation of the
  ``[0, 1]`` coordinate over the calibration corpus. Higher means the
  prompts occupy more of the axis rather than collapsing to the median.

It also reports the **between-axis correlation** of the calibrated
coordinates, which directly measures whether the harm and hallucination
axes carry independent structure or collapse toward a diagonal. A
``--decorrelate`` flag additionally reports the post-Gram-Schmidt
correlation.

Everything here is read-only with respect to the package: the candidate
estimators are computed locally so the script can be run as a
measurement before deciding whether to port any change into
``safety_trait_space.py``. It runs offline against a toy encoder for
smoke-testing and against the real MiniLM encoder on ``doob``.

Example
-------

.. code-block:: console

   # Offline smoke test with the built-in toy encoder.
   $ python scripts/diagnose_axis_separability.py --toy

   # Real run on doob, with a figure and decorrelation report.
   $ python scripts/diagnose_axis_separability.py \\
         --config configs/benchmark/router/safety_truth.yaml \\
         --decorrelate \\
         --output-stem scripts/figures/axis_separability
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# Candidate axis estimators
# -----------------------------------------------------------------------------

def _fit_direction_mean_diff(
    pos: np.ndarray, neg: np.ndarray,
) -> np.ndarray:
    """Unit-normalised difference of class centroids (current estimator).

    This reproduces the direction used by
    :func:`infl_ens.data.benchmarks.safety_trait_space._learn_axis`.

    :param pos: Positive-class embeddings, shape ``(P, D)``.
    :type pos: numpy.ndarray
    :param neg: Negative-class embeddings, shape ``(N, D)``.
    :type neg: numpy.ndarray
    :returns: Unit-norm axis direction, shape ``(D,)``.
    :rtype: numpy.ndarray
    """
    d = pos.mean(axis=0) - neg.mean(axis=0)
    return _unit(d)


def _fit_direction_lda(
    pos: np.ndarray, neg: np.ndarray, *, shrinkage: float = 0.1,
) -> np.ndarray:
    """Fisher linear-discriminant direction with shrinkage covariance.

    Computes :math:`\\Sigma_w^{-1}(\\mu_+ - \\mu_-)` where
    :math:`\\Sigma_w` is the pooled within-class covariance regularised
    toward a scaled identity:
    :math:`(1 - \\lambda)\\Sigma_w + \\lambda \\bar\\tau I`, with
    :math:`\\bar\\tau` the mean diagonal of :math:`\\Sigma_w`. Shrinkage
    is essential because MiniLM's 384-dim covariance is ill-conditioned
    relative to the per-class sample count.

    :param pos: Positive-class embeddings, shape ``(P, D)``.
    :type pos: numpy.ndarray
    :param neg: Negative-class embeddings, shape ``(N, D)``.
    :type neg: numpy.ndarray
    :param shrinkage: Shrinkage intensity :math:`\\lambda \\in [0, 1]`.
        ``0`` is plain LDA, ``1`` collapses to ``mean_diff``.
    :type shrinkage: float
    :returns: Unit-norm axis direction, shape ``(D,)``.
    :rtype: numpy.ndarray
    """
    mu_pos = pos.mean(axis=0)
    mu_neg = neg.mean(axis=0)
    d = pos.shape[1]
    cov_pos = np.cov(pos, rowvar=False) if len(pos) > 1 else np.zeros((d, d))
    cov_neg = np.cov(neg, rowvar=False) if len(neg) > 1 else np.zeros((d, d))
    n_pos, n_neg = len(pos), len(neg)
    sw = ((n_pos - 1) * cov_pos + (n_neg - 1) * cov_neg) / max(
        n_pos + n_neg - 2, 1
    )
    tau = float(np.trace(sw) / d) if d else 0.0
    sw_reg = (1.0 - shrinkage) * sw + shrinkage * tau * np.eye(d)
    try:
        direction = np.linalg.solve(sw_reg, mu_pos - mu_neg)
    except np.linalg.LinAlgError:
        direction = mu_pos - mu_neg
    return _unit(direction)


def _unit(v: np.ndarray) -> np.ndarray:
    """Return ``v`` scaled to unit norm (no-op on near-zero input).

    :param v: Input vector.
    :type v: numpy.ndarray
    :returns: Unit-norm vector, or ``v`` unchanged if its norm underflows.
    :rtype: numpy.ndarray
    """
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


# -----------------------------------------------------------------------------
# Calibration of raw projections to [0, 1]
# -----------------------------------------------------------------------------

def _calibrate_median(
    proj_pos: np.ndarray, proj_neg: np.ndarray,
) -> tuple[float, float]:
    """Calibrate lo/hi from class medians (current estimator).

    :param proj_pos: Raw projections of positives onto the axis.
    :type proj_pos: numpy.ndarray
    :param proj_neg: Raw projections of negatives onto the axis.
    :type proj_neg: numpy.ndarray
    :returns: ``(lo, hi)`` mapping negatives' median to ``0`` and
        positives' median to ``1``.
    :rtype: tuple[float, float]
    """
    return float(np.median(proj_neg)), float(np.median(proj_pos))


def _calibrate_percentile(
    proj_pos: np.ndarray, proj_neg: np.ndarray, *, q: float = 5.0,
) -> tuple[float, float]:
    """Calibrate lo/hi from tail percentiles for fuller axis coverage.

    :param proj_pos: Raw projections of positives onto the axis.
    :type proj_pos: numpy.ndarray
    :param proj_neg: Raw projections of negatives onto the axis.
    :type proj_neg: numpy.ndarray
    :param q: Lower percentile; the upper is ``100 - q``. Lo is the
        ``q``-th percentile of negatives, hi the ``(100 - q)``-th of
        positives.
    :type q: float
    :returns: ``(lo, hi)`` calibration endpoints.
    :rtype: tuple[float, float]
    """
    return float(np.percentile(proj_neg, q)), float(
        np.percentile(proj_pos, 100.0 - q)
    )


def _apply_calibration(
    raw: np.ndarray, lo: float, hi: float,
) -> np.ndarray:
    """Map raw projections through the lo/hi calibration into ``[0, 1]``.

    :param raw: Raw projected values.
    :type raw: numpy.ndarray
    :param lo: Value mapped to ``0``.
    :type lo: float
    :param hi: Value mapped to ``1``.
    :type hi: float
    :returns: Clipped coordinates in ``[0, 1]``.
    :rtype: numpy.ndarray
    """
    span = max(hi - lo, 1e-12)
    return np.clip((raw - lo) / span, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Separability metrics
# -----------------------------------------------------------------------------

def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) identity.

    :param scores: 1-D score per example (higher = more positive).
    :type scores: numpy.ndarray
    :param labels: Binary labels (``1`` positive, ``0`` negative).
    :type labels: numpy.ndarray
    :returns: AUC in ``[0, 1]``; ``0.5`` is chance.
    :rtype: float
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties.
    _assign_tie_ranks(scores, ranks)
    r_pos = ranks[labels == 1].sum()
    n_p, n_n = len(pos), len(neg)
    return float((r_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def _assign_tie_ranks(scores: np.ndarray, ranks: np.ndarray) -> None:
    """Average ranks across tied score groups, in place.

    :param scores: Score per example.
    :type scores: numpy.ndarray
    :param ranks: Provisional integer ranks, modified in place.
    :type ranks: numpy.ndarray
    """
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    i = 0
    n = len(s_sorted)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1


def _cohens_d(scores: np.ndarray, labels: np.ndarray) -> float:
    """Cohen's d: standardised mean gap between positive and negative.

    :param scores: 1-D score per example.
    :type scores: numpy.ndarray
    :param labels: Binary labels (``1`` / ``0``).
    :type labels: numpy.ndarray
    :returns: Standardised mean difference; ``0`` is no separation.
    :rtype: float
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    pooled = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0)
    if pooled < 1e-12:
        return float("nan")
    return float((pos.mean() - neg.mean()) / pooled)


@dataclass(frozen=True)
class AxisReport:
    """Per-axis, per-estimator separability metrics.

    :param axis_name: Axis label, e.g. ``'harm'``.
    :type axis_name: str
    :param estimator: Estimator key, e.g. ``'lda_pct'``.
    :type estimator: str
    :param auc: ROC AUC of the projected score vs. label.
    :type auc: float
    :param cohens_d: Standardised mean gap between classes.
    :type cohens_d: float
    :param coord_std: Std-dev of the calibrated ``[0, 1]`` coordinate over
        the calibration corpus.
    :type coord_std: float
    :param frac_saturated: Fraction of calibration coordinates clipped to
        exactly ``0`` or ``1`` (mass piling on the box edges).
    :type frac_saturated: float
    """

    axis_name: str
    estimator: str
    auc: float
    cohens_d: float
    coord_std: float
    frac_saturated: float


# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------

_ESTIMATORS: tuple[str, ...] = ("mean_diff", "lda", "mean_diff_pct", "lda_pct")


def _fit_and_score_axis(
    emb: np.ndarray,
    scores: np.ndarray,
    cal_emb: np.ndarray,
    *,
    axis_name: str,
    threshold: float,
    shrinkage: float,
) -> tuple[dict[str, np.ndarray], list[AxisReport]]:
    """Fit all four estimators for one axis and score them.

    :param emb: Labelled embeddings for axis fitting, shape ``(M, D)``.
    :type emb: numpy.ndarray
    :param scores: Per-row benchmark score in ``[0, 1]``, shape ``(M,)``.
    :type scores: numpy.ndarray
    :param cal_emb: Calibration-corpus embeddings, shape ``(C, D)``. Used
        to measure calibrated-coordinate spread / saturation.
    :type cal_emb: numpy.ndarray
    :param axis_name: Axis label for reporting.
    :type axis_name: str
    :param threshold: Score threshold splitting positives / negatives.
    :type threshold: float
    :param shrinkage: LDA shrinkage intensity.
    :type shrinkage: float
    :returns: ``(coords, reports)`` where ``coords`` maps each estimator
        key to its calibrated coordinate over ``cal_emb`` (for the
        between-axis correlation step), and ``reports`` lists one
        :class:`AxisReport` per estimator.
    :rtype: tuple[dict[str, numpy.ndarray], list[AxisReport]]
    """
    labels = (scores >= threshold).astype(int)
    pos = emb[labels == 1]
    neg = emb[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        raise ValueError(
            f"axis {axis_name!r}: need >=2 per class "
            f"(got {len(pos)} pos, {len(neg)} neg)"
        )

    directions = {
        "mean_diff": _fit_direction_mean_diff(pos, neg),
        "lda": _fit_direction_lda(pos, neg, shrinkage=shrinkage),
    }

    coords: dict[str, np.ndarray] = {}
    reports: list[AxisReport] = []

    for est in _ESTIMATORS:
        base = "lda" if est.startswith("lda") else "mean_diff"
        w = directions[base]
        proj_fit = emb @ w
        proj_cal = cal_emb @ w
        if est.endswith("_pct"):
            lo, hi = _calibrate_percentile(proj_fit[labels == 1], proj_fit[labels == 0])
        else:
            lo, hi = _calibrate_median(proj_fit[labels == 1], proj_fit[labels == 0])

        coord_fit = _apply_calibration(proj_fit, lo, hi)
        coord_cal = _apply_calibration(proj_cal, lo, hi)
        coords[est] = coord_cal

        sat = float(np.mean((coord_cal <= 1e-9) | (coord_cal >= 1.0 - 1e-9)))
        reports.append(
            AxisReport(
                axis_name=axis_name,
                estimator=est,
                auc=_auc(proj_fit, labels),
                cohens_d=_cohens_d(proj_fit, labels),
                coord_std=float(coord_cal.std()),
                frac_saturated=sat,
            )
        )

    return coords, reports


def _gram_schmidt_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of ``b`` against the residual after removing ``a``.

    Mirrors a Gram-Schmidt decorrelation of the second axis on the first
    and reports the residual correlation (≈0 means the axes were already
    near-orthogonal in coordinate space).

    :param a: First axis coordinates, shape ``(C,)``.
    :type a: numpy.ndarray
    :param b: Second axis coordinates, shape ``(C,)``.
    :type b: numpy.ndarray
    :returns: Pearson correlation of ``a`` with the part of ``b``
        orthogonal to ``a`` (after centering).
    :rtype: float
    """
    ac = a - a.mean()
    bc = b - b.mean()
    denom = float(ac @ ac)
    if denom < 1e-12:
        return float("nan")
    beta = float(ac @ bc) / denom
    resid = bc - beta * ac
    return _pearson(ac, resid)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient.

    :param x: First sample.
    :type x: numpy.ndarray
    :param y: Second sample.
    :type y: numpy.ndarray
    :returns: Correlation in ``[-1, 1]``; ``nan`` on degenerate input.
    :rtype: float
    """
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt((xc @ xc) * (yc @ yc)))
    return float((xc @ yc) / denom) if denom > 1e-12 else float("nan")


# -----------------------------------------------------------------------------
# Toy fixtures for offline smoke testing
# -----------------------------------------------------------------------------

def _toy_inputs(
    seed: int = 0,
) -> tuple[list, Callable[[Sequence[str]], np.ndarray]]:
    """Build two toy benchmark splits and a deterministic toy encoder.

    The encoder hashes tokens into a fixed-dim space and injects a weak,
    overlapping class signal — deliberately *not* cleanly separable — so
    the diagnostic exercises the same diffuse-cloud regime seen on the
    real data.

    :param seed: RNG seed.
    :type seed: int
    :returns: ``(splits, encoder)`` ready to feed the diagnostic.
    :rtype: tuple[list, Callable[[Sequence[str]], numpy.ndarray]]
    """
    from infl_ens.data.benchmarks.base import BenchmarkSplit

    rng = np.random.default_rng(seed)
    dim = 64
    harm_dir = _unit(rng.standard_normal(dim))
    hall_dir = _unit(rng.standard_normal(dim))

    def make_split(name: str, axis: str, direction: np.ndarray, n: int):
        labels = rng.integers(0, 2, size=n)
        prompts = [f"{name}-{i}-{int(l)}" for i, l in enumerate(labels)]
        scores = labels.astype(float)
        return BenchmarkSplit(
            name=name, prompts=prompts, scores=scores,
            axis_name=axis, responses=list(prompts),
        ), direction, labels

    harm_split, _, _ = make_split("beavertails", "harm", harm_dir, 300)
    hall_split, _, _ = make_split("halueval", "hallucination", hall_dir, 300)

    def encoder(texts: Sequence[str]) -> np.ndarray:
        out = np.empty((len(texts), dim), dtype=float)
        for i, t in enumerate(texts):
            h = abs(hash(t)) % (2**32)
            local = np.random.default_rng(h).standard_normal(dim)
            label = 1.0 if t.endswith("-1") else 0.0
            axis = harm_dir if t.startswith("beavertails") else hall_dir
            # Weak class signal buried in noise => low but non-zero AUC.
            out[i] = local + 0.6 * (label - 0.5) * axis
        return out

    return [harm_split, hall_split], encoder


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _plot(
    coords_by_axis: list[tuple[str, dict[str, np.ndarray]]],
    output_stem: str,
) -> None:
    """Scatter the calibrated 2-D coordinates per estimator.

    Renders one panel per estimator showing the calibration corpus in the
    (axis-0, axis-1) plane, so the diffuse-vs-clustered difference is
    visible directly. Writes ``<stem>.pdf`` and ``<stem>.png``.

    :param coords_by_axis: Per-axis ``(axis_name, {estimator: coords})``.
    :type coords_by_axis: list[tuple[str, dict[str, numpy.ndarray]]]
    :param output_stem: Output path stem (no extension).
    :type output_stem: str
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(coords_by_axis) < 2:
        return
    (name0, c0), (name1, c1) = coords_by_axis[0], coords_by_axis[1]

    fig, axes = plt.subplots(1, len(_ESTIMATORS), figsize=(4 * len(_ESTIMATORS), 4))
    for ax, est in zip(np.atleast_1d(axes), _ESTIMATORS):
        ax.scatter(c0[est], c1[est], s=4, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(name0)
        ax.set_ylabel(name1)
        ax.set_title(est)
        ax.grid(alpha=0.3)
    fig.suptitle("Calibrated trait coordinates by axis estimator")
    fig.tight_layout()
    out = Path(output_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Config / encoder loading (real run)
# -----------------------------------------------------------------------------

def _load_from_config(
    config: Path,
) -> tuple[list, Callable[[Sequence[str]], np.ndarray], Optional[list]]:
    """Load benchmark splits, the encoder, and the calibration corpus.

    :param config: Path to a router YAML (e.g. ``safety_truth.yaml``).
    :type config: pathlib.Path
    :returns: ``(splits, encoder, calibration_corpus)``; the corpus may be
        ``None`` (then the concatenated split prompts are used).
    :rtype: tuple[list, Callable, list | None]
    :raises ImportError: If PyYAML or the data subpackage are unavailable.
    """
    import yaml

    from infl_ens.data import SentenceTransformerEncoder
    from infl_ens.data.benchmarks import load_beavertails, load_halueval

    cfg = yaml.safe_load(config.read_text())
    ts = cfg.get("trait_space", {})
    encoder_name = ts.get("encoder", "sentence-transformers/all-MiniLM-L6-v2")
    encoder = SentenceTransformerEncoder(encoder_name)

    splits = [load_beavertails(), load_halueval()]
    calibration_corpus = None  # default: concatenated prompts
    return splits, encoder, calibration_corpus


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _format_table(reports: list[AxisReport]) -> str:
    """Render the per-axis/estimator reports as a fixed-width table.

    :param reports: Flat list of reports across axes and estimators.
    :type reports: list[AxisReport]
    :returns: Printable table string.
    :rtype: str
    """
    header = (
        f"{'axis':<16}{'estimator':<16}{'AUC':>8}{'Cohen d':>10}"
        f"{'coord_std':>12}{'frac_sat':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in reports:
        lines.append(
            f"{r.axis_name:<16}{r.estimator:<16}{r.auc:>8.3f}"
            f"{r.cohens_d:>10.3f}{r.coord_std:>12.3f}{r.frac_saturated:>10.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Optional argument vector; defaults to ``sys.argv[1:]``.
    :type argv: list[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        help="Router YAML; loads BeaverTails + HaluEval and the encoder.",
    )
    parser.add_argument(
        "--toy", action="store_true",
        help="Use the built-in toy encoder/splits (offline smoke test).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Score threshold for the positive/negative split.",
    )
    parser.add_argument(
        "--shrinkage", type=float, default=0.1,
        help="LDA covariance shrinkage intensity in [0, 1].",
    )
    parser.add_argument(
        "--decorrelate", action="store_true",
        help="Also report post-Gram-Schmidt between-axis correlation.",
    )
    parser.add_argument(
        "--output-stem", default=None,
        help="If set, write <stem>.{pdf,png} of the coordinate scatter.",
    )
    args = parser.parse_args(argv)

    if not args.toy and args.config is None:
        parser.error("pass --toy or --config")

    if args.toy:
        splits, encoder = _toy_inputs()
        calibration_corpus = None
    else:
        splits, encoder, calibration_corpus = _load_from_config(args.config)

    if calibration_corpus is None:
        calibration_corpus = [p for s in splits for p in s.prompts]
    cal_emb = np.asarray(encoder(list(calibration_corpus)), dtype=float)

    all_reports: list[AxisReport] = []
    coords_by_axis: list[tuple[str, dict[str, np.ndarray]]] = []

    for split in splits:
        emb = np.asarray(
            encoder(list(split.prompts + split.responses)), dtype=float
        )
        if split.responses:
            emb = emb[len(split.prompts):]
        coords, reports = _fit_and_score_axis(
            emb, split.scores, cal_emb,
            axis_name=split.axis_name,
            threshold=args.threshold,
            shrinkage=args.shrinkage,
        )
        all_reports.extend(reports)
        coords_by_axis.append((split.axis_name, coords))

    print(_format_table(all_reports))

    if len(coords_by_axis) >= 2:
        (n0, c0), (n1, c1) = coords_by_axis[0], coords_by_axis[1]
        print()
        print("Between-axis coordinate correlation (calibration corpus):")
        for est in _ESTIMATORS:
            raw = _pearson(c0[est], c1[est])
            line = f"  {est:<16} corr({n0}, {n1}) = {raw:+.3f}"
            if args.decorrelate:
                resid = _gram_schmidt_corr(c0[est], c1[est])
                line += f"   post-GS = {resid:+.3f}"
            print(line)

    if args.output_stem is not None:
        _plot(coords_by_axis, args.output_stem)
        print(f"\nwrote {args.output_stem}.{{pdf,png}}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
