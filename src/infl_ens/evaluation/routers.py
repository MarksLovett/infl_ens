"""Pure routing policies for comparing a fixed family of experts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Mapping, Sequence

import numpy as np

from infl_ens.inflgame.router.allocation import allocation_weights


def normalize_routing_weights(weights: np.ndarray) -> np.ndarray:
    """Validate and row-normalize an ``(M, K)`` routing matrix.

    :param weights: Nonnegative routing scores.
    :type weights: numpy.ndarray
    :returns: Row-stochastic routing weights.
    :rtype: numpy.ndarray
    :raises ValueError: If the input is not finite, nonnegative, or 2-D.
    """
    out = np.asarray(weights, dtype=float)
    if out.ndim != 2:
        raise ValueError(f"routing weights must be 2-D (M, K), got {out.shape}")
    if not np.isfinite(out).all() or np.any(out < 0):
        raise ValueError("routing weights must be finite and nonnegative")
    row_sum = out.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("every routing row must have positive mass")
    return out / row_sum


def gaussian_router(
    positions: np.ndarray,
    coordinates: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    """Compute the Gaussian allocation router in row-stochastic form.

    :param positions: Expert positions, shape ``(K, L)``.
    :type positions: numpy.ndarray
    :param coordinates: Prompt coordinates, shape ``(M, L)``.
    :type coordinates: numpy.ndarray
    :param sigma: Isotropic Gaussian reach.
    :type sigma: float
    :returns: Routing weights, shape ``(M, K)``.
    :rtype: numpy.ndarray
    :raises ValueError: If ``sigma`` is not positive.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    pos = np.asarray(positions, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    if pos.ndim != 2 or coords.ndim != 2 or pos.shape[1] != coords.shape[1]:
        raise ValueError("positions and coordinates must be aligned 2-D matrices")
    cov = float(sigma) ** 2 * np.eye(pos.shape[1])
    return normalize_routing_weights(allocation_weights(pos, coords, cov).T)


def nearest_centroid_router(
    centroids: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Route each prompt to its nearest centroid.

    :param centroids: Expert centroids, shape ``(K, L)``.
    :type centroids: numpy.ndarray
    :param coordinates: Prompt coordinates, shape ``(M, L)``.
    :type coordinates: numpy.ndarray
    :returns: One-hot routing matrix, shape ``(M, K)``.
    :rtype: numpy.ndarray
    """
    centers = np.asarray(centroids, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    if centers.ndim != 2 or coords.ndim != 2 or centers.shape[1] != coords.shape[1]:
        raise ValueError("centroids and coordinates must be aligned 2-D matrices")
    nearest = np.argmin(((coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    out = np.zeros((coords.shape[0], centers.shape[0]), dtype=float)
    out[np.arange(coords.shape[0]), nearest] = 1.0
    return out


def metadata_router(
    labels: Sequence[str],
    expert_names: Sequence[str],
    *,
    mapping: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Construct a one-hot benchmark-metadata router.

    :param labels: Benchmark label per record.
    :type labels: Sequence[str]
    :param expert_names: Available expert names.
    :type expert_names: Sequence[str]
    :param mapping: Optional benchmark-to-expert mapping.  Without it, an
        expert must be named either exactly as the benchmark or
        ``domain-<benchmark>``.
    :type mapping: Mapping[str, str] | None
    :returns: One-hot routing matrix, shape ``(M, K)``.
    :rtype: numpy.ndarray
    :raises KeyError: If a label has no corresponding expert.
    """
    names = list(expert_names)
    index = {name: i for i, name in enumerate(names)}
    out = np.zeros((len(labels), len(names)), dtype=float)
    for row, label in enumerate(labels):
        target = mapping.get(label) if mapping is not None else None
        candidates = [target, label, f"domain-{label}"]
        selected = next((name for name in candidates if name in index), None)
        if selected is None:
            raise KeyError(f"no metadata expert for benchmark {label!r}")
        out[row, index[selected]] = 1.0
    return out


def uniform_router(n_records: int, n_experts: int) -> np.ndarray:
    """Return uniform routing weights.

    :param n_records: Number of records.
    :type n_records: int
    :param n_experts: Number of experts.
    :type n_experts: int
    :returns: Matrix shape ``(n_records, n_experts)``.
    :rtype: numpy.ndarray
    """
    if n_records < 0 or n_experts < 1:
        raise ValueError("n_records must be nonnegative and n_experts positive")
    return np.full((n_records, n_experts), 1.0 / n_experts, dtype=float)


def permute_router(weights: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    """Apply an expert-column permutation to a routing matrix.

    :param weights: Routing weights, shape ``(M, K)``.
    :type weights: numpy.ndarray
    :param permutation: A permutation of ``range(K)``.
    :type permutation: Sequence[int]
    :returns: Permuted row-stochastic matrix.
    :rtype: numpy.ndarray
    """
    base = normalize_routing_weights(weights)
    order = np.asarray(permutation, dtype=int)
    if order.shape != (base.shape[1],) or set(order.tolist()) != set(range(base.shape[1])):
        raise ValueError("permutation must contain every expert column exactly once")
    return base[:, order]


def hindsight_oracle_router(expert_nll: np.ndarray) -> np.ndarray:
    """Select the minimum-NLL expert independently for every record.

    :param expert_nll: Expert NLL matrix, shape ``(M, K)``.
    :type expert_nll: numpy.ndarray
    :returns: One-hot hindsight routing matrix.
    :rtype: numpy.ndarray
    """
    nll = np.asarray(expert_nll, dtype=float)
    if nll.ndim != 2 or not np.isfinite(nll).all():
        raise ValueError("expert_nll must be a finite 2-D matrix")
    winner = np.argmin(nll, axis=1)
    out = np.zeros_like(nll, dtype=float)
    out[np.arange(nll.shape[0]), winner] = 1.0
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class LinearRouter:
    """A fitted linear prompt router.

    :param coefficients: Matrix shape ``(L, K)``.
    :type coefficients: numpy.ndarray
    :param intercept: Vector shape ``(K,)``.
    :type intercept: numpy.ndarray
    :param mean: Feature standardization mean.
    :type mean: numpy.ndarray
    :param scale: Feature standardization scale.
    :type scale: numpy.ndarray
    :param objective: ``classification``, ``soft``, or ``regression``.
    :type objective: str
    """

    coefficients: np.ndarray
    intercept: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    objective: str

    def weights(self, coordinates: np.ndarray) -> np.ndarray:
        """Predict row-stochastic expert weights.

        :param coordinates: Prompt coordinates, shape ``(M, L)``.
        :type coordinates: numpy.ndarray
        :returns: Routing weights, shape ``(M, K)``.
        :rtype: numpy.ndarray
        """
        x = (np.asarray(coordinates, dtype=float) - self.mean) / self.scale
        raw = x @ self.coefficients + self.intercept
        if self.objective == "regression":
            raw = -raw
        return normalize_routing_weights(_softmax(raw))


def fit_linear_router(
    coordinates: np.ndarray,
    expert_nll: np.ndarray,
    *,
    objective: str = "classification",
    temperature: float = 0.1,
    l2: float = 1e-3,
    learning_rate: float = 0.05,
    max_steps: int = 2000,
    seed: int = 0,
) -> LinearRouter:
    """Fit a deterministic linear router from validation NLL targets.

    :param coordinates: Validation trait coordinates, shape ``(M, L)``.
    :type coordinates: numpy.ndarray
    :param expert_nll: Validation expert NLL, shape ``(M, K)``.
    :type expert_nll: numpy.ndarray
    :param objective: ``classification``, ``soft``, or ``regression``.
    :type objective: str
    :param temperature: Soft-target temperature for ``objective='soft'``.
    :type temperature: float
    :param l2: L2 penalty.
    :type l2: float
    :param learning_rate: Gradient-descent learning rate.
    :type learning_rate: float
    :param max_steps: Maximum full-batch optimization steps.
    :type max_steps: int
    :param seed: Initialization seed.
    :type seed: int
    :returns: Fitted router.
    :rtype: LinearRouter
    """
    x_raw = np.asarray(coordinates, dtype=float)
    nll = np.asarray(expert_nll, dtype=float)
    if x_raw.ndim != 2 or nll.ndim != 2 or x_raw.shape[0] != nll.shape[0]:
        raise ValueError("coordinates and expert_nll must be aligned 2-D matrices")
    if objective not in {"classification", "soft", "regression"}:
        raise ValueError("objective must be classification, soft, or regression")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    mean = x_raw.mean(axis=0)
    scale = x_raw.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    x = (x_raw - mean) / scale
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1e-3, size=(x.shape[1], nll.shape[1]))
    b = np.zeros(nll.shape[1], dtype=float)
    if objective == "classification":
        target = np.zeros_like(nll)
        target[np.arange(len(nll)), np.argmin(nll, axis=1)] = 1.0
    elif objective == "soft":
        target = _softmax(-(nll - nll.min(axis=1, keepdims=True)) / temperature)
    else:
        target = nll
    for step in range(max_steps):
        pred = x @ w + b
        if objective == "regression":
            error = pred - target
            grad_w = x.T @ error / len(x) + l2 * w
            grad_b = error.mean(axis=0)
        else:
            error = _softmax(pred) - target
            grad_w = x.T @ error / len(x) + l2 * w
            grad_b = error.mean(axis=0)
        rate = learning_rate / np.sqrt(1.0 + step / 100.0)
        w -= rate * grad_w
        b -= rate * grad_b
        if max(float(np.abs(grad_w).max()), float(np.abs(grad_b).max())) < 1e-8:
            break
    return LinearRouter(w, b, mean, scale, objective)


def stratified_folds(
    labels: Sequence[str],
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> list[np.ndarray]:
    """Build deterministic approximately stratified validation folds.

    :param labels: Stratum label per record.
    :type labels: Sequence[str]
    :param n_folds: Number of folds.
    :type n_folds: int
    :param seed: Shuffle seed.
    :type seed: int
    :returns: Boolean holdout mask per fold.
    :rtype: list[numpy.ndarray]
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    label_array = np.asarray(labels, dtype=str)
    folds = [np.zeros(len(labels), dtype=bool) for _ in range(n_folds)]
    rng = np.random.default_rng(seed)
    for label in sorted(set(labels)):
        rows = np.flatnonzero(label_array == label)
        rows = rows[rng.permutation(len(rows))]
        for offset, row in enumerate(rows):
            folds[offset % n_folds][row] = True
    return folds


def fit_best_linear_router(
    coordinates: np.ndarray,
    expert_nll: np.ndarray,
    bench_labels: Sequence[str],
    *,
    objective: str,
    temperatures: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0),
    l2_values: Sequence[float] = (0.0, 1e-4, 1e-3, 1e-2),
    n_folds: int = 5,
    seed: int = 0,
    max_steps: int = 2000,
) -> tuple[LinearRouter, dict[str, float | str]]:
    """Select linear-router hyperparameters by stratified validation CV.

    :param coordinates: Validation coordinates, shape ``(M, L)``.
    :type coordinates: numpy.ndarray
    :param expert_nll: Validation expert NLL, shape ``(M, K)``.
    :type expert_nll: numpy.ndarray
    :param bench_labels: Validation benchmark labels.
    :type bench_labels: Sequence[str]
    :param objective: ``classification``, ``soft``, or ``regression``.
    :type objective: str
    :param temperatures: Soft-target temperature grid.
    :type temperatures: Sequence[float]
    :param l2_values: L2 grid.
    :type l2_values: Sequence[float]
    :param n_folds: Cross-validation folds.
    :type n_folds: int
    :param seed: Fitting seed.
    :type seed: int
    :param max_steps: Optimization steps per fit.
    :type max_steps: int
    :returns: ``(refitted_router, selection_metadata)``.
    :rtype: tuple[LinearRouter, dict[str, float | str]]
    """
    x = np.asarray(coordinates, dtype=float)
    nll = np.asarray(expert_nll, dtype=float)
    folds = stratified_folds(bench_labels, n_folds=n_folds, seed=seed)
    tau_grid = tuple(temperatures) if objective == "soft" else (1.0,)
    best: tuple[float, float, float] | None = None
    for temperature in tau_grid:
        for l2 in l2_values:
            losses: list[float] = []
            for fold_index, holdout in enumerate(folds):
                train = ~holdout
                model = fit_linear_router(
                    x[train],
                    nll[train],
                    objective=objective,
                    temperature=float(temperature),
                    l2=float(l2),
                    max_steps=max_steps,
                    seed=seed + fold_index,
                )
                weights = model.weights(x[holdout])
                losses.append(float(np.mean(np.sum(weights * nll[holdout], axis=1))))
            candidate = (float(np.mean(losses)), float(temperature), float(l2))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    cv_nll, temperature, l2 = best
    fitted = fit_linear_router(
        x,
        nll,
        objective=objective,
        temperature=temperature,
        l2=l2,
        max_steps=max_steps,
        seed=seed,
    )
    return fitted, {
        "objective": objective,
        "temperature": temperature,
        "l2": l2,
        "cv_expected_nll": cv_nll,
    }


def validation_selected_permutation(
    base_weights: np.ndarray,
    expert_nll: np.ndarray,
) -> tuple[tuple[int, ...], dict[str, float]]:
    """Enumerate expert permutations and select by validation expected NLL.

    :param base_weights: Base validation router, shape ``(M, K)``.
    :type base_weights: numpy.ndarray
    :param expert_nll: Validation NLL matrix, shape ``(M, K)``.
    :type expert_nll: numpy.ndarray
    :returns: Best permutation and distribution summary.
    :rtype: tuple[tuple[int, ...], dict[str, float]]
    """
    base = normalize_routing_weights(base_weights)
    nll = np.asarray(expert_nll, dtype=float)
    if base.shape != nll.shape:
        raise ValueError("base_weights and expert_nll must have identical shape")
    values: list[tuple[float, tuple[int, ...]]] = []
    for permutation in permutations(range(base.shape[1])):
        value = float(np.mean(np.sum(base[:, permutation] * nll, axis=1)))
        values.append((value, tuple(int(x) for x in permutation)))
    values.sort()
    scores = np.asarray([value for value, _ in values], dtype=float)
    return values[0][1], {
        "n_permutations": float(len(values)),
        "min_expected_nll": float(scores.min()),
        "median_expected_nll": float(np.median(scores)),
        "max_expected_nll": float(scores.max()),
    }


def fit_simplex_stacking(
    expert_nll: np.ndarray,
    token_counts: np.ndarray,
    *,
    learning_rate: float = 0.1,
    max_steps: int = 2000,
) -> np.ndarray:
    """Fit global sequence-mixture weights on validation records.

    :param expert_nll: Validation NLL matrix, shape ``(M, K)``.
    :type expert_nll: numpy.ndarray
    :param token_counts: Supervised token counts, shape ``(M,)``.
    :type token_counts: numpy.ndarray
    :param learning_rate: Logit-gradient learning rate.
    :type learning_rate: float
    :param max_steps: Optimization steps.
    :type max_steps: int
    :returns: Simplex weight vector, shape ``(K,)``.
    :rtype: numpy.ndarray
    """
    nll = np.asarray(expert_nll, dtype=float)
    counts = np.asarray(token_counts, dtype=float)
    if nll.ndim != 2 or counts.shape != (nll.shape[0],):
        raise ValueError("expert_nll and token_counts have incompatible shapes")
    log_prob = -np.clip(counts, 1.0, None)[:, None] * nll
    row_weight = 1.0 / np.clip(counts, 1.0, None)
    row_weight = row_weight / row_weight.sum()
    logits = np.zeros(nll.shape[1], dtype=float)
    for step in range(max_steps):
        prior = _softmax(logits[None, :])[0]
        joint = log_prob + np.log(np.clip(prior, np.finfo(float).tiny, None))[None, :]
        posterior = _softmax(joint)
        grad = ((prior[None, :] - posterior) * row_weight[:, None]).sum(axis=0)
        logits -= learning_rate / np.sqrt(1.0 + step / 100.0) * grad
        if float(np.abs(grad).max()) < 1e-9:
            break
    return _softmax(logits[None, :])[0]


def low_support_mask(
    coordinates: np.ndarray,
    labels: Sequence[str],
    train_centroids: Mapping[str, np.ndarray],
    train_mean: np.ndarray,
    train_scale: np.ndarray,
    *,
    fraction: float = 0.2,
) -> np.ndarray:
    """Select the furthest fixed fraction within each benchmark.

    :param coordinates: Evaluation coordinates, shape ``(M, L)``.
    :type coordinates: numpy.ndarray
    :param labels: Benchmark labels aligned with rows.
    :type labels: Sequence[str]
    :param train_centroids: Training centroid for every benchmark.
    :type train_centroids: Mapping[str, numpy.ndarray]
    :param train_mean: Global training-coordinate mean.
    :type train_mean: numpy.ndarray
    :param train_scale: Global training-coordinate standard deviation.
    :type train_scale: numpy.ndarray
    :param fraction: Fraction selected within each benchmark.
    :type fraction: float
    :returns: Boolean selection mask.
    :rtype: numpy.ndarray
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    coords = np.asarray(coordinates, dtype=float)
    scale = np.where(np.asarray(train_scale, dtype=float) > 1e-12, train_scale, 1.0)
    standardized = (coords - np.asarray(train_mean, dtype=float)) / scale
    label_array = np.asarray(labels, dtype=str)
    out = np.zeros(len(labels), dtype=bool)
    for label in sorted(set(labels)):
        rows = np.flatnonzero(label_array == label)
        centroid = (np.asarray(train_centroids[label], dtype=float) - train_mean) / scale
        distance = np.linalg.norm(standardized[rows] - centroid, axis=1)
        keep = max(1, int(np.ceil(fraction * len(rows))))
        chosen = rows[np.argsort(distance, kind="stable")[-keep:]]
        out[chosen] = True
    return out


def build_validation_router_suite(
    validation_coordinates: np.ndarray,
    validation_nll: np.ndarray,
    validation_token_counts: np.ndarray,
    validation_labels: Sequence[str],
    validation_native_weights: np.ndarray,
    evaluation_coordinates: np.ndarray,
    evaluation_labels: Sequence[str],
    evaluation_native_weights: np.ndarray,
    expert_names: Sequence[str],
    *,
    seed: int = 0,
    max_steps: int = 750,
    include_centroid: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Fit the common validation-only router suite and predict evaluation weights.

    :param validation_coordinates: Validation trait coordinates.
    :type validation_coordinates: numpy.ndarray
    :param validation_nll: Validation expert NLL matrix.
    :type validation_nll: numpy.ndarray
    :param validation_token_counts: Validation supervised-token counts.
    :type validation_token_counts: numpy.ndarray
    :param validation_labels: Validation benchmark labels.
    :type validation_labels: Sequence[str]
    :param validation_native_weights: Native validation router.
    :type validation_native_weights: numpy.ndarray
    :param evaluation_coordinates: Held-out evaluation coordinates.
    :type evaluation_coordinates: numpy.ndarray
    :param evaluation_labels: Held-out benchmark labels.
    :type evaluation_labels: Sequence[str]
    :param evaluation_native_weights: Native held-out router.
    :type evaluation_native_weights: numpy.ndarray
    :param expert_names: Expert column names.
    :type expert_names: Sequence[str]
    :param seed: Deterministic fit seed.
    :type seed: int
    :param max_steps: Optimization steps per linear fit.
    :type max_steps: int
    :param include_centroid: Include soft and hard versions of the supplied
        native geometric router. Disable for neural mixtures without fixed
        expert centroids.
    :type include_centroid: bool
    :returns: ``(evaluation_weight_matrices, fit_metadata)``.
    :rtype: tuple[dict[str, numpy.ndarray], dict[str, object]]
    """
    suites: dict[str, np.ndarray] = {}
    if include_centroid:
        suites["centroid_soft"] = normalize_routing_weights(
            evaluation_native_weights
        )
        hard = np.zeros_like(evaluation_native_weights, dtype=float)
        hard[
            np.arange(len(hard)),
            np.argmax(evaluation_native_weights, axis=1),
        ] = 1.0
        suites["centroid_hard"] = hard
    metadata: dict[str, object] = {}

    for objective in ("classification", "soft", "regression"):
        model, selected = fit_best_linear_router(
            validation_coordinates,
            validation_nll,
            validation_labels,
            objective=objective,
            seed=seed,
            max_steps=max_steps,
        )
        name = f"fitted_{objective}"
        suites[name] = model.weights(evaluation_coordinates)
        metadata[name] = selected

    stacking = fit_simplex_stacking(validation_nll, validation_token_counts)
    suites["simplex_stacking"] = np.broadcast_to(
        stacking, (len(evaluation_coordinates), len(stacking)),
    ).copy()
    metadata["simplex_stacking"] = {"weights": stacking.tolist()}

    permutation, distribution = validation_selected_permutation(
        validation_native_weights, validation_nll,
    )
    suites["validation_selected_permutation"] = permute_router(
        evaluation_native_weights, permutation,
    )
    metadata["validation_selected_permutation"] = {
        "permutation": list(permutation),
        **distribution,
    }

    try:
        validation_metadata = metadata_router(validation_labels, expert_names)
        suites["metadata"] = metadata_router(evaluation_labels, expert_names)
    except KeyError:
        validation_metadata = None
    if validation_metadata is not None:
        pseudo_nll = 1.0 - validation_metadata
        label_model, selected = fit_best_linear_router(
            validation_coordinates,
            pseudo_nll,
            validation_labels,
            objective="classification",
            seed=seed,
            max_steps=max_steps,
        )
        suites["prompt_to_label"] = label_model.weights(evaluation_coordinates)
        metadata["prompt_to_label"] = selected
    return suites, metadata


__all__ = [
    "LinearRouter",
    "build_validation_router_suite",
    "fit_linear_router",
    "fit_best_linear_router",
    "fit_simplex_stacking",
    "gaussian_router",
    "hindsight_oracle_router",
    "low_support_mask",
    "metadata_router",
    "nearest_centroid_router",
    "normalize_routing_weights",
    "permute_router",
    "stratified_folds",
    "uniform_router",
    "validation_selected_permutation",
]
