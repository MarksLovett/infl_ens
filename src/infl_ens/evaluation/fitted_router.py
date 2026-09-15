"""A fitted prompt-level router over trait-space coordinates.

The game router places experts in trait space and allocates prompts by
Gaussian affinity; it is never fitted to expert quality. This module is the
predictive counterpart: a multinomial logistic regression from a prompt's
trait coordinates to the expert with the **lowest NLL** on that prompt,
fitted on the validation pool and scored on test. It is the prompt-level
analogue of the learned gate in PEFT mixture-of-experts baselines and is
reported for every routed arm alongside the oracle and the game router.

Pure numpy: L2-regularised softmax regression trained by full-batch
gradient descent with a fixed step, deterministic given its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FittedRouter:
    """Softmax regression weights over trait coordinates.

    :param weights: Coefficients, shape ``(L, K)``.
    :type weights: numpy.ndarray
    :param bias: Intercepts, shape ``(K,)``.
    :type bias: numpy.ndarray
    :param mean: Feature means used for standardisation, shape ``(L,)``.
    :type mean: numpy.ndarray
    :param scale: Feature scales used for standardisation, shape ``(L,)``.
    :type scale: numpy.ndarray
    :param train_accuracy: Argmax accuracy on the fitting set.
    :type train_accuracy: float
    :param n_iter: Gradient steps taken.
    :type n_iter: int
    """

    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    train_accuracy: float
    n_iter: int

    @property
    def n_experts(self) -> int:
        """Number of output classes."""
        return int(self.bias.shape[0])

    def predict_proba(self, coords: np.ndarray) -> np.ndarray:
        """Class probabilities for each prompt.

        :param coords: Trait coordinates, shape ``(M, L)``.
        :type coords: numpy.ndarray
        :returns: Probabilities, shape ``(M, K)``, rows summing to one.
        :rtype: numpy.ndarray
        """
        x = (np.asarray(coords, dtype=float) - self.mean) / self.scale
        return _softmax(x @ self.weights + self.bias)

    def predict(self, coords: np.ndarray) -> np.ndarray:
        """Argmax expert index per prompt.

        :param coords: Trait coordinates, shape ``(M, L)``.
        :type coords: numpy.ndarray
        :returns: Expert indices, shape ``(M,)``.
        :rtype: numpy.ndarray
        """
        return np.argmax(self.predict_proba(coords), axis=1)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_argmin_router(
    coords: np.ndarray,
    nll: np.ndarray,
    *,
    l2: float = 1e-2,
    learning_rate: float = 0.5,
    n_iter: int = 2000,
    tol: float = 1e-7,
) -> FittedRouter:
    """Fit a softmax router predicting the argmin-NLL expert per prompt.

    Minimises the mean cross-entropy between the router's distribution and
    the one-hot argmin-NLL label, plus :math:`\\tfrac{\\lambda}{2}\\|W\\|^2`.
    Features are standardised per axis; the fit is full-batch gradient
    descent with a fixed step and stops early when the loss change falls
    below ``tol``.

    :param coords: Validation trait coordinates, shape ``(M, L)``.
    :type coords: numpy.ndarray
    :param nll: Per-prompt expert NLL, shape ``(M, K)``.
    :type nll: numpy.ndarray
    :param l2: Ridge penalty on the weights (not the bias).
    :type l2: float
    :param learning_rate: Gradient step size.
    :type learning_rate: float
    :param n_iter: Maximum number of steps.
    :type n_iter: int
    :param tol: Early-stopping threshold on the loss decrease.
    :type tol: float
    :returns: The fitted router.
    :rtype: FittedRouter
    :raises ValueError: On shape mismatches or fewer than two experts.
    """
    x = np.asarray(coords, dtype=float)
    nll = np.asarray(nll, dtype=float)
    if x.ndim != 2 or nll.ndim != 2 or x.shape[0] != nll.shape[0]:
        raise ValueError(
            f"coords {x.shape} and nll {nll.shape} must be 2-D with matching rows"
        )
    m, n_feat = x.shape
    k = nll.shape[1]
    if k < 2:
        raise ValueError("need at least two experts to fit a router")
    if m == 0:
        raise ValueError("cannot fit a router on an empty pool")

    labels = np.argmin(nll, axis=1)
    y = np.zeros((m, k), dtype=float)
    y[np.arange(m), labels] = 1.0

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    xs = (x - mean) / scale

    w = np.zeros((n_feat, k), dtype=float)
    b = np.zeros(k, dtype=float)
    prev = np.inf
    steps = 0
    while steps < n_iter:
        steps += 1
        p = _softmax(xs @ w + b)
        loss = -np.mean(np.sum(y * np.log(p + 1e-12), axis=1)) + 0.5 * l2 * np.sum(w * w)
        g = (p - y) / m
        w -= learning_rate * (xs.T @ g + l2 * w)
        b -= learning_rate * g.sum(axis=0)
        if prev - loss < tol:
            break
        prev = loss

    acc = float(np.mean(np.argmax(_softmax(xs @ w + b), axis=1) == labels))
    return FittedRouter(
        weights=w, bias=b, mean=mean, scale=scale, train_accuracy=acc, n_iter=steps,
    )


__all__ = ["FittedRouter", "fit_argmin_router"]
