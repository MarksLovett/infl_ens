"""Utilities for working with empirical resource distributions :math:`B(b)`.

These helpers are kernel-agnostic: they operate on a ``(grid, weights)`` pair
returned by :class:`infl_ens.data.trait_space.TraitSpace` and produce scalar
or vector summaries used in the stability analysis of the symmetric Nash
equilibrium (Corollary 8 and its multivariate generalisation, Lovett & Fu 2024).

Per AGENTS.md §4 rule 4, this module imports no siblings — only NumPy.
"""

from __future__ import annotations

import numpy as np


def weighted_mean(grid: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Resource-weighted mean :math:`\\mathbb{E}_B[b]`.

    :param grid: Trait grid points, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param weights: Probability weights at each grid point, shape ``(K,)``.
        Renormalised internally if they do not sum to one.
    :type weights: numpy.ndarray
    :returns: Mean vector, shape ``(L,)``.
    :rtype: numpy.ndarray
    """
    w = weights / weights.sum()
    return np.einsum("k,kl->l", w, grid)


def weighted_covariance(grid: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Resource-weighted covariance :math:`\\Sigma_B`.

    :param grid: Trait grid points, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param weights: Probability weights at each grid point, shape ``(K,)``.
    :type weights: numpy.ndarray
    :returns: Covariance matrix, shape ``(L, L)``.
    :rtype: numpy.ndarray
    """
    w = weights / weights.sum()
    mu = np.einsum("k,kl->l", w, grid)
    c = grid - mu
    return np.einsum("k,kl,km->lm", w, c, c)


def gaussian_stability_threshold(
    n_agents: int,
    grid: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Closed-form first-bifurcation threshold :math:`\\sigma_0^*`.

    Implements the multivariate generalisation of Corollary 8:

    .. math::

        \\sigma_0^* \\;=\\; \\sqrt{\\frac{N - 2}{N - 1}\\,\\Lambda_{\\max}(\\Sigma_B)},

    where :math:`\\Lambda_{\\max}` is the largest eigenvalue of the resource
    covariance. Below this competitive reach the symmetric Nash equilibrium
    of the MV-Gaussian influencer's game destabilises along the principal
    direction of :math:`\\Sigma_B`.

    For ``n_agents == 2`` the threshold is identically zero (Theorem 5: the
    symmetric equilibrium is stable in every dimension).

    :param n_agents: Number of agents :math:`N`. Must satisfy ``N >= 2``.
    :type n_agents: int
    :param grid: Trait grid points, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param weights: Resource weights at each grid point.
    :type weights: numpy.ndarray
    :returns: Critical competitive reach :math:`\\sigma_0^*`.
    :rtype: float
    :raises ValueError: If ``n_agents < 2``.
    """
    if n_agents < 2:
        raise ValueError(f"n_agents must be >= 2, got {n_agents}")
    if n_agents == 2:
        return 0.0
    cov = weighted_covariance(grid, weights)
    lam_max = float(np.linalg.eigvalsh(cov)[-1])
    return float(np.sqrt((n_agents - 2) / (n_agents - 1) * lam_max))
