"""Allocation-rule computation for the influencer-game router.

Implements the proportional-share allocation rule

.. math::

    G_i(\\mathbf{x}, b) \\;=\\; \\frac{f_i(x_i, b)}{\\sum_{j=1}^{N} f_j(x_j, b)}

(eqn. 1, Lovett & Fu 2024) and the resulting expected utility

.. math::

    u_i(\\mathbf{x}) \\;=\\; \\sum_{b \\in \\mathbb{B}} B(b)\\, G_i(\\mathbf{x}, b)

(eqn. 2). The influence kernel is the multivariate Gaussian with a shared
covariance :math:`\\Sigma`, preserving the kernel-symmetry assumption of
Theorems 1–5.

.. note::
    AGENTS.md §4 rule 2 reserves the canonical reward to
    :mod:`infl_ens.inflgame.env`. This module computes :math:`G_i` and
    :math:`u_i` directly for routing-time convenience; when integrating with
    the trainer, route position updates through ``env`` rather than calling
    these functions twice.
"""

from __future__ import annotations

import numpy as np


def _mv_gaussian_log_kernel(
    positions: np.ndarray,
    grid: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Log multivariate-Gaussian influence values per (agent, trait) pair.

    Returns the unnormalised log-kernel

    .. math::

        \\log f_i(x_i, b) \\;=\\; -\\tfrac{1}{2}(x_i - b)^\\top \\Sigma^{-1} (x_i - b),

    omitting the agent-independent normalising constant (which cancels in
    :math:`G_i`).

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param grid: Trait points, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param cov: Shared covariance :math:`\\Sigma`, shape ``(L, L)``.
    :type cov: numpy.ndarray
    :returns: Log-kernel values, shape ``(N, K)``.
    :rtype: numpy.ndarray
    """
    diff = positions[:, None, :] - grid[None, :, :]                # (N, K, L)
    sol = np.linalg.solve(cov, diff[..., None])[..., 0]            # (N, K, L)
    quad = np.einsum("nkl,nkl->nk", diff, sol)
    return -0.5 * quad


def allocation_weights(
    positions: np.ndarray,
    grid: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Allocation matrix :math:`G_i(\\mathbf{x}, b_k)` for every (agent, trait) pair.

    Computed in log-space (softmax over agents at each trait) for numerical
    stability. The returned matrix has columns that sum to one across agents.

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param grid: Trait points, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param cov: Shared MV-Gaussian covariance :math:`\\Sigma`, shape ``(L, L)``.
        For isotropic kernels, pass ``sigma**2 * np.eye(L)``.
    :type cov: numpy.ndarray
    :returns: Allocation matrix ``G`` with ``G[i, k] = G_i(x, b_k)``;
        ``G.sum(axis=0)`` is identically 1. Shape ``(N, K)``.
    :rtype: numpy.ndarray
    """
    log_f = _mv_gaussian_log_kernel(positions, grid, cov)
    log_f_max = log_f.max(axis=0, keepdims=True)
    f = np.exp(log_f - log_f_max)
    return f / f.sum(axis=0, keepdims=True)


def expected_utilities(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Expected utility :math:`u_i(\\mathbf{x})` per agent.

    Implements :math:`u_i = \\sum_k B(b_k)\\, G_i(\\mathbf{x}, b_k)`. Utilities
    sum to one in expectation because :math:`\\sum_i G_i \\equiv 1` at every
    trait.

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param grid: Trait points, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param weights: Resource weights :math:`B(b_k)`, shape ``(K,)``, summing
        to one.
    :type weights: numpy.ndarray
    :param cov: Shared covariance :math:`\\Sigma`, shape ``(L, L)``.
    :type cov: numpy.ndarray
    :returns: Utility vector, shape ``(N,)``.
    :rtype: numpy.ndarray
    """
    G = allocation_weights(positions, grid, cov)
    return G @ weights


def strategic_routing_weights(
    positions: np.ndarray,
    grid: np.ndarray,
    cov: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-trait :math:`G_i(1 - G_i)`-weighted routing probabilities.

    The strategic gradient of agent :math:`i`'s expected utility under the
    multivariate-Gaussian influencer's game is

    .. math::

        \\nabla_{x_i} u_i \\;=\\;
        \\sum_b B(b)\\, G_i(\\mathbf{x}, b)\\,(1 - G_i(\\mathbf{x}, b))\\,
        \\Sigma^{-1}(b - x_i).

    The factor :math:`G_i(1-G_i)` down-weights traits where agent :math:`i`
    already dominates (large :math:`G_i`) and up-weights contested traits
    (intermediate :math:`G_i`). Routing queries according to

    .. math::

        p_i(\\mathbf{x}, b) \\;=\\;
        \\frac{G_i(\\mathbf{x}, b)\\,(1 - G_i(\\mathbf{x}, b))}
             {\\sum_j G_j(\\mathbf{x}, b)\\,(1 - G_j(\\mathbf{x}, b))}

    matches the strategic gradient in expectation: an agent's expected SFT
    drift becomes :math:`\\propto \\nabla_{x_i} u_i` (up to the constant
    :math:`\\Sigma^{-1}` metric and a normalising factor), rather than the
    naive proportional drift :math:`\\propto \\sum_b B(b) G_i (b - x_i)`.

    Columns sum to one across agents and the routing reduces to uniform
    (:math:`p_i \\equiv 1/N`) at clone-start (where :math:`G_i \\equiv 1/N`
    everywhere), so the strategic correction is invisible until positions
    have started to separate.

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param grid: Trait points, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param cov: Shared MV-Gaussian covariance :math:`\\Sigma`, shape ``(L, L)``.
    :type cov: numpy.ndarray
    :param eps: Tiny floor for numerical stability when all
        :math:`G_i(1-G_i)` at some trait approach zero (e.g. one agent has
        :math:`G_i \\to 1` and the rest have :math:`G_i \\to 0`); falls back
        to uniform routing at that trait.
    :type eps: float
    :returns: Strategic routing matrix ``P`` with ``P[i, k]`` the strategic
        routing probability of agent ``i`` at trait ``b_k``. Shape ``(N, K)``;
        ``P.sum(axis=0)`` is identically one.
    :rtype: numpy.ndarray
    """
    G = allocation_weights(positions, grid, cov)              # (N, K)
    w = G * (1.0 - G)                                          # (N, K)
    col_sum = w.sum(axis=0, keepdims=True)                     # (1, K)
    # Where the column sums to ~0 (degenerate dominance), fall back to
    # uniform routing rather than divide by zero.
    n_agents = positions.shape[0]
    fallback = np.full_like(w, 1.0 / n_agents)
    safe = col_sum > eps
    return np.where(safe, w / np.where(safe, col_sum, 1.0), fallback)


def top_k_allocation_weights(
    G: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Top-:math:`k` sparsified, per-query renormalised allocation weights.

    Used by the **soft (dense) routing** mode of the closed loop. Whereas
    hard routing samples a single agent per query from
    :math:`G_i(\\mathbf{x}, b)`, soft routing trains *every* agent on the
    query with a per-example loss weight equal to its (renormalised)
    allocation share. To cap the :math:`\\mathcal{O}(N)` blow-up in SFT
    compute, only the ``top_k`` most-allocated agents per query are kept;
    their weights are renormalised so each query column sums to one (mass
    per query is conserved).

    Setting ``top_k == 1`` recovers a deterministic hard assignment (one
    agent per query at unit weight — the :math:`\\arg\\max_i G_i` winner);
    ``top_k >= N`` keeps the full dense allocation.

    :param G: Allocation matrix :math:`G_i(\\mathbf{x}, b_k)`, shape
        ``(N, K)``. Columns are expected to sum to one across agents (as
        returned by :func:`allocation_weights`) but this is not required —
        the output is renormalised regardless.
    :type G: numpy.ndarray
    :param top_k: Number of highest-allocation agents to retain per query.
        Must satisfy ``top_k >= 1``.
    :type top_k: int
    :returns: Sparsified weight matrix ``W`` with the same shape as ``G``;
        each column has at most ``top_k`` non-zero entries and sums to one
        (except an all-zero query column, which is left at zero). Shape
        ``(N, K)``.
    :rtype: numpy.ndarray
    :raises ValueError: If ``top_k < 1`` or ``G`` is not 2-D.
    """
    keep = _top_k_keep_mask(G, top_k)
    w = np.where(keep, G, 0.0)
    col_sum = w.sum(axis=0, keepdims=True)
    safe = col_sum > 0.0
    return np.where(safe, w / np.where(safe, col_sum, 1.0), 0.0)


def _top_k_keep_mask(G: np.ndarray, top_k: int) -> np.ndarray:
    """Boolean support of the ``top_k`` largest entries per query column.

    Factored out of :func:`top_k_allocation_weights` so any other top-``k``
    consumer sees a bitwise identical support, ties included.

    :param G: Allocation matrix, shape ``(N, K)``.
    :type G: numpy.ndarray
    :param top_k: Agents to retain per query; ``top_k >= N`` keeps all.
    :type top_k: int
    :returns: Boolean mask with the same shape as ``G``.
    :rtype: numpy.ndarray
    :raises ValueError: If ``top_k < 1`` or ``G`` is not 2-D.
    """
    if G.ndim != 2:
        raise ValueError(f"G must be 2-D (N, K), got shape {G.shape}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    n_agents = G.shape[0]
    if top_k >= n_agents:
        return np.ones_like(G, dtype=bool)
    keep = np.zeros_like(G, dtype=bool)
    top_idx = np.argpartition(-G, top_k - 1, axis=0)[:top_k]     # (top_k, K)
    np.put_along_axis(keep, top_idx, True, axis=0)
    return keep


def matched_centroid_mass(G: np.ndarray) -> np.ndarray:
    """Gradient-matched centroid mass :math:`G_i(1 - G_i)`, dense over every query.

    .. math::

        m_i(b) \\;=\\; G_i(\\mathbf{x}, b)\\,\\bigl(1 - G_i(\\mathbf{x}, b)\\bigr)

    This is the un-normalised coefficient of the strategic gradient
    :func:`utility_gradient`. Under hard routing the sampling step already
    contributes one factor :math:`G_i` to the expected centroid drift, so
    the ``(1 - G_i)`` re-weight completes the coefficient on the routed
    subset. Under soft routing nothing is sampled, so the centroid must
    carry the full :math:`G_i(1 - G_i)` — and it may do so over the
    **entire** batch, not merely the top-``k`` queries an agent trains on:
    the position step is a weighted mean of already-projected coordinates,
    so using every query costs nothing, removes any truncation bias, and
    is the Rao–Blackwellised (lower-variance) form of the hard-routing
    estimator with the same expectation
    :math:`\\sum_b B(b)\\,G_i(1-G_i)\\,(b - x_i) \\propto \\Sigma\\,\\nabla_{x_i} u_i`.

    The mass is deliberately **not** renormalised per query: the per-agent
    sum normalisation applied by the weighted centroid is a positive
    scalar, so the drift direction stays exactly parallel to the utility
    gradient (isotropic :math:`\\Sigma`); a per-query renormaliser would
    tilt it.

    Always pass the clone-level allocation: every agent steps along its
    own gradient, twins included as competitors. Co-located clones have
    identical rows and therefore identical steps, which is how the
    theory's stable pairs persist without being enforced.

    :param G: Allocation matrix :math:`G_i(\\mathbf{x}, b_k)`, shape
        ``(N, K)``, columns summing to one across agents.
    :type G: numpy.ndarray
    :returns: Centroid mass matrix, shape ``(N, K)``.
    :rtype: numpy.ndarray
    :raises ValueError: If ``G`` is not 2-D.
    """
    if G.ndim != 2:
        raise ValueError(f"G must be 2-D (N, K), got shape {G.shape}")
    return G * (1.0 - G)


def group_allocation_weights(
    G: np.ndarray,
    group_index: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Sum per-agent allocation rows into per-group rows.

    .. math::

        G_p(\\mathbf{x}, b) \\;=\\; \\sum_{i \\in p} G_i(\\mathbf{x}, b)

    Used by **soft routing over merge groups**: when a group's members are
    co-located at one shared position and *every* group has the same size,
    the summed share equals the allocation :math:`G_p` of a single agent
    placed at that shared position in the ``n_groups``-agent game (the common
    member count cancels in the numerator and denominator of eqn. 1). With
    unequal group sizes the sum is still the group's total share of the
    query, but it is no longer the allocation of an equivalent single agent.

    Columns of the result sum to one whenever the columns of ``G`` do.

    :param G: Allocation matrix :math:`G_i(\\mathbf{x}, b_k)`, shape
        ``(N, K)``, as returned by :func:`allocation_weights`.
    :type G: numpy.ndarray
    :param group_index: Group id per agent row, shape ``(N,)``, with values
        in ``[0, n_groups)``.
    :type group_index: numpy.ndarray
    :param n_groups: Number of groups :math:`P`.
    :type n_groups: int
    :returns: Group allocation matrix, shape ``(P, K)``.
    :rtype: numpy.ndarray
    :raises ValueError: If ``G`` is not 2-D, ``group_index`` does not have
        one entry per agent row, ``n_groups < 1``, or any group id lies
        outside ``[0, n_groups)``.
    """
    if G.ndim != 2:
        raise ValueError(f"G must be 2-D (N, K), got shape {G.shape}")
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1, got {n_groups}")
    idx = np.asarray(group_index, dtype=int)
    if idx.shape != (G.shape[0],):
        raise ValueError(
            f"group_index must have shape ({G.shape[0]},), got {idx.shape}"
        )
    if idx.size and (idx.min() < 0 or idx.max() >= n_groups):
        raise ValueError(
            f"group_index values must lie in [0, {n_groups}), got "
            f"[{int(idx.min())}, {int(idx.max())}]"
        )
    out = np.zeros((n_groups, G.shape[1]), dtype=float)
    np.add.at(out, idx, G)
    return out


def empirical_utility(
    positions: np.ndarray,
    query_coords: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Empirical-pool utility :math:`\\hat u_i = \\frac{1}{M}\\sum_m G_i(\\mathbf{x}, b^\\ast_m)`.

    Whereas :func:`expected_utilities` evaluates
    :math:`u_i = \\sum_k B_\\text{KDE}(b_k)\\, G_i(\\mathbf{x}, b_k)` against the
    KDE-smoothed resource density on a discretised grid, this function
    averages :math:`G_i` against the **actual embedded query corpus**,
    treating each query as a unit-mass sample from the empirical resource
    distribution. The two coincide in the limit of (large corpus) ∩
    (fine grid) ∩ (small KDE bandwidth); they can disagree noticeably for
    small corpora, coarse grids, or wide Scott's-rule bandwidths.

    The empirical-pool utility is what the proportional routing mechanism
    actually consumes: for a batch of :math:`M` queries IID from the pool,

    .. math::

        \\mathbb{E}\\!\\left[\\frac{N_i}{M}\\right]
        \\;=\\; \\hat u_i(\\mathbf{x}),

    where :math:`N_i` is the count of queries routed to agent :math:`i`
    under ``policy='proportional'``.

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param query_coords: Embedded query coordinates in trait space, shape
        ``(M, L)``. Typically obtained via
        ``trait_space.project(list_of_query_strings)``.
    :type query_coords: numpy.ndarray
    :param cov: Shared MV-Gaussian covariance :math:`\\Sigma`, shape
        ``(L, L)``.
    :type cov: numpy.ndarray
    :returns: Per-agent empirical utility, shape ``(N,)``. Sums to one.
    :rtype: numpy.ndarray
    :raises ValueError: If ``query_coords`` is empty.
    """
    if query_coords.shape[0] == 0:
        raise ValueError("query_coords must be non-empty")
    G = allocation_weights(positions, query_coords, cov)        # (N, M)
    return G.mean(axis=1)


def utility_gradient(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Closed-form gradient :math:`\\nabla_{x_i} u_i(\\mathbf{x})`.

    Using

    .. math::

        \\nabla_{x_i} \\log f_i \\;=\\; -\\Sigma^{-1}(x_i - b),
        \\quad
        \\nabla_{x_i} G_i \\;=\\; G_i\\,(1 - G_i)\\, \\nabla_{x_i} \\log f_i,

    we have

    .. math::

        \\nabla_{x_i} u_i \\;=\\; \\sum_k B(b_k)\\,
            G_i(\\mathbf{x}, b_k)\\,(1 - G_i(\\mathbf{x}, b_k))\\,
            \\big(-\\Sigma^{-1}(x_i - b_k)\\big).

    Note this is the gradient of *only player i*'s own utility with respect
    to *their own* position — what the gradient-ascent dynamics
    :math:`\\nabla_t x_i = \\nabla_{x_i} u_i` consume.

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param grid: Trait points, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param weights: Resource weights :math:`B(b_k)`, shape ``(K,)``.
    :type weights: numpy.ndarray
    :param cov: Shared covariance :math:`\\Sigma`, shape ``(L, L)``.
    :type cov: numpy.ndarray
    :returns: Stacked gradient matrix, shape ``(N, L)``; row ``i`` is
        :math:`\\nabla_{x_i} u_i`.
    :rtype: numpy.ndarray
    """
    diff = positions[:, None, :] - grid[None, :, :]                  # (N, K, L)
    sol = np.linalg.solve(cov, diff[..., None])[..., 0]              # (N, K, L)
    G = allocation_weights(positions, grid, cov)                     # (N, K)
    coeff = weights[None, :] * G * (1.0 - G)                         # (N, K)
    return -np.einsum("nk,nkl->nl", coeff, sol)
