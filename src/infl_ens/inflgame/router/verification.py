"""Numerical verification of routing drift vs strategic gradient.

Covers the hard-routing rules (strategic, canonical + ``(1-G)`` re-weight,
canonical naive) and the soft (dense, top-``k``) rules: the historical
renormalised-share centroid (``position_update: naive``) and the
theory-matched dense :math:`G_i(1-G_i)` mass over the whole batch
(``position_update: theory_matched``), per agent — including co-located
clone pairs, whose members take identical independent steps. The
Monte-Carlo section also shows the variance reduction of the dense rule
over the hard ``(1-G)`` rule at equal expectation.
"""

from __future__ import annotations

import numpy as np

from infl_ens.inflgame.router.allocation import (
    allocation_weights,
    group_allocation_weights,
    matched_centroid_mass,
    strategic_routing_weights,
    top_k_allocation_weights,
    utility_gradient,
)


def _build_landscape(
    n_grid: int = 25,
    *,
    sigma_b: float = 0.18,
    centers: tuple[tuple[float, float], ...] = ((0.25, 0.25), (0.75, 0.75)),
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 2-D bimodal Gaussian resource landscape on the unit square.

    :param n_grid: Number of grid points per axis.
    :type n_grid: int
    :param sigma_b: Width of each resource mode.
    :type sigma_b: float
    :param centers: Mode centres inside :math:`[0, 1]^2`.
    :type centers: tuple[tuple[float, float], ...]
    :returns: ``(grid, weights)`` with ``grid`` shape ``(K, 2)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    axes = np.linspace(0.0, 1.0, n_grid)
    gx, gy = np.meshgrid(axes, axes, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    density = np.zeros(grid.shape[0])
    for c in centers:
        diff = grid - np.array(c)
        density += np.exp(-0.5 * (diff ** 2).sum(axis=-1) / sigma_b ** 2)
    weights = density / density.sum()
    return grid, weights


def _normalised_drift(
    positions: np.ndarray,
    grid: np.ndarray,
    mass: np.ndarray,
) -> np.ndarray:
    """Mass-weighted centroid minus position, per agent row of ``mass``."""
    Z = mass.sum(axis=1, keepdims=True)
    diff = grid[None, :, :] - positions[:, None, :]
    return (mass[..., None] * diff).sum(axis=1) / np.maximum(Z, 1e-30)


def _expected_drift_strategic(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Expected centroid-blend drift under strategic routing."""
    P = strategic_routing_weights(positions, grid, cov)
    return _normalised_drift(positions, grid, weights[None, :] * P)


def _expected_drift_canonical_reweighted(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Expected drift under canonical routing with per-query :math:`(1-G_i)` weighting."""
    G = allocation_weights(positions, grid, cov)
    return _normalised_drift(positions, grid, weights[None, :] * G * (1.0 - G))


def _expected_drift_canonical_naive(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Expected drift under canonical routing with an unweighted centroid SFT."""
    G = allocation_weights(positions, grid, cov)
    return _normalised_drift(positions, grid, weights[None, :] * G)


def _expected_drift_topk_naive(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    """Expected soft-routing drift with the renormalised top-``k`` share centroid.

    This is ``routing_mode: soft`` with ``position_update: naive``: every
    query is assigned to its top-``k`` agents deterministically and the
    centroid mass is the per-query renormalised share. There is no sampling
    factor, so the mass carries a bare :math:`G_i` (times the per-query
    renormaliser) rather than :math:`G_i(1-G_i)`.
    """
    G = allocation_weights(positions, grid, cov)
    mass = weights[None, :] * top_k_allocation_weights(G, top_k)
    return _normalised_drift(positions, grid, mass)


def _expected_drift_dense_matched(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Expected soft-routing drift with the dense :math:`G_i(1-G_i)` centroid mass.

    This is ``routing_mode: soft`` with ``position_update: theory_matched``:
    the mass is applied to every query regardless of ``soft_top_k``, so the
    drift equals the gradient coefficient exactly and is parallel to
    :math:`\\nabla_{x_i} u_i` under isotropic :math:`\\Sigma`. Analytically
    it coincides with :func:`_expected_drift_canonical_reweighted`; the two
    differ only in finite-batch variance (see :func:`_monte_carlo_drift`).
    """
    G = allocation_weights(positions, grid, cov)
    mass = weights[None, :] * matched_centroid_mass(G)
    return _normalised_drift(positions, grid, mass)


def _expected_drift_naive_pairs(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
    group_index: np.ndarray,
    n_groups: int,
    *,
    top_k: int,
) -> np.ndarray:
    """Expected drift of each pair's shared position under the naive pair rule.

    The historical soft-pairs ablation arm: clone allocations are summed per
    group (:func:`group_allocation_weights`), top-``k`` renormalised, and the
    resulting share is the centroid mass of ONE step written to both
    members. Rows are groups; the drift is measured from the shared position
    (the position of the first member).

    :returns: Drift per group, shape ``(P, L)``.
    :rtype: numpy.ndarray
    """
    G = allocation_weights(positions, grid, cov)
    G_group = group_allocation_weights(G, group_index, n_groups)
    mass = weights[None, :] * top_k_allocation_weights(G_group, top_k)
    idx = np.asarray(group_index, dtype=int)
    group_pos = np.stack(
        [positions[np.flatnonzero(idx == p)[0]] for p in range(n_groups)],
        axis=0,
    )
    return _normalised_drift(group_pos, grid, mass)


def _cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    """Cosine similarity between two 1-D vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / max(na * nb, eps))


def _ess_per_query(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Per-agent ESS ratio for canonical+reweight vs strategic routing."""
    G = allocation_weights(positions, grid, cov)
    P = strategic_routing_weights(positions, grid, cov)
    u_strat = (weights[None, :] * P).sum(axis=1)
    num = (weights * (G * (1.0 - G))).sum(axis=1) ** 2
    den = (weights * (G * (1.0 - G) ** 2)).sum(axis=1)
    ess_per_M = num / np.maximum(den, 1e-30)
    return ess_per_M / np.maximum(u_strat, 1e-30)


def _monte_carlo_drift(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
    *,
    batch_size: int = 4096,
    n_trials: int = 200,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Finite-batch Monte-Carlo realisations of the four drift rules.

    ``dense_matched`` uses every sampled query with mass
    :math:`G_i(1-G_i)` (soft theory-matched); ``canonical_reweight`` uses
    only the queries a categorical :math:`G` draw routed to the agent, with
    weight :math:`1-G_i` (hard theory-matched). They share the same
    expectation; the dense rule is the conditional expectation of the hard
    rule given the batch, so its spread is never larger.
    """
    rng = np.random.default_rng(seed)
    n_agents, _ = positions.shape
    g_strat = strategic_routing_weights(positions, grid, cov)
    g = allocation_weights(positions, grid, cov)
    cum_b = np.cumsum(weights)

    strat = np.zeros((n_trials, n_agents, positions.shape[1]))
    can_rw = np.zeros((n_trials, n_agents, positions.shape[1]))
    can_naive = np.zeros((n_trials, n_agents, positions.shape[1]))
    dense = np.zeros((n_trials, n_agents, positions.shape[1]))

    for t in range(n_trials):
        u = rng.random(batch_size)
        k_idx = np.searchsorted(cum_b, u)
        b_samp = grid[k_idx]

        p_at_k = g_strat[:, k_idx].T
        cum_p = np.cumsum(p_at_k, axis=-1)
        r = rng.random((batch_size, 1))
        a_strat = (r < cum_p).argmax(axis=-1)

        g_at_k = g[:, k_idx].T
        cum_g = np.cumsum(g_at_k, axis=-1)
        r2 = rng.random((batch_size, 1))
        a_canon = (r2 < cum_g).argmax(axis=-1)

        for i in range(n_agents):
            mask_s = a_strat == i
            if mask_s.any():
                strat[t, i] = b_samp[mask_s].mean(axis=0) - positions[i]

            mask_c = a_canon == i
            if mask_c.any():
                w_i = (1.0 - g_at_k[mask_c, i])
                w_sum = w_i.sum()
                if w_sum > 1e-30:
                    can_rw[t, i] = (
                        (w_i[:, None] * b_samp[mask_c]).sum(0) / w_sum
                        - positions[i]
                    )
                can_naive[t, i] = b_samp[mask_c].mean(axis=0) - positions[i]

            m_i = g_at_k[:, i] * (1.0 - g_at_k[:, i])
            m_sum = m_i.sum()
            if m_sum > 1e-30:
                dense[t, i] = (
                    (m_i[:, None] * b_samp).sum(0) / m_sum - positions[i]
                )

    return {
        "strategic": strat,
        "canonical_reweight": can_rw,
        "canonical_naive": can_naive,
        "dense_matched": dense,
    }


def _report_config(
    label: str,
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
    *,
    mc_seed: int,
    soft_top_ks: tuple[int, ...] = (),
) -> None:
    """Print analytic and Monte-Carlo diagnostics for one agent configuration.

    :param soft_top_ks: Soft-routing ``top_k`` values for which the naive
        renormalised-share centroid is reported (one column each); the
        dense matched rule does not depend on ``top_k`` and gets a single
        column.
    :type soft_top_ks: tuple[int, ...]
    """
    print(f"\n=== {label} ===")
    grad = utility_gradient(positions, grid, weights, cov)
    d_strat = _expected_drift_strategic(positions, grid, weights, cov)
    d_rw = _expected_drift_canonical_reweighted(positions, grid, weights, cov)
    d_naive = _expected_drift_canonical_naive(positions, grid, weights, cov)
    d_dense = _expected_drift_dense_matched(positions, grid, weights, cov)
    d_soft_naive = {
        k: _expected_drift_topk_naive(positions, grid, weights, cov, top_k=k)
        for k in soft_top_ks
    }
    ess = _ess_per_query(positions, grid, weights, cov)

    print("positions:")
    for i, pos in enumerate(positions):
        print(f"  agent-{i}: {pos}")

    print("\nCosine similarity of expected drift with grad u_i:")
    header = (
        f"  {'agent':<8} {'strategic':>11} {'canon+rw':>11} "
        f"{'canon naive':>13} {'dense match':>13}"
    )
    for k in soft_top_ks:
        header += f" {f'top{k} naive':>12}"
    print(header)
    for i in range(positions.shape[0]):
        row = (
            f"  {i:<8} "
            f"{_cos(d_strat[i], grad[i]):>11.4f} "
            f"{_cos(d_rw[i], grad[i]):>11.4f} "
            f"{_cos(d_naive[i], grad[i]):>13.4f} "
            f"{_cos(d_dense[i], grad[i]):>13.4f}"
        )
        for k in soft_top_ks:
            row += f" {_cos(d_soft_naive[k][i], grad[i]):>12.4f}"
        print(row)

    print("\nESS ratio (canonical+reweight ÷ strategic batch count):")
    for i, ratio in enumerate(ess):
        print(f"  agent-{i}: {ratio:.3f}")

    mc = _monte_carlo_drift(positions, grid, weights, cov, seed=mc_seed)
    print("\nMonte-Carlo (batch=4096, n_trials=200):")
    print(
        f"  {'agent':<6} {'rule':<22} {'cos vs grad':>12} "
        f"{'||mean drift||':>15} {'||drift|| std':>14}"
    )
    for i in range(positions.shape[0]):
        for name, arr in mc.items():
            mean_drift = arr[:, i].mean(axis=0)
            std_norm = np.linalg.norm(
                arr[:, i] - mean_drift[None, :], axis=-1
            ).std()
            print(
                f"  {i:<6} {name:<22} "
                f"{_cos(mean_drift, grad[i]):>12.4f} "
                f"{np.linalg.norm(mean_drift):>15.4f} "
                f"{std_norm:>14.4f}"
            )


def _report_group_config(
    label: str,
    pair_positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
    *,
    top_k: int,
) -> None:
    """Print the soft-pairs alignment check for co-located clone pairs.

    Two clones per pair sit at ``pair_positions``. Under the kept design
    every clone takes its own dense matched step in the ``2P``-player game
    (twin included as a competitor); the report shows that each clone's
    drift is parallel to its own utility gradient and that the two members
    of a pair receive bitwise identical drifts, so the pair persists
    without any shared step. The naive column is the historical pair rule
    (renormalised top-``k`` group share, one step per pair), compared with
    the members' common gradient.
    """
    print(f"\n=== {label} ===")
    n_pairs = pair_positions.shape[0]
    clones = np.repeat(pair_positions, 2, axis=0)
    group_index = np.repeat(np.arange(n_pairs), 2)
    grad_clones = utility_gradient(clones, grid, weights, cov)
    d_match = _expected_drift_dense_matched(clones, grid, weights, cov)
    d_naive = _expected_drift_naive_pairs(
        clones, grid, weights, cov, group_index, n_pairs, top_k=top_k,
    )
    print(
        f"per-clone dense match vs own grad u_i in the {2 * n_pairs}-player "
        f"game; naive pair rule at top_k={top_k}:"
    )
    print(
        f"  {'pair':<6} {'clone':<6} {'naive pair':>11} {'dense match':>12} "
        f"{'|d_twin - d|':>13}"
    )
    for p in range(n_pairs):
        i, j = 2 * p, 2 * p + 1
        for c in (i, j):
            twin = j if c == i else i
            print(
                f"  {p:<6} {c:<6} {_cos(d_naive[p], grad_clones[c]):>11.4f} "
                f"{_cos(d_match[c], grad_clones[c]):>12.4f} "
                f"{np.linalg.norm(d_match[c] - d_match[twin]):>13.2e}"
            )


