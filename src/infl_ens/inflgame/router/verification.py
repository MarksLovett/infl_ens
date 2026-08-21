"""Numerical verification of routing drift vs strategic gradient."""

from __future__ import annotations

import numpy as np

from infl_ens.inflgame.router.allocation import (
    allocation_weights,
    strategic_routing_weights,
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


def _expected_drift_strategic(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Expected centroid-blend drift under strategic routing."""
    P = strategic_routing_weights(positions, grid, cov)
    mass = weights[None, :] * P
    Z = mass.sum(axis=1, keepdims=True)
    diff = grid[None, :, :] - positions[:, None, :]
    return (mass[..., None] * diff).sum(axis=1) / np.maximum(Z, 1e-30)


def _expected_drift_canonical_reweighted(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Expected drift under canonical routing with per-query :math:`(1-G_i)` weighting."""
    G = allocation_weights(positions, grid, cov)
    mass = weights[None, :] * G * (1.0 - G)
    Z = mass.sum(axis=1, keepdims=True)
    diff = grid[None, :, :] - positions[:, None, :]
    return (mass[..., None] * diff).sum(axis=1) / np.maximum(Z, 1e-30)


def _expected_drift_canonical_naive(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Expected drift under canonical routing with an unweighted centroid SFT."""
    G = allocation_weights(positions, grid, cov)
    mass = weights[None, :] * G
    Z = mass.sum(axis=1, keepdims=True)
    diff = grid[None, :, :] - positions[:, None, :]
    return (mass[..., None] * diff).sum(axis=1) / np.maximum(Z, 1e-30)


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
    """Finite-batch Monte-Carlo realisations of the three drifts."""
    rng = np.random.default_rng(seed)
    n_agents, _ = positions.shape
    g_strat = strategic_routing_weights(positions, grid, cov)
    g = allocation_weights(positions, grid, cov)
    cum_b = np.cumsum(weights)

    strat = np.zeros((n_trials, n_agents, positions.shape[1]))
    can_rw = np.zeros((n_trials, n_agents, positions.shape[1]))
    can_naive = np.zeros((n_trials, n_agents, positions.shape[1]))

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

    return {
        "strategic": strat,
        "canonical_reweight": can_rw,
        "canonical_naive": can_naive,
    }


def _report_config(
    label: str,
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
    *,
    mc_seed: int,
) -> None:
    """Print analytic and Monte-Carlo diagnostics for one agent configuration."""
    print(f"\n=== {label} ===")
    grad = utility_gradient(positions, grid, weights, cov)
    d_strat = _expected_drift_strategic(positions, grid, weights, cov)
    d_rw = _expected_drift_canonical_reweighted(positions, grid, weights, cov)
    d_naive = _expected_drift_canonical_naive(positions, grid, weights, cov)
    ess = _ess_per_query(positions, grid, weights, cov)

    print("positions:")
    for i, pos in enumerate(positions):
        print(f"  agent-{i}: {pos}")

    print("\nCosine similarity of expected drift with grad u_i:")
    header = f"  {'agent':<8} {'strategic':>11} {'canon+rw':>11} {'canon naive':>13}"
    print(header)
    for i in range(positions.shape[0]):
        print(
            f"  {i:<8} "
            f"{_cos(d_strat[i], grad[i]):>11.4f} "
            f"{_cos(d_rw[i], grad[i]):>11.4f} "
            f"{_cos(d_naive[i], grad[i]):>13.4f}"
        )

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


def run_reweighted_drift_report() -> None:
    """Run the comparison across three off-symmetry configurations."""
    grid, weights = _build_landscape()
    sigma = 0.20
    cov = sigma ** 2 * np.eye(2)

    positions_a = np.array([
        [0.50, 0.40],
        [0.50, 0.55],
        [0.45, 0.50],
    ])
    positions_b = np.array([
        [0.28, 0.28],
        [0.72, 0.72],
        [0.50, 0.50],
    ])
    positions_c = np.array([
        [0.20, 0.20],
        [0.30, 0.30],
        [0.75, 0.75],
    ])

    _report_config(
        "Configuration A: near symmetric Nash",
        positions_a, grid, weights, cov, mc_seed=0,
    )
    _report_config(
        "Configuration B: one agent per mode + contested centre",
        positions_b, grid, weights, cov, mc_seed=1,
    )
    _report_config(
        "Configuration C: two agents on one mode + one alone",
        positions_c, grid, weights, cov, mc_seed=2,
    )
