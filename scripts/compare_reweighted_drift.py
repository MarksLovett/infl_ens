"""Compare routing-rule alignment with the closed-form strategic gradient.

This script numerically verifies three claims about routing / SFT-update
equivalences for the multivariate-Gaussian influencer's game.

1. **Canonical routing with per-query reweighting**:
   :math:`p_i(\\mathbf{x}, b) = G_i(\\mathbf{x}, b)` (game-faithful
   assignment) with per-query SFT loss weight
   :math:`w_i(b) = 1 - G_i(\\mathbf{x}, b)`. The expected centroid-blend
   drift is

   .. math::

       \\mathbb{E}[\\Delta x_i]
       \\;=\\; \\beta\\,
       \\frac{\\sum_b B(b)\\, G_i(b)\\,(1-G_i(b))\\,(b - x_i)}
            {\\sum_b B(b)\\, G_i(b)\\,(1-G_i(b))}
       \\;\\propto\\; \\Sigma\\,\\nabla_{x_i} u_i.

   Direction matches :math:`\\nabla_{x_i} u_i` **exactly** under isotropic
   :math:`\\Sigma`; under anisotropic :math:`\\Sigma` the drift is the
   gradient mapped through the metric.

2. **Strategic routing**:
   :math:`p_i^{strat}(b) = G_i(1-G_i) / Z(b)` with
   :math:`Z(b) = \\sum_j G_j(1-G_j)`, unweighted centroid SFT. Approximate
   match to :math:`\\nabla_{x_i} u_i`; the per-trait :math:`1/Z(b)` factor
   stays inside the trait sum and biases the direction slightly.

3. **Canonical naive**: :math:`p_i = G_i`, unweighted centroid. The
   :math:`(1-G_i)` factor is missing entirely and the drift diverges from
   :math:`\\nabla_{x_i} u_i` once agents have separated.

Trade-off: canonical + reweight matches the gradient exactly but assigns
weight :math:`\\to 0` to queries from traits where the routed agent already
dominates, shrinking effective sample size. Strategic routing has a small
directional bias but uses every routed query at unit weight. The script
prints both the analytic cosine alignment and a finite-batch Monte-Carlo
estimate of the variance penalty.

Run with::

    python scripts/compare_reweighted_drift.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infl_ens.inflgame.router.allocation import (  # noqa: E402
    allocation_weights,
    strategic_routing_weights,
    utility_gradient,
)


# -----------------------------------------------------------------------------
# Landscape construction
# -----------------------------------------------------------------------------

def _build_landscape(
    n_grid: int = 25,
    *,
    sigma_b: float = 0.18,
    centers: tuple[tuple[float, float], ...] = ((0.25, 0.25), (0.75, 0.75)),
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 2-D bimodal Gaussian resource landscape on the unit square.

    :param n_grid: Number of grid points per axis. The full grid has
        ``n_grid ** 2`` points.
    :type n_grid: int
    :param sigma_b: Width (standard deviation) of each resource mode.
    :type sigma_b: float
    :param centers: Mode centres inside :math:`[0, 1]^2`.
    :type centers: tuple[tuple[float, float], ...]
    :returns: Pair ``(grid, weights)`` with ``grid`` shape ``(K, 2)`` and
        ``weights`` shape ``(K,)`` summing to one.
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


# -----------------------------------------------------------------------------
# Analytic expected-drift formulas
# -----------------------------------------------------------------------------

def _expected_drift_strategic(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Expected centroid-blend drift under strategic routing.

    Routing is :math:`p_i^{strat}(b) = G_i(1-G_i) / Z(b)` (per-trait
    normalised); SFT centroid is unweighted. Computed in the
    :math:`M \\to \\infty` limit.

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param grid: Trait grid, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param weights: Resource weights :math:`B(b_k)`, shape ``(K,)``.
    :type weights: numpy.ndarray
    :param cov: Shared MV-Gaussian covariance, shape ``(L, L)``.
    :type cov: numpy.ndarray
    :returns: Per-agent expected drift, shape ``(N, L)``.
    :rtype: numpy.ndarray
    """
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
    """Expected drift under canonical routing with per-query :math:`(1 - G_i)` weighting.

    Routing is :math:`p_i(b) = G_i(b)` and each routed query is weighted by
    :math:`w_i(b) = 1 - G_i(b)` in the weighted-centroid SFT step. The
    joint mass on (trait, agent) is :math:`B(b)\\,G_i(b)\\,(1-G_i(b))`,
    matching the integrand of :math:`\\nabla_{x_i} u_i`.

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param grid: Trait grid, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param weights: Resource weights :math:`B(b_k)`, shape ``(K,)``.
    :type weights: numpy.ndarray
    :param cov: Shared MV-Gaussian covariance, shape ``(L, L)``.
    :type cov: numpy.ndarray
    :returns: Per-agent expected drift, shape ``(N, L)``.
    :rtype: numpy.ndarray
    """
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
    """Expected drift under canonical routing with an unweighted centroid SFT.

    The original Lovett & Fu allocation :math:`p_i = G_i` plus a vanilla
    SFT centroid step. Missing the softmax-Jacobian factor :math:`(1-G_i)`
    that the gradient carries.

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param grid: Trait grid, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param weights: Resource weights :math:`B(b_k)`, shape ``(K,)``.
    :type weights: numpy.ndarray
    :param cov: Shared MV-Gaussian covariance, shape ``(L, L)``.
    :type cov: numpy.ndarray
    :returns: Per-agent expected drift, shape ``(N, L)``.
    :rtype: numpy.ndarray
    """
    G = allocation_weights(positions, grid, cov)
    mass = weights[None, :] * G
    Z = mass.sum(axis=1, keepdims=True)
    diff = grid[None, :, :] - positions[:, None, :]
    return (mass[..., None] * diff).sum(axis=1) / np.maximum(Z, 1e-30)


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def _cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    """Cosine similarity between two 1-D vectors.

    :param a: First vector.
    :type a: numpy.ndarray
    :param b: Second vector.
    :type b: numpy.ndarray
    :param eps: Floor for the denominator to avoid division by zero.
    :type eps: float
    :returns: Cosine of the angle between ``a`` and ``b`` in ``[-1, 1]``.
    :rtype: float
    """
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / max(na * nb, eps))


def _ess_per_query(
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Expected effective sample size per batch query for canonical+reweight vs strategic.

    Canonical+reweight gives agent :math:`i` an expected weighted
    effective-sample-size

    .. math::

        \\mathrm{ESS}_i^{\\text{can+rw}} \\;\\approx\\; M\\,
        \\frac{\\left[\\sum_b B(b)\\, G_i(b)\\,(1-G_i(b))\\right]^2}
             {\\sum_b B(b)\\, G_i(b)\\,(1-G_i(b))^2}.

    Strategic routing gives agent :math:`i` an expected unit-weight count
    :math:`M\\,u_i^{\\text{strat}} = M \\sum_b B(b)\\, p_i^{strat}(b)`.
    The ratio :math:`\\rho_i = \\mathrm{ESS}_i^{\\text{can+rw}} / (M\\,
    u_i^{\\text{strat}})` measures the relative efficiency. :math:`\\rho_i
    \\approx 1` means parity; :math:`\\rho_i < 1` means canonical+reweight
    is throwing away effective samples on dominated traits that strategic
    routing would have used at full weight.

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param grid: Trait grid, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param weights: Resource weights :math:`B(b_k)`, shape ``(K,)``.
    :type weights: numpy.ndarray
    :param cov: Shared MV-Gaussian covariance, shape ``(L, L)``.
    :type cov: numpy.ndarray
    :returns: Per-agent ESS ratio, shape ``(N,)``.
    :rtype: numpy.ndarray
    """
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
    """Finite-batch Monte-Carlo realisations of the three drifts.

    For each trial, samples a fresh batch of ``batch_size`` queries from
    :math:`B`, routes under each rule, and computes the realised drift
    (unweighted centroid - position, or weighted-centroid - position).

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param grid: Trait grid, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param weights: Resource weights :math:`B(b_k)`, shape ``(K,)``.
    :type weights: numpy.ndarray
    :param cov: Shared MV-Gaussian covariance, shape ``(L, L)``.
    :type cov: numpy.ndarray
    :param batch_size: Number of queries per trial.
    :type batch_size: int
    :param n_trials: Number of independent trials.
    :type n_trials: int
    :param seed: RNG seed.
    :type seed: int
    :returns: Dictionary with keys ``strategic``, ``canonical_reweight``,
        ``canonical_naive``; each value an array of shape
        ``(n_trials, N, L)`` of realised drifts.
    :rtype: dict[str, numpy.ndarray]
    """
    rng = np.random.default_rng(seed)
    N, L = positions.shape
    G_strat = strategic_routing_weights(positions, grid, cov)
    G = allocation_weights(positions, grid, cov)
    cumB = np.cumsum(weights)

    strat = np.zeros((n_trials, N, L))
    can_rw = np.zeros((n_trials, N, L))
    can_naive = np.zeros((n_trials, N, L))

    for t in range(n_trials):
        u = rng.random(batch_size)
        k_idx = np.searchsorted(cumB, u)
        b_samp = grid[k_idx]

        P_at_k = G_strat[:, k_idx].T
        cumP = np.cumsum(P_at_k, axis=-1)
        r = rng.random((batch_size, 1))
        a_strat = (r < cumP).argmax(axis=-1)

        G_at_k = G[:, k_idx].T
        cumG = np.cumsum(G_at_k, axis=-1)
        r2 = rng.random((batch_size, 1))
        a_canon = (r2 < cumG).argmax(axis=-1)

        for i in range(N):
            mask_s = a_strat == i
            if mask_s.any():
                strat[t, i] = b_samp[mask_s].mean(axis=0) - positions[i]

            mask_c = a_canon == i
            if mask_c.any():
                w_i = (1.0 - G_at_k[mask_c, i])
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


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def _report_config(
    label: str,
    positions: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
    *,
    mc_seed: int,
) -> None:
    """Print analytic and Monte-Carlo diagnostics for one agent configuration.

    :param label: Human-readable header for the section.
    :type label: str
    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param grid: Trait grid, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param weights: Resource weights :math:`B(b_k)`, shape ``(K,)``.
    :type weights: numpy.ndarray
    :param cov: Shared MV-Gaussian covariance, shape ``(L, L)``.
    :type cov: numpy.ndarray
    :param mc_seed: RNG seed for the Monte-Carlo trials.
    :type mc_seed: int
    """
    print(f"\n=== {label} ===")
    grad = utility_gradient(positions, grid, weights, cov)
    d_strat = _expected_drift_strategic(positions, grid, weights, cov)
    d_rw = _expected_drift_canonical_reweighted(positions, grid, weights, cov)
    d_naive = _expected_drift_canonical_naive(positions, grid, weights, cov)
    ess = _ess_per_query(positions, grid, weights, cov)

    print("positions:")
    for i, p in enumerate(positions):
        print(f"  agent-{i}: {p}")

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
    for i, r in enumerate(ess):
        print(f"  agent-{i}: {r:.3f}")

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


def main() -> None:
    """Run the comparison across three off-symmetry configurations.

    :returns: None.
    :rtype: None
    """
    grid, weights = _build_landscape()
    sigma = 0.20
    cov = sigma ** 2 * np.eye(2)

    # Configuration A: agents slightly off the symmetric Nash point.
    positions_a = np.array([
        [0.50, 0.40],
        [0.50, 0.55],
        [0.45, 0.50],
    ])

    # Configuration B: one agent near each mode and one in the contested
    # centre — the regime where the (1-G_i) factor matters most.
    positions_b = np.array([
        [0.28, 0.28],
        [0.72, 0.72],
        [0.50, 0.50],
    ])

    # Configuration C: two agents crowding the same mode and one alone —
    # a (2, 1)-style asymmetric configuration.
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


if __name__ == "__main__":
    main()
