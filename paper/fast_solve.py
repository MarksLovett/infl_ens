"""A faster, mathematically identical inner loop for the offline theory solve.

This exists only for the offline re-solve in ``true_theory_positions.py``. It
does not touch ``src/`` and changes no experiment: the shipped solver stays
exactly as every run used it.

Why it is needed
----------------
``router_training.train_router_positions`` costs ~445 ms per step at N=14 agents
against 28,355 support points, which is ~2 h per arm. Three things account for
almost all of it, and none of them are inherent:

1. ``utility_gradient`` calls ``np.linalg.solve(cov, diff[..., None])`` to apply
   :math:`\\Sigma^{-1}`. Every run here has :math:`\\Sigma = \\sigma^2 I`, so that
   is a scalar divide wearing a 7x7 LAPACK solve: 132 ms against 4 ms.
2. It materialises an ``(N, K, L)`` array to hold :math:`\\Sigma^{-1}(x_i - b_k)`
   and contracts it with ``einsum``. The contraction is algebraically
   ``coeff.sum(k) * x_i - coeff @ grid``, which needs no such temporary at all.
3. The loop computes the allocation a second time per step purely to append to a
   ``traj_u`` diagnostic that ``run_gradient_ascent_theory`` then discards.

Removing those three gives the same positions from the same arithmetic path,
about 15x faster. ``assert_matches_reference`` is the guard: it runs both
implementations and compares trajectories.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def fast_train_router_positions(trait_space, agents, cfg, seed: int = 0) -> dict[str, Any]:
    """Drop-in replacement for ``train_router_positions``, isotropic sigma only.

    :raises ValueError: If ``cfg.sigma`` is not scalar. Anisotropic reach is
        supported by the shipped solver and deliberately not reimplemented here.
    """
    sigma_arr = np.atleast_1d(np.asarray(cfg.sigma, dtype=float))
    if sigma_arr.size != 1:
        raise ValueError("fast path covers isotropic sigma only; "
                         f"got shape {sigma_arr.shape}")
    inv = 1.0 / float(sigma_arr.item()) ** 2

    grid = np.ascontiguousarray(trait_space.grid, dtype=float)      # (K, L)
    weights = np.asarray(trait_space.weights, dtype=float)          # (K,)
    grid_sq = np.einsum("kl,kl->k", grid, grid)                     # (K,)

    pos = np.stack([a.position for a in agents], axis=0).astype(float)
    traj_pos = [pos.copy()]
    converged = False
    t = 0

    for t in range(1, cfg.n_steps + 1):
        # Squared distances as one GEMM instead of an (N, K, L) difference.
        pos_sq = np.einsum("nl,nl->n", pos, pos)
        d2 = pos_sq[:, None] - 2.0 * (pos @ grid.T) + grid_sq[None, :]
        np.maximum(d2, 0.0, out=d2)

        log_f = d2
        log_f *= -0.5 * inv
        log_f -= log_f.max(axis=0, keepdims=True)
        f = np.exp(log_f)
        G = f / f.sum(axis=0, keepdims=True)                        # (N, K)

        coeff = weights[None, :] * G * (1.0 - G)                    # (N, K)
        # grad_i = -inv * sum_k coeff_ik (x_i - b_k)
        #        = -inv * (x_i * sum_k coeff_ik  -  sum_k coeff_ik b_k)
        grad = -inv * (coeff.sum(axis=1)[:, None] * pos - coeff @ grid)

        new_pos = pos + cfg.learning_rate * grad
        if cfg.clip_to_box:
            np.clip(new_pos, 0.0, 1.0, out=new_pos)

        step = float(np.max(np.abs(new_pos - pos)))
        pos = new_pos
        traj_pos.append(pos.copy())
        if step < cfg.tol:
            converged = True
            break

    for a, p in zip(agents, pos):
        a.position = p
    return {
        "positions": np.stack(traj_pos, axis=0),
        "converged": converged,
        "n_steps": int(t),
    }


def assert_matches_reference(n_agents: int = 5, n_grid: int = 400, L: int = 7,
                             n_steps: int = 60, sigma: float = 0.2,
                             atol: float = 1e-10) -> float:
    """Run both implementations on a small problem and compare trajectories.

    :returns: Max absolute difference in position across all steps.
    """
    import sys
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from infl_ens.data.trait_space import TraitSpace
    from infl_ens.training.router_training import (
        RouterTrainingConfig, train_router_positions,
    )
    from infl_ens.inflgame.router.core import RouterAgent

    rng = np.random.default_rng(0)
    space = TraitSpace(
        grid=rng.random((n_grid, L)),
        weights=np.full(n_grid, 1.0 / n_grid),
        project=lambda _q: (_ for _ in ()).throw(RuntimeError("unused")),
    )
    start = rng.random((n_agents, L))
    cfg = RouterTrainingConfig(sigma=sigma, learning_rate=0.02,
                               n_steps=n_steps, tol=0.0, clip_to_box=True)

    def _agents():
        return [RouterAgent(name=f"a{i}", position=start[i].copy())
                for i in range(n_agents)]

    ref = train_router_positions(space, _agents(), cfg, seed=0)["positions"]
    got = fast_train_router_positions(space, _agents(), cfg, seed=0)["positions"]
    assert ref.shape == got.shape, (ref.shape, got.shape)
    diff = float(np.abs(ref - got).max())
    assert diff <= atol, f"fast path diverges from the reference by {diff:g}"
    return diff


if __name__ == "__main__":
    print(f"max |fast - reference| = {assert_matches_reference():.3e}")
