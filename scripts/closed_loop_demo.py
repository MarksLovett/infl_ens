"""Closed-loop demo: identical clones differentiate into specialists.

Starting from :math:`N` identical clones of a base model, this script
simulates the closed loop

::

    sample queries  ──>  router routes  ──>  each agent "trains"
        ▲                                           │
        │                                           ▼
        └─────────  position re-estimated  ◀────────┘

"Training" is the toy update ``x_i <- (1 - lr) x_i + lr * centroid(queries_i)``,
a tractable stand-in for SFT (the agent's capability drifts toward the
queries it processed). Real use would replace this with actual fine-tuning
followed by ``agent.update_position_from_corpus(...)`` on the resulting
training corpus or an eval set.

Two regimes are run:

- ``sigma > sigma_0*``: symmetric Nash is locally stable; clones stay
  near the resource-weighted mean and the maximum pairwise distance does
  not grow (no symmetry-breaking).
- ``sigma < sigma_0*``: symmetric configuration is unstable; small
  sampling noise amplifies into specialist clusters.

Run with ``python closed_loop_demo.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(ROOT))

from infl_ens.data.trait_space import build_trait_space, position_from_corpus
from infl_ens.inflgame.router import InfluencerRouter, RouterAgent
from infl_ens.utils.resource import gaussian_stability_threshold


def toy_encoder(texts: list[str]) -> np.ndarray:
    """Deterministic hash-bag encoder (no external model dependency)."""
    vocab_dim = 16

    def vec(w: str) -> np.ndarray:
        return np.random.default_rng(abs(hash(w)) % (2 ** 32)).standard_normal(vocab_dim)

    out = []
    for t in texts:
        ws = t.lower().split()
        out.append(np.mean([vec(w) for w in ws], axis=0) if ws else np.zeros(vocab_dim))
    return np.stack(out, axis=0)


def make_corpus() -> list[str]:
    """Build a thematic corpus with three distinguishable clusters."""
    math = [
        "solve integral", "derivative sine", "prove irrational",
        "determinant matrix", "factor polynomial", "evaluate limit",
        "compute eigenvalues", "diagonalize matrix", "taylor expansion",
        "fourier transform", "gradient descent", "convex optimization",
    ]
    code = [
        "python function reverse string", "implement quicksort rust",
        "fix segfault c", "refactor callback async", "explain stack trace",
        "write unit test pytest", "debug null pointer java",
        "javascript promise chain", "go channels concurrency",
        "rust lifetime annotation", "typescript generic constraint",
        "kubernetes deployment yaml",
    ]
    creative = [
        "haiku ocean", "story lonely robot", "sonnet wedding",
        "melancholy poem autumn", "fairy tale child", "myth moon",
        "ballad sailor", "limerick cat", "elegy lost city",
        "short story horror cabin", "novella dystopian future",
        "essay nature solitude",
    ]
    return math + code + creative


def run_closed_loop(
    *,
    sigma_fraction: float,
    n_rounds: int = 80,
    batch_size: int = 64,
    lr: float = 0.15,
    n_agents: int = 3,
    init_noise: float = 1e-4,
    seed: int = 0,
) -> dict:
    """Run the closed loop at ``sigma = sigma_fraction * sigma_0*``.

    :param sigma_fraction: Multiplier on :math:`\\sigma_0^*`. Values below 1
        put the system in the bifurcation regime; values above 1 in the
        symmetric-stable regime.
    :type sigma_fraction: float
    :param n_rounds: Closed-loop iterations.
    :type n_rounds: int
    :param batch_size: Queries sampled per round.
    :type batch_size: int
    :param lr: Position step size per round.
    :type lr: float
    :param n_agents: Number of clones at start.
    :type n_agents: int
    :param init_noise: Std of Gaussian perturbation added to identical
        starting positions. With exactly equal positions the closed loop is
        marginal and slow to break symmetry; a tiny noise floor reflects
        real-world floating-point / training-data variance.
    :type init_noise: float
    :param seed: RNG seed.
    :type seed: int
    :returns: Dictionary with the spread history (max pairwise distance per
        round), final positions, threshold, and final allocation entropy.
    :rtype: dict
    """
    rng = np.random.default_rng(seed)

    corpus = make_corpus()
    space = build_trait_space(
        corpus, toy_encoder,
        anchors=["mathematics", "programming code", "creative writing"],
        n_grid=12,
    )

    # All clones start at the resource-weighted mean.
    x0 = space.mean.copy()
    agents = [
        RouterAgent(
            name=f"clone-{i}",
            position=x0 + init_noise * rng.standard_normal(space.L),
        )
        for i in range(n_agents)
    ]

    sigma_star = gaussian_stability_threshold(n_agents, space.grid, space.weights)
    sigma = sigma_fraction * sigma_star
    router = InfluencerRouter(space, agents, sigma=sigma, policy="proportional")

    spread_history = []
    for r in range(n_rounds):
        # Sample queries from the corpus (toy proxy for sampling from B(b)).
        batch_idx = rng.integers(0, len(corpus), size=batch_size)
        queries = [corpus[i] for i in batch_idx]

        # Route, then "train" each agent by moving its position toward the
        # centroid of queries it received this round.
        choices = router.route_batch(queries, rng=rng)
        for agent in agents:
            agent_qs = [q for q, c in zip(queries, choices) if c.name == agent.name]
            if not agent_qs:
                continue
            agent.update_position_from_corpus(
                agent_qs, space.project, blend=lr,
            )

        positions = router.positions
        pairwise = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        spread_history.append(float(pairwise.max()))

    # Final allocation entropy averaged over the resource grid.
    from infl_ens.inflgame.router.allocation import allocation_weights
    G = allocation_weights(router.positions, space.grid, router.cov)  # (N, K)
    H_grid = -np.sum(G * np.log(G + 1e-30), axis=0)  # (K,)
    mean_entropy = float((space.weights * H_grid).sum())
    log_N = float(np.log(n_agents))

    return {
        "sigma_star": float(sigma_star),
        "sigma": float(sigma),
        "spread_history": np.asarray(spread_history),
        "final_positions": router.positions,
        "mean_allocation_entropy": mean_entropy,
        "uniform_entropy": log_N,
    }


def main() -> None:
    for frac in [1.5, 0.5]:
        label = "ABOVE" if frac > 1 else "BELOW"
        print(f"\n=== sigma = {frac} * sigma_0*  ({label} threshold) ===")
        result = run_closed_loop(sigma_fraction=frac, seed=0)
        h = result["spread_history"]
        print(f"  sigma_0*               = {result['sigma_star']:.4f}")
        print(f"  sigma                  = {result['sigma']:.4f}")
        print(f"  spread start           = {h[0]:.4f}")
        print(f"  spread mid             = {h[len(h)//2]:.4f}")
        print(f"  spread end             = {h[-1]:.4f}")
        print(f"  growth factor          = {h[-1] / max(h[0], 1e-12):.1f}x")
        print(f"  mean alloc entropy     = {result['mean_allocation_entropy']:.4f}"
              f"  (uniform = {result['uniform_entropy']:.4f})")
        print(f"  final positions:")
        for i, p in enumerate(result["final_positions"]):
            print(f"    clone-{i}: {p}")


if __name__ == "__main__":
    main()
