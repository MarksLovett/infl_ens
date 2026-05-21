"""Self-contained smoke test for the router design.

Runs the full pipeline with a toy deterministic encoder so the modules can
be exercised without any external embedding model. This is not a unit test;
it's a sanity check that the pieces connect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make /src importable
ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(ROOT))

from infl_ens.data.trait_space import build_trait_space
from infl_ens.inflgame.router import (
    InfluencerRouter,
    RouterAgent,
    expected_utilities,
)
from infl_ens.training.router_training import (
    RouterTrainingConfig,
    train_router_positions,
)
from infl_ens.utils.resource import gaussian_stability_threshold


def toy_encoder(texts):
    """Hash-bag encoder: each word contributes a deterministic 16-dim vec."""
    rng_base = np.random.default_rng(0)
    vocab_dim = 16

    def vec_for_word(w):
        # Deterministic per-word vector.
        h = abs(hash(w)) % (2 ** 32)
        r = np.random.default_rng(h)
        return r.standard_normal(vocab_dim)

    out = []
    for t in texts:
        words = t.lower().split()
        if not words:
            out.append(rng_base.standard_normal(vocab_dim))
            continue
        v = np.mean([vec_for_word(w) for w in words], axis=0)
        out.append(v)
    return np.stack(out, axis=0)


def main() -> None:
    np.random.seed(0)

    # 1. Calibration corpus: a mix of math, code, and creative prompts.
    math_queries = [
        "solve the integral of x squared",
        "what is the derivative of sine x",
        "prove that the square root of two is irrational",
        "compute the determinant of a three by three matrix",
        "factor this polynomial expression",
        "evaluate the limit as x approaches zero",
    ]
    code_queries = [
        "write a python function to reverse a string",
        "implement quicksort in rust",
        "fix this segfault in c code",
        "refactor this javascript callback into async await",
        "explain this stack trace",
        "write a unit test for this function",
    ]
    creative_queries = [
        "write a haiku about the ocean",
        "tell me a short story about a lonely robot",
        "compose a sonnet for a wedding",
        "draft a melancholy poem about autumn",
        "write a fairy tale for a child",
        "invent a myth about the moon",
    ]
    corpus = math_queries + code_queries + creative_queries

    # 2. Build trait space with three interpretable anchors.
    space = build_trait_space(
        queries=corpus,
        encoder=toy_encoder,
        anchors=["mathematics", "programming code", "creative writing"],
        n_grid=16,
    )
    print(f"Trait space:  L={space.L}, K={space.K}, axes={space.axis_labels}")
    print(f"  weighted mean        = {space.mean}")
    print(f"  weighted cov (diag)  = {np.diag(space.covariance)}")

    # 3. Initialise three agents from their calibration sets.
    agents = [
        RouterAgent.from_calibration("math-tuned",     math_queries,     space.project),
        RouterAgent.from_calibration("code-tuned",     code_queries,     space.project),
        RouterAgent.from_calibration("creative-tuned", creative_queries, space.project),
    ]
    for a in agents:
        print(f"  init position [{a.name:>14}] = {a.position}")

    # 4. Pick sigma above the stability threshold for a sanity baseline.
    sigma_star = gaussian_stability_threshold(
        len(agents), space.grid, space.weights,
    )
    sigma = 1.5 * max(sigma_star, 0.05)
    print(f"sigma_0* = {sigma_star:.4f},  using sigma = {sigma:.4f}")

    router = InfluencerRouter(space, agents, sigma=sigma, policy="argmax")
    print(f"is_stable = {router.is_stable()}")
    print(f"utilities pre-train  = {router.expected_utilities()}")

    # 5. Train positions for one game-theoretic refinement pass.
    cfg = RouterTrainingConfig(sigma=sigma, learning_rate=1e-2, n_steps=2000)
    info = train_router_positions(space, agents, cfg, seed=0)
    print(f"trained in {info['n_steps']} steps, converged={info['converged']}")
    for a in agents:
        print(f"  post position [{a.name:>14}] = {a.position}")
    print(f"utilities post-train = {router.expected_utilities()}")

    # 6. Route a few test queries.
    test_queries = [
        "what is the integral of cosine",                # math
        "debug this null pointer exception in java",     # code
        "write a poem about the autumn wind",            # creative
    ]
    for q in test_queries:
        chosen = router.route(q)
        print(f"  '{q[:40]:<40}' -> {chosen.name}")


if __name__ == "__main__":
    main()
