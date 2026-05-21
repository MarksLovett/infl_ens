"""Compare three estimators of agent utility :math:`u_i(\\mathbf{x})`.

The proportional routing mechanism (``policy='proportional'``) yields a
sample count :math:`N_i` whose expected share of a batch is *not* the
grid-evaluated :math:`u_i` reported by
:func:`infl_ens.inflgame.router.allocation.expected_utilities`. It is the
**empirical-pool utility**

.. math::

    \\hat u_i(\\mathbf{x}) \\;=\\; \\frac{1}{|\\mathcal{C}|}
        \\sum_{q \\in \\mathcal{C}} G_i(\\mathbf{x}, b^\\ast_q),

i.e. :math:`G_i` averaged against the actual embedded queries in the pool
:math:`\\mathcal{C}`, with no KDE smoothing or grid discretisation. This
script prints all three quantities side-by-side so you can see whether the
KDE-smoothed grid estimate matches the empirical pool estimate on a given
trait space, and how close a finite batch sample comes to either.

Two modes:

- ``--mode toy``: self-contained demo using a hash-bag encoder and three
  thematic clusters (math, code, creative). Runs offline.
- ``--mode safety``: full BeaverTails×HaluEval pipeline. Requires the
  benchmark data files and ``sentence-transformers`` to be installed.

Outputs a CSV per round at ``--output`` if specified, plus a printed
summary.

Run with ``python scripts/compare_utility_estimators.py --mode toy``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(ROOT))

from infl_ens.data.trait_space import build_trait_space
from infl_ens.inflgame.router import (
    InfluencerRouter,
    RouterAgent,
    allocation_weights,
    empirical_utility,
)
from infl_ens.utils.resource import gaussian_stability_threshold


# -----------------------------------------------------------------------------
# Encoders
# -----------------------------------------------------------------------------

def _toy_encoder(texts: Sequence[str]) -> np.ndarray:
    """Deterministic 16-dim hash-bag encoder (no external dependencies).

    :param texts: Queries to embed.
    :type texts: Sequence[str]
    :returns: Embedding matrix, shape ``(len(texts), 16)``.
    :rtype: numpy.ndarray
    """
    vocab_dim = 16

    def vec(w: str) -> np.ndarray:
        return np.random.default_rng(abs(hash(w)) % (2 ** 32)).standard_normal(vocab_dim)

    out = []
    for t in texts:
        ws = t.lower().split()
        out.append(np.mean([vec(w) for w in ws], axis=0) if ws else np.zeros(vocab_dim))
    return np.stack(out, axis=0)


def _sentence_transformer_encoder(model_name: str):
    """Lazy-build a sentence-transformers encoder callable.

    :param model_name: HuggingFace model identifier.
    :type model_name: str
    :returns: Callable mapping ``list[str] -> numpy.ndarray``.
    :rtype: Callable[[Sequence[str]], numpy.ndarray]
    """
    from infl_ens.data.encoders import SentenceTransformerEncoder
    enc = SentenceTransformerEncoder(model_name=model_name)
    return lambda texts: enc(list(texts))


# -----------------------------------------------------------------------------
# Mode: toy
# -----------------------------------------------------------------------------

_TOY_MATH = [
    "solve integral", "prove irrational", "factor polynomial", "derivative sine",
    "eigenvalues matrix", "fourier transform", "taylor expansion", "convex optimization",
]
_TOY_CODE = [
    "quicksort rust", "null pointer java", "python async promise",
    "lifetime annotation rust", "generic constraint typescript", "kubernetes yaml deploy",
    "segfault c debug", "unit test pytest",
]
_TOY_CREA = [
    "haiku ocean", "sonnet wedding", "story lonely robot", "poem autumn melancholy",
    "ballad sailor", "limerick cat", "myth moon", "elegy lost city",
]


def _build_toy_setup() -> tuple:
    """Build the toy three-cluster setup.

    :returns: Tuple ``(router, corpus, sigma_star)`` ready for comparison.
    :rtype: tuple
    """
    corpus = _TOY_MATH + _TOY_CODE + _TOY_CREA
    space = build_trait_space(
        queries=corpus,
        encoder=_toy_encoder,
        anchors=["mathematics", "programming", "creative writing"],
        n_grid=12,
    )
    agents = [
        RouterAgent.from_calibration("math",     _TOY_MATH, space.project),
        RouterAgent.from_calibration("code",     _TOY_CODE, space.project),
        RouterAgent.from_calibration("creative", _TOY_CREA, space.project),
    ]
    sigma_star = gaussian_stability_threshold(len(agents), space.grid, space.weights)
    sigma = 0.5 * max(sigma_star, 0.05)
    router = InfluencerRouter(space, agents, sigma=sigma, policy="proportional")
    return router, corpus, sigma_star


# -----------------------------------------------------------------------------
# Mode: safety
# -----------------------------------------------------------------------------

def _build_safety_setup(args: argparse.Namespace) -> tuple:
    """Build the BeaverTails×HaluEval trait space and three clones.

    :param args: Parsed CLI args (``beavertails``, ``halueval``, ``encoder``,
        ``n_grid``, ``sigma_fraction``, ``max_records``).
    :type args: argparse.Namespace
    :returns: Tuple ``(router, corpus, sigma_star)``.
    :rtype: tuple
    """
    from infl_ens.data.benchmarks import (
        build_safety_trait_space,
        load_beavertails,
        load_halueval,
    )

    splits = [
        load_beavertails(args.beavertails, max_records=args.max_records),
        load_halueval(args.halueval, max_records=args.max_records),
    ]
    encoder = _sentence_transformer_encoder(args.encoder)
    space = build_safety_trait_space(splits, encoder, n_grid=args.n_grid)

    corpus = [p for s in splits for p in s.prompts]
    n_agents = 3
    agents = [
        RouterAgent(name=f"clone-{i}", position=space.mean.copy())
        for i in range(n_agents)
    ]
    # Break symmetry minimally so different agents have visible G columns.
    rng = np.random.default_rng(0)
    for a in agents:
        a.position = a.position + 1e-2 * rng.standard_normal(space.L)
        a.position = np.clip(a.position, 0.0, 1.0)

    sigma_star = gaussian_stability_threshold(n_agents, space.grid, space.weights)
    sigma = args.sigma_fraction * max(sigma_star, 0.05)
    router = InfluencerRouter(space, agents, sigma=sigma, policy="proportional")
    return router, corpus, sigma_star


# -----------------------------------------------------------------------------
# Comparison core
# -----------------------------------------------------------------------------

def compare_estimators(
    router: InfluencerRouter,
    corpus: Sequence[str],
    *,
    batch_size: int = 1024,
    n_trials: int = 8,
    seed: int = 0,
) -> dict:
    """Run all three utility estimators for one router/corpus pair.

    :param router: Configured router (positions and sigma already set).
    :type router: InfluencerRouter
    :param corpus: Full query pool (the empirical resource distribution).
    :type corpus: Sequence[str]
    :param batch_size: Queries per routing trial.
    :type batch_size: int
    :param n_trials: Independent batches over which to average the
        proportional-routing share. Reduces Monte Carlo noise.
    :type n_trials: int
    :param seed: RNG seed.
    :type seed: int
    :returns: Dictionary with keys ``u_grid``, ``u_pool``, ``shares_mean``,
        ``shares_std``, ``positions``, ``sigma``.
    :rtype: dict
    """
    rng = np.random.default_rng(seed)
    corpus = list(corpus)

    # 1. Grid utility u_grid: G @ B_KDE (existing expected_utilities).
    u_grid = router.expected_utilities()

    # 2. Empirical-pool utility u_pool: G averaged over the full pool.
    full_coords = router.trait_space.project(corpus)
    u_pool = empirical_utility(router.positions, full_coords, router.cov)

    # 3. Proportional-routing share averaged over n_trials batches.
    N = len(router.agents)
    shares = np.zeros((n_trials, N))
    for t in range(n_trials):
        idx = rng.integers(0, len(corpus), size=batch_size)
        batch = [corpus[i] for i in idx]
        choices = router.route_batch(batch, rng=rng)
        counts = np.array(
            [sum(1 for c in choices if c.name == a.name) for a in router.agents]
        )
        shares[t] = counts / batch_size

    return {
        "u_grid": u_grid,
        "u_pool": u_pool,
        "shares_mean": shares.mean(axis=0),
        "shares_std": shares.std(axis=0),
        "positions": router.positions.copy(),
        "sigma": router.sigma_scalar,
    }


def _print_report(
    result: dict,
    *,
    agent_names: list[str],
    sigma_star: float,
    batch_size: int,
    n_trials: int,
) -> None:
    """Pretty-print the comparison.

    :param result: Output of :func:`compare_estimators`.
    :type result: dict
    :param agent_names: Display names, in router order.
    :type agent_names: list[str]
    :param sigma_star: Stability threshold for context.
    :type sigma_star: float
    :param batch_size: Batch size used.
    :type batch_size: int
    :param n_trials: Trial count used.
    :type n_trials: int
    """
    print(f"\nsigma_0* = {sigma_star:.4f}, sigma used = {result['sigma']:.4f}")
    print(f"batch_size = {batch_size}, n_trials = {n_trials}")
    print(f"\n{'agent':<14} {'u_grid':>10} {'u_pool':>10} "
          f"{'share_mean':>12} {'share_std':>10}")
    print("-" * 60)
    for i, name in enumerate(agent_names):
        print(
            f"{name:<14} "
            f"{result['u_grid'][i]:>10.4f} "
            f"{result['u_pool'][i]:>10.4f} "
            f"{result['shares_mean'][i]:>12.4f} "
            f"{result['shares_std'][i]:>10.4f}"
        )
    print("-" * 60)
    delta_grid_pool = float(np.max(np.abs(result["u_grid"] - result["u_pool"])))
    delta_pool_share = float(np.max(np.abs(result["u_pool"] - result["shares_mean"])))
    print(f"max |u_grid - u_pool|       = {delta_grid_pool:.4f}"
          f"  (KDE-vs-empirical smoothing gap)")
    print(f"max |u_pool - share_mean|   = {delta_pool_share:.4f}"
          f"  (Poisson-Binomial finite-batch noise)")


def _write_csv(result: dict, agent_names: list[str], path: Path) -> None:
    """Write the per-agent comparison to CSV.

    :param result: Output of :func:`compare_estimators`.
    :type result: dict
    :param agent_names: Display names.
    :type agent_names: list[str]
    :param path: CSV path. Parent directories are created if needed.
    :type path: pathlib.Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["agent", "u_grid", "u_pool", "share_mean", "share_std"])
        for i, name in enumerate(agent_names):
            w.writerow([
                name,
                f"{result['u_grid'][i]:.6f}",
                f"{result['u_pool'][i]:.6f}",
                f"{result['shares_mean'][i]:.6f}",
                f"{result['shares_std'][i]:.6f}",
            ])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    :returns: Configured argparse parser.
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="Compare grid, empirical-pool, and finite-batch estimators "
                    "of the influencer-game utility u_i."
    )
    p.add_argument("--mode", choices=["toy", "safety"], default="toy")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-trials", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None,
                   help="Optional CSV output path.")
    # safety-only flags
    p.add_argument("--beavertails", type=Path, default=None,
                   help="Path to BeaverTails JSONL (safety mode).")
    p.add_argument("--halueval", type=Path, default=None,
                   help="Directory containing HaluEval JSON files (safety mode).")
    p.add_argument("--encoder", type=str,
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--n-grid", type=int, default=32)
    p.add_argument("--sigma-fraction", type=float, default=0.5)
    p.add_argument("--max-records", type=int, default=2000)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional argv override; ``None`` reads ``sys.argv[1:]``.
    :type argv: list[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    args = _build_parser().parse_args(argv)

    if args.mode == "toy":
        router, corpus, sigma_star = _build_toy_setup()
    else:
        if args.beavertails is None or args.halueval is None:
            print("safety mode requires --beavertails and --halueval", file=sys.stderr)
            return 2
        router, corpus, sigma_star = _build_safety_setup(args)

    result = compare_estimators(
        router, corpus,
        batch_size=args.batch_size,
        n_trials=args.n_trials,
        seed=args.seed,
    )
    agent_names = [a.name for a in router.agents]
    _print_report(
        result,
        agent_names=agent_names,
        sigma_star=sigma_star,
        batch_size=args.batch_size,
        n_trials=args.n_trials,
    )
    if args.output is not None:
        _write_csv(result, agent_names, args.output)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
