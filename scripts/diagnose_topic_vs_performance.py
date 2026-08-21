"""Test whether content/topic separation predicts per-prompt agent skill.

Topic separation (clustering prompts by semantic content) is an
*unsupervised, content-based* partition — the same family as the PCA/ICA
decomposition. It optimises for **content coherence**, not **performance
heterogeneity**, and the two coincide only if agent competence happens to
track topic. The jailbreak/ToxicChat result is a counterexample: a
coherent topic with no distinct performance niche. So before treating
topic clusters as a routing axis (or a warm-start for the learned
projection), the question to settle is empirical and optimiser-independent:

    *Does any content partition predict which agent wins a prompt?*

This script answers it directly. It clusters the pooled prompt embeddings
into ``K`` topics (``K = N`` agents — the direct "could topics replace the
router" setting) with both KMeans (hard) and a diagonal-covariance GMM
(soft), then measures agreement between the topic partition and the
**winner-agent partition** (winner = lowest-NLL agent per prompt, hard
argmin):

- **Mutual information** ``I(topic; winner)`` and its normalised form
  (NMI). ``NMI -> 0`` means topic and skill are independent — topic
  routing cannot specialise no matter how clean the topics look.
- **Topic-routing NLL**: assign each topic to the agent that is best *on
  average over that topic's prompts*, route accordingly, and report the
  resulting mean NLL vs. the generalist (uniform) and the per-prompt
  oracle. This is the concrete "would topic separation beat the
  generalist" number, on the same footing as
  :mod:`scripts.prototype_learned_trait_projection`.

Crucially every metric is reported **per benchmark as well as pooled**:
the pooled gap can look healthy purely from harm/hallucination while
jailbreak contributes nothing, which would reproduce the exact picture the
investigation has been chasing. A single pooled number hides that.

Because it depends only on the embeddings and the per-agent NLL matrix —
not on any optimiser — this test *arbitrates* a 0%-recovery result from
the learned-projection prototype: high topic↔winner MI but 0% prototype
recovery ⇒ the prototype's optimiser / load-balance penalty failed (try
``--load-balance 0``); near-zero MI ⇒ no content partition exploits the
data and the fix is upstream (clone differentiation), full stop.

Read-only with respect to the package. ``--toy`` runs offline.

Examples
--------

.. code-block:: console

   # Offline smoke test.
   $ python scripts/diagnose_topic_vs_performance.py --toy

   # Real run on doob, with per-benchmark breakdown.
   $ python scripts/diagnose_topic_vs_performance.py \\
         --embeddings results/prompt_embeddings.npy \\
         --nll-matrix results/per_agent_nll.npy \\
         --benchmark-sizes 256,256,256,256 \\
         --benchmark-names harm,hallucination,jailbreak,privacy \\
         --output-stem scripts/figures/topic_vs_performance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# Clustering (KMeans hard + diagonal-Gaussian GMM soft), dependency-free
# -----------------------------------------------------------------------------

def _kmeans(
    x: np.ndarray, k: int, *, n_iter: int = 100, n_init: int = 5, seed: int = 0,
) -> np.ndarray:
    """KMeans hard assignment with k-means++ init and restarts.

    :param x: Data matrix, shape ``(M, D)``.
    :type x: numpy.ndarray
    :param k: Number of clusters.
    :type k: int
    :param n_iter: Lloyd iterations per restart.
    :type n_iter: int
    :param n_init: Random restarts; the lowest-inertia solution is kept.
    :type n_init: int
    :param seed: RNG seed.
    :type seed: int
    :returns: Hard cluster labels in ``[0, k)``, shape ``(M,)``.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        centres = _kpp_init(x, k, rng)
        labels = np.zeros(len(x), dtype=int)
        for _ in range(n_iter):
            d2 = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            new = d2.argmin(axis=1)
            if np.array_equal(new, labels):
                labels = new
                break
            labels = new
            for c in range(k):
                m = labels == c
                if m.any():
                    centres[c] = x[m].mean(axis=0)
        inertia = float(
            ((x - centres[labels]) ** 2).sum()
        )
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels


def _kpp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding.

    :param x: Data matrix, shape ``(M, D)``.
    :type x: numpy.ndarray
    :param k: Number of centres.
    :type k: int
    :param rng: RNG.
    :type rng: numpy.random.Generator
    :returns: Initial centres, shape ``(k, D)``.
    :rtype: numpy.ndarray
    """
    centres = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d2 = np.min(
            [((x - c) ** 2).sum(axis=1) for c in centres], axis=0
        )
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(len(x), 1.0 / len(x))
        centres.append(x[rng.choice(len(x), p=probs)])
    return np.stack(centres, axis=0)


def _gmm_soft(
    x: np.ndarray, k: int, *, n_iter: int = 80, seed: int = 0,
) -> np.ndarray:
    """Diagonal-covariance GMM, returning soft responsibilities.

    :param x: Data matrix, shape ``(M, D)``.
    :type x: numpy.ndarray
    :param k: Number of components.
    :type k: int
    :param n_iter: EM iterations.
    :type n_iter: int
    :param seed: RNG seed (init from a KMeans run).
    :type seed: int
    :returns: Responsibility matrix, shape ``(M, k)``, rows sum to one.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    labels = _kmeans(x, k, seed=seed)
    means = np.stack(
        [x[labels == c].mean(axis=0) if (labels == c).any()
         else x[rng.integers(len(x))] for c in range(k)], axis=0
    )
    var = np.stack([x.var(axis=0) + 1e-3 for _ in range(k)], axis=0)
    weights = np.full(k, 1.0 / k)
    for _ in range(n_iter):
        # E-step: log responsibilities.
        log_r = np.zeros((len(x), k))
        for c in range(k):
            diff2 = (x - means[c]) ** 2
            log_r[:, c] = (
                np.log(weights[c] + 1e-12)
                - 0.5 * np.sum(np.log(2 * np.pi * var[c]))
                - 0.5 * np.sum(diff2 / var[c], axis=1)
            )
        log_r -= log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r)
        r /= r.sum(axis=1, keepdims=True)
        # M-step.
        nk = r.sum(axis=0) + 1e-12
        weights = nk / len(x)
        means = (r.T @ x) / nk[:, None]
        for c in range(k):
            var[c] = (r[:, c][:, None] * (x - means[c]) ** 2).sum(axis=0) / nk[c]
            var[c] = np.clip(var[c], 1e-4, None)
    return r


# -----------------------------------------------------------------------------
# Mutual information between two hard partitions
# -----------------------------------------------------------------------------

def _mutual_information(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mutual information and normalised MI between two label vectors.

    :param a: First label vector, shape ``(M,)``.
    :type a: numpy.ndarray
    :param b: Second label vector, shape ``(M,)``.
    :type b: numpy.ndarray
    :returns: ``(mi_nats, nmi)`` where ``nmi`` is normalised by the mean of
        the two marginal entropies (``0`` independent, ``1`` identical).
    :rtype: tuple[float, float]
    """
    m = len(a)
    a_vals = np.unique(a)
    b_vals = np.unique(b)
    joint = np.zeros((len(a_vals), len(b_vals)))
    ai = {v: i for i, v in enumerate(a_vals)}
    bi = {v: i for i, v in enumerate(b_vals)}
    for x, y in zip(a, b):
        joint[ai[x], bi[y]] += 1.0
    joint /= m
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    mi = 0.0
    for i in range(len(a_vals)):
        for j in range(len(b_vals)):
            if joint[i, j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (pa[i] * pb[j] + 1e-12))
    ha = -np.sum(pa * np.log(pa + 1e-12))
    hb = -np.sum(pb * np.log(pb + 1e-12))
    denom = 0.5 * (ha + hb)
    nmi = float(mi / denom) if denom > 1e-12 else 0.0
    return float(mi), nmi


# -----------------------------------------------------------------------------
# Topic routing: assign each topic to its best-on-average agent
# -----------------------------------------------------------------------------

def _topic_routing_nll(
    topic_labels: np.ndarray, nll: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Route each topic to the agent best on that topic's prompts.

    :param topic_labels: Hard topic labels, shape ``(M,)``.
    :type topic_labels: numpy.ndarray
    :param nll: Per-prompt per-agent NLL, shape ``(M, N)``.
    :type nll: numpy.ndarray
    :returns: ``(routed_nll, topic_to_agent)`` where ``topic_to_agent``
        maps each topic id to its assigned agent index.
    :rtype: tuple[float, numpy.ndarray]
    """
    topics = np.unique(topic_labels)
    topic_to_agent = {}
    total = 0.0
    for t in topics:
        mask = topic_labels == t
        mean_per_agent = nll[mask].mean(axis=0)   # (N,)
        best = int(mean_per_agent.argmin())
        topic_to_agent[t] = best
        total += nll[mask, best].sum()
    routed = total / len(topic_labels)
    mapping = np.array([topic_to_agent[t] for t in topics])
    return float(routed), mapping


# -----------------------------------------------------------------------------
# Per-benchmark slicing
# -----------------------------------------------------------------------------

def _block_slices(sizes: Sequence[int]) -> list[slice]:
    """Contiguous row slices for each benchmark block.

    :param sizes: Per-benchmark row counts, in row order.
    :type sizes: Sequence[int]
    :returns: One slice per block.
    :rtype: list[slice]
    """
    out, start = [], 0
    for n in sizes:
        out.append(slice(start, start + n))
        start += n
    return out


# -----------------------------------------------------------------------------
# Toy fixture
# -----------------------------------------------------------------------------

def _toy(seed: int = 0):
    """Synthetic data where topic partly — but not fully — tracks skill.

    Three of four benchmark blocks have competence aligned with a content
    cluster (topic predicts winner); one block (index 2, the "jailbreak"
    analogue) has its winner assigned independently of content, so its
    per-block MI should be ~0 while the others are high. This reproduces
    the per-benchmark split the real run is expected to show.

    :param seed: RNG seed.
    :type seed: int
    :returns: ``(emb, nll, sizes, names)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, list[int], list[str]]
    """
    rng = np.random.default_rng(seed)
    n_per, dim, n_agents = 200, 48, 4
    names = ["harm", "hallucination", "jailbreak", "privacy"]
    centres = rng.standard_normal((4, dim)) * 2.5
    emb_blocks, nll_blocks, sizes = [], [], []
    for b in range(4):
        e = centres[b] + rng.standard_normal((n_per, dim))
        nll = 1.5 + 0.2 * rng.standard_normal((n_per, n_agents))
        if b == 2:
            # jailbreak analogue: winner independent of content -> low MI.
            winners = rng.integers(0, n_agents, size=n_per)
        else:
            # winner tracks the block (content-aligned) for most prompts.
            winners = np.full(n_per, b)
            flip = rng.random(n_per) < 0.2
            winners[flip] = rng.integers(0, n_agents, size=flip.sum())
        nll[np.arange(n_per), winners] -= 0.6  # make the winner actually best
        emb_blocks.append(e)
        nll_blocks.append(nll)
        sizes.append(n_per)
    return (np.concatenate(emb_blocks), np.concatenate(nll_blocks),
            sizes, names)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _plot(
    names: list[str],
    nmi_per: list[float],
    routed_gap_frac: list[float],
    output_stem: str,
) -> None:
    """Bar chart of per-benchmark topic↔winner NMI and routed-gap fraction.

    :param names: Benchmark names.
    :type names: list[str]
    :param nmi_per: Per-benchmark normalised MI.
    :type nmi_per: list[float]
    :param routed_gap_frac: Per-benchmark fraction of the oracle gap that
        topic routing recovers.
    :type routed_gap_frac: list[float]
    :param output_stem: Output path stem.
    :type output_stem: str
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(names, nmi_per, color="#4c72b0")
    ax1.set_ylabel("NMI(topic; winner)")
    ax1.set_title("Topic ↔ best-agent agreement")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(alpha=0.3, axis="y")
    ax2.bar(names, routed_gap_frac, color="#55a868")
    ax2.set_ylabel("oracle-gap fraction recovered")
    ax2.set_title("Topic routing vs generalist")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = Path(output_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Optional argument vector; defaults to ``sys.argv[1:]``.
    :type argv: list[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy", action="store_true",
                        help="Use the built-in 4-benchmark fixture.")
    parser.add_argument("--embeddings", type=Path,
                        help="(M, D) .npy prompt embeddings.")
    parser.add_argument("--nll-matrix", type=Path,
                        help="(M, N) .npy per-prompt per-agent NLL.")
    parser.add_argument("--benchmark-sizes", default=None,
                        help="Comma-separated per-benchmark row counts.")
    parser.add_argument("--benchmark-names", default=None,
                        help="Comma-separated benchmark names.")
    parser.add_argument("--k", type=int, default=None,
                        help="Topic count; defaults to N agents.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--output-stem", default=None,
                        help="If set, write <stem>.{pdf,png}.")
    args = parser.parse_args(argv)

    if args.toy:
        emb, nll, sizes, names = _toy(seed=args.seed)
    else:
        if args.embeddings is None or args.nll_matrix is None:
            parser.error("pass --toy, or both --embeddings and --nll-matrix")
        emb = np.load(args.embeddings)
        nll = np.load(args.nll_matrix)
        if args.benchmark_sizes:
            sizes = [int(s) for s in args.benchmark_sizes.split(",")]
        else:
            sizes = [emb.shape[0]]
        if args.benchmark_names:
            names = args.benchmark_names.split(",")
        else:
            names = [f"block{i}" for i in range(len(sizes))]
    if sum(sizes) != emb.shape[0]:
        parser.error(f"benchmark-sizes sum {sum(sizes)} != rows {emb.shape[0]}")

    n_agents = nll.shape[1]
    k = args.k or n_agents

    # Standardise embeddings before clustering (variance-scale invariance).
    emb_s = (emb - emb.mean(axis=0)) / (emb.std(axis=0) + 1e-9)
    km = _kmeans(emb_s, k, seed=args.seed)
    gmm_r = _gmm_soft(emb_s, k, seed=args.seed)
    gmm_hard = gmm_r.argmax(axis=1)
    winner = nll.argmin(axis=1)

    def _report(label: str, topics: np.ndarray, rows: slice) -> tuple[float, float]:
        mi, nmi = _mutual_information(topics[rows], winner[rows])
        routed, _ = _topic_routing_nll(topics[rows], nll[rows])
        uniform = float(nll[rows].mean())
        oracle = float(nll[rows].min(axis=1).mean())
        frac = (uniform - routed) / (uniform - oracle) if uniform - oracle > 1e-9 else 0.0
        print(f"    {label:<10} NMI={nmi:.3f}  routed={routed:.4f}  "
              f"uniform={uniform:.4f}  oracle={oracle:.4f}  gap_recovered={frac:+.1%}")
        return nmi, frac

    print(f"Topic↔winner agreement (K={k} topics, N={n_agents} agents)")
    print(f"  winner = lowest-NLL agent (hard argmin)\n")
    print("  POOLED:")
    full = slice(0, emb.shape[0])
    _report("kmeans", km, full)
    _report("gmm", gmm_hard, full)

    print("\n  PER BENCHMARK (within-block clustering):")
    print("    (global topics rarely sub-divide a single benchmark, so MI is")
    print("     measured against a fresh K-cluster fit *within* each block.)")
    nmi_per, frac_per = [], []
    for name, sl in zip(names, _block_slices(sizes)):
        block_emb = emb_s[sl]
        block_nll = nll[sl]
        block_winner = winner[sl]
        # A within-block winner needs >1 distinct winner to have any MI;
        # if every prompt in the block has the same best agent, topic
        # cannot add information and MI is undefined (report n/a).
        if len(np.unique(block_winner)) < 2:
            print(f"    {name:<14} winner constant in block -> MI n/a; "
                  f"routed gap below")
            block_topics = _kmeans(block_emb, k, seed=args.seed)
            routed, _ = _topic_routing_nll(block_topics, block_nll)
            uniform = float(block_nll.mean())
            oracle = float(block_nll.min(axis=1).mean())
            frac = (uniform - routed) / (uniform - oracle) if uniform - oracle > 1e-9 else 0.0
            print(f"    {name:<14} (MI n/a) routed={routed:.4f}  "
                  f"uniform={uniform:.4f}  oracle={oracle:.4f}  "
                  f"gap_recovered={frac:+.1%}")
            nmi_per.append(float("nan"))
            frac_per.append(frac)
            continue
        block_topics = _kmeans(block_emb, k, seed=args.seed)
        mi, nmi = _mutual_information(block_topics, block_winner)
        routed, _ = _topic_routing_nll(block_topics, block_nll)
        uniform = float(block_nll.mean())
        oracle = float(block_nll.min(axis=1).mean())
        frac = (uniform - routed) / (uniform - oracle) if uniform - oracle > 1e-9 else 0.0
        print(f"    {name:<14} NMI={nmi:.3f}  routed={routed:.4f}  "
              f"uniform={uniform:.4f}  oracle={oracle:.4f}  "
              f"gap_recovered={frac:+.1%}")
        nmi_per.append(nmi)
        frac_per.append(frac)

    print("\n  Read: NMI~0 AND gap_recovered<=0 => topic independent of skill "
          "(fix is upstream).")
    print("        NMI high but prototype recovered 0% => prototype optimiser/"
          "penalty failed, not the data.")

    if args.output_stem is not None:
        _plot(names, nmi_per, frac_per, args.output_stem)
        print(f"\nwrote {args.output_stem}.{{pdf,png}}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
