"""Prototype a learned, performance-correlated trait projection.

Motivation
----------
Pinning each trait axis to a *named benchmark* (harm=BeaverTails,
jailbreak=ToxicChat, ...) makes axis admissibility a property to audit
after the fact, and the jailbreak/ToxicChat investigation showed some
benchmarks have no separable niche in MiniLM space. The agents then fail
to specialise: routed and generalist training give the same per-benchmark
NLL because the router cannot send each clone a *distinct, performance-
relevant* slice of the corpus.

This prototype inverts the construction. Instead of fixing axes to
benchmarks and checking separability, it **learns** a low-dimensional
trait projection :math:`\\phi_\\theta:\\,e \\mapsto b \\in \\mathbb{R}^L`
(``e`` the raw encoder embedding) so that the trait coordinate correlates
with *which agent performs best* — while keeping the influencer-game
routing layer (agent positions, resource density :math:`B(b)`, the shared
MV-Gaussian kernel, and hence the Nash analysis) completely intact on top
of :math:`\\phi_\\theta`'s output.

Architecture (framework-preserving MoE gate)
---------------------------------------------
- :math:`\\phi_\\theta(e) = W e + c`, a learned **linear** bottleneck to
  ``L`` dims. Linear keeps each axis an interpretable direction in
  embedding space (comparable by cosine to the LDA axes) and keeps the
  projected coordinate a genuine point in the same trait space the kernel
  consumes — so ``allocation_weights(positions, phi(e), cov)`` is
  unchanged and Theorems 1–5 still apply in :math:`\\phi_\\theta`'s output
  space.
- The gate is the existing kernel allocation
  :math:`G_i \\propto f_i(x_i, \\phi_\\theta(e))`; **only** :math:`\\phi`
  is new. Agent positions :math:`x_i` and :math:`B(b)` are defined in the
  projected space exactly as before.

Training signal (continuous performance gap)
--------------------------------------------
:math:`\\phi_\\theta` is fit so the routing it induces sends each prompt to
the agent that *actually* does best on it. With per-prompt per-agent NLL
:math:`\\ell_{mi}`, the routed expected NLL is
:math:`\\sum_i G_i(\\phi_\\theta(e_m))\\,\\ell_{mi}`; minimising its mean
over the corpus makes :math:`\\phi_\\theta` place prompts so the kernel
prefers the locally-best agent. This is a *continuous* (regression-style)
signal — no hard winner labels — and it is differentiable through the
softmax gate. Two guards keep it honest:

- **Collapse guard.** A learned gate trained on ensemble loss can route
  everything to the single best-on-average agent, reproducing the
  no-specialisation symptom. A load-balance penalty (KL of mean gate load
  to uniform) plus the option to score under the strategic
  :math:`G_i(1-G_i)` weighting counteracts this.
- **Geometry guard.** ``W`` is row-orthonormalised after each step so the
  axes stay non-degenerate (no collapse to a 1-D line), mirroring the
  Gram-Schmidt decorrelation used for the LDA axes.

Online continuity
------------------
``--online`` updates :math:`\\theta` under a Robbins-Monro step-size decay
with Polyak-Ruppert averaging — the same stochastic-approximation schedule
as :class:`infl_ens.inflgame.router.online.OnlineRouter`, applied to the
projection parameters rather than to agent positions. The trait axes then
drift *continuously* as the performance landscape shifts, instead of being
re-fit from scratch per round, which keeps the fixed-point reading of the
dynamics intact.

This script is **read-only with respect to the package**: it reimplements
a minimal copy of the kernel allocation locally so nothing in
``inflgame`` / ``safety_trait_space`` is modified. It is a measurement /
feasibility prototype to decide whether a learned projection recovers
specialisation *before* porting any of it into the package. ``--toy`` runs
fully offline with synthetic agents whose competence varies across a
latent factor.

Examples
--------

.. code-block:: console

   # Offline feasibility check: does a learned phi recover (2,2) +
   # specialisation on synthetic heterogeneous agents?
   $ python scripts/prototype_learned_trait_projection.py --toy \\
         --n-agents 4 --dim-latent 2 --online --output-stem \\
         scripts/figures/learned_projection_toy

   # Real run on doob from a per-prompt per-agent NLL matrix.
   $ python scripts/prototype_learned_trait_projection.py \\
         --embeddings results/prompt_embeddings.npy \\
         --nll-matrix results/per_agent_nll.npy \\
         --n-traits 2 --epochs 300 --load-balance 0.1
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# -----------------------------------------------------------------------------
# Minimal local copy of the influencer kernel (read-only wrt the package)
# -----------------------------------------------------------------------------

def _allocation_weights(
    positions: np.ndarray, coords: np.ndarray, cov: np.ndarray,
) -> np.ndarray:
    """Allocation matrix :math:`G_i(x, b)` (softmax of the log-kernel).

    Local reimplementation of
    :func:`infl_ens.inflgame.router.allocation.allocation_weights` so the
    prototype does not import or modify the package.

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param coords: Trait coordinates, shape ``(M, L)``.
    :type coords: numpy.ndarray
    :param cov: Shared MV-Gaussian covariance, shape ``(L, L)``.
    :type cov: numpy.ndarray
    :returns: ``G`` with ``G[i, m] = G_i(x, b_m)``; columns sum to one.
        Shape ``(N, M)``.
    :rtype: numpy.ndarray
    """
    diff = positions[:, None, :] - coords[None, :, :]      # (N, M, L)
    sol = np.linalg.solve(cov, diff[..., None])[..., 0]    # (N, M, L)
    log_f = -0.5 * np.einsum("nml,nml->nm", diff, sol)     # (N, M)
    log_f -= log_f.max(axis=0, keepdims=True)
    f = np.exp(log_f)
    return f / f.sum(axis=0, keepdims=True)


def _strategic_weights(
    positions: np.ndarray, coords: np.ndarray, cov: np.ndarray, *, eps: float = 1e-12,
) -> np.ndarray:
    """Per-trait :math:`G_i(1-G_i)` routing weights (strategic gradient match).

    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param coords: Trait coordinates, shape ``(M, L)``.
    :type coords: numpy.ndarray
    :param cov: Shared covariance, shape ``(L, L)``.
    :type cov: numpy.ndarray
    :param eps: Degenerate-column floor.
    :type eps: float
    :returns: Strategic routing matrix, shape ``(N, M)``; columns sum to one.
    :rtype: numpy.ndarray
    """
    g = _allocation_weights(positions, coords, cov)
    w = g * (1.0 - g)
    col = w.sum(axis=0, keepdims=True)
    n = positions.shape[0]
    fallback = np.full_like(w, 1.0 / n)
    safe = col > eps
    return np.where(safe, w / np.where(safe, col, 1.0), fallback)


# -----------------------------------------------------------------------------
# Learned linear projection phi_theta
# -----------------------------------------------------------------------------

@dataclass
class Projection:
    """Learned linear trait projection :math:`\\phi(e) = W e + c`.

    :param w: Weight matrix, shape ``(L, D)`` (rows are axis directions).
    :type w: numpy.ndarray
    :param c: Bias, shape ``(L,)``.
    :type c: numpy.ndarray
    """

    w: np.ndarray
    c: np.ndarray

    def project(self, emb: np.ndarray) -> np.ndarray:
        """Map embeddings to trait coordinates.

        :param emb: Embeddings, shape ``(M, D)``.
        :type emb: numpy.ndarray
        :returns: Trait coordinates, shape ``(M, L)``.
        :rtype: numpy.ndarray
        """
        return emb @ self.w.T + self.c

    def orthonormalize(self) -> None:
        """Row-orthonormalise ``W`` (geometry guard against axis collapse).

        Keeps the ``L`` axes linearly independent and unit-scaled so the
        projected coordinate cannot degenerate to a lower-dimensional
        subspace during training.
        """
        q, _ = np.linalg.qr(self.w.T)        # (D, L) orthonormal columns
        self.w = q.T


# -----------------------------------------------------------------------------
# Loss + gradient (routed expected NLL with load-balance + geometry guards)
# -----------------------------------------------------------------------------

def _routed_loss_and_grad(
    proj: Projection,
    emb: np.ndarray,
    nll: np.ndarray,
    positions: np.ndarray,
    cov: np.ndarray,
    *,
    load_balance: float,
    fd_eps: float = 1e-4,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Routed expected NLL plus a finite-difference gradient on ``W``, ``c``.

    The objective is

    .. math::

        \\mathcal{L}(\\theta) = \\frac{1}{M}\\sum_m \\sum_i
            G_i(x, \\phi_\\theta(e_m))\\,\\ell_{mi}
        \\;+\\; \\lambda\\,\\mathrm{KL}\\!\\big(\\bar G \\,\\|\\, \\tfrac1N \\mathbf 1\\big),

    where :math:`\\bar G_i = \\tfrac1M\\sum_m G_i(\\cdot)` is the mean gate
    load and the KL term is the collapse guard. The gradient is taken by
    central finite differences: the projection is tiny
    (``L x D`` + ``L`` parameters with ``L`` small), so FD is robust and
    keeps the prototype dependency-free (no autodiff). Porting to the
    package would swap this for an analytic / autodiff gradient.

    :param proj: Current projection.
    :type proj: Projection
    :param emb: Embeddings, shape ``(M, D)``.
    :type emb: numpy.ndarray
    :param nll: Per-prompt per-agent NLL, shape ``(M, N)``.
    :type nll: numpy.ndarray
    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param cov: Shared covariance, shape ``(L, L)``.
    :type cov: numpy.ndarray
    :param load_balance: KL load-balance penalty weight :math:`\\lambda`.
    :type load_balance: float
    :param fd_eps: Finite-difference step.
    :type fd_eps: float
    :returns: ``(loss, grad_w, grad_c, mean_load)``.
    :rtype: tuple[float, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """

    def loss_of(w: np.ndarray, c: np.ndarray) -> tuple[float, np.ndarray]:
        coords = emb @ w.T + c                       # (M, L)
        g = _allocation_weights(positions, coords, cov)   # (N, M)
        routed = (g.T * nll).sum(axis=1).mean()      # scalar expected NLL
        mean_load = g.mean(axis=1)                   # (N,)
        n = positions.shape[0]
        kl = float(np.sum(mean_load * np.log(np.clip(mean_load * n, 1e-12, None))))
        return float(routed + load_balance * kl), mean_load

    base, mean_load = loss_of(proj.w, proj.c)

    grad_w = np.zeros_like(proj.w)
    for r in range(proj.w.shape[0]):
        for cc in range(proj.w.shape[1]):
            wp = proj.w.copy(); wp[r, cc] += fd_eps
            wm = proj.w.copy(); wm[r, cc] -= fd_eps
            lp, _ = loss_of(wp, proj.c)
            lm, _ = loss_of(wm, proj.c)
            grad_w[r, cc] = (lp - lm) / (2 * fd_eps)

    grad_c = np.zeros_like(proj.c)
    for r in range(proj.c.shape[0]):
        cp = proj.c.copy(); cp[r] += fd_eps
        cm = proj.c.copy(); cm[r] -= fd_eps
        lp, _ = loss_of(proj.w, cp)
        lm, _ = loss_of(proj.w, cm)
        grad_c[r] = (lp - lm) / (2 * fd_eps)

    return base, grad_w, grad_c, mean_load


# -----------------------------------------------------------------------------
# Position update (agents move to the centroid of the prompts routed to them)
# -----------------------------------------------------------------------------

def _update_positions(
    emb: np.ndarray,
    proj: Projection,
    positions: np.ndarray,
    cov: np.ndarray,
    *,
    strategic: bool,
) -> np.ndarray:
    """Move each agent to its routing-weighted centroid in trait space.

    Mirrors the package's position-from-corpus update: agent ``i``'s new
    position is the weighted mean of projected prompt coordinates, weighted
    by its routing share (``G`` or the strategic ``G(1-G)``). This is the
    measurement update — position reflects observed routing, not a
    strategic gradient step — consistent with the project's principle that
    positions are set by post-routing corpus centroids.

    :param emb: Embeddings, shape ``(M, D)``.
    :type emb: numpy.ndarray
    :param proj: Current projection.
    :type proj: Projection
    :param positions: Current agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param cov: Shared covariance, shape ``(L, L)``.
    :type cov: numpy.ndarray
    :param strategic: If ``True`` weight by :math:`G_i(1-G_i)`, else by
        :math:`G_i`.
    :type strategic: bool
    :returns: Updated positions, shape ``(N, L)``.
    :rtype: numpy.ndarray
    """
    coords = proj.project(emb)                            # (M, L)
    p = (_strategic_weights if strategic else _allocation_weights)(
        positions, coords, cov
    )                                                     # (N, M)
    wsum = p.sum(axis=1, keepdims=True)
    wsum = np.where(wsum > 1e-12, wsum, 1.0)
    return (p @ coords) / wsum                            # (N, L)


# -----------------------------------------------------------------------------
# Diagnostics on the learned configuration
# -----------------------------------------------------------------------------

def _specialisation_margin(nll: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    """Routed vs. uniform expected NLL — does routing beat the generalist?

    :param nll: Per-prompt per-agent NLL, shape ``(M, N)``.
    :type nll: numpy.ndarray
    :param g: Gate matrix, shape ``(N, M)``.
    :type g: numpy.ndarray
    :returns: ``(routed_nll, uniform_nll)``; routed < uniform means routing
        extracts specialisation the generalist (uniform mix) cannot.
    :rtype: tuple[float, float]
    """
    routed = float((g.T * nll).sum(axis=1).mean())
    uniform = float(nll.mean())
    return routed, uniform


def _oracle_nll(nll: np.ndarray) -> float:
    """Best-achievable routed NLL: each prompt to its lowest-NLL agent.

    :param nll: Per-prompt per-agent NLL, shape ``(M, N)``.
    :type nll: numpy.ndarray
    :returns: Mean per-prompt minimum NLL (the routing oracle bound).
    :rtype: float
    """
    return float(nll.min(axis=1).mean())


def _heterogeneity(nll: np.ndarray) -> float:
    """Performance heterogeneity: gap between mean and oracle NLL.

    If this is ~0 the per-prompt best agent is no better than the average
    agent — there is no performance gap for any projection to exploit, and
    no division method can manufacture specialisation. A precondition
    check, surfaced before trusting any learned-projection result.

    :param nll: Per-prompt per-agent NLL, shape ``(M, N)``.
    :type nll: numpy.ndarray
    :returns: ``mean_nll - oracle_nll`` (>=0); larger = more exploitable
        heterogeneity.
    :rtype: float
    """
    return float(nll.mean() - _oracle_nll(nll))


# -----------------------------------------------------------------------------
# Toy fixture: synthetic agents whose competence varies across a latent factor
# -----------------------------------------------------------------------------

def _toy(
    n_agents: int, dim_latent: int, n_prompts: int, dim_emb: int, seed: int,
):
    """Synthetic embeddings + a per-prompt per-agent NLL with real structure.

    Each agent is competent in a Gaussian neighbourhood of a latent
    competence centre; a prompt's NLL under an agent grows with distance
    from that agent's centre in a latent factor that **is** linearly
    embedded in the (higher-dim, noisy) embedding. So a *learnable* linear
    projection can recover the latent factor and route prompts to their
    locally-best agent, whereas a fixed benchmark-pinned axis generally
    cannot. This is the regime where a learned projection should help.

    :param n_agents: Number of agents ``N``.
    :type n_agents: int
    :param dim_latent: True latent competence dimensionality ``L*``.
    :type dim_latent: int
    :param n_prompts: Number of prompts ``M``.
    :type n_prompts: int
    :param dim_emb: Embedding dimensionality ``D`` (>= latent).
    :type dim_emb: int
    :param seed: RNG seed.
    :type seed: int
    :returns: ``(emb, nll, true_latent)`` with shapes ``(M, D)``,
        ``(M, N)``, ``(M, L*)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_prompts, dim_latent))
    centres = rng.standard_normal((n_agents, dim_latent)) * 1.4
    d2 = ((latent[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)  # (M, N)
    # NLL: floor + competence falloff with latent distance + noise.
    nll = 1.4 + 0.55 * d2 + 0.05 * rng.standard_normal(d2.shape)
    # Embed latent linearly into a noisy higher-dim space.
    basis = rng.standard_normal((dim_latent, dim_emb))
    emb = latent @ basis + 0.6 * rng.standard_normal((n_prompts, dim_emb))
    return emb, nll, latent


# -----------------------------------------------------------------------------
# Training loop (offline + optional online stochastic-approximation track)
# -----------------------------------------------------------------------------

def _train(
    emb: np.ndarray,
    nll: np.ndarray,
    *,
    n_traits: int,
    n_agents: int,
    sigma: float,
    epochs: int,
    lr: float,
    load_balance: float,
    strategic: bool,
    online: bool,
    seed: int,
) -> tuple[Projection, np.ndarray, np.ndarray, dict]:
    """Alternate projection-gradient and position-centroid updates.

    Offline mode uses a fixed learning rate. ``online`` mode replaces it
    with a Robbins-Monro decay :math:`a_t = lr / (1 + t)^{0.7}` and keeps a
    Polyak-Ruppert running average of the projection parameters — the same
    stochastic-approximation schedule as the package ``OnlineRouter`` — so
    the trait axes drift continuously rather than re-fit per round.

    :param emb: Embeddings, shape ``(M, D)``.
    :type emb: numpy.ndarray
    :param nll: Per-prompt per-agent NLL, shape ``(M, N)``.
    :type nll: numpy.ndarray
    :param n_traits: Trait-space dimensionality ``L``.
    :type n_traits: int
    :param n_agents: Number of agents ``N``.
    :type n_agents: int
    :param sigma: Competitive reach (isotropic).
    :type sigma: float
    :param epochs: Training iterations.
    :type epochs: int
    :param lr: Base learning rate.
    :type lr: float
    :param load_balance: KL load-balance penalty weight.
    :type load_balance: float
    :param strategic: Use :math:`G(1-G)` weighting for position updates.
    :type strategic: bool
    :param online: Use the Robbins-Monro / Polyak-Ruppert schedule.
    :type online: bool
    :param seed: RNG seed.
    :type seed: int
    :returns: ``(projection, positions, gate, history)`` where ``history``
        holds per-epoch ``loss`` and ``min_load`` lists.
    :rtype: tuple[Projection, numpy.ndarray, numpy.ndarray, dict]
    """
    rng = np.random.default_rng(seed)
    d = emb.shape[1]
    proj = Projection(
        w=rng.standard_normal((n_traits, d)) / np.sqrt(d),
        c=np.zeros(n_traits),
    )
    proj.orthonormalize()
    coords = proj.project(emb)
    # Initialise positions spread over the projected cloud.
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    positions = lo + (hi - lo) * rng.random((n_agents, n_traits))
    cov = sigma**2 * np.eye(n_traits)

    avg_w = proj.w.copy()
    avg_c = proj.c.copy()
    history = {"loss": [], "min_load": []}

    for t in range(epochs):
        loss, gw, gc, mean_load = _routed_loss_and_grad(
            proj, emb, nll, positions, cov, load_balance=load_balance,
        )
        step = lr / (1.0 + t) ** 0.7 if online else lr
        proj.w -= step * gw
        proj.c -= step * gc
        proj.orthonormalize()
        if online:
            avg_w += (proj.w - avg_w) / (t + 1)
            avg_c += (proj.c - avg_c) / (t + 1)
        positions = _update_positions(
            emb, proj, positions, cov, strategic=strategic,
        )
        history["loss"].append(loss)
        history["min_load"].append(float(mean_load.min()))

    if online:
        proj = Projection(w=avg_w, c=avg_c)
        proj.orthonormalize()
        positions = _update_positions(
            emb, proj, positions, cov, strategic=strategic,
        )

    gate = _allocation_weights(positions, proj.project(emb), cov)
    return proj, positions, gate, history


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _plot(
    coords: np.ndarray,
    positions: np.ndarray,
    history: dict,
    output_stem: str,
) -> None:
    """Plot the learned 2-D trait cloud + agent positions, and the loss curve.

    :param coords: Projected prompt coordinates, shape ``(M, L)``.
    :type coords: numpy.ndarray
    :param positions: Agent positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param history: Training history dict with ``loss`` / ``min_load``.
    :type history: dict
    :param output_stem: Output path stem (no extension).
    :type output_stem: str
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    if coords.shape[1] >= 2:
        ax1.scatter(coords[:, 0], coords[:, 1], s=4, alpha=0.25, label="prompts")
        ax1.scatter(positions[:, 0], positions[:, 1], s=160, marker="*",
                    edgecolor="k", zorder=5, label="agents")
        ax1.set_xlabel("learned trait 0")
        ax1.set_ylabel("learned trait 1")
    else:
        ax1.hist(coords[:, 0], bins=40, alpha=0.6)
        for p in positions[:, 0]:
            ax1.axvline(p, color="k", lw=1)
        ax1.set_xlabel("learned trait 0")
    ax1.set_title("Learned trait space + agent positions")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(history["loss"], label="routed expected NLL + λ·KL")
    ax2.plot(history["min_load"], label="min agent load")
    ax2.set_xlabel("epoch")
    ax2.set_title("Training")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(alpha=0.3)

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
                        help="Use synthetic heterogeneous agents (offline).")
    parser.add_argument("--embeddings", type=Path,
                        help="(M, D) .npy of prompt embeddings (real run).")
    parser.add_argument("--nll-matrix", type=Path,
                        help="(M, N) .npy of per-prompt per-agent NLL.")
    parser.add_argument("--n-traits", type=int, default=2,
                        help="Learned trait-space dimensionality L.")
    parser.add_argument("--n-agents", type=int, default=4,
                        help="Number of agents N (toy only; real N=nll cols).")
    parser.add_argument("--dim-latent", type=int, default=2,
                        help="Toy: true latent competence dimensionality.")
    parser.add_argument("--n-prompts", type=int, default=600,
                        help="Toy: number of prompts.")
    parser.add_argument("--dim-emb", type=int, default=48,
                        help="Toy: embedding dimensionality.")
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Competitive reach (isotropic).")
    parser.add_argument("--epochs", type=int, default=200, help="Iterations.")
    parser.add_argument("--lr", type=float, default=0.5, help="Base LR.")
    parser.add_argument("--load-balance", type=float, default=0.1,
                        help="KL load-balance penalty weight (collapse guard).")
    parser.add_argument("--strategic", action="store_true",
                        help="Use G(1-G) weighting for position updates.")
    parser.add_argument("--online", action="store_true",
                        help="Robbins-Monro decay + Polyak-Ruppert averaging.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--output-stem", default=None,
                        help="If set, write <stem>.{pdf,png}.")
    args = parser.parse_args(argv)

    if args.toy:
        emb, nll, _ = _toy(
            args.n_agents, args.dim_latent, args.n_prompts,
            args.dim_emb, args.seed,
        )
        n_agents = args.n_agents
    else:
        if args.embeddings is None or args.nll_matrix is None:
            parser.error("pass --toy, or both --embeddings and --nll-matrix")
        emb = np.load(args.embeddings)
        nll = np.load(args.nll_matrix)
        if emb.shape[0] != nll.shape[0]:
            parser.error(
                f"row mismatch: embeddings {emb.shape[0]} vs nll {nll.shape[0]}"
            )
        n_agents = nll.shape[1]

    # Precondition: is there any performance heterogeneity to exploit?
    het = _heterogeneity(nll)
    print("Precondition — performance heterogeneity:")
    print(f"  mean agent NLL   : {nll.mean():.4f}")
    print(f"  oracle (per-prompt best) NLL : {_oracle_nll(nll):.4f}")
    print(f"  heterogeneity gap : {het:.4f}"
          + ("  (OK: exploitable)" if het > 1e-3
             else "  (~0: NO gap to exploit; specialisation impossible)"))

    proj, positions, gate, history = _train(
        emb, nll,
        n_traits=args.n_traits, n_agents=n_agents, sigma=args.sigma,
        epochs=args.epochs, lr=args.lr, load_balance=args.load_balance,
        strategic=args.strategic, online=args.online, seed=args.seed,
    )

    routed, uniform = _specialisation_margin(nll, gate)
    oracle = _oracle_nll(nll)
    print()
    print("Learned-projection routing result:")
    print(f"  uniform / generalist NLL : {uniform:.4f}")
    print(f"  routed NLL (learned phi) : {routed:.4f}")
    print(f"  oracle NLL (upper bound) : {oracle:.4f}")
    if uniform - oracle > 1e-9:
        recovered = (uniform - routed) / (uniform - oracle)
        print(f"  specialisation recovered : {recovered:.1%} of oracle gap")
    print(f"  min agent load (>0 => no collapse) : {gate.mean(axis=1).min():.3f}")

    load = gate.mean(axis=1)
    print()
    print("Per-agent mean routing load (uniform would be "
          f"{1.0 / n_agents:.3f}):")
    for i, l in enumerate(load):
        print(f"  agent {i} : {l:.3f}")

    if args.output_stem is not None:
        _plot(proj.project(emb), positions, history, args.output_stem)
        print(f"\nwrote {args.output_stem}.{{pdf,png}}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
