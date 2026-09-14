"""Naive start versus the game equilibrium.

Every arm in the original study begins from a *naive start*: the resource
measure the theory solves against, ``B(b)``, is near-uniform over the
``{0, 0.5, 1}^7`` lattice rather than a density of the prompt cloud, so the
initial positions carry almost no information about where the prompts actually
are. (``_kde_on_grid`` divides each grid point by its own nearest-prompt kernel
value, which flattens it; per-axis variance 0.159 and anisotropy 1.29, against
the prompt cloud's 0.084 and 3.10.)

The theory-equilibrium arms replace that with an equilibrium of the same game
solved against the empirical prompt distribution, polished to a true fixed
point. One factor moves; split, sigma, rounds, routing rule and generalist are
all held. So this is a clean ablation of what the initialisation is worth --
not a bug report.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from figures_lib import (
    C_ACCENT, C_ORACLE, C_POOLED, C_ROUTER, C_WARN, INK, INK_2, INK_3,
    PAIR_ARMS, PAIR_THRESHOLD, TEXT_WIDTH, THEORY_EQ_PAIR_ARMS, better,
    contrast, despine, diagnostics, save,
)


def _rows():
    """Matched (K, naive, equilibrium) contrasts, skipping any arm not scored."""
    eq = dict(THEORY_EQ_PAIR_ARMS)
    out = []
    for k, base_run in PAIR_ARMS:
        b, t = diagnostics(base_run), diagnostics(eq.get(k, ""))
        if b is None or t is None:
            continue
        out.append((k, contrast(b), contrast(t)))
    return out


def fig_naive_vs_theory() -> None:
    """Routing gain from a naive start and from the game equilibrium.

    The upper panel is the gain curve under each initialisation; the lower is
    their difference, against the noise floor measured from the arms where the
    two agree. The effect is confined to the arms below the original threshold:
    where routing capacity is scarce, where the pairs start decides what they
    become; where it is plentiful, the game reaches the same place regardless.
    """
    rows = _rows()
    if len(rows) < 3:
        return
    ks = np.array([k for k, _b, _t in rows], dtype=float)
    gb = np.array([b["gain"] for _k, b, _t in rows])
    gt = np.array([t["gain"] for _k, _b, t in rows])
    ob = np.array([b["envelope"] for _k, b, _t in rows])
    ot = np.array([t["envelope"] for _k, _b, t in rows])
    d = gt - gb

    below = ks <= PAIR_THRESHOLD
    above = ~below
    sd = float(np.std(d[above], ddof=1)) if above.sum() > 1 else float("nan")

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(TEXT_WIDTH, 4.0), sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.13})

    for a in (ax, ax2):
        a.axvspan(ks.min() - 0.4, PAIR_THRESHOLD, color=C_ACCENT, alpha=0.05, lw=0)
        a.axvline(PAIR_THRESHOLD, color=INK_3, ls=":", lw=0.9, zorder=1)

    ax.plot(ks, ot, "-", color=C_ORACLE, lw=0.9, alpha=0.55, zorder=2)
    ax.plot(ks, ob, "--", color=C_ORACLE, lw=0.9, alpha=0.55, zorder=2)
    ax.fill_between(ks, gb, gt, where=gt >= gb, color=C_ACCENT, alpha=0.16, lw=0,
                    zorder=2, label="gain from the equilibrium start")
    ax.plot(ks, gb, "s--", color=C_POOLED, lw=1.2, ms=3.4, mew=0,
            label="naive start", zorder=4)
    ax.plot(ks, gt, "o-", color=C_ROUTER, lw=1.5, ms=3.6, mew=0,
            label="game equilibrium", zorder=5)
    ax.axhline(0, color=INK_3, lw=0.8, zorder=1)
    ax.set_ylabel("router $-$ pooled (nats)", labelpad=2)
    better(ax, "y", "up")
    ax.legend(loc="lower right", frameon=False, fontsize=6.4, handlelength=2.0)
    ax.text(PAIR_THRESHOLD - 0.12, ax.get_ylim()[1] * 0.97, "scarce capacity",
            ha="right", va="top", fontsize=6.0, color=INK_2, style="italic")
    ax.set_title("Thin blue lines are the oracle ceilings (solid: equilibrium, "
                 "dashed: naive)", loc="left", fontsize=6.6, pad=4, color=INK_2)

    if np.isfinite(sd):
        ax2.axhspan(-2 * sd, 2 * sd, color=INK_3, alpha=0.16, lw=0, zorder=1)
        ax2.text(ks.max(), 2 * sd, r"  $\pm2\sigma$ of the arms that agree",
                 va="bottom", ha="right", fontsize=5.8, color=INK_2)
    ax2.axhline(0, color=INK_3, lw=0.8, zorder=2)
    ax2.bar(ks, d, width=0.52, zorder=3,
            color=[C_ACCENT if k <= PAIR_THRESHOLD else C_POOLED for k in ks])
    for k, v in zip(ks, d):
        ax2.text(k, v + (0.0015 if v >= 0 else -0.0015), "{:+.3f}".format(v),
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=5.8,
                 color=INK if k <= PAIR_THRESHOLD else INK_2)
    ax2.set_ylabel("equilibrium $-$ naive", labelpad=2)
    better(ax2, "y", "up")
    ax2.set_xlabel(r"routing pairs $K$", labelpad=2)
    ax2.set_xticks(ks)
    ax2.set_xlim(ks.min() - 0.5, ks.max() + 0.5)

    for a in (ax, ax2):
        a.grid(zorder=0)
        a.set_axisbelow(True)
        despine(a)

    save(fig, "fig_naive_vs_theory", meta={
        "figure": "naive start vs game equilibrium, by pair count",
        "seed": "seed0",
        "threshold": PAIR_THRESHOLD,
        "below_mean": float(d[below].mean()) if below.any() else None,
        "above_mean": float(d[above].mean()) if above.any() else None,
        "above_sd": sd,
        "per_k": {int(k): round(float(v), 4) for k, v in zip(ks, d)},
        "caveat": "one data split per point",
    })


THEORY_EQ_FIGURES = {"naive_vs_theory": fig_naive_vs_theory}
