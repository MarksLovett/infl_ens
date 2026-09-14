"""Tables rendered as figures.

Some of these results are read, not seen: ten reach settings differing in the
third decimal are a table, and drawing them as two nearly-flat lines asks the
reader to decode a picture that carries less information than the numbers. This
module renders those as paper-quality tables so they still ship as PDF + PNG
through the one ``make_figures.py`` entry point.

Shading is per column. Each column is min-max scaled within itself, so colour
ranks rows inside a column and never compares across columns -- the columns are
in different units.
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from figures_lib import (
    INK, INK_2, TEXT_WIDTH, despine, drift_stats, save, sigma_contrasts, summarise,
)

HEAT = mcolors.LinearSegmentedColormap.from_list(
    "seq", ["#eaf1fb", "#9ec5f4", "#2a78d6"])


def _cell(mean: float, sd: float, n: int, pct: bool) -> tuple[str, str]:
    """Main value and its spread, formatted for a two-line cell."""
    if pct:
        return "{:.1f}%".format(mean * 100), ("" if n < 2 else
                                              "$\\pm${:.1f}".format(sd * 100))
    return "{:+.4f}".format(mean), ("" if n < 2 else "$\\pm${:.4f}".format(sd))


def fig_sigma_table() -> None:
    """The reach sweep as a table: router gain, envelope, capture, agreement.

    Replaces the two line panels and the stacked-bar envelope figure. Those
    plotted quantities that differ by ~0.005 nats across most of the sweep, so
    the eye could not separate the settings and the envelope bars needed a
    second figure to say what one column says here.

    Every entry is a within-split difference from that split's own pooled
    generalist. Absolute NLL is not comparable across data splits -- each has
    its own test partition and its own generalist -- so the raw numbers are
    deliberately absent.
    """
    by_sigma = sigma_contrasts()
    sigmas = sorted(by_sigma, reverse=True)

    # column key, header, is-a-percentage, shade it
    spec = [
        ("gain", "router $-$ pooled", False, True),
        ("envelope", "oracle $-$ pooled", False, True),
        ("capture", "captured", True, True),
        ("agreement", "argmax agreement", True, False),
    ]

    means = np.full((len(sigmas), len(spec)), np.nan)
    sds = np.zeros_like(means)
    ns = np.zeros(len(sigmas), dtype=int)
    for ri, s in enumerate(sigmas):
        rows = by_sigma[s]
        ns[ri] = len(rows)
        for ci, (key, _h, _p, _sh) in enumerate(spec):
            mu, sd, n = summarise([r.get(key) for r in rows])
            if n:
                means[ri, ci], sds[ri, ci] = mu, sd

    shade = np.full_like(means, np.nan)
    for ci, (_k, _h, _p, do_shade) in enumerate(spec):
        if not do_shade:
            continue
        col = means[:, ci]
        lo, hi = np.nanmin(col), np.nanmax(col)
        shade[:, ci] = (col - lo) / (hi - lo) if hi > lo else 0.5

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.55))
    fig.subplots_adjust(left=0.145, right=0.995, top=0.875, bottom=0.235)
    ax.imshow(np.nan_to_num(shade, nan=0.0), cmap=HEAT, aspect="auto",
              vmin=0, vmax=1)

    best = {ci: int(np.nanargmax(means[:, ci])) for ci in range(len(spec))}
    for ri in range(means.shape[0]):
        for ci, (_k, _h, pct, do_shade) in enumerate(spec):
            if np.isnan(means[ri, ci]):
                continue
            top, bot = _cell(means[ri, ci], sds[ri, ci], ns[ri], pct)
            dark = do_shade and shade[ri, ci] > 0.62
            color = "white" if dark else INK
            ax.text(ci, ri - 0.15, top, ha="center", va="center", fontsize=6.4,
                    color=color,
                    fontweight="bold" if best[ci] == ri else "normal")
            if bot:
                ax.text(ci, ri + 0.22, bot, ha="center", va="center",
                        fontsize=5.1,
                        color="#dbe7f7" if dark else INK_2)

    ax.set_xticks(range(len(spec)))
    ax.set_xticklabels([h for _k, h, _p, _s in spec], fontsize=6.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(sigmas)))
    ax.set_yticklabels(
        ["{:.2f}".format(s) + ("  $n{=}$" + str(ns[ri]) if ns[ri] else "")
         for ri, s in enumerate(sigmas)], fontsize=6.5)
    ax.set_ylabel(r"competitive reach $\sigma/\hat\sigma_0^*$", fontsize=7,
                  labelpad=4)
    ax.tick_params(length=0)
    despine(ax, keep=())

    ax.set_title(r"Higher is better in every column ($\uparrow$)."
                 "  Bold marks the best setting in a column.",
                 loc="left", fontsize=7.0, pad=18)

    fig.text(0.005, 0.20,
             "Mean $\\pm$ sample SD over the data splits scored for that "
             "setting; $n$ is on each row. Every entry is a difference from "
             "that split's\nown pooled generalist, because absolute NLL is not "
             "comparable across splits. Shading is min-max within a column "
             "only.\nCapture is (router $-$ pooled)/(oracle $-$ pooled): the "
             "share of the available headroom the router actually took. It "
             "peaks at\n$\\sigma/\\hat\sigma_0^*=0.20$ and collapses at $0.70$, "
             "where the specialists stop separating.",
             fontsize=5.6, color=INK_2, va="top")

    save(fig, "fig_sigma_table", meta={
        "figure": "reach sweep table (replaces sigma_replicates and gain_envelope)",
        "n_per_sigma": {"{:.2f}".format(s): int(ns[i])
                        for i, s in enumerate(sigmas)},
        "splits": sorted({r["seed"] for rows in by_sigma.values() for r in rows}),
        "columns": [k for k, _h, _p, _s in spec],
        "shading": "per-column min-max",
    })


TABLE_FIGURES = {"sigma_table": fig_sigma_table}


# ------------------------------------------------- theory drift: two baselines
DRIFT_ROWS = [
    ("reach 0.05", "seven_axis_sigma005_full"),
    ("reach 0.50", "seven_axis_soft_full_pairs"),
    ("reach 0.70", "seven_axis_sigma07_full"),
    ("2 pairs", "seven_axis_2pair_sigma05"),
    ("5 pairs", "seven_axis_5pair_sigma05"),
]


def fig_theory_drift() -> None:
    """How far the trained pairs sit from each of the two theory solves.

    Every run was initialised from the *approximate* solve, because the shipped
    ``B(b)`` is not a density of the prompt cloud -- ``_kde_on_grid`` divides each
    grid point by its own nearest-prompt kernel value, so what the solve saw is
    close to a uniform prior over the lattice. The empirical-``B`` solve is the
    same game against the real distribution, computed offline for comparison.
    No run was re-trained and no experiment changed.

    The two distance columns answer the only question here: how far training
    moved the pairs from where they started, and how far they ended up from
    where the theory would have put them had it seen the data.
    """
    rows = []
    for label, run in DRIFT_ROWS:
        d = drift_stats(run)
        if d is not None:
            rows.append((label, d))
    if not rows:
        return

    spec = [
        ("mean_l2", r"$\|$trained $-$ approx$\|$", True),
        ("mean_l2_true", r"$\|$trained $-$ empirical$\|$", True),
        ("theory_gap", r"$\|$approx $-$ empirical$\|$", False),
        ("radius_final", "trained radius", False),
    ]
    means = np.full((len(rows), len(spec)), np.nan)
    for ri, (_l, d) in enumerate(rows):
        for ci, (key, _h, _s) in enumerate(spec):
            v = d.get(key)
            if v is not None:
                means[ri, ci] = float(v)

    shade = np.full_like(means, np.nan)
    for ci, (_k, _h, do_shade) in enumerate(spec):
        if not do_shade:
            continue
        col = means[:, ci]
        if np.isnan(col).all():
            continue
        lo, hi = np.nanmin(col), np.nanmax(col)
        shade[:, ci] = (col - lo) / (hi - lo) if hi > lo else 0.5
    # A cell with no number must not be shaded as if it had one.
    shade[np.isnan(means)] = np.nan

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.15))
    fig.subplots_adjust(left=0.175, right=0.995, top=0.80, bottom=0.30)
    cmap = HEAT.copy()
    cmap.set_bad("#f4f5f6")
    ax.imshow(np.ma.masked_invalid(shade), cmap=cmap, aspect="auto",
              vmin=0, vmax=1)

    for ri in range(means.shape[0]):
        for ci, (_k, _h, do_shade) in enumerate(spec):
            if np.isnan(means[ri, ci]):
                ax.text(ci, ri, "pending", ha="center", va="center",
                        fontsize=5.6, style="italic", color=INK_2)
                continue
            dark = do_shade and shade[ri, ci] > 0.62
            ax.text(ci, ri, "{:.3f}".format(means[ri, ci]), ha="center",
                    va="center", fontsize=6.4, color="white" if dark else INK)

    ax.set_xticks(range(len(spec)))
    ax.set_xticklabels([h for _k, h, _s in spec], fontsize=6.3)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        ["{}  ($K{{=}}${})".format(lbl, d["n_pairs"]) for lbl, d in rows],
        fontsize=6.5)
    ax.tick_params(length=0)
    despine(ax, keep=())
    ax.set_title("Mean L2 per pair, split 0. These are distances, not scores: "
                 "neither column has a good direction.",
                 loc="left", fontsize=6.8, pad=16)

    fig.text(0.005, 0.255,
             "Pairs are Hungarian-matched before differencing, because the two "
             "solves index pairs independently. The space has diagonal "
             "$\sqrt{7}\approx2.65$.\nEvery run was initialised from the "
             "approximate solve; the empirical-$B$ solve is offline comparison "
             "only. Shaded columns are min-max within the column.",
             fontsize=5.6, color=INK_2, va="top")

    save(fig, "fig_theory_drift", meta={
        "figure": "distance from trained positions to each theory solve",
        "seed": "seed0",
        "rows": {lbl: {k: (None if np.isnan(means[ri, ci]) else
                           round(float(means[ri, ci]), 4))
                       for ci, (k, _h, _s) in enumerate(spec)}
                 for ri, (lbl, _d) in enumerate(rows)},
        "matched_by": "Hungarian on final positions",
    })


TABLE_FIGURES["theory_drift"] = fig_theory_drift


# Paired naive-vs-equilibrium reach table; see figures_init_table for why it
# lives in its own module.
from figures_init_table import fig_sigma_init_table  # noqa: E402

TABLE_FIGURES["sigma_init_table"] = fig_sigma_init_table
