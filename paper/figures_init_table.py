"""Naive start against the game equilibrium, at matched competitive reach.

This is the one comparison in the study that is genuinely paired. Both
initialisations are scored on the identical test partition (n=5671) against the
identical pooled generalist (2.0152 nats), with only the starting positions
moved, so each row is a within-prompt contrast rather than a cross-split one --
the usual "absolute NLL is not comparable across splits" caveat does not bite
here, and the difference column is meaningful in a way the sigma sweep's is not.

Kept in its own module because it is the only table that pairs two run families
rather than summarising one, and its column logic has nothing in common with
``fig_sigma_table``.
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from figures_lib import (
    INK, INK_2, SIGMA_DEGENERATE, TEXT_WIDTH, despine, monotone_steps,
    naive_sigma_contrasts, roughness, save, theory_eq_sigma_contrasts,
)

HEAT = mcolors.LinearSegmentedColormap.from_list(
    "seq", ["#eaf1fb", "#9ec5f4", "#2a78d6"])
# Differences are signed and small, so they get a colormap where a sign change
# is a hue change rather than a shade change.
DIVERGE = mcolors.LinearSegmentedColormap.from_list(
    "div", ["#b0341f", "#f2ece8", "#0b8f84"])

# column key, header, is-a-difference
SPEC = [
    ("gain", "naive", False),
    ("gain", "equil.", False),
    ("gain", "difference", True),
    ("gain_mixture", "naive", False),
    ("gain_mixture", "equil.", False),
    ("gain_mixture", "difference", True),
]


def fig_sigma_init_table() -> None:
    """Paired reach table: naive vs equilibrium, expected and mixture.

    Read the difference columns, not the mean of them. On the expected metric
    the equilibrium is ahead at eight of the ten reaches, but the two it loses
    include the largest single difference in the sweep, so the mean difference
    is about +0.0007 while the median is about +0.0027.

    The gap between mean and median is reach 0.20, and it would be wrong to
    dismiss that as an outlier: the naive start beats its own split's mean at
    reach 0.20 in *all four* naive data splits, so the advantage replicates.
    What does not replicate is its size -- seed 0's excess over its split mean
    is +0.0220 against +0.0069, +0.0075 and +0.0042 on the other three. The
    equilibrium arm at reach 0.20 sits exactly at its own sweep mean, so the
    whole difference is a naive bump rather than an equilibrium dip. On a
    typical split the difference there would be nearer -0.008 than -0.0215,
    which is why the median is the better summary of this column.

    The advantage does not carry to the mixture columns, where the equilibrium
    is ahead at only four of ten and the mean difference is about -0.0019.
    Mixing recovers much of what a mis-placed specialist costs, so a better
    starting geometry buys least where the combination rule is already
    forgiving -- which is why the win counts are reported per metric below and
    not pooled into one headline.

    What the equilibrium reliably changes is not the level but the shape. The
    footer reports mean absolute second difference along the reach axis, which
    annihilates trends and so measures zigzag alone. That is the honest
    statistic here: the mixture column trends upward with reach under *both*
    initialisations, so a standard deviation charges both curves for the same
    real trend, lands at about 0.0156 either way, and reports no difference at
    all -- even though only the equilibrium's climb is near-monotone (7 of 8
    steps, against 4 of 8).
    """
    eq_by, nv_by = theory_eq_sigma_contrasts(), naive_sigma_contrasts()
    eq, nv = eq_by.get("seed0", {}), nv_by.get("seed0", {})
    sigmas = sorted(set(eq) & set(nv), reverse=True)
    if not sigmas:
        return

    vals = np.full((len(sigmas), len(SPEC)), np.nan)
    for ri, sg in enumerate(sigmas):
        for ci, (key, head, is_diff) in enumerate(SPEC):
            a, b = nv[sg].get(key), eq[sg].get(key)
            if is_diff:
                if a is not None and b is not None:
                    vals[ri, ci] = b - a
            elif head == "naive" and a is not None:
                vals[ri, ci] = a
            elif head == "equil." and b is not None:
                vals[ri, ci] = b

    rgba = np.ones((len(sigmas), len(SPEC), 4))
    for ci, (_k, _h, is_diff) in enumerate(SPEC):
        col = vals[:, ci]
        if np.all(np.isnan(col)):
            continue
        if is_diff:
            scale = np.nanmax(np.abs(col)) or 1.0
            rgba[:, ci] = DIVERGE(np.nan_to_num((col / scale + 1) / 2, nan=0.5))
        else:
            lo, hi = np.nanmin(col), np.nanmax(col)
            t = (col - lo) / (hi - lo) if hi > lo else np.full_like(col, 0.5)
            rgba[:, ci] = HEAT(np.nan_to_num(t, nan=0.0))

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 4.35))
    fig.subplots_adjust(left=0.150, right=0.995, top=0.752, bottom=0.315)
    ax.imshow(rgba, aspect="auto")

    for ri in range(len(sigmas)):
        for ci in range(len(SPEC)):
            if np.isnan(vals[ri, ci]):
                continue
            r, g, b = rgba[ri, ci, :3]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(ci, ri, "{:+.4f}".format(vals[ri, ci]), ha="center",
                    va="center", fontsize=6.3,
                    color="white" if lum < 0.55 else INK)

    ax.set_xticks(range(len(SPEC)))
    ax.set_xticklabels([h for _k, h, _d in SPEC], fontsize=6.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(sigmas)))
    ax.set_yticklabels(
        ["{:.2f}".format(sg) + ("  (degenerate)" if sg == SIGMA_DEGENERATE else "")
         for sg in sigmas], fontsize=6.5)
    ax.set_ylabel(r"competitive reach $\sigma/\hat\sigma_0^*$", fontsize=7,
                  labelpad=4)
    ax.tick_params(length=0)
    despine(ax, keep=())

    for xc, lab in ((1.0, "commit to one expert (expected)"),
                    (4.0, "sequence-level mixture")):
        ax.text(xc, -1.68, lab, ha="center", va="center", fontsize=6.8,
                color=INK)
    ax.plot([2.5, 2.5], [-0.5, len(sigmas) - 0.5], color="white", lw=1.8,
            clip_on=False)

    ax.set_title("Higher is better everywhere ($\\uparrow$).  Difference is "
                 "equilibrium $-$ naive, so $>0$ favours the equilibrium.",
                 loc="left", fontsize=6.8, pad=36)

    # Ascending reach: the table rows run high-to-low for reading, but a
    # "rises with reach" count is only meaningful in increasing order.
    # Roughness is reversal-invariant; the step counts are not.
    keep = sorted(sg for sg in sigmas if sg != SIGMA_DEGENERATE)
    lines, stats = [], {}
    for key, name in (("gain", "expected"), ("gain_mixture", "mixture")):
        a = [nv[sg].get(key) for sg in keep]
        b = [eq[sg].get(key) for sg in keep]
        if any(v is None for v in a + b):
            continue
        ra, rb = roughness(a), roughness(b)
        ua, da = monotone_steps(a)
        ub, db = monotone_steps(b)
        stats[name] = {"roughness_naive": ra, "roughness_equilibrium": rb,
                       "ratio": ra / rb if rb else None,
                       "rising_naive": [ua, ua + da],
                       "rising_equilibrium": [ub, ub + db]}
        lines.append(
            "{:9s} roughness  naive {:.5f}  vs  equilibrium {:.5f}   "
            "({:.1f}$\\times$ smoother);   steps rising with reach "
            "{}/{} vs {}/{}".format(
                name, ra, rb, (ra / rb) if rb else float("nan"),
                ua, ua + da, ub, ub + db))

    # Win counts are reported per metric, not pooled: the equilibrium's edge is
    # specific to committing to one expert. Under sequence-level mixing the two
    # initialisations are level, because mixing recovers much of what a
    # mis-placed specialist loses, so a single headline count would hide the
    # one asymmetry in the table worth noticing.
    tally = {}
    for name, ci in (("expected", 2), ("mixture", 5)):
        d = [v for v in vals[:, ci] if not np.isnan(v)]
        tally[name] = {"n": len(d), "wins": sum(1 for v in d if v > 0),
                       "mean": float(np.mean(d)), "median": float(np.median(d))}
    diffs = [v for v in vals[:, 2] if not np.isnan(v)]
    wins = tally["expected"]["wins"]

    fig.text(0.005, 0.272,
             "Paired: identical test partition ($n{{=}}5671$), identical pooled "
             "generalist ($2.0152$ nats), seed 0. Only the starting\n"
             "positions move.\n"
             "Expected: equilibrium ahead at {} of {}; mean ${:+.4f}$ but "
             "median ${:+.4f}$. The gap between them is reach 0.20,\n"
             "where the naive start is genuinely good: it beats its own split's "
             "mean in all four naive splits. Seed 0\nexaggerates the size "
             "(excess $+0.0220$ against $+0.0069$, $+0.0075$, $+0.0042$), so "
             "the median is the better\nsummary -- but the naive advantage "
             "there is real, not noise.  Mixture: ahead at only {} of {}, mean\n"
             "${:+.4f}$. The equilibrium's edge is specific to committing to "
             "one expert; under mixing the two starts\nare level, and only the "
             "smoothness survives.\n"
             "Roughness is mean $|$second difference$|$ along the reach axis; "
             "it cancels trends, so it measures zigzag, not\nspread. The "
             "mixture columns both trend upward with reach, so SD is\ninflated "
             "equally for both and reports no difference; only the "
             "equilibrium's climb is\nnear-monotone.\n{}\n"
             "Reach 0.70 is excluded from these footer statistics: its "
             "equilibrium is degenerate, placing seven pairs on\nfive distinct "
             "positions.".format(
                 wins, tally["expected"]["n"], tally["expected"]["mean"],
                 tally["expected"]["median"], tally["mixture"]["wins"],
                 tally["mixture"]["n"], tally["mixture"]["mean"],
                 "\n".join(lines)),
             fontsize=5.5, color=INK_2, va="top", linespacing=1.45)

    save(fig, "fig_sigma_init_table", meta={
        "figure": "naive start vs game equilibrium at matched reach (paired)",
        "split": "seed0",
        "paired_basis": "same test pool (n=5671), same generalist (2.0152)",
        "n_reaches": len(sigmas),
        
        "by_metric": tally,
        "smoothness": stats,
        "excluded_from_footer": SIGMA_DEGENERATE,
        "dseed1_equilibrium_arms_scored": len(eq_by.get("dseed1", {})),
    })
