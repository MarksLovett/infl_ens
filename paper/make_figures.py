"""Build every figure in the manuscript.

    python paper/make_figures.py              # all figures
    python paper/make_figures.py --only sigma_nll
    python paper/make_figures.py --list

Each figure reads the run outputs under ``results/`` (and ``paper/data/`` for the
replicate splits that live on the training host) and writes a vector PDF into
``paper/figures/``. Nothing is hand-edited afterwards.
"""
from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from figures_lib import (
    AXES_LONG, AXES_SHORT, AXIS_COLORS, C_ACCENT, C_ORACLE, C_POOLED, C_ROUTER,
    C_WARN, GRID, INK, INK_2, INK_3, PAIR_ARMS, ROUTING_ARMS, SCALE_ARMS, SEQ,
    SIGMA_ARMS, SIGMA_STAR, TEXT_WIDTH, contrast, despine, diagnostics,
    history_slim, intra_inter, match_pairs, mean_radius, per_benchmark_gain,
    positions, prompt_coords, save, seeds_for, sigma_contrasts, summarise,
    better,
    use_paper_style,
)
from figures_extra import EXTRA_FIGURES
from figures_ensemble import ENSEMBLE_FIGURES
from figures_tables import TABLE_FIGURES
from figures_theory_eq import THEORY_EQ_FIGURES

SIGMA_TEX = r"$\sigma/\hat\sigma_0^*$"


def fig_routing_ablation() -> None:
    """Five routing designs against the shared generalist baseline."""
    rows = []
    for label, run, sel, k, weight in ROUTING_ARMS:
        d = diagnostics(run)
        if d is None:
            raise SystemExit(f"missing routing diagnostics for {run}")
        rows.append((label, d, sel, k, weight))
    pooled = rows[0][1]["pooled_nll"]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 2.15), gridspec_kw={"width_ratios": [2.6, 1.0], "wspace": 0.08})
    y = np.arange(len(rows))[::-1]

    ax.axvline(pooled, color=C_POOLED, lw=0.9, ls=(0, (3, 2.5)), zorder=2)
    ax.text(pooled - 0.003, -0.55, "pooled generalist", fontsize=6.5,
            color=INK_2, ha="right", va="center")
    for i, (label, d, *_rest) in enumerate(rows):
        o, l = d["oracle_routing_nll"], d["learned_routing_nll"]
        yy = y[i]
        ax.plot([o, l], [yy, yy], color="#c3cbca", lw=1.6, solid_capstyle="round", zorder=3)
        band = C_ACCENT if l < pooled else C_WARN
        ax.plot([min(l, pooled), max(l, pooled)], [yy, yy], color=band, lw=3.4,
                alpha=0.20, solid_capstyle="butt", zorder=2)
        ax.plot([o], [yy], "o", color=C_ORACLE, mec="white", mew=0.7, zorder=5)
        ax.plot([l], [yy], "s", color=C_ROUTER, mec="white", mew=0.7, zorder=5)
        gap = l - pooled
        ax.text(2.083, yy, f"{gap:+.4f}", fontsize=6.8, va="center", ha="right",
                color=C_WARN if gap > 0 else C_ACCENT, family="monospace")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(1.875, 2.086)
    ax.set_ylim(-0.95, len(rows) - 0.35)
    ax.set_xlabel("mean NLL")
    better(ax, "x", "down")
    ax.set_xticks([1.90, 1.95, 2.00])
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.text(2.083, y.max() + 0.60, "router $-$ pooled", fontsize=6.5, color=INK_3,
            ha="right", va="center")
    handles = [Line2D([], [], marker="o", ls="", color=C_ORACLE, mec="white", mew=0.7),
               Line2D([], [], marker="s", ls="", color=C_ROUTER, mec="white", mew=0.7)]
    ax.legend(handles, ["oracle", "router"], loc="upper left",
              bbox_to_anchor=(0.0, 1.14), ncol=2, handletextpad=0.3, columnspacing=1.1)

    agree = [d["routing_agreement_argmax"] for _, d, *_ in rows]
    colors = [C_WARN if a < 0.6 else "#9aa4a3" for a in agree]
    ax2.barh(y, agree, height=0.5, color=colors, zorder=3)
    for i, a in enumerate(agree):
        ax2.text(a + 0.02, y[i], f"{a:.3f}", va="center", fontsize=6.8,
                 color=C_WARN if a < 0.6 else INK_2, family="monospace")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(-0.95, len(rows) - 0.35)
    ax2.set_xlabel("router / oracle agreement")
    better(ax2, "x", "up")
    ax2.grid(axis="x", zorder=0)
    ax2.set_axisbelow(True)
    despine(ax2, keep=("bottom",))
    ax2.tick_params(axis="y", length=0)
    save(fig, "fig_routing_ablation")


def fig_trait_space() -> None:
    """The resource landscape and where the pairs settle in it."""
    coords = prompt_coords()
    pos = positions("seven_axis_soft_full_pairs")
    shown = [(0, 1), (2, 3), (4, 6)]

    fig = plt.figure(figsize=(TEXT_WIDTH, 1.95))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.18], wspace=0.42,
                          left=0.055, right=0.985, top=0.86, bottom=0.20)
    for panel, (i, j) in enumerate(shown):
        ax = fig.add_subplot(gs[0, panel])
        ax.hexbin(coords[:, i], coords[:, j], gridsize=26, cmap="Greys",
                  bins="log", linewidths=0, mincnt=1)
        ax.plot(pos[:, i], pos[:, j], "o", color=C_ROUTER, ms=4.2, mec="white",
                mew=0.8, zorder=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel(AXES_SHORT[i], labelpad=1.5)
        ax.set_ylabel(AXES_SHORT[j], labelpad=1.5)
        despine(ax)

    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(pos.T, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(pos.shape[0]))
    ax.set_xticklabels([str(i) for i in range(pos.shape[0])], fontsize=6)
    ax.set_yticks(range(len(AXES_SHORT)))
    ax.set_yticklabels(AXES_SHORT, fontsize=6)
    ax.set_xlabel("pair", labelpad=1.5)
    despine(ax, keep=())
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.04)
    cb.set_label("coordinate", fontsize=6, labelpad=2)
    cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.outline.set_linewidth(0.4)
    save(fig, "fig_trait_space")


def _sigma_table():
    out = []
    for factor, run in SIGMA_ARMS:
        d = diagnostics(run)
        if d is None:
            raise SystemExit(f"missing routing diagnostics for {run}")
        ia, ie = intra_inter(run)
        out.append({
            "factor": factor, "sigma": factor * SIGMA_STAR,
            "oracle": d["oracle_routing_nll"], "router": d["learned_routing_nll"],
            "pooled": d["pooled_nll"], "agree": d["routing_agreement_argmax"],
            "intra": ia, "inter": ie, "radius": mean_radius(positions(run)),
        })
    return sorted(out, key=lambda r: r["factor"])


def fig_sigma_nll() -> None:
    """Routing NLL against competitive reach."""
    rows = _sigma_table()
    x = np.array([r["factor"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.05),
                             gridspec_kw={"wspace": 0.46})
    ax, ax2, ax3 = axes

    for key, color, marker, label in (
            ("oracle", C_ORACLE, "o", "oracle"),
            ("router", C_ROUTER, "s", "router"),
            ("pooled", C_POOLED, "^", "pooled")):
        ax.plot(x, [r[key] for r in rows], marker=marker, color=color, label=label,
                mec="white", mew=0.6, ms=3.4)
    ax.set_ylabel("mean NLL", labelpad=2)
    better(ax, "y", "down")
    ax.legend(loc="center left", borderpad=0.15, labelspacing=0.2, handletextpad=0.4)
    ax.set_title("routing NLL", loc="left", fontsize=7.5, pad=4)

    gain = np.array([r["router"] - r["pooled"] for r in rows])
    ax2.axhline(0, color=C_POOLED, lw=0.8, ls=(0, (3, 2.5)))
    ax2.plot(x, gain, "s-", color=C_ROUTER, mec="white", mew=0.6, ms=3.4)
    worst = int(np.argmax(gain))
    ax2.annotate(r"nearest $\hat\sigma_0^*$", xy=(x[worst], gain[worst]),
                 xytext=(x[worst] - 0.30, gain[worst] + 0.014), fontsize=6.2,
                 color=C_ACCENT, arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=0.7))
    ax2.set_ylabel(r"router $-$ pooled", labelpad=2)
    better(ax2, "y", "down")
    ax2.set_title("margin over generalist", loc="left", fontsize=7.5, pad=4)

    ax3.plot(x, [r["intra"] for r in rows], "o-", color=C_ORACLE, mec="white",
             mew=0.6, ms=3.4, label="own axis")
    ax3.plot(x, [r["inter"] for r in rows], "s-", color=C_WARN, mec="white",
             mew=0.6, ms=3.4, label="other six")
    ax3.set_ylabel("mean NLL", labelpad=2)
    better(ax3, "y", "down")
    ax3.legend(loc="center right", borderpad=0.15, labelspacing=0.2, handletextpad=0.4, framealpha=0.9, frameon=True, edgecolor="none", facecolor="white")
    ax3.set_title("specialisation", loc="left", fontsize=7.5, pad=4)

    for a in axes:
        a.set_xlabel(r"reach $\sigma/\hat\sigma_0^*$", labelpad=2)
        a.set_xlim(0, 0.75)
        a.grid(zorder=0)
        a.set_axisbelow(True)
        despine(a)
    save(fig, "fig_sigma_nll")


def fig_positions() -> None:
    """Pair positions at three competitive reaches."""
    shown = [(0.70, "seven_axis_sigma07_full"),
             (0.40, "seven_axis_sigma04_full"),
             (0.05, "seven_axis_sigma005_full")]
    P = {f: positions(run) for f, run in shown}
    styles = [(SEQ[0], "-"), (SEQ[1], (0, (5, 2))), (SEQ[2], (0, (1.4, 1.8)))]

    n = 7
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angc = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=(TEXT_WIDTH, 1.55))
    gs = fig.add_gridspec(1, 7, wspace=0.42, left=0.015, right=0.845,
                          top=0.80, bottom=0.04)
    for pi in range(n):
        ax = fig.add_subplot(gs[0, pi], polar=True)
        for (f, _run), (color, ls) in zip(shown, styles):
            v = P[f][pi]
            ax.plot(angc, np.concatenate([v, v[:1]]), color=color, ls=ls, lw=1.0)
        ax.set_ylim(0, 1)
        ax.set_xticks(ang)
        ax.set_xticklabels(AXES_SHORT, fontsize=4.2)
        ax.set_yticks([0.5])
        ax.set_yticklabels([])
        ax.grid(lw=0.35)
        ax.tick_params(pad=-4)
        ax.spines["polar"].set_linewidth(0.5)
        ax.set_title(f"pair {pi}", fontsize=6.2, pad=3)

    handles = [plt.Line2D([], [], color=c, ls=l, lw=1.3) for c, l in styles]
    labels = [rf"$\sigma/\hat\sigma_0^*={f:.2f}$   radius {mean_radius(P[f]):.3f}"
              for f, _ in shown]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.855, 0.48),
               frameon=False, fontsize=6.2, handlelength=2.2, labelspacing=0.5)
    save(fig, "fig_positions")


def fig_replication() -> None:
    """Routing gain across every scored data split.

    Read live from ``results/`` rather than from a checked-in snapshot, so the
    error bars tighten by themselves as further replicates land. Absolute NLL is
    not comparable across splits -- each rebuilds the partition and trains its
    own generalist -- so the quantity plotted is the within-split margin.
    """
    rows = sigma_contrasts()
    # Ascending reach: this panel is a curve now, and a curve needs its x axis
    # to run the way the quantity does.
    factors = sorted(rows)
    splits = sorted({r["seed"] for f in factors for r in rows[f]},
                    key=lambda s: (s != "seed0", s))
    by = {f: {r["seed"]: r for r in rows[f]} for f in factors}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.15),
                                  gridspec_kw={"width_ratios": [1.35, 1.0],
                                               "wspace": 0.34})

    x = np.array(factors)
    per_split = np.full((len(splits), len(factors)), np.nan)
    for si, split in enumerate(splits):
        for fi, f in enumerate(factors):
            if split in by[f]:
                per_split[si, fi] = -by[f][split]["gain"]

    # Mean with a +/-1 SD band is the reading; the individual splits go behind
    # it thin and unlabelled, so the spread is visible without four legend
    # entries competing with the shape.
    mean = np.nanmean(per_split, axis=0)
    sd = np.nanstd(per_split, axis=0, ddof=1)
    ax.fill_between(x, mean - sd, mean + sd, color=C_ACCENT, alpha=0.16, lw=0)
    for si in range(len(splits)):
        ax.plot(x, per_split[si], color=INK_3, lw=0.6, alpha=0.75,
                zorder=2)
    ax.plot(x, mean, color=C_ACCENT, lw=1.5, zorder=3,
            marker="o", ms=2.8, mec="white", mew=0.5)

    # The one setting that is separable from the rest gets said, not implied.
    worst = int(np.nanargmax(mean))
    ax.annotate("collapse at {:.2f}".format(x[worst]),
                xy=(x[worst], mean[worst]), xytext=(-4, 9),
                textcoords="offset points", ha="right", fontsize=6.0,
                color=C_WARN)

    ax.set_xlabel("competitive reach " + SIGMA_TEX)
    ax.set_ylabel("router $-$ pooled NLL")
    better(ax, "y", "down")
    # The sweep is denser near 0.5 than a linear axis has room to label, so the
    # ticks mark every arm and only the uncrowded ones are named.
    labelled = {0.05, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70}
    ax.set_xticks(x)
    ax.set_xticklabels(["{:.2f}".format(f) if f in labelled else ""
                        for f in x], fontsize=6.0)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    base = 0.50
    contrasts = [f for f in factors if f != base]
    means, errs = [], []
    for f in contrasts:
        shared = [s for s in splits if s in by[f] and s in by[base]]
        diff = np.array([(-by[f][s]["gain"]) - (-by[base][s]["gain"])
                         for s in shared])
        means.append(float(diff.mean()) if diff.size else np.nan)
        errs.append(float(diff.std(ddof=1) / math.sqrt(diff.size))
                    if diff.size > 1 else 0.0)
    yy = np.arange(len(contrasts))[::-1]
    ax2.axvline(0, color=C_POOLED, lw=0.8, ls=(0, (3, 2.5)))
    ax2.errorbar(means, yy, xerr=errs, fmt="o", color=C_ROUTER, ecolor=INK_3,
                 elinewidth=0.8, capsize=2, mec="white", mew=0.6)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(["{:.2f} vs 0.50".format(f) for f in contrasts],
                        fontsize=6.2)
    ax2.set_xlabel("paired difference in margin")
    ax2.grid(axis="x", zorder=0)
    ax2.set_axisbelow(True)
    despine(ax2, keep=("bottom",))
    ax2.tick_params(axis="y", length=0)
    save(fig, "fig_replication", meta={
        "figure": "routing margin across data splits",
        "splits": splits, "n_splits": len(splits),
        "sigmas": [float(f) for f in factors],
        "baseline_for_contrasts": base,
    })


# --------------------------------------------------------------- new figures
def fig_pair_count() -> None:
    """Routing NLL against the number of routing pairs (seed 0)."""
    rows = [(k, contrast(diagnostics(run))) for k, run in PAIR_ARMS
            if diagnostics(run) is not None]
    x = np.array([k for k, _ in rows])
    pooled = np.array([c["pooled"] for _, c in rows])
    learned = np.array([c["learned"] for _, c in rows])
    oracle = np.array([c["oracle"] for _, c in rows])

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(TEXT_WIDTH, 3.4), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.16})

    ax.fill_between(x, pooled, learned, where=learned <= pooled, color=C_ACCENT,
                    alpha=0.14, lw=0, label="gain", interpolate=True)
    ax.fill_between(x, pooled, learned, where=learned > pooled, color=C_WARN,
                    alpha=0.18, lw=0, label="routing loses", interpolate=True)
    ax.fill_between(x, learned, oracle, color=C_ORACLE, alpha=0.10, lw=0,
                    label="headroom left")
    ax.axhline(pooled.mean(), color=C_POOLED, lw=0.9, ls=(0, (3, 2.5)),
               label="pooled generalist")
    ax.plot(x, learned, "s-", color=C_ROUTER, ms=3.4, mec="white", mew=0.6,
            label="router", zorder=4)
    ax.plot(x, oracle, "o-", color=C_ORACLE, ms=3.4, mec="white", mew=0.6,
            label="oracle", zorder=4)
    ax.set_ylabel("mean NLL", labelpad=2)
    better(ax, "y", "down")
    ax.legend(loc="lower left", ncol=2, borderpad=0.15, labelspacing=0.25,
              columnspacing=1.1, handletextpad=0.4, fontsize=6.2)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    ax2.plot(x, [c["capture"] * 100 for _, c in rows], "o-", color=C_ACCENT,
             ms=3.4, mec="white", mew=0.6, label="captured")
    ax2.plot(x, [c["agreement"] * 100 for _, c in rows], "s--", color=INK_3,
             ms=3.0, mec="white", mew=0.6, label="oracle agreement")
    ax2.axhline(0, color=INK_3, lw=0.6)
    ax2.set_ylabel("percent", labelpad=2)
    better(ax2, "y", "up")
    ax2.set_xlabel(r"routing pairs (clones $=2\times$ pairs)", labelpad=2)
    ax2.set_xticks(x)
    ax2.legend(loc="lower right", ncol=2, borderpad=0.15, handletextpad=0.4,
               columnspacing=1.1, fontsize=6.2)
    ax2.grid(zorder=0)
    ax2.set_axisbelow(True)
    despine(ax2)

    for a in (ax, ax2):
        a.axvline(4.5, color=INK_3, ls=":", lw=0.9, zorder=0)
    ax.annotate("threshold", xy=(4.5, ax.get_ylim()[0]), xytext=(3, 8),
                textcoords="offset points", fontsize=6.0, color=INK_3)

    save(fig, "fig_pair_count", meta={
        "figure": "pair-count ablation", "seed": "seed0",
        "pairs": [int(v) for v in x],
        "runs": {str(k): run for k, run in PAIR_ARMS},
    })


def fig_sigma_replicates() -> None:
    """The reach sweep averaged over data splits: margin on top, capture below.

    Both series are improvements over each split's *own* pooled generalist, so
    the generalist is the zero line by construction and the axis runs the
    opposite way to a raw NLL plot -- which is why the direction is stated on it.
    The companion ``fig_sigma_table`` carries the same numbers, plus the
    envelope, for reading rather than eyeballing.
    """
    by_sigma = sigma_contrasts()
    sigmas = sorted(by_sigma)
    x = np.arange(len(sigmas))
    n_splits = max(len(rows) for rows in by_sigma.values())

    def band(key):
        mus, sds = [], []
        for s in sigmas:
            mu, sd, _n = summarise([r.get(key) for r in by_sigma[s]])
            mus.append(mu)
            sds.append(sd)
        return np.array(mus), np.array(sds)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(TEXT_WIDTH, 3.5), sharex=True,
        gridspec_kw={"height_ratios": [1.55, 1.0], "hspace": 0.12})

    ax.axhline(0, color=INK_3, ls="--", lw=0.9, label="pooled generalist")
    for key, label, color, marker in (
            ("envelope", "oracle (ceiling)", C_ORACLE, "o"),
            ("gain", "router", C_ROUTER, "s")):
        mus, sds = band(key)
        ax.plot(x, mus, "-", color=color, lw=1.3, marker=marker, ms=3.0,
                mew=0, label=label, zorder=3)
        ax.fill_between(x, mus - sds, mus + sds, color=color, alpha=0.13,
                        lw=0, zorder=1)
        ax.errorbar(x, mus, yerr=sds, fmt="none", ecolor=color, elinewidth=0.8,
                    capsize=1.8, capthick=0.8, zorder=4)
    ax.set_ylabel("improvement over generalist (nats)", labelpad=2)
    better(ax, "y", "up", newline=True)
    ax.legend(ncol=3, frameon=False, fontsize=6.3, loc="lower center",
              handlelength=1.9, columnspacing=1.4)

    mus, sds = band("capture")
    mus, sds = mus * 100, sds * 100
    ax2.plot(x, mus, "-", color=C_ACCENT, lw=1.3, marker="o", ms=3.0, mew=0,
             zorder=3)
    ax2.fill_between(x, mus - sds, mus + sds, color=C_ACCENT, alpha=0.13,
                     lw=0, zorder=1)
    ax2.errorbar(x, mus, yerr=sds, fmt="none", ecolor=C_ACCENT, elinewidth=0.8,
                 capsize=1.8, capthick=0.8, zorder=4)
    ax2.set_ylabel("captured (%)", labelpad=2)
    better(ax2, "y", "up")
    ax2.set_xticks(x)
    ax2.set_xticklabels([("{:.2f}".format(s)).rstrip("0").rstrip(".")
                         for s in sigmas])
    ax2.set_xlabel("competitive reach " + SIGMA_TEX, labelpad=2)

    for a in (ax, ax2):
        a.grid(zorder=0)
        a.set_axisbelow(True)
        despine(a)
    ax.set_title(r"mean $\pm$ 1 SD over {} data splits".format(n_splits),
                 loc="left", fontsize=7.5, pad=4)

    save(fig, "fig_sigma_replicates", meta={
        "figure": "reach sweep over data splits",
        "n_per_sigma": {"{:.2f}".format(s): len(by_sigma[s]) for s in sigmas},
        "splits": sorted({r["seed"] for rows in by_sigma.values() for r in rows}),
    })


FIGURES = {
    "routing_ablation": fig_routing_ablation,
    "trait_space": fig_trait_space,
    "sigma_nll": fig_sigma_nll,
    "positions": fig_positions,
    "replication": fig_replication,
    "pair_count": fig_pair_count,
    "sigma_replicates": fig_sigma_replicates,
}

# Figures added for the results walkthrough live in their own module; they are
# folded in here so `python make_figures.py` stays the one regeneration command.
FIGURES.update(EXTRA_FIGURES)
FIGURES.update(ENSEMBLE_FIGURES)
FIGURES.update(TABLE_FIGURES)
FIGURES.update(THEORY_EQ_FIGURES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES),
                        help="Build only the named figures.")
    parser.add_argument("--list", action="store_true", help="List figure names and exit.")
    args = parser.parse_args()
    if args.list:
        print("\n".join(sorted(FIGURES)))
        return
    use_paper_style()
    for name in (args.only or sorted(FIGURES)):
        print(f"{name}:")
        FIGURES[name]()


if __name__ == "__main__":
    main()
