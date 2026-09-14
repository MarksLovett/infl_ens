"""Figures added for the results walkthrough.

Kept in their own module so ``make_figures.py`` stays readable; they are
registered into its ``FIGURES`` table, so ``python make_figures.py`` remains the
one command that regenerates everything.

Two conventions carried from the manuscript figures: sizes are in inches against
the ICLR text block, and colour is never the only channel that carries meaning.
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from figures_lib import (
    AXES_LONG, AXIS_COLORS, C_ACCENT, C_ORACLE, C_POOLED, C_ROUTER, C_WARN,
    INK, INK_2, INK_3, better, PAIR_ARMS, ROUTING_ARMS, SCALE_ARMS, SEQ, SIGMA_ARMS,
    SIGMA_RUN_ALIASES, TEXT_WIDTH, contrast, despine, diagnostics, history_slim,
    true_theory,
    match_pairs, per_benchmark_gain, save, seeds_for, sigma_contrasts, summarise,
)
from radar_lib import draw_polygon, emphasise_spoke, radar_axes

# Benchmark rows in trait-axis order, with the axis each one defines.
BENCH_ORDER = ["beavertails", "halueval", "jbb_behaviors", "ai4privacy",
               "orbench", "prompt_injection", "do_not_answer"]
BENCH_LABELS = ["harm", "hallucination", "jailbreak (n=40)", "privacy",
                "over-refusal", "injection", "policy"]


# ------------------------------------------------------------------- radars
def fig_radar_sigma() -> None:
    """One radar per pair, three competitive reaches overlaid (seed 0).

    Pairs are matched across runs by nearest final position to the 0.50 run.
    Pair index comes from the theory solve's ordering and is not an identity, so
    overlaying by index would silently compare different specialists.
    """
    shown = [(0.05, "seven_axis_sigma005_full"),
             (0.50, "seven_axis_soft_full_pairs"),
             (0.70, "seven_axis_sigma07_full")]
    hist = {f: history_slim(run) for f, run in shown}
    ref = hist[0.50]["positions"][-1]
    perms, costs = {}, {}
    for f, _ in shown:
        perms[f], costs[f] = match_pairs(ref, hist[f]["positions"][-1])
    styles = [(SEQ[2], (0, (1.4, 1.6))), (SEQ[1], (0, (5, 2))), (SEQ[0], "-")]

    n_pairs = ref.shape[0]
    fig = plt.figure(figsize=(7.4, 1.85))
    gs = fig.add_gridspec(1, n_pairs, wspace=1.05, left=0.02, right=0.845,
                          top=0.80, bottom=0.06)
    for pi in range(n_pairs):
        ax = radar_axes(fig, gs[0, pi], title="pair {}".format(pi))
        for (f, _run), (color, ls) in zip(shown, styles):
            draw_polygon(ax, hist[f]["positions"][-1][perms[f][pi]],
                         color=color, ls=ls, lw=1.0, fill_alpha=0.06)

    handles = [plt.Line2D([], [], color=c, ls=l, lw=1.3) for c, l in styles]
    labels = [r"$\sigma/\hat\sigma_0^*={:.2f}$".format(f) for f, _ in shown]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.855, 0.46),
               frameon=False, fontsize=6.2, handlelength=2.2, labelspacing=0.55)
    # sigma_0* is computed from the shipped B, which is near-uniform over the
    # lattice rather than a density of the prompt cloud. The absolute reaches the
    # runs used are unchanged; only the constant they are quoted against is
    # approximate, so it is marked with a hat wherever it appears.
    fig.text(0.855, 0.10,
             r"$\hat\sigma_0^*=0.410$ (approximate $B$)" "\n"
             r"empirical $B$ gives $0.369$",
             ha="left", va="bottom", fontsize=5.4, color=INK_2)
    save(fig, "fig_radar_sigma", meta={
        "figure": "pair positions vs competitive reach",
        "seed": "seed0", "reference": "sigma 0.50",
        "matched_by": "nearest final position (Hungarian)",
        "assignment": {"{:.2f}".format(f): [int(i) for i in perms[f]] for f, _ in shown},
        "assignment_cost": {"{:.2f}".format(f): costs[f] for f, _ in shown},
        "sigma0_note": "quoted against the approximate sigma_0* = 0.40977; "
                       "the empirical measure gives 0.36905",
    })


def fig_radar_pairs() -> None:
    """Pair positions as the ensemble grows: 2, 5 and 7 pairs (seed 0).

    Each trained position is drawn in the hue of the axis that pair specialises
    on, with that spoke's label bolded so the hue is redundant rather than
    load-bearing. The theory-solved starting position sits behind in grey, so
    the panel shows how far training moved the pair from where the game placed it.
    """
    rows = [(2, "seven_axis_2pair_sigma05"), (5, "seven_axis_5pair_sigma05"),
            (7, "seven_axis_soft_full_pairs")]
    hist = [(k, run, history_slim(run)) for k, run in rows]
    span, wide = 2, 14                      # 7 panels * 2 columns, rows centred

    fig = plt.figure(figsize=(7.4, 4.05))
    gs = fig.add_gridspec(len(hist), wide, wspace=1.15, hspace=0.70,
                          left=0.075, right=0.975, top=0.855, bottom=0.10)
    have_true = False
    for ri, (k, run, h) in enumerate(hist):
        order = np.argsort(h["dominant_axis"])
        off = (wide - k * span) // 2
        # The empirical-measure solve is a separate solve, so its pair index is
        # not the trained one; match before overlaying.
        tt = true_theory(run)
        true_pos = None
        if tt is not None and len(tt["positions"]) == h["positions"].shape[1]:
            perm, _cost = match_pairs(h["positions"][-1], tt["positions"])
            true_pos = tt["positions"][perm]
            have_true = True
        for ci, pi in enumerate(order):
            ax = radar_axes(fig, gs[ri, off + ci * span:off + (ci + 1) * span])
            dom = int(h["dominant_axis"][pi])
            color = AXIS_COLORS[dom] if 0 <= dom < len(AXIS_COLORS) else INK_2
            if h["theory_positions"] is not None:
                draw_polygon(ax, h["theory_positions"][pi], color=INK_3,
                             ls=(0, (2, 1.5)), lw=0.7, zorder=2)
            if true_pos is not None:
                draw_polygon(ax, true_pos[pi], color=C_WARN,
                             ls=(0, (4, 1.2, 1, 1.2)), lw=0.85, zorder=3)
            draw_polygon(ax, h["positions"][-1][pi], color=color, lw=1.1,
                         fill_alpha=0.10, zorder=4)
            emphasise_spoke(ax, dom, color)
        fig.text(0.015, 0.865 - (ri + 0.5) * (0.755 / len(hist)),
                 "{} pairs".format(k), rotation=90, va="center", ha="center",
                 fontsize=7.5)

    handles = [plt.Line2D([], [], color=INK_2, lw=1.1),
               plt.Line2D([], [], color=INK_3, ls=(0, (2, 1.5)), lw=0.7)]
    labels = ["trained (round 11)", r"theory init, approximate $B$"]
    if have_true:
        handles.append(plt.Line2D([], [], color=C_WARN,
                                  ls=(0, (4, 1.2, 1, 1.2)), lw=0.85))
        labels.append(r"theory, empirical $B$")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.005),
               ncol=3, frameon=False, fontsize=6.5, handlelength=2.0)
    fig.text(0.5, 0.012,
             "The runs were initialised from the approximate solve. The "
             "empirical-$B$ solve is shown for comparison only; no run was "
             "re-trained.",
             ha="center", va="bottom", fontsize=5.8, color=INK_2, style="italic")
    save(fig, "fig_radar_pairs", meta={
        "figure": "pair positions vs pair count", "seed": "seed0",
        "rows": [k for k, _ in rows], "axis_order": AXES_LONG,
        "dominant_axis": {str(k): [int(d) for d in h["dominant_axis"]]
                          for k, _r, h in hist},
        "empirical_theory_overlay": have_true,
        "caveat": ("theory init used B from _kde_on_grid, which is near-uniform "
                   "over the lattice; the empirical-B solve is the comparison"),
    })


# ------------------------------------------------------------- per-benchmark
SIGMA_ASC = sorted(SIGMA_ARMS, key=lambda t: t[0])


def _sigma_bench_matrix() -> np.ndarray:
    """Mean per-benchmark gain at each reach, averaged over scored splits."""
    out = np.full((len(BENCH_ORDER), len(SIGMA_ASC)), np.nan)
    for ci, (factor, run) in enumerate(SIGMA_ASC):
        acc: dict[str, dict[str, float]] = {}
        for candidate in SIGMA_RUN_ALIASES.get(factor, [run]):
            for sub in seeds_for(candidate):
                for bench, val in per_benchmark_gain(candidate, sub).items():
                    acc.setdefault(bench, {})[sub] = val
        for ri, bench in enumerate(BENCH_ORDER):
            vals = list(acc.get(bench, {}).values())
            if vals:
                out[ri, ci] = float(np.mean(vals))
    return out


def _pair_bench_matrix() -> np.ndarray:
    out = np.full((len(BENCH_ORDER), len(PAIR_ARMS)), np.nan)
    for ci, (_k, run) in enumerate(PAIR_ARMS):
        gains = per_benchmark_gain(run)
        for ri, bench in enumerate(BENCH_ORDER):
            if bench in gains:
                out[ri, ci] = gains[bench]
    return out


def fig_per_benchmark() -> None:
    """Where the routing gain actually comes from, axis by axis.

    A single headline number hides that routing helps enormously on injection
    and jailbreak while consistently *hurting* privacy. One diverging scale is
    shared by both panels so the two are directly comparable.
    """
    m_sigma, m_pairs = _sigma_bench_matrix(), _pair_bench_matrix()
    lim = float(np.nanmax(np.abs(np.concatenate([m_sigma.ravel(), m_pairs.ravel()]))))
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gain", ["#7a1f1f", "#e34948", "#f0efec", "#2a78d6", "#0d366b"])

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 2.4),
        gridspec_kw={"width_ratios": [len(SIGMA_ASC), len(PAIR_ARMS)],
                     "wspace": 0.06})
    panels = (
        (ax, m_sigma, ["{:g}".format(f) for f, _ in SIGMA_ASC],
         r"reach $\sigma/\hat\sigma_0^*$"),
        (ax2, m_pairs, [str(k) for k, _ in PAIR_ARMS], "routing pairs"),
    )
    im = None
    for axis, mat, cols, xlabel in panels:
        im = axis.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        axis.set_xticks(range(len(cols)))
        axis.set_xticklabels(cols, fontsize=5.8)
        axis.set_xlabel(xlabel, labelpad=2)
        for ri in range(mat.shape[0]):
            for ci in range(mat.shape[1]):
                if np.isnan(mat[ri, ci]):
                    continue
                axis.text(ci, ri, "{:+.2f}".format(mat[ri, ci]).replace("0.", "."),
                          ha="center", va="center", fontsize=4.5,
                          color="white" if abs(mat[ri, ci]) > 0.62 * lim else INK)
        despine(axis, keep=())
        axis.tick_params(length=0)
    ax.set_yticks(range(len(BENCH_LABELS)))
    ax.set_yticklabels(BENCH_LABELS, fontsize=6)
    ax2.set_yticks([])
    ax.set_title("across reach (mean over splits)", loc="left", fontsize=7)
    ax2.set_title("across pair count (seed 0)", loc="left", fontsize=7)
    cb = fig.colorbar(im, ax=[ax, ax2], fraction=0.028, pad=0.02)
    cb.set_label(r"routing gain, nats ($\uparrow$ better)",
             fontsize=6, labelpad=2)
    cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.outline.set_linewidth(0.4)
    save(fig, "fig_per_benchmark", meta={
        "figure": "per-benchmark routing gain", "benchmarks": BENCH_ORDER,
        "sigmas": [f for f, _ in SIGMA_ASC], "pairs": [k for k, _ in PAIR_ARMS],
    })


# ----------------------------------------------------------- model families
def fig_model_families() -> None:
    """Base model sweep as a heat table.

    Only the ratio columns are shaded. Per-token NLL is not comparable across
    tokenizers (Qwen ~152k, Llama 128k, Gemma 262k), so shading those would
    invite exactly the cross-family comparison the numbers cannot support.
    """
    rows = []
    for label, run, params, family in SCALE_ARMS:
        diag = diagnostics(run)
        if diag is not None:
            rows.append((label, family, params, contrast(diag)))

    cols = ["captured", "agreement", "pooled", "router", "oracle"]
    shaded = [True, True, False, False, False]
    mat = np.array([[c["capture"], c["agreement"], c["pooled"],
                     c["learned"], c["oracle"]] for _l, _f, _p, c in rows])

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 1.85))
    # Scale the shading on the trustworthy rows only. Gemma's capture is
    # inflated by an unconverged generalist, and letting it set the top of the
    # scale flattens the Qwen ladder into one indistinguishable block. Its own
    # cell is clipped to the top rather than dropped, and it stays daggered.
    trusted = np.array(["Gemma" not in lbl for lbl, _f, _p, _c in rows])
    shade = np.full_like(mat, np.nan)
    for ci, use in enumerate(shaded):
        if use:
            col = mat[:, ci]
            ref = col[trusted] if trusted.any() else col
            lo, hi = np.nanmin(ref), np.nanmax(ref)
            shade[:, ci] = np.clip((col - lo) / (hi - lo), 0.0, 1.0) if hi > lo else 0.5
    ax.imshow(shade, cmap=mcolors.LinearSegmentedColormap.from_list(
        "seq", ["#eaf1fb", "#9ec5f4", "#2a78d6"]), aspect="auto", vmin=0, vmax=1)

    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            val = mat[ri, ci]
            txt = "{:.1f}%".format(val * 100) if ci < 2 else "{:.4f}".format(val)
            dark = shaded[ci] and shade[ri, ci] > 0.62
            ax.text(ci, ri, txt, ha="center", va="center", fontsize=6.2,
                    color="white" if dark else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=6.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lbl + (" $\\dagger$" if "Gemma" in lbl else "")
                        for lbl, _f, _p, _c in rows], fontsize=6.5)
    ax.tick_params(length=0)
    despine(ax, keep=())
    ax.set_title("base model at $\\sigma/\\hat\\sigma_0^*=0.50$, round 11, seed 0",
                 loc="left", fontsize=7.5, pad=6)
    fig.text(0.005, -0.10,
             "Shaded columns are ratios and compare across families. Per-token NLL "
             "does not: Qwen ~152k tokens, Llama 128k, Gemma 262k.\n"
             "$\\dagger$ the Gemma generalist had not converged at twelve rounds, "
             "so its margin is inflated.",
             fontsize=5.6, color=INK_2, va="top")
    save(fig, "fig_model_families", meta={
        "figure": "base model sweep", "seed": "seed0",
        "models": [{"label": l, "family": f, "params_b": p,
                    "capture": c["capture"], "gain": c["gain"]}
                   for l, f, p, c in rows],
        "caveat": "per-token NLL not comparable across tokenizers",
    })


# --------------------------------------------------------- routing mechanism
def fig_routing_ensemble() -> None:
    """The five routing designs, as gain and as share of headroom captured.

    Complements the manuscript's ``fig_routing_ablation`` (which shows the NLL
    line-up) by separating how much specialisation each design produced from how
    much of it the router actually exploited.
    """
    rows = []
    for label, run, sel, k, weight in ROUTING_ARMS:
        diag = diagnostics(run)
        if diag is not None:
            rows.append((label, sel, k, weight, contrast(diag)))
    y = np.arange(len(rows))[::-1]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 2.1), sharey=True,
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.08})

    gains = [c["gain"] for *_r, c in rows]
    ax.barh(y, gains, height=0.55,
            color=[C_ACCENT if g > 0 else C_WARN for g in gains], zorder=3)
    ax.axvline(0, color=INK_3, lw=0.8)
    for yy, g in zip(y, gains):
        ax.text(g + (0.002 if g > 0 else -0.002), yy, "{:+.4f}".format(g),
                va="center", ha="left" if g > 0 else "right", fontsize=6.2,
                family="monospace", color=C_ACCENT if g > 0 else C_WARN)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(r"router $-$ pooled (nats)", labelpad=2)
    better(ax, "x", "up")
    ax.set_xlim(min(gains) - 0.028, max(gains) + 0.028)

    caps = [c["capture"] * 100 for *_r, c in rows]
    ax2.barh(y, caps, height=0.55,
             color=[C_ROUTER if c > 0 else C_WARN for c in caps], zorder=3)
    ax2.axvline(0, color=INK_3, lw=0.8)
    for yy, c in zip(y, caps):
        ax2.text(c + (1.2 if c > 0 else -1.2), yy, "{:.1f}%".format(c),
                 va="center", ha="left" if c > 0 else "right", fontsize=6.2,
                 family="monospace", color=INK_2)
    ax2.set_xlabel("headroom captured (%)", labelpad=2)
    better(ax2, "x", "up")
    ax2.set_xlim(min(caps) - 14, max(caps) + 14)

    for a in (ax, ax2):
        a.grid(axis="x", zorder=0)
        a.set_axisbelow(True)
        despine(a, keep=("bottom",))
        a.tick_params(axis="y", length=0)
    save(fig, "fig_routing_ensemble", meta={
        "figure": "routing design: gain and capture", "seed": "seed0",
        "arms": [{"label": l, "selection": s, "k": k, "loss": w,
                  "gain": c["gain"], "capture": c["capture"]}
                 for l, s, k, w, c in rows],
    })


EXTRA_FIGURES = {
    "radar_sigma": fig_radar_sigma,
    "radar_pairs": fig_radar_pairs,
    "per_benchmark": fig_per_benchmark,
    "model_families": fig_model_families,
    "routing_ensemble": fig_routing_ensemble,
}
