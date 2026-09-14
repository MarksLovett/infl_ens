r"""The ensemble mixture figure.

``learned_routing_nll`` is :math:`\sum_p G_p \mathrm{NLL}_p` -- the expected NLL
of *sampling one* expert. That is not an ensemble. The sequence-level mixture

.. math::

    -\frac{1}{n_i} \log \sum_p w_p P_p(y_i)

is what "ensemble" means, and by Jensen it is never worse: ``oracle <= mixture
<= expected``. This module plots both against the oracle ceiling on every axis
of the study, together with the uniform control :math:`w = 1/K` -- if uniform
mixing matches :math:`G`-weighted mixing, the router contributes nothing to the
ensemble, which is the cheapest available check on the central claim.

Kept separate from ``figures_extra`` because it is the one figure that depends
on metrics added after most runs were scored, so it has to degrade gracefully
while the re-score is still queued.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figures_lib import (
    better, C_ACCENT, C_ORACLE, C_POOLED, C_ROUTER, C_WARN, INK_2, INK_3, PAIR_ARMS,
    ROUTING_ARMS, SCALE_ARMS, TEXT_WIDTH, contrast, despine, diagnostics, save,
    sigma_contrasts, summarise,
)

# Series drawn in every panel: key, legend label, colour, linestyle, linewidth.
# Order is the Jensen ordering, best first, so the legend reads as the sandwich.
ENS_SERIES = [
    ("oracle", "oracle (ceiling)", C_ORACLE, "-", 1.0),
    ("mixture", r"mixture, $w=G$", C_ACCENT, "-", 1.6),
    ("uniform", r"mixture, $w=1/K$", C_POOLED, "--", 1.1),
    ("expected", "expected (sample one)", C_ROUTER, "-", 1.3),
]


def ens_gains(c: dict | None) -> dict:
    """The four ensemble quantities, each as improvement over its own pooled.

    Every panel spans arms with their own generalist -- and panels (d)/(e) span
    three tokenizers, where per-token NLL is not comparable at all -- so only
    these within-arm differences are plotted. A ``None`` marks a run scored
    before the mixture metrics existed.
    """
    if c is None:
        return {k: None for k, *_ in ENS_SERIES}
    return {
        "expected": c["gain"],
        "mixture": c.get("gain_mixture"),
        "uniform": c.get("gain_uniform"),
        "oracle": c["envelope"],
    }


def _as_array(vals) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in vals])


def _stamp_pending(ax, gains_list) -> bool:
    """Mark a panel whose runs all predate the re-score; True if pending."""
    if any(g.get("mixture") is not None for g in gains_list):
        return False
    ax.text(0.5, 0.5, "mixture pending re-score", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.5, style="italic", color=C_WARN,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_WARN, lw=0.6,
                      alpha=0.92), zorder=10)
    return True


def _categorical_panel(ax, labels, gains_list, xlabel=None, rotate=0) -> None:
    """One x position per arm, the four series, and the Jensen gap shaded."""
    x = np.arange(len(labels))
    have = {k: [g[k] for g in gains_list] for k, *_ in ENS_SERIES}

    exp_v, mix_v = _as_array(have["expected"]), _as_array(have["mixture"])
    ok = ~np.isnan(exp_v) & ~np.isnan(mix_v)
    if ok.any():
        ax.fill_between(x, exp_v, mix_v, where=ok, color=C_ACCENT, alpha=0.14,
                        lw=0, zorder=1, label="Jensen gap")

    for key, label, color, ls, lw in ENS_SERIES:
        v = _as_array(have[key])
        if np.isnan(v).all():
            continue
        ax.plot(x, v, ls, color=color, lw=lw, marker="o", ms=2.6, mew=0,
                label=label, zorder=3)

    ax.axhline(0, color=INK_3, lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate,
                       ha="right" if rotate else "center")
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=2)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    _stamp_pending(ax, gains_list)


def _sigma_panel(ax, meta: dict) -> None:
    """Reach sweep, mean +/- SD across data splits."""
    by_sigma = sigma_contrasts()
    sigmas = sorted(by_sigma)
    x = np.arange(len(sigmas))
    n_per: dict[str, int] = {}

    for key, label, color, ls, lw in ENS_SERIES:
        mus, sds = [], []
        for s in sigmas:
            mu, sd, n = summarise([ens_gains(c)[key] for c in by_sigma[s]])
            mus.append(mu if n else np.nan)
            sds.append(sd if n else np.nan)
            if key == "expected":
                n_per["{:.2f}".format(s)] = n
        mus, sds = np.array(mus), np.array(sds)
        if np.isnan(mus).all():
            continue
        ax.plot(x, mus, ls, color=color, lw=lw, marker="o", ms=2.6, mew=0,
                label=label, zorder=3)
        ax.fill_between(x, mus - sds, mus + sds, color=color, alpha=0.12,
                        lw=0, zorder=1)

    ax.axhline(0, color=INK_3, lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["{:.2f}".format(s) for s in sigmas])
    ax.set_xlabel(r"competitive reach $\sigma$", labelpad=2)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    _stamp_pending(ax, [ens_gains(c) for cs in by_sigma.values() for c in cs])
    meta["sigma"] = {"n_per_sigma": n_per}


def _routing_panel(ax) -> None:
    """The five routing designs as grouped horizontal bars."""
    rows = []
    for label, run, *_rest in ROUTING_ARMS:
        diag = diagnostics(run)
        rows.append((label, ens_gains(contrast(diag) if diag else None)))

    y = np.arange(len(rows))[::-1]
    height = 0.19
    for i, (key, label, color, _ls, _lw) in enumerate(ENS_SERIES):
        vals = [g[key] for _, g in rows]
        if all(v is None for v in vals):
            continue
        ax.barh(y + (1.5 - i) * height,
                [0.0 if v is None else v for v in vals], height=height,
                color=color, label=label, zorder=3,
                hatch="//" if key == "uniform" else None,
                edgecolor="white", lw=0.3)

    ax.axvline(0, color=INK_3, lw=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([lbl for lbl, _ in rows], fontsize=6.4)
    ax.set_xlabel("improvement over pooled (nats)", labelpad=2)
    better(ax, "x", "up")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    _stamp_pending(ax, [g for _, g in rows])


def fig_ensemble() -> None:
    r"""Ensemble NLL across reach, pair count, routing design, size and family.

    Read the uniform control with sequence length in mind: the weights enter the
    mixture only as :math:`-\log w_{\mathrm{best}} / n_i`, so at the measured
    mean of ~68 supervised tokens the mixture already sits close to oracle
    selection and the two weightings must converge. The router's contribution is
    carried by the *expected* metric -- where committing to a single expert is
    the decision actually being made -- and by the short-sequence benchmarks in
    the per-benchmark breakdown.
    """
    fig = plt.figure(figsize=(TEXT_WIDTH, 7.6))
    gs = fig.add_gridspec(4, 6, hspace=0.75, wspace=0.55,
                          left=0.125, right=0.985, top=0.925, bottom=0.075)
    ax_s = fig.add_subplot(gs[0, :])
    ax_p = fig.add_subplot(gs[1, :])
    ax_r = fig.add_subplot(gs[2, :])
    ax_d = fig.add_subplot(gs[3, :3])
    ax_e = fig.add_subplot(gs[3, 3:])

    meta: dict = {
        "figure": "ensemble mixture NLL across the study",
        "note": "sequence-level mixture of completed sequences, "
                "not token-level mixing at generation time",
        "panels": {},
    }

    _sigma_panel(ax_s, meta["panels"])
    ax_s.set_title(r"(a) competitive reach: mean $\pm$ SD over data splits",
                   loc="left", fontsize=7.2, pad=3)

    pair_rows = []
    for k, run in PAIR_ARMS:
        diag = diagnostics(run)
        pair_rows.append((k, ens_gains(contrast(diag) if diag else None)))
    _categorical_panel(ax_p, [str(k) for k, _ in pair_rows],
                       [g for _, g in pair_rows], xlabel=r"routing pairs $K$")
    ax_p.set_title("(b) pair count (split 0)", loc="left", fontsize=7.2, pad=3)

    _routing_panel(ax_r)
    ax_r.set_title("(c) routing design (split 0)", loc="left", fontsize=7.2, pad=3)

    size_arms = [a for a in SCALE_ARMS if a[3] == "Qwen2.5"]
    fam_arms = [a for a in SCALE_ARMS
                if a[0] in {"Qwen2.5-1.5B", "Llama-3.2-1B", "Gemma-3-1B"}]
    short = {"Qwen2.5-1.5B": "Qwen 1.5B", "Qwen2.5-3B": "Qwen 3B",
             "Qwen2.5-7B": "Qwen 7B", "Llama-3.2-1B": "Llama 1B",
             "Gemma-3-1B": "Gemma 1B"}
    for ax, arms, title in ((ax_d, size_arms, "(d) model size"),
                            (ax_e, fam_arms, "(e) model family")):
        rows = []
        for label, run, _b, _fam in arms:
            diag = diagnostics(run)
            rows.append((label, ens_gains(contrast(diag) if diag else None)))
        _categorical_panel(ax, [short[lbl] for lbl, _ in rows],
                           [g for _, g in rows], rotate=30)
        ax.set_title(title, loc="left", fontsize=7.2, pad=3)

    ax_e.text(1.0, -0.52,
              "per-token NLL is not comparable across tokenizers; the\n"
              "difference from each model's own pooled baseline is",
              transform=ax_e.transAxes, ha="right", va="top", fontsize=5.6,
              color=INK_2, style="italic")

    for ax in (ax_s, ax_p, ax_d):
        ax.set_ylabel("nats vs. pooled", labelpad=2)
        better(ax, "y", "up")

    handles, labels = ax_s.get_legend_handles_labels()
    if not handles:
        handles, labels = ax_p.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right",
                   bbox_to_anchor=(0.985, 1.005), ncol=2, frameon=False,
                   fontsize=6.2, handlelength=1.8, columnspacing=1.1)

    scored = [run for _k, run in PAIR_ARMS
              if (d := diagnostics(run)) is not None
              and d.get("ensemble_mixture_nll") is not None]
    meta["mixture_available"] = bool(scored)
    if not scored:
        meta["caveat"] = ("mixture panels pending the routing re-score; "
                          "expected and oracle series are final")
    save(fig, "fig_ensemble", meta=meta)


ENSEMBLE_FIGURES = {"ensemble": fig_ensemble}
