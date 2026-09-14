#!/usr/bin/env python
"""Publication figures for the sigma sweep and the pair-count ablation.

Reads ``routing_ensemble_diagnostics.json`` straight out of the results tree, so
re-running after another replicate lands simply tightens the error bars -- no
numbers are hardcoded.

    python scripts/paper_figures.py                  # both figures -> figures/paper/
    python scripts/paper_figures.py --out figures/x  # elsewhere
    python scripts/paper_figures.py --only sigma     # or: pairs

METHODOLOGICAL NOTE, and the reason the sigma figure is drawn the way it is:
absolute NLL is **not** comparable across data seeds. Each seed rebuilds the
70/10/20 split, so its test pool differs and its pooled generalist sits at a
different level (measured range 1.9841-1.9991). Averaging raw NLL across seeds
would fold pool difficulty into the error bars. Every quantity plotted here is
therefore a **within-seed contrast**:

    gain     = pooled - learned      (what routing won)
    envelope = pooled - oracle       (what was there to win)
    capture  = gain / envelope       (the fraction it got)

The pair-count figure is different: all its arms share the single seed-0 pool,
so absolute NLL *is* comparable there and is plotted directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colourblind-safe categorical slots, validated for CVD separation and contrast
# in both light and dark rendering. Keep this order.
C_SPEC = "#2a78d6"   # specialist / learned routing
C_ORAC = "#eb6834"   # oracle ceiling
C_CAP = "#1baf7a"    # capture
C_GEN = "#565d68"    # generalist reference

SIGMAS = {"07": 0.70, "06": 0.60, "055": 0.55, "05": 0.50, "045": 0.45,
          "04": 0.40, "03": 0.30, "02": 0.20, "01": 0.10, "005": 0.05}
SEEDS = ["seed0", "dseed1", "dseed2", "dseed3", "dseed4",
         "dseed5", "dseed6", "dseed7", "dseed8", "dseed9"]
PAIR_ARMS = {2: "seven_axis_2pair_sigma05", 3: "seven_axis_3pair_sigma05",
             4: "seven_axis_4pair_sigma05", 5: "seven_axis_5pair_sigma05",
             6: "seven_axis_6pair_sigma05", 7: "seven_axis_soft_full_pairs",
             8: "seven_axis_8pair_sigma05", 9: "seven_axis_9pair_sigma05"}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8f96", "axes.linewidth": 0.8,
        "grid.color": "#dfe1de", "grid.linewidth": 0.6,
        "legend.frameon": False, "lines.linewidth": 1.8,
    })


def _read(path: Path) -> dict | None:
    """Return the flat diagnostics block, or None if the arm is not scored."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)["flat"]
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return None


def load_sigma(root: Path) -> list[dict]:
    """Per-sigma contrasts across every data seed that is fully scored."""
    rows = []
    for key, sigma in SIGMAS.items():
        gains, envs, caps, used = [], [], [], []
        for seed in SEEDS:
            cands = [root / f"seven_axis_sigma{key}_full" / seed / "routing_ensemble_diagnostics.json"]
            if key == "05":   # sigma 0.50 at seed0 lives under the original arm name
                cands.append(root / "seven_axis_soft_full_pairs" / seed / "routing_ensemble_diagnostics.json")
            flat = next((f for f in (_read(c) for c in cands) if f), None)
            if flat is None:
                continue
            po, le, orc = flat["pooled_nll"], flat["learned_routing_nll"], flat["oracle_routing_nll"]
            gains.append(po - le)
            envs.append(po - orc)
            caps.append((po - le) / (po - orc))
            used.append(seed)
        if gains:
            rows.append(dict(sigma=sigma, n=len(gains), seeds=used,
                             gain=np.mean(gains), gain_sd=np.std(gains, ddof=1) if len(gains) > 1 else 0.0,
                             env=np.mean(envs), env_sd=np.std(envs, ddof=1) if len(envs) > 1 else 0.0,
                             cap=np.mean(caps), cap_sd=np.std(caps, ddof=1) if len(caps) > 1 else 0.0))
    return sorted(rows, key=lambda r: r["sigma"])


def load_pairs(root: Path) -> list[dict]:
    rows = []
    for pairs, name in sorted(PAIR_ARMS.items()):
        flat = _read(root / name / "seed0" / "routing_ensemble_diagnostics.json")
        if flat is None:
            continue
        po, le, orc = flat["pooled_nll"], flat["learned_routing_nll"], flat["oracle_routing_nll"]
        rows.append(dict(pairs=pairs, clones=2 * pairs, pooled=po, learned=le, oracle=orc,
                         gain=po - le, env=po - orc, cap=(po - le) / (po - orc),
                         agree=flat.get("routing_agreement_argmax")))
    return rows


def fig_sigma(rows: list[dict], out: Path) -> None:
    """Specialist and oracle improvement over each seed's own generalist."""
    x = np.array([r["sigma"] for r in rows])
    n = max(r["n"] for r in rows)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})

    for key, sdkey, colour, label in ((("env"), "env_sd", C_ORAC, "Oracle routing (ceiling)"),
                                      (("gain"), "gain_sd", C_SPEC, "Specialist ensemble (learned routing)")):
        m = np.array([r[key] for r in rows])
        s = np.array([r[sdkey] for r in rows])
        ax.fill_between(x, m - s, m + s, color=colour, alpha=0.15, linewidth=0)
        ax.errorbar(x, m, yerr=s, color=colour, marker="o", markersize=4.5,
                    capsize=3, elinewidth=1.2, markeredgecolor="white",
                    markeredgewidth=0.8, label=label, zorder=3)

    ax.axhline(0, color=C_GEN, linestyle="--", linewidth=1.5,
               label="Pooled generalist (per-seed, = 0 by construction)")
    # Note the sense flips between the two figures: this one plots an improvement
    # (higher is better), the pair-count figure plots raw NLL (lower is better).
    ax.set_ylabel("improvement over that seed's generalist\n(nats — higher is better)")
    ax.grid(axis="y")
    # Park the legend in the empty band between the zero line and the specialist
    # series; "lower center" alone drops it straight onto both.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.09), ncol=1)
    ax.set_ylim(bottom=min(-0.004, ax.get_ylim()[0]))
    ax.set_title(f"Competitive reach sweep — mean ± 1 SD over {n} data seeds", loc="left")

    cm = np.array([r["cap"] for r in rows]) * 100
    cs = np.array([r["cap_sd"] for r in rows]) * 100
    ax2.fill_between(x, cm - cs, cm + cs, color=C_CAP, alpha=0.15, linewidth=0)
    ax2.errorbar(x, cm, yerr=cs, color=C_CAP, marker="o", markersize=4.5, capsize=3,
                 elinewidth=1.2, markeredgecolor="white", markeredgewidth=0.8)
    ax2.set_ylabel("capture  (%, higher is better)")
    ax2.set_xlabel(r"$\sigma$ as a fraction of $\sigma_0^*$")
    ax2.grid(axis="y")
    # Tick every measured sigma; the auto-locator drops 0.05 and 0.45/0.55.
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{v:g}" for v in x], fontsize=8)
    ax2.set_title("Share of available headroom captured", loc="left", fontsize=9.5)

    for suffix in ("pdf", "png"):
        fig.savefig(out / f"sigma_sweep.{suffix}")
    plt.close(fig)
    print(f"wrote {out}/sigma_sweep.pdf and .png   (n={n} seeds, {len(rows)} sigma values)")


def fig_pairs(rows: list[dict], out: Path) -> None:
    """The three NLLs against pair count. One shared seed-0 pool, so absolute NLL is fair."""
    x = np.array([r["pairs"] for r in rows])
    pooled = np.array([r["pooled"] for r in rows])
    learned = np.array([r["learned"] for r in rows])
    oracle = np.array([r["oracle"] for r in rows])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})

    ax.fill_between(x, pooled, learned, color=C_SPEC, alpha=0.13, linewidth=0, label="gain")
    ax.fill_between(x, learned, oracle, color=C_ORAC, alpha=0.11, linewidth=0, label="headroom left")
    ax.axhline(pooled.mean(), color=C_GEN, linestyle="--", linewidth=1.5,
               label="Pooled generalist (one fixed model, never retrained)")
    ax.plot(x, learned, color=C_SPEC, marker="o", markersize=4.5, markeredgecolor="white",
            markeredgewidth=0.8, label="Specialist ensemble (learned routing)", zorder=3)
    ax.plot(x, oracle, color=C_ORAC, marker="o", markersize=4.5, markeredgecolor="white",
            markeredgewidth=0.8, label="Oracle routing (ceiling)", zorder=3)
    ax.set_ylabel("mean NLL  (lower is better)")
    ax.grid(axis="y")
    # Lower-left is the only region both curves vacate; "upper right" buries the
    # legend in the gain band and collides with the threshold label.
    ax.legend(loc="lower left", ncol=1, fontsize=8)
    ax.set_title("Pair-count ablation — seed 0, shared generalist", loc="left")

    ax2.plot(x, [r["cap"] * 100 for r in rows], color=C_CAP, marker="o", markersize=4.5,
             markeredgecolor="white", markeredgewidth=0.8, label="capture")
    if all(r["agree"] is not None for r in rows):
        ax2.plot(x, [r["agree"] * 100 for r in rows], color=C_GEN, marker="s", markersize=3.8,
                 linestyle="--", linewidth=1.4, label="oracle agreement")
    ax2.axhline(0, color="#8a8f96", linewidth=0.8)
    ax2.set_ylabel("percent")
    ax2.set_xlabel("routing pairs   (clones = 2 × pairs)")
    ax2.set_xticks(x)
    ax2.grid(axis="y")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, fontsize=8)

    # The threshold is the finding; mark it on both panels rather than in a caption.
    for a in (ax, ax2):
        a.axvline(4.5, color="#8a8f96", linestyle=":", linewidth=1.1, zorder=0)
    # Anchor the label to the BOTTOM of the upper panel: the top is occupied by
    # the generalist line and the title.
    ax.annotate("threshold", xy=(4.5, ax.get_ylim()[0]), xytext=(4, 9),
                textcoords="offset points", ha="left", fontsize=8, color="#6b7078")

    for suffix in ("pdf", "png"):
        fig.savefig(out / f"pair_count.{suffix}")
    plt.close(fig)
    print(f"wrote {out}/pair_count.pdf and .png   ({len(rows)} arms)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="results tree (default: results)")
    ap.add_argument("--out", default="figures/paper", help="output dir (default: figures/paper)")
    ap.add_argument("--only", choices=["sigma", "pairs"], help="render just one figure")
    args = ap.parse_args()

    root = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _style()

    if args.only != "pairs":
        rows = load_sigma(root)
        if rows:
            fig_sigma(rows, out)
            print("  seeds used:", ", ".join(rows[0]["seeds"]))
        else:
            print("no scored sigma arms found under", root)

    if args.only != "sigma":
        rows = load_pairs(root)
        if rows:
            fig_pairs(rows, out)
        else:
            print("no scored pair-count arms found under", root)


if __name__ == "__main__":
    main()
