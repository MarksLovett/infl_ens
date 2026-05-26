"""Compare position-step policies on stability-test sweep results.

Reads histories under::

    <root>/<mode_slug>/sigma*/seed*/

and reports final pairwise spread, mean effective blend, and collapse flags.
Writes ``summary.csv`` and an overview figure under ``<figure-root>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_SIGMA_RE = re.compile(r"^sigma(?P<val>[0-9.]+)$", re.IGNORECASE)
_SEED_RE = re.compile(r"^seed(?P<val>\d+)$", re.IGNORECASE)
_MODE_RE = re.compile(r"^mode_(?P<slug>.+)$", re.IGNORECASE)


def pairwise_spread(pos: np.ndarray) -> float:
    """Mean pairwise L2 distance among agents.

    :param pos: ``(N, L)`` positions.
    :type pos: numpy.ndarray
    :returns: Mean off-diagonal distance.
    :rtype: float
    """
    n = pos.shape[0]
    if n < 2:
        return 0.0
    dists = [
        float(np.linalg.norm(pos[i] - pos[j]))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return float(np.mean(dists))


def load_final_spread(history_path: Path) -> tuple[float, float, int]:
    """Final spread and mean effective blend from a history file.

    :param history_path: Path to ``history.json``.
    :type history_path: pathlib.Path
    :returns: ``(final_spread, mean_blend_eff, n_rounds)``.
    :rtype: tuple[float, float, int]
    """
    with history_path.open(encoding="utf-8") as fh:
        records = json.load(fh)
    if records[-1].get("pairwise_spread") is not None:
        spread = float(records[-1]["pairwise_spread"])
    else:
        names = list(records[-1]["positions"].keys())
        pos = np.stack(
            [np.asarray(records[-1]["positions"][n]) for n in names], axis=0,
        )
        spread = pairwise_spread(pos)

    blends: list[float] = []
    for rec in records:
        ab = rec.get("agent_blend_effective") or {}
        for vals in ab.values():
            blends.extend(vals)
    mean_blend = float(np.mean(blends)) if blends else float("nan")
    return spread, mean_blend, len(records) - 1


def discover_cells(root: Path) -> list[dict]:
    """Walk the stability-test tree.

    :param root: Results root containing ``mode_*`` subdirs.
    :type root: pathlib.Path
    :returns: Row dicts for the summary table.
    :rtype: list[dict]
    """
    rows: list[dict] = []
    for mode_dir in sorted(root.iterdir()):
        if not mode_dir.is_dir():
            continue
        mm = _MODE_RE.match(mode_dir.name)
        if not mm:
            continue
        mode_slug = mm.group("slug")
        for sigma_dir in sorted(mode_dir.iterdir()):
            if not sigma_dir.is_dir():
                continue
            sm = _SIGMA_RE.match(sigma_dir.name)
            if not sm:
                continue
            sigma_frac = float(sm.group("val"))
            for seed_dir in sorted(sigma_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                sd = _SEED_RE.match(seed_dir.name)
                if not sd:
                    continue
                hist = seed_dir / "history.json"
                if not hist.is_file():
                    continue
                spread, mean_blend, n_rounds = load_final_spread(hist)
                rows.append({
                    "mode": mode_slug,
                    "sigma_fraction": sigma_frac,
                    "seed": int(sd.group("val")),
                    "n_rounds": n_rounds,
                    "final_spread": spread,
                    "mean_blend_effective": mean_blend,
                    "collapsed": spread < 0.45,
                })
    return rows


def plot_overview(rows: Sequence[dict], output_stem: Path) -> None:
    """Bar chart of mean final spread by mode and sigma.

    :param rows: Summary rows.
    :type rows: Sequence[dict]
    :param output_stem: Filename stem (no extension).
    :type output_stem: pathlib.Path
    """
    import matplotlib.pyplot as plt

    modes = sorted({r["mode"] for r in rows})
    sigmas = sorted({r["sigma_fraction"] for r in rows})
    n_modes = len(modes)
    n_sig = len(sigmas)
    fig, axes = plt.subplots(1, n_sig, figsize=(4 * n_sig, 4), squeeze=False)
    collapse_thresh = 0.45

    for j, sigma in enumerate(sigmas):
        ax = axes[0, j]
        means, stds, labels = [], [], []
        for mode in modes:
            vals = [
                r["final_spread"]
                for r in rows
                if r["mode"] == mode and r["sigma_fraction"] == sigma
            ]
            if not vals:
                continue
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
            labels.append(mode.replace("_", " "))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, alpha=0.85)
        ax.axhline(collapse_thresh, color="red", ls="--", lw=1,
                   label=f"collapse < {collapse_thresh}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("final pairwise spread")
        ax.set_title(f"σ/σ₀* = {sigma:g}")
        ax.grid(True, axis="y", alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("position-step stability test (mean ± std over seeds)")
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional CLI args.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--figure-root", type=Path, required=True)
    args = p.parse_args(argv)

    rows = discover_cells(args.root)
    if not rows:
        print(f"no runs under {args.root}", file=sys.stderr)
        return 1

    fig_dir = args.figure_root
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_path = fig_dir / "summary.csv"
    fields = [
        "mode", "sigma_fraction", "seed", "n_rounds",
        "final_spread", "mean_blend_effective", "collapsed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    plot_overview(rows, fig_dir / "spread_by_mode_and_sigma")
    print(f"wrote {fig_dir / 'spread_by_mode_and_sigma'}.{{pdf,png}}")

    # Console recommendation
    print("\n--- mean final spread (higher = better separation) ---")
    for sigma in sorted({r["sigma_fraction"] for r in rows}):
        print(f"\n  sigma_fraction = {sigma}")
        for mode in sorted({r["mode"] for r in rows}):
            vals = [
                r["final_spread"]
                for r in rows
                if r["mode"] == mode and r["sigma_fraction"] == sigma
            ]
            if not vals:
                continue
            coll = sum(1 for r in rows
                       if r["mode"] == mode and r["sigma_fraction"] == sigma
                       and r["collapsed"])
            print(
                f"    {mode:<22}  spread={np.mean(vals):.3f} ± {np.std(vals):.3f}"
                f"  collapsed={coll}/{len(vals)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
