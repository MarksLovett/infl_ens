"""Compare large-batch / static pool centroids vs small-batch simulation.

Reads ``results/large_batch_static_analysis/<mode>/sigma*/seed*/`` and
optionally the earlier ``batch_256`` reference under
``position_step_stability_test/mode_static``.
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

_SIGMA_RE = re.compile(r"^sigma(?P<val>[0-9.]+)$", re.IGNORECASE)
_SEED_RE = re.compile(r"^seed(?P<val>\d+)$", re.IGNORECASE)
_COLLAPSE_THRESH = 0.45


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
    return float(np.mean([
        np.linalg.norm(pos[i] - pos[j])
        for i in range(n) for j in range(i + 1, n)
    ]))


def final_spread_from_history(path: Path) -> tuple[float, str]:
    """Load final spread and centroid mode from a history file.

    :param path: ``history.json`` path.
    :type path: pathlib.Path
    :returns: ``(spread, centroid_mode)``.
    :rtype: tuple[float, str]
    """
    with path.open(encoding="utf-8") as fh:
        records = json.load(fh)
    last = records[-1]
    if last.get("pairwise_spread") is not None:
        spread = float(last["pairwise_spread"])
    else:
        names = list(last["positions"].keys())
        pos = np.stack(
            [np.asarray(last["positions"][n]) for n in names], axis=0,
        )
        spread = pairwise_spread(pos)
    mode = str(last.get("centroid_mode", "batch"))
    return spread, mode


def discover(root: Path, mode_slug: str) -> list[dict]:
    """Collect rows under ``root/mode_slug/`` or ``root`` if flat layout.

    :param root: Results root.
    :type root: pathlib.Path
    :param mode_slug: Subdirectory name, e.g. ``full_pool``.
    :type mode_slug: str
    :returns: Summary rows.
    :rtype: list[dict]
    """
    base = root / mode_slug if (root / mode_slug).is_dir() else root
    rows: list[dict] = []
    if not base.is_dir():
        return rows
    for sigma_dir in sorted(base.iterdir()):
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
            spread, cm = final_spread_from_history(hist)
            rows.append({
                "mode": mode_slug,
                "centroid_mode": cm,
                "sigma_fraction": sigma_frac,
                "seed": int(sd.group("val")),
                "final_spread": spread,
                "collapsed": spread < _COLLAPSE_THRESH,
            })
    return rows


def plot_comparison(rows: Sequence[dict], output_stem: Path) -> None:
    """Grouped bar chart of mean spread by mode and sigma.

    :param rows: All summary rows.
    :type rows: Sequence[dict]
    :param output_stem: Filename stem without extension.
    :type output_stem: pathlib.Path
    """
    import matplotlib.pyplot as plt

    modes = sorted({r["mode"] for r in rows})
    sigmas = sorted({r["sigma_fraction"] for r in rows})
    fig, axes = plt.subplots(1, len(sigmas), figsize=(5 * len(sigmas), 4),
                             squeeze=False)

    for j, sigma in enumerate(sigmas):
        ax = axes[0, j]
        labels, means, stds = [], [], []
        for mode in modes:
            vals = [
                r["final_spread"]
                for r in rows
                if r["mode"] == mode and r["sigma_fraction"] == sigma
            ]
            if not vals:
                continue
            labels.append(mode.replace("_", "\n"))
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, alpha=0.88)
        ax.axhline(_COLLAPSE_THRESH, color="red", ls="--", lw=1,
                   label=f"collapse < {_COLLAPSE_THRESH}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("final pairwise spread")
        ax.set_title(f"σ/σ₀* = {sigma:g}")
        ax.grid(True, axis="y", alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)

    fig.suptitle("batch size / static centroid analysis (mean ± std, 5 seeds)")
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def print_table(rows: Sequence[dict]) -> None:
    """Print console summary.

    :param rows: Summary rows.
    :type rows: Sequence[dict]
    """
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
            coll = sum(1 for v in vals if v < _COLLAPSE_THRESH)
            print(
                f"    {mode:<18}  spread={np.mean(vals):.3f} ± {np.std(vals):.3f}"
                f"  collapsed={coll}/{len(vals)}"
            )


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
    p.add_argument(
        "--reference-root", type=Path, default=None,
        help="Optional prior sweep, e.g. position_step_stability_test/mode_static",
    )
    args = p.parse_args(argv)

    mode_slugs = [
        "batch_256", "batch_large", "full_pool", "expected_pool",
        "baseline", "init_noise_1e-2", "pool_and_noise",
    ]
    rows: list[dict] = []
    for slug in mode_slugs:
        rows.extend(discover(args.root, slug))

    if args.reference_root is not None:
        ref = args.reference_root
        if ref.is_dir():
            for sigma_dir in sorted(ref.iterdir()):
                if not sigma_dir.is_dir() or not _SIGMA_RE.match(sigma_dir.name):
                    continue
                sigma_frac = float(_SIGMA_RE.match(sigma_dir.name).group("val"))
                for seed_dir in sorted(sigma_dir.iterdir()):
                    sd = _SEED_RE.match(seed_dir.name)
                    if not sd:
                        continue
                    hist = seed_dir / "history.json"
                    if hist.is_file():
                        spread, _ = final_spread_from_history(hist)
                        rows.append({
                            "mode": "prior_batch_256",
                            "centroid_mode": "batch",
                            "sigma_fraction": sigma_frac,
                            "seed": int(sd.group("val")),
                            "final_spread": spread,
                            "collapsed": spread < _COLLAPSE_THRESH,
                        })

    if not rows:
        print(f"no results under {args.root}", file=sys.stderr)
        return 1

    args.figure_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.figure_root / "summary.csv"
    fields = ["mode", "centroid_mode", "sigma_fraction", "seed",
              "final_spread", "collapsed"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    plot_comparison(rows, args.figure_root / "spread_by_batch_mode")
    print(f"wrote {args.figure_root / 'spread_by_batch_mode'}.{{pdf,png}}")
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
