"""Summarize position-only stability sweeps (batch size, step modes)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from infl_ens.training.pool_dynamics import pairwise_spread
from infl_ens.training.sweep_aggregate import COLLAPSE_SPREAD_THRESH, MODE_STEP_DIR_RE
from infl_ens.utils.sweep_discovery import SEED_DIR_RE, SIGMA_DIR_RE
from infl_ens.vis.sweeps import plot_spread_by_mode_sigma

_SIGMA_RE = SIGMA_DIR_RE
_SEED_RE = SEED_DIR_RE


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


def load_final_spread_and_blend(path: Path) -> tuple[float, float, int]:
    """Final spread, mean effective blend, and round count from history.

    :param path: ``history.json`` path.
    :type path: pathlib.Path
    :returns: ``(final_spread, mean_blend_eff, n_rounds)``.
    :rtype: tuple[float, float, int]
    """
    with path.open(encoding="utf-8") as fh:
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


def discover_batch_mode_rows(root: Path, mode_slug: str) -> list[dict]:
    """Collect summary rows under ``root/mode_slug/`` or flat ``root/``.

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
                "collapsed": spread < COLLAPSE_SPREAD_THRESH,
            })
    return rows


def discover_position_step_rows(root: Path) -> list[dict]:
    """Walk a position-step stability tree under ``mode_*`` subdirs.

    :param root: Results root containing ``mode_*`` subdirs.
    :type root: pathlib.Path
    :returns: Summary row dicts.
    :rtype: list[dict]
    """
    rows: list[dict] = []
    for mode_dir in sorted(root.iterdir()):
        if not mode_dir.is_dir():
            continue
        mm = MODE_STEP_DIR_RE.match(mode_dir.name)
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
                spread, mean_blend, n_rounds = load_final_spread_and_blend(hist)
                rows.append({
                    "mode": mode_slug,
                    "sigma_fraction": sigma_frac,
                    "seed": int(sd.group("val")),
                    "n_rounds": n_rounds,
                    "final_spread": spread,
                    "mean_blend_effective": mean_blend,
                    "collapsed": spread < COLLAPSE_SPREAD_THRESH,
                })
    return rows


def print_spread_summary_table(rows: Sequence[dict]) -> None:
    """Print mean final spread grouped by mode and sigma.

    :param rows: Summary rows with ``mode``, ``sigma_fraction``, ``final_spread``.
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
            coll = sum(1 for v in vals if v < COLLAPSE_SPREAD_THRESH)
            print(
                f"    {mode:<22}  spread={np.mean(vals):.3f} ± {np.std(vals):.3f}"
                f"  collapsed={coll}/{len(vals)}"
            )


def write_spread_summary_csv(
    rows: Sequence[dict],
    path: Path,
    *,
    fieldnames: Sequence[str],
) -> None:
    """Write summary rows to CSV.

    :param rows: Summary rows.
    :type rows: Sequence[dict]
    :param path: Output CSV path.
    :type path: pathlib.Path
    :param fieldnames: CSV column names.
    :type fieldnames: Sequence[str]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_batch_size_static_comparison(
    root: Path,
    figure_root: Path,
    *,
    reference_root: Optional[Path] = None,
) -> list[dict]:
    """Aggregate batch-size / static-centroid sweep results.

    :param root: Sweep results root.
    :type root: pathlib.Path
    :param figure_root: Directory for CSV and figures.
    :type figure_root: pathlib.Path
    :param reference_root: Optional prior ``batch_256`` reference sweep.
    :type reference_root: pathlib.Path | None
    :returns: All summary rows.
    :rtype: list[dict]
    """
    mode_slugs = [
        "batch_256", "batch_large", "full_pool", "expected_pool",
        "baseline", "init_noise_1e-2", "pool_and_noise",
    ]
    rows: list[dict] = []
    for slug in mode_slugs:
        rows.extend(discover_batch_mode_rows(root, slug))

    if reference_root is not None and reference_root.is_dir():
        for sigma_dir in sorted(reference_root.iterdir()):
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
                        "collapsed": spread < COLLAPSE_SPREAD_THRESH,
                    })

    if not rows:
        return rows

    figure_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode", "centroid_mode", "sigma_fraction", "seed",
        "final_spread", "collapsed",
    ]
    csv_path = figure_root / "summary.csv"
    write_spread_summary_csv(rows, csv_path, fieldnames=fields)
    print(f"wrote {csv_path}")

    stem = figure_root / "spread_by_batch_mode"
    plot_spread_by_mode_sigma(
        rows,
        output_stem=stem,
        suptitle="batch size / static centroid analysis (mean ± std, 5 seeds)",
        mode_label_fn=lambda m: m.replace("_", "\n"),
    )
    print(f"wrote {stem}.{{pdf,png}}")
    print_spread_summary_table(rows)
    return rows


def run_position_step_modes_comparison(
    root: Path,
    figure_root: Path,
) -> list[dict]:
    """Aggregate position-step stability sweep results.

    :param root: Sweep results root with ``mode_*`` subdirs.
    :type root: pathlib.Path
    :param figure_root: Directory for CSV and figures.
    :type figure_root: pathlib.Path
    :returns: All summary rows.
    :rtype: list[dict]
    """
    rows = discover_position_step_rows(root)
    if not rows:
        return rows

    figure_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode", "sigma_fraction", "seed", "n_rounds",
        "final_spread", "mean_blend_effective", "collapsed",
    ]
    csv_path = figure_root / "summary.csv"
    write_spread_summary_csv(rows, csv_path, fieldnames=fields)
    print(f"wrote {csv_path}")

    stem = figure_root / "spread_by_mode_and_sigma"
    plot_spread_by_mode_sigma(
        rows,
        output_stem=stem,
        suptitle="position-step stability test (mean ± std over seeds)",
        mode_label_fn=lambda m: m.replace("_", " "),
        rotate_xticks=25,
    )
    print(f"wrote {stem}.{{pdf,png}}")
    print_spread_summary_table(rows)
    return rows
