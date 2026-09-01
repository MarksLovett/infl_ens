"""Family x scale NLL summary for the model scale-family sweep.

Pure builders (records in, a :class:`matplotlib.figure.Figure` or written
table paths out) for the headline of the scale-family experiment: one
held-out NLL per ``(family, scale)`` cell, arranged as a family x scale
grid. The renderer in :mod:`infl_ens.figures.render` reads each cell's
route-then-score report and feeds the aggregated cells here.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from infl_ens.figures.style import apply_paper_style


@dataclass(frozen=True)
class CellNLL:
    """One family x scale cell's held-out NLL summary.

    :param family: Model family label (grid row).
    :type family: str
    :param scale: Nominal scale bucket (grid column).
    :type scale: str
    :param learned_nll: Learned-routing ensemble NLL (the headline metric).
    :type learned_nll: float
    :param pooled_nll: Pooled-generalist NLL, if available.
    :type pooled_nll: float | None
    :param oracle_nll: Oracle-routing NLL, if available.
    :type oracle_nll: float | None
    """

    family: str
    scale: str
    learned_nll: float
    pooled_nll: Optional[float] = None
    oracle_nll: Optional[float] = None


def _ordered(values: Sequence[str], present: Sequence[str]) -> list[str]:
    """Keep ``values`` order but drop entries not in ``present`` (extras appended)."""
    seen = list(dict.fromkeys(present))
    ordered = [v for v in values if v in seen]
    ordered += [v for v in seen if v not in ordered]
    return ordered


def _grid(
    cells: Sequence[CellNLL],
    families: Optional[Sequence[str]],
    scales: Optional[Sequence[str]],
) -> tuple[list[str], list[str], np.ndarray]:
    """Build the ``(families, scales, matrix)`` grid of learned NLL.

    :param cells: Per-cell summaries.
    :type cells: Sequence[CellNLL]
    :param families: Preferred row order (``None`` = first-seen order).
    :type families: Sequence[str] | None
    :param scales: Preferred column order (``None`` = first-seen order).
    :type scales: Sequence[str] | None
    :returns: ``(family_order, scale_order, matrix)`` with ``NaN`` for
        missing cells.
    :rtype: tuple[list[str], list[str], numpy.ndarray]
    """
    fam_present = [c.family for c in cells]
    sc_present = [c.scale for c in cells]
    fam_order = _ordered(families or [], fam_present)
    sc_order = _ordered(scales or [], sc_present)
    matrix = np.full((len(fam_order), len(sc_order)), np.nan, dtype=float)
    for c in cells:
        i = fam_order.index(c.family)
        j = sc_order.index(c.scale)
        matrix[i, j] = c.learned_nll
    return fam_order, sc_order, matrix


def plot_family_scale_nll(
    cells: Sequence[CellNLL],
    *,
    families: Optional[Sequence[str]] = None,
    scales: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    metric_label: str = r"Learned-routing NLL $\downarrow$",
) -> Figure:
    """Heatmap of held-out learned-routing NLL over the family x scale grid.

    :param cells: Per-cell NLL summaries (one per ``(family, scale)``).
    :type cells: Sequence[CellNLL]
    :param families: Preferred row order; defaults to first-seen order.
    :type families: Sequence[str] | None
    :param scales: Preferred column order; defaults to first-seen order.
    :type scales: Sequence[str] | None
    :param title: Optional figure title.
    :type title: str | None
    :param metric_label: Colorbar label.
    :type metric_label: str
    :returns: Matplotlib figure.
    :rtype: matplotlib.figure.Figure
    :raises ValueError: If ``cells`` is empty.
    """
    if not cells:
        raise ValueError("plot_family_scale_nll needs at least one cell")
    apply_paper_style()
    fam_order, sc_order, matrix = _grid(cells, families, scales)

    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(sc_order), 1.4 + 0.9 * len(fam_order)))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("viridis_r").with_extremes(bad="#dddddd")
    im = ax.imshow(masked, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(sc_order)))
    ax.set_yticks(np.arange(len(fam_order)))
    ax.set_xticklabels(sc_order)
    ax.set_yticklabels(fam_order)
    ax.set_xlabel("scale")
    ax.set_ylabel("family")
    ax.set_xticks(np.arange(len(sc_order) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(fam_order) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    finite = matrix[np.isfinite(matrix)]
    mid = float(finite.mean()) if finite.size else 0.0
    for i in range(len(fam_order)):
        for j in range(len(sc_order)):
            v = matrix[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "--", ha="center", va="center", color="#666666")
                continue
            ax.text(
                j, i, f"{v:.3f}",
                ha="center", va="center",
                color="white" if v > mid else "black",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def write_family_scale_table(
    cells: Sequence[CellNLL],
    output_stem: str | Path,
    *,
    families: Optional[Sequence[str]] = None,
    scales: Optional[Sequence[str]] = None,
    label: str = "family x scale",
) -> dict[str, Path]:
    """Write the per-cell learned/pooled/oracle NLL as csv / md / tex / json.

    :param cells: Per-cell NLL summaries.
    :type cells: Sequence[CellNLL]
    :param output_stem: Path without extension; ``.csv/.md/.tex/.json`` are written.
    :type output_stem: str | pathlib.Path
    :param families: Preferred row order.
    :type families: Sequence[str] | None
    :param scales: Preferred column order.
    :type scales: Sequence[str] | None
    :param label: Caption label.
    :type label: str
    :returns: Mapping of format to written path.
    :rtype: dict[str, pathlib.Path]
    :raises ValueError: If ``cells`` is empty.
    """
    if not cells:
        raise ValueError("write_family_scale_table needs at least one cell")
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fam_order, sc_order, _ = _grid(cells, families, scales)
    by_cell: dict[tuple[str, str], CellNLL] = {(c.family, c.scale): c for c in cells}

    def _fmt(v: Optional[float]) -> str:
        return "--" if v is None or not np.isfinite(v) else f"{v:.4f}"

    rows = [
        (fam, sc, by_cell.get((fam, sc)))
        for fam in fam_order
        for sc in sc_order
    ]
    header = ["family", "scale", "learned_nll", "pooled_nll", "oracle_nll"]
    written: dict[str, Path] = {}

    csv_path = stem.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for fam, sc, c in rows:
            writer.writerow([
                fam, sc,
                _fmt(None if c is None else c.learned_nll),
                _fmt(None if c is None else c.pooled_nll),
                _fmt(None if c is None else c.oracle_nll),
            ])
    written["csv"] = csv_path

    md_lines = [
        f"# Family x scale held-out NLL — {label}",
        "",
        "Learned-routing NLL (lower is better); pooled/oracle shown where scored.",
        "",
        "| Family | Scale | Learned | Pooled | Oracle |",
        "|---|---|---|---|---|",
    ]
    for fam, sc, c in rows:
        md_lines.append(
            f"| {fam} | {sc} | {_fmt(None if c is None else c.learned_nll)} "
            f"| {_fmt(None if c is None else c.pooled_nll)} "
            f"| {_fmt(None if c is None else c.oracle_nll)} |"
        )
    md_lines.append("")
    md_path = stem.with_suffix(".md")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    written["md"] = md_path

    tex_lines = [
        "% Generated by infl_ens.figures.scale_family",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\begin{tabular}{llrrr}",
        "    \\toprule",
        "    Family & Scale & Learned & Pooled & Oracle \\\\",
        "    \\midrule",
    ]
    for fam, sc, c in rows:
        fam_tex = fam.replace("_", "\\_")
        sc_tex = sc.replace("_", "\\_")
        tex_lines.append(
            f"    {fam_tex} & {sc_tex} & "
            f"{_fmt(None if c is None else c.learned_nll)} & "
            f"{_fmt(None if c is None else c.pooled_nll)} & "
            f"{_fmt(None if c is None else c.oracle_nll)} \\\\"
        )
    tex_lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        f"  \\caption{{Family x scale held-out learned-routing NLL — {label}. "
        f"Lower is better.}}",
        "  \\label{tab:family-scale-nll}",
        "\\end{table}",
    ]
    tex_path = stem.with_suffix(".tex")
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    written["tex"] = tex_path

    json_path = stem.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "label": label,
                "families": fam_order,
                "scales": sc_order,
                "cells": [
                    {
                        "family": c.family,
                        "scale": c.scale,
                        "learned_nll": c.learned_nll,
                        "pooled_nll": c.pooled_nll,
                        "oracle_nll": c.oracle_nll,
                    }
                    for c in cells
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    written["json"] = json_path
    return written


__all__ = [
    "CellNLL",
    "plot_family_scale_nll",
    "write_family_scale_table",
]
