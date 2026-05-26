"""Aggregate seed × sigma closed-loop sweeps into mean ± std figures.

Scans a nested layout::

    <root>/sigma0.5/seed0/history.json
    <root>/sigma0.5/seed1/history.json
    ...

For each ``sigma*`` slug, computes statistics across seeds and writes
aggregate plots under ``<figure-root>/aggregate/``:

- ``by_sigma/<slug>/trajectory_mean_std`` — trait-space trajectories
- ``by_sigma/<slug>/probe_margin_mean_std`` — cross-batch margin μ(r)
- ``by_sigma/<slug>/theory_gap_mean_std`` — per-round theory↔SFT gap (if JSONs exist)
- ``overview/metrics_vs_sigma`` — final spread, margin, gap vs σ_fraction
- ``summary.csv`` — tabular metrics for all (sigma, seed) cells

Run after :file:`scripts/run_position_only_seed_sigma_sweep.sh` or any
compatible launcher that uses the same directory naming.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

_SIGMA_RE = re.compile(r"^sigma(?P<val>[0-9.]+)$", re.IGNORECASE)
_ROUND_RE = re.compile(r"^r(?P<val>\d+)$", re.IGNORECASE)
_SEED_RE = re.compile(r"^seed(?P<val>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class RunCell:
    """One trained run in the sweep grid.

    :param group_slug: Directory name, e.g. ``sigma0.5`` or ``r20``.
    :type group_slug: str
    :param group_value: Parsed sweep coordinate (σ fraction or ``n_rounds``).
    :type group_value: float
    :param group_kind: ``sigma`` or ``round``.
    :type group_kind: str
    :param seed: Training RNG seed.
    :type seed: int
    :param run_dir: Directory containing ``history.json``.
    :type run_dir: pathlib.Path
    """

    group_slug: str
    group_value: float
    group_kind: str
    seed: int
    run_dir: Path

    @property
    def sigma_slug(self) -> str:
        """Backward-compatible alias when ``group_kind=='sigma'``."""
        return self.group_slug

    @property
    def sigma_fraction(self) -> float:
        """Backward-compatible alias when ``group_kind=='sigma'``."""
        return self.group_value


def _load_history(path: Path) -> list[dict]:
    """Load ``history.json``.

    :param path: Path to the history file.
    :type path: pathlib.Path
    :returns: Per-round records.
    :rtype: list[dict]
    """
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _agent_order(records: Sequence[dict]) -> list[str]:
    """Agent names in first-round order.

    :param records: History records.
    :type records: Sequence[dict]
    :returns: Ordered agent names.
    :rtype: list[str]
    """
    return list(records[0]["positions"].keys())


def _trajectories(records: Sequence[dict], names: Sequence[str]) -> np.ndarray:
    """Stack positions to ``(T, N, L)``.

    :param records: History records.
    :type records: Sequence[dict]
    :param names: Agent order.
    :type names: Sequence[str]
    :returns: Position tensor.
    :rtype: numpy.ndarray
    """
    return np.stack(
        [
            np.stack([np.asarray(r["positions"][n]) for n in names], axis=0)
            for r in records
        ],
        axis=0,
    )


def discover_runs(root: Path, *, layout: str = "auto") -> list[RunCell]:
    """Find all sweep cells under *root*.

    Supports ``sigma*/seed*`` (seed × σ) and ``r*/seed*`` (seed × rounds).

    :param root: Sweep results root.
    :type root: pathlib.Path
    :param layout: ``auto``, ``sigma_seed``, or ``round_seed``.
    :type layout: str
    :returns: Discovered run cells.
    :rtype: list[RunCell]
    """
    if layout == "auto":
        sigma_cells = _discover_group_seed(root, _SIGMA_RE, "sigma")
        if sigma_cells:
            return sigma_cells
        return _discover_group_seed(root, _ROUND_RE, "round")
    if layout == "sigma_seed":
        return _discover_group_seed(root, _SIGMA_RE, "sigma")
    if layout == "round_seed":
        return _discover_group_seed(root, _ROUND_RE, "round")
    raise ValueError(f"unknown layout {layout!r}")


def _discover_group_seed(
    root: Path,
    pattern: re.Pattern[str],
    kind: str,
) -> list[RunCell]:
    """Discover ``<group>/seed*/history.json`` cells.

    :param root: Sweep root.
    :type root: pathlib.Path
    :param pattern: Regex for group directory names.
    :type pattern: re.Pattern[str]
    :param kind: ``sigma`` or ``round``.
    :type kind: str
    :returns: Run cells sorted by group then seed.
    :rtype: list[RunCell]
    """
    cells: list[RunCell] = []
    if not root.is_dir():
        return cells
    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir():
            continue
        m = pattern.match(group_dir.name)
        if not m:
            continue
        group_val = float(m.group("val"))
        for seed_dir in sorted(group_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            sm = _SEED_RE.match(seed_dir.name)
            if not sm:
                continue
            hist = seed_dir / "history.json"
            if not hist.is_file():
                continue
            cells.append(
                RunCell(
                    group_slug=group_dir.name,
                    group_value=group_val,
                    group_kind=kind,
                    seed=int(sm.group("val")),
                    run_dir=seed_dir,
                )
            )
    return cells


def pairwise_spread(pos: np.ndarray) -> float:
    """Mean pairwise L2 distance among agents at one time slice.

    :param pos: Positions ``(N, L)``.
    :type pos: numpy.ndarray
    :returns: Mean off-diagonal distance.
    :rtype: float
    """
    n = pos.shape[0]
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(pos[i] - pos[j])))
    return float(np.mean(dists))


def margins_from_probe_csv(path: Path) -> dict[int, float]:
    """Cross-batch margin μ(r) = off_diag_mean − diag_mean per round.

    :param path: ``probe.csv`` from :mod:`probe_sft_capability`.
    :type path: pathlib.Path
    :returns: ``{round: margin}``.
    :rtype: dict[int, float]
    """
    by_round: dict[int, list[tuple[str, str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            r = int(row["round"])
            i_name = str(row["agent_i"]).strip()
            j_name = str(row["agent_j"]).strip()
            nll = float(row["nll"])
            by_round.setdefault(r, []).append((i_name, j_name, nll))
    out: dict[int, float] = {}
    for r, triples in by_round.items():
        diag = [nll for i, j, nll in triples if i == j]
        off = [nll for i, j, nll in triples if i != j]
        if diag and off:
            out[r] = float(np.mean(off) - np.mean(diag))
    return out


def find_probe_csv(cell: RunCell, figure_root: Optional[Path]) -> Optional[Path]:
    """Locate ``probe.csv`` for a run cell.

    :param cell: Run metadata.
    :type cell: RunCell
    :param figure_root: Optional figures tree with ``per_run/`` layout.
    :type figure_root: pathlib.Path | None
    :returns: Path if found.
    :rtype: pathlib.Path | None
    """
    candidates = [
        cell.run_dir / "probe.csv",
    ]
    if figure_root is not None:
        candidates.append(
            figure_root
            / "per_run"
            / cell.group_slug
            / f"seed{cell.seed}"
            / "probe.csv"
        )
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_theory_gaps(path: Path) -> dict[str, float]:
    """Per-agent theory↔SFT endpoint gaps from summary JSON.

    :param path: ``theory_vs_sft.json``.
    :type path: pathlib.Path
    :returns: ``{agent_name: gap}``.
    :rtype: dict[str, float]
    """
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return {a["name"]: float(a["gap"]) for a in data["agents"]}


def _mean_std_stack(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Align along axis 0 and compute mean/std.

    :param arrays: Same-shaped arrays, one per seed.
    :type arrays: list[numpy.ndarray]
    :returns: ``(mean, std)`` with ``std`` using ddof=0; NaN if only one seed.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    stack = np.stack(arrays, axis=0)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0) if stack.shape[0] > 1 else np.zeros_like(mean)
    return mean, std


def plot_trajectory_mean_std(
    *,
    sigma_fraction: float,
    names: Sequence[str],
    pos_mean: np.ndarray,
    pos_std: np.ndarray,
    theo_mean: Optional[np.ndarray],
    axis_labels: tuple[str, str],
    title: Optional[str],
    output_stem: Path,
    n_seeds: int,
) -> None:
    """Trait-space trajectories: mean line and ±1σ band per clone.

    :param sigma_fraction: σ / σ₀* for the legend and title.
    :type sigma_fraction: float
    :param names: Clone names.
    :type names: Sequence[str]
    :param pos_mean: Mean positions ``(T, N, L)``.
    :type pos_mean: numpy.ndarray
    :param pos_std: Std positions ``(T, N, L)``.
    :type pos_std: numpy.ndarray
    :param theo_mean: Optional mean theory endpoints ``(N, L)`` (one point per agent).
    :type theo_mean: numpy.ndarray | None
    :param axis_labels: Trait axis names.
    :type axis_labels: tuple[str, str]
    :param title: Figure suptitle.
    :type title: str | None
    :param output_stem: Filename stem (no extension).
    :type output_stem: pathlib.Path
    :param n_seeds: Number of seeds aggregated (for legend).
    :type n_seeds: int
    """
    import matplotlib.pyplot as plt

    T, N, L = pos_mean.shape
    if L != 2:
        raise ValueError(f"requires L=2, got L={L}")

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(N, 3)))
    sigma_label = f"σ/σ₀* = {sigma_fraction:g}"

    for i, name in enumerate(names):
        xs, ys = pos_mean[:, i, 0], pos_mean[:, i, 1]
        xs_s, ys_s = pos_std[:, i, 0], pos_std[:, i, 1]
        ax.plot(
            xs, ys, "-", color=colors[i], lw=2.0,
            label=f"{name} (mean, {n_seeds} seeds)",
        )
        ax.fill_between(xs, ys - ys_s, ys + ys_s, color=colors[i], alpha=0.2)
        ax.scatter(xs[0], ys[0], color=colors[i], marker="o", s=50,
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.scatter(xs[-1], ys[-1], color=colors[i], marker="*", s=140,
                   edgecolor="black", linewidth=0.5, zorder=3)
        if theo_mean is not None:
            tx, ty = theo_mean[i, 0], theo_mean[i, 1]
            ax.scatter(
                tx, ty, color=colors[i], marker="X", s=120,
                edgecolor="black", linewidth=0.6, zorder=4,
                label="theory NE (mean)" if i == 0 else None,
            )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"trait trajectories — {sigma_label}  •  ○ start, ★ SFT end, ✕ theory")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.patches import Patch
    handles.append(Patch(facecolor="gray", alpha=0.25, label="±1σ across seeds"))
    ax.legend(handles=handles, labels=labels + ["±1σ across seeds"],
              loc="best", fontsize=8, frameon=True)
    if title:
        fig.suptitle(title)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_series_mean_std(
    *,
    rounds: np.ndarray,
    series_mean: np.ndarray,
    series_std: np.ndarray,
    ylabel: str,
    title: str,
    output_stem: Path,
    n_seeds: int,
) -> None:
    """Single time series with mean line and shaded ±1σ.

    :param rounds: Round indices.
    :type rounds: numpy.ndarray
    :param series_mean: Mean values per round.
    :type series_mean: numpy.ndarray
    :param series_std: Std per round.
    :type series_std: numpy.ndarray
    :param ylabel: Y-axis label.
    :type ylabel: str
    :param title: Panel title (include σ and seed count).
    :type title: str
    :param output_stem: Filename stem.
    :type output_stem: pathlib.Path
    :param n_seeds: Seeds aggregated.
    :type n_seeds: int
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(rounds, series_mean, "-o", color="C0", lw=2,
            label=f"mean over {n_seeds} seeds")
    ax.fill_between(
        rounds,
        series_mean - series_std,
        series_mean + series_std,
        color="C0",
        alpha=0.25,
        label="±1σ across seeds",
    )
    ax.set_xlabel("round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_overview(
    *,
    x_values: np.ndarray,
    spread_mean: np.ndarray,
    spread_std: np.ndarray,
    margin_mean: np.ndarray,
    margin_std: np.ndarray,
    gap_mean: np.ndarray,
    gap_std: np.ndarray,
    n_seeds: int,
    output_stem: Path,
    title: Optional[str],
    xlabel: str = r"$\sigma / \sigma_0^*$",
) -> None:
    """Three-panel overview: final spread, final margin, mean theory gap vs *x*.

    :param x_values: Sweep coordinate (σ fraction or ``n_rounds``).
    :type x_values: numpy.ndarray
    :param spread_mean: Mean final pairwise spread per σ.
    :type spread_mean: numpy.ndarray
    :param spread_std: Std of final spread across seeds.
    :type spread_std: numpy.ndarray
    :param margin_mean: Mean final probe margin per σ.
    :type margin_mean: numpy.ndarray
    :param margin_std: Std of final margin.
    :type margin_std: numpy.ndarray
    :param gap_mean: Mean theory gap (L2, averaged over agents) per σ.
    :type gap_mean: numpy.ndarray
    :param gap_std: Std of mean gap across seeds.
    :type gap_std: numpy.ndarray
    :param n_seeds: Seeds per σ cell.
    :type n_seeds: int
    :param output_stem: Filename stem.
    :type output_stem: pathlib.Path
    :param title: Optional suptitle.
    :type title: str | None
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = [
        (spread_mean, spread_std, "final pairwise spread", "trait separation"),
        (margin_mean, margin_std, r"final probe margin $\mu(r)$", "cross-batch NLL margin"),
        (gap_mean, gap_std, "mean theory↔SFT gap", r"$\|\mathbf{x}_\mathrm{SFT}-\mathbf{x}_\mathrm{theo}\|_2$"),
    ]
    for ax, (ym, ys, ylab, subt) in zip(axes, panels):
        ax.errorbar(
            x_values, ym, yerr=ys, fmt="o-", capsize=4, lw=1.8,
            label=f"mean ± std ({n_seeds} seeds)",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        ax.set_title(subt)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def aggregate(
    cells: Sequence[RunCell],
    figure_root: Path,
    *,
    axis_labels: tuple[str, str],
    title: Optional[str],
) -> Path:
    """Build all aggregate figures and ``summary.csv``.

    :param cells: Discovered run cells.
    :type cells: Sequence[RunCell]
    :param figure_root: Base figure directory for this sweep.
    :type figure_root: pathlib.Path
    :param axis_labels: Trait-space axis names.
    :type axis_labels: tuple[str, str]
    :param title: Optional global suptitle.
    :type title: str | None
    :returns: Path to written ``summary.csv``.
    :rtype: pathlib.Path
    """
    group_kind = cells[0].group_kind
    agg_dir = figure_root / "aggregate"
    by_group_dir = agg_dir / f"by_{group_kind}"
    overview_dir = agg_dir / "overview"
    overview_xlabel = (
        r"$\sigma / \sigma_0^*$" if group_kind == "sigma" else "closed-loop rounds"
    )
    overview_stem = (
        "metrics_vs_sigma" if group_kind == "sigma" else "metrics_vs_rounds"
    )

    by_group: dict[str, list[RunCell]] = {}
    for c in cells:
        by_group.setdefault(c.group_slug, []).append(c)

    summary_rows: list[dict] = []

    overview_x: list[float] = []
    overview_spread_m: list[float] = []
    overview_spread_s: list[float] = []
    overview_margin_m: list[float] = []
    overview_margin_s: list[float] = []
    overview_gap_m: list[float] = []
    overview_gap_s: list[float] = []
    n_seeds_ref = 0

    for group_slug in sorted(by_group.keys(), key=lambda s: by_group[s][0].group_value):
        group = sorted(by_group[group_slug], key=lambda c: c.seed)
        group_val = group[0].group_value
        n_seeds = len(group)
        n_seeds_ref = max(n_seeds_ref, n_seeds)
        out_dir = by_group_dir / group_slug
        if group_kind == "sigma":
            coord_title = f"σ/σ₀* = {group_val:g}"
        else:
            coord_title = f"n_rounds = {int(group_val)}"

        # --- trajectories ---
        trajs: list[np.ndarray] = []
        names: list[str] = []
        min_T = None
        for cell in group:
            rec = _load_history(cell.run_dir / "history.json")
            if not names:
                names = _agent_order(rec)
            pos = _trajectories(rec, names)
            if min_T is None:
                min_T = pos.shape[0]
            else:
                min_T = min(min_T, pos.shape[0])
            trajs.append(pos)

        trajs = [t[:min_T] for t in trajs]
        pos_mean, pos_std = _mean_std_stack(trajs)

        theo_ends: list[np.ndarray] = []
        gaps_per_seed: list[float] = []
        for cell in group:
            tjson = cell.run_dir / "theory_vs_sft.json"
            if tjson.is_file():
                with tjson.open(encoding="utf-8") as fh:
                    data = json.load(fh)
                theo = np.stack(
                    [np.asarray(a["theory_end"]) for a in data["agents"]], axis=0
                )
                theo_ends.append(theo)
                gaps_per_seed.append(
                    float(np.mean([float(a["gap"]) for a in data["agents"]]))
                )

        theo_mean = None
        if theo_ends:
            theo_mean = np.stack(theo_ends, axis=0).mean(axis=0)

        plot_trajectory_mean_std(
            sigma_fraction=group_val,
            names=names,
            pos_mean=pos_mean,
            pos_std=pos_std,
            theo_mean=theo_mean,
            axis_labels=axis_labels,
            title=title or coord_title,
            output_stem=out_dir / "trajectory_mean_std",
            n_seeds=n_seeds,
        )
        print(f"wrote {out_dir / 'trajectory_mean_std'}.{{pdf,png}}")

        # --- per-seed metrics + spreads over time ---
        rounds = np.arange(min_T)
        spreads_over_time: list[np.ndarray] = []
        margins_by_seed: list[dict[int, float]] = []

        for cell in group:
            rec = _load_history(cell.run_dir / "history.json")
            pos = _trajectories(rec, names)[:min_T]
            spreads = np.array([pairwise_spread(pos[t]) for t in range(min_T)])
            spreads_over_time.append(spreads)

            probe_path = find_probe_csv(cell, figure_root)
            margin_final = float("nan")
            if probe_path is not None:
                margins = margins_from_probe_csv(probe_path)
                margins_by_seed.append(margins)
                if margins:
                    last_r = max(margins)
                    margin_final = margins[last_r]
            else:
                margins_by_seed.append({})

            gap_mean_seed = float("nan")
            tjson = cell.run_dir / "theory_vs_sft.json"
            if tjson.is_file():
                g = load_theory_gaps(tjson)
                gap_mean_seed = float(np.mean(list(g.values())))

            summary_rows.append({
                "group_slug": group_slug,
                "group_value": group_val,
                "group_kind": group_kind,
                "seed": cell.seed,
                "n_rounds": min_T - 1,
                "final_spread": float(spreads[-1]),
                "final_margin": margin_final,
                "mean_theory_gap": gap_mean_seed,
            })

        spread_mean, spread_std = _mean_std_stack(spreads_over_time)
        plot_series_mean_std(
            rounds=rounds,
            series_mean=spread_mean,
            series_std=spread_std,
            ylabel="pairwise L2 spread",
            title=f"trait spread vs round — {coord_title} (N={n_seeds} seeds)",
            output_stem=out_dir / "spread_vs_round_mean_std",
            n_seeds=n_seeds,
        )

        # --- probe margins ---
        if margins_by_seed and any(margins_by_seed):
            # align rounds present in all seeds that have probe data
            common_rounds = None
            margin_arrays: list[np.ndarray] = []
            for m in margins_by_seed:
                if not m:
                    continue
                rs = sorted(m.keys())
                if common_rounds is None:
                    common_rounds = rs
                else:
                    common_rounds = sorted(set(common_rounds) & set(rs))
            if common_rounds:
                margin_arrays = [
                    np.array([m[r] for r in common_rounds])
                    for m in margins_by_seed
                    if m
                ]
                m_mean, m_std = _mean_std_stack(margin_arrays)
                plot_series_mean_std(
                    rounds=np.array(common_rounds),
                    series_mean=m_mean,
                    series_std=m_std,
                    ylabel=r"margin $\mu(r)$ (off − diag NLL)",
                    title=f"probe margin vs round — {coord_title} (N={len(margin_arrays)} seeds)",
                    output_stem=out_dir / "probe_margin_mean_std",
                    n_seeds=len(margin_arrays),
                )

        # --- theory gap bar summary per sigma ---
        if gaps_per_seed:
            g_mean = float(np.mean(gaps_per_seed))
            g_std = float(np.std(gaps_per_seed)) if len(gaps_per_seed) > 1 else 0.0
            overview_gap_m.append(g_mean)
            overview_gap_s.append(g_std)
        else:
            overview_gap_m.append(float("nan"))
            overview_gap_s.append(float("nan"))

        final_spreads = [float(s[-1]) for s in spreads_over_time]
        overview_x.append(group_val)
        overview_spread_m.append(float(np.mean(final_spreads)))
        overview_spread_s.append(
            float(np.std(final_spreads)) if len(final_spreads) > 1 else 0.0
        )

        final_margins = [
            row["final_margin"]
            for row in summary_rows
            if row["group_slug"] == group_slug and not np.isnan(row["final_margin"])
        ]
        if final_margins:
            overview_margin_m.append(float(np.mean(final_margins)))
            overview_margin_s.append(
                float(np.std(final_margins)) if len(final_margins) > 1 else 0.0
            )
        else:
            overview_margin_m.append(float("nan"))
            overview_margin_s.append(float("nan"))

    # --- overview ---
    if overview_x:
        order = np.argsort(overview_x)
        x_arr = np.array(overview_x)[order]
        plot_overview(
            x_values=x_arr,
            spread_mean=np.array(overview_spread_m)[order],
            spread_std=np.array(overview_spread_s)[order],
            margin_mean=np.array(overview_margin_m)[order],
            margin_std=np.array(overview_margin_s)[order],
            gap_mean=np.array(overview_gap_m)[order],
            gap_std=np.array(overview_gap_s)[order],
            n_seeds=n_seeds_ref,
            output_stem=overview_dir / overview_stem,
            title=title,
            xlabel=overview_xlabel,
        )
        print(f"wrote {overview_dir / overview_stem}.{{pdf,png}}")

    # --- CSV ---
    csv_path = agg_dir / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_slug", "group_value", "group_kind", "seed", "n_rounds",
        "final_spread", "final_margin", "mean_theory_gap",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {csv_path}")
    return csv_path


def _build_parser() -> argparse.ArgumentParser:
    """CLI parser.

    :returns: Configured parser.
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="Aggregate seed×sigma sweep runs into mean±std figures.",
    )
    p.add_argument(
        "--root", type=Path, required=True,
        help="Sweep results root (sigma*/seed* or r*/seed*).",
    )
    p.add_argument(
        "--layout", choices=("auto", "sigma_seed", "round_seed"), default="auto",
        help="Directory layout under --root.",
    )
    p.add_argument(
        "--figure-root", type=Path, required=True,
        help="Figure tree for this sweep (writes aggregate/ subfolder).",
    )
    p.add_argument("--config", type=Path, default=None, help="Unused; reserved.")
    p.add_argument(
        "--axis-labels", nargs=2, default=["harm", "hallucination"],
        help="Trait-space axis labels.",
    )
    p.add_argument("--title", type=str, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional CLI args.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    args = _build_parser().parse_args(argv)
    cells = discover_runs(args.root, layout=args.layout)
    if not cells:
        print(f"no runs found under {args.root}", file=sys.stderr)
        return 1

    kind = cells[0].group_kind
    n_group = len({c.group_slug for c in cells})
    n_seed = len({c.seed for c in cells})
    print(f"found {len(cells)} cells ({n_group} {kind} groups × up to {n_seed} seeds)")

    aggregate(
        cells,
        args.figure_root,
        axis_labels=(args.axis_labels[0], args.axis_labels[1]),
        title=args.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
