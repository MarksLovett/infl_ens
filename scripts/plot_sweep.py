"""Aggregate a sweep of closed-loop runs into a single comparison figure.

Reads multiple ``history.json`` files produced by ``scripts/run_sweep.sh`` and
renders a grid of trajectory subplots plus a summary CSV that classifies each
run's final equilibrium type (e.g. ``(2, 2)``, ``(2, 1, 1)``,
``(1, 1, 1, 1)``).

Equilibrium-type classification is a simple single-linkage clustering of the
final agent positions in trait space, with a configurable distance threshold.
Output cluster sizes are sorted descending so different runs that produce the
same partition give the same tag.

If ``--with-theory`` is set, each run's ``theory_vs_sft.json`` (written by
``run_sweep.sh`` when ``POST_THEORY=1``) is loaded and the theoretical Nash
endpoints are overlaid as faint markers on each panel.

Run with::

    python scripts/plot_sweep.py \\
        --root results/sweep_seeds \\
        --mode seeds \\
        --output-stem scripts/figures/sweep_seeds \\
        --with-theory
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = ROOT / "scripts" / "figures"


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def _discover_runs(root: Path, mode: str) -> list[dict]:
    """Find sub-runs under ``root`` and parse their slug-encoded parameter.

    Sub-directories whose names start with the mode prefix
    (``seed``/``sigma``/``kde``) are treated as sweep runs. Each must contain
    a ``history.json``.

    :param root: Sweep root directory.
    :type root: pathlib.Path
    :param mode: Sweep mode, one of ``'seeds'``, ``'sigma'``, ``'kde'``.
    :type mode: str
    :returns: List of dicts with keys ``slug``, ``value``, ``history_path``,
        ``theory_path`` (``None`` if absent).
    :rtype: list[dict]
    """
    prefix = {"seeds": "seed", "sigma": "sigma", "kde": "kde"}[mode]
    runs: list[dict] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        history = d / "history.json"
        if not history.exists():
            continue
        raw = d.name[len(prefix):]
        try:
            value: float = float(raw) if mode != "seeds" else float(int(raw))
        except ValueError:
            continue
        theory = d / "theory_vs_sft.json"
        runs.append({
            "slug": d.name,
            "value": value,
            "history_path": history,
            "theory_path": theory if theory.exists() else None,
        })
    runs.sort(key=lambda r: r["value"])
    return runs


def _load_history(path: Path) -> list[dict]:
    """Load a single ``history.json``.

    :param path: File path.
    :type path: pathlib.Path
    :returns: Per-round dictionaries.
    :rtype: list[dict]
    """
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _trajectory(records: Sequence[dict]) -> tuple[list[str], np.ndarray]:
    """Stack per-round positions into a ``(T, N, L)`` array.

    :param records: Loaded history records.
    :type records: Sequence[dict]
    :returns: Tuple of ``(agent_names, positions)``.
    :rtype: tuple[list[str], numpy.ndarray]
    """
    names = list(records[0]["positions"].keys())
    pos = np.stack(
        [np.stack([np.asarray(r["positions"][n]) for n in names], axis=0)
         for r in records],
        axis=0,
    )
    return names, pos


# -----------------------------------------------------------------------------
# Equilibrium-type classification
# -----------------------------------------------------------------------------

def classify_equilibrium(
    positions: np.ndarray,
    *,
    threshold: float = 0.1,
) -> tuple[int, ...]:
    """Classify a final configuration by single-linkage clustering of positions.

    Returns a tuple of cluster sizes sorted descending. Examples:

    - ``(2, 2)``: two pairs of clones at distinct positions.
    - ``(2, 1, 1)``: one pair plus two singletons.
    - ``(4,)``: all clones collapsed onto one point.

    :param positions: Final positions, shape ``(N, L)``.
    :type positions: numpy.ndarray
    :param threshold: Maximum L2 distance for two positions to be considered
        in the same cluster. Default 0.1 is appropriate for a :math:`[0, 1]^L`
        trait space.
    :type threshold: float
    :returns: Tuple of cluster sizes, sorted descending.
    :rtype: tuple[int, ...]
    """
    N = positions.shape[0]
    parent = list(range(N))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(positions[i] - positions[j]) < threshold:
                union(i, j)

    sizes: dict[int, int] = {}
    for i in range(N):
        sizes[find(i)] = sizes.get(find(i), 0) + 1
    return tuple(sorted(sizes.values(), reverse=True))


# -----------------------------------------------------------------------------
# Per-run summary
# -----------------------------------------------------------------------------

def _summarise_run(run: dict, *, cluster_threshold: float) -> dict:
    """Compute summary stats for a single run.

    :param run: Run descriptor from :func:`_discover_runs`.
    :type run: dict
    :param cluster_threshold: Threshold for equilibrium-type classification.
    :type cluster_threshold: float
    :returns: Summary with keys ``slug``, ``value``, ``names``,
        ``positions`` (T×N×L), ``equilibrium_type``, ``u_pool_final``,
        ``share_final``, ``u_grid_final``, ``theory_positions`` (or None),
        ``theory_eq_type`` (or None).
    :rtype: dict
    """
    records = _load_history(run["history_path"])
    names, pos = _trajectory(records)
    last = records[-1]
    eq_type = classify_equilibrium(pos[-1], threshold=cluster_threshold)
    summary = {
        "slug": run["slug"],
        "value": run["value"],
        "names": names,
        "positions": pos,
        "u_grid_final": np.asarray(last["u_grid"]),
        "u_pool_final": np.asarray(last["u_pool"]),
        "share_final": np.asarray(last["observed_share"]),
        "equilibrium_type": eq_type,
        "n_rounds": len(records),
        "theory_positions": None,
        "theory_eq_type": None,
        "theory_u_pool": None,
    }
    if run["theory_path"] is not None:
        with run["theory_path"].open("r", encoding="utf-8") as fh:
            theo = json.load(fh)
        # Re-order to match SFT name order.
        by_name = {a["name"]: a for a in theo["agents"]}
        theo_pos = np.stack(
            [np.asarray(by_name[n]["theory_end"]) for n in names], axis=0,
        )
        theo_u = np.array([by_name[n]["u_pool_theory"] for n in names])
        summary["theory_positions"] = theo_pos
        summary["theory_eq_type"] = classify_equilibrium(
            theo_pos, threshold=cluster_threshold,
        )
        summary["theory_u_pool"] = theo_u
        summary["sigma"] = float(theo["sigma"])
        summary["sigma_star"] = float(theo["sigma_star"])
    return summary


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_sweep(
    summaries: list[dict],
    *,
    mode: str,
    axis_labels: tuple[str, str] = ("harm", "hallucination"),
    title: Optional[str] = None,
    with_theory: bool = False,
):
    """Render a grid of per-run trajectory panels with summary text.

    :param summaries: Output of :func:`_summarise_run` per run.
    :type summaries: list[dict]
    :param mode: Sweep mode (used for panel titles).
    :type mode: str
    :param axis_labels: Trait-space axis names.
    :type axis_labels: tuple[str, str]
    :param title: Optional figure suptitle.
    :type title: str | None
    :param with_theory: Whether to overlay theoretical Nash endpoints.
    :type with_theory: bool
    :returns: Matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n_runs = len(summaries)
    n_cols = min(n_runs, 4)
    n_rows = int(np.ceil(n_runs / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 4.0 * n_rows + 0.5),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    label_fmt = {"seeds": "seed={:.0f}", "sigma": "σ_frac={:.2f}",
                 "kde": "h={:.3f}"}[mode]

    for ax, s in zip(axes_flat, summaries):
        names = s["names"]
        pos = s["positions"]
        n_agents = pos.shape[1]
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(n_agents, 3)))
        for i, name in enumerate(names):
            xs, ys = pos[:, i, 0], pos[:, i, 1]
            ax.plot(xs, ys, "--", color=colors[i], lw=1.5, alpha=0.9,
                    label=name if ax is axes_flat[0] else None)
            ax.scatter(xs[0], ys[0], color=colors[i], marker="o",
                       s=40, edgecolor="black", linewidth=0.5, zorder=3)
            ax.scatter(xs[-1], ys[-1], color=colors[i], marker="*",
                       s=160, edgecolor="black", linewidth=0.6, zorder=4)
            if with_theory and s["theory_positions"] is not None:
                tx, ty = s["theory_positions"][i]
                ax.scatter([tx], [ty], color=colors[i], marker="X",
                           s=110, edgecolor="black", linewidth=0.6,
                           alpha=0.65, zorder=4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        eq = "(" + ", ".join(str(x) for x in s["equilibrium_type"]) + ")"
        title_str = f"{label_fmt.format(s['value'])}   →   {eq}"
        if with_theory and s["theory_eq_type"] is not None:
            theo_eq = "(" + ", ".join(str(x) for x in s["theory_eq_type"]) + ")"
            title_str += f"\ntheory: {theo_eq}"
        ax.set_title(title_str, fontsize=10)

    # Hide unused axes
    for ax in axes_flat[n_runs:]:
        ax.set_visible(False)

    # Shared legend on the first panel
    if summaries:
        axes_flat[0].legend(loc="best", fontsize=8, frameon=True)
        # Shared axis labels on outer panels
        for ax in axes[-1, :]:
            ax.set_xlabel(axis_labels[0])
        for ax in axes[:, 0]:
            ax.set_ylabel(axis_labels[1])

    if title is not None:
        fig.suptitle(title, fontsize=12)
    return fig


# -----------------------------------------------------------------------------
# CSV summary
# -----------------------------------------------------------------------------

def _write_csv(summaries: list[dict], path: Path) -> None:
    """Write a one-row-per-run CSV summary of the sweep.

    :param summaries: Output of :func:`_summarise_run` per run.
    :type summaries: list[dict]
    :param path: CSV path. Parent directories are created if needed.
    :type path: pathlib.Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n_agents = len(summaries[0]["names"]) if summaries else 0
    header = [
        "slug", "value", "n_rounds",
        "equilibrium_type_sft", "equilibrium_type_theory",
    ]
    for n in summaries[0]["names"] if summaries else []:
        header.extend([
            f"{n}_x", f"{n}_y",
            f"{n}_u_pool", f"{n}_share",
        ])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for s in summaries:
            row = [
                s["slug"],
                s["value"],
                s["n_rounds"],
                "(" + ", ".join(str(x) for x in s["equilibrium_type"]) + ")",
                (("(" + ", ".join(str(x) for x in s["theory_eq_type"]) + ")")
                 if s["theory_eq_type"] is not None else ""),
            ]
            for i, _ in enumerate(s["names"]):
                row.extend([
                    f"{s['positions'][-1, i, 0]:.6f}",
                    f"{s['positions'][-1, i, 1]:.6f}",
                    f"{s['u_pool_final'][i]:.6f}",
                    f"{s['share_final'][i]:.6f}",
                ])
            w.writerow(row)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser.

    :returns: Configured parser.
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="Aggregate a sweep of closed-loop runs into one figure + CSV."
    )
    p.add_argument("--root", type=Path, required=True,
                   help="Sweep root directory (e.g. results/sweep_seeds).")
    p.add_argument("--mode", choices=["seeds", "sigma", "kde"], required=True)
    p.add_argument("--cluster-threshold", type=float, default=0.1,
                   help="L2 distance below which two positions are considered "
                        "in the same cluster.")
    p.add_argument("--axis-labels", nargs=2, default=["harm", "hallucination"])
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--with-theory", action="store_true",
                   help="Overlay theoretical NE endpoints from theory_vs_sft.json.")
    p.add_argument("--output-stem", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None,
                   help="Optional CSV summary path. Defaults to "
                        "<output-stem>.csv.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional argument vector.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    args = _build_parser().parse_args(argv)
    runs = _discover_runs(args.root, args.mode)
    if not runs:
        print(f"no runs found under {args.root} matching mode={args.mode}",
              file=sys.stderr)
        return 1
    print(f"discovered {len(runs)} runs in {args.root}:")
    for r in runs:
        print(f"  {r['slug']:<14}  value={r['value']:<8}  "
              f"theory={'yes' if r['theory_path'] else 'no'}")

    summaries = [
        _summarise_run(r, cluster_threshold=args.cluster_threshold)
        for r in runs
    ]

    # Print a quick table to stdout
    print(f"\n{'slug':<14} {'value':>8} {'SFT eq':<14} {'theory eq':<14}")
    print("-" * 60)
    for s in summaries:
        sft_eq = "(" + ", ".join(str(x) for x in s["equilibrium_type"]) + ")"
        theo_eq = (("(" + ", ".join(str(x) for x in s["theory_eq_type"]) + ")")
                   if s["theory_eq_type"] is not None else "—")
        print(f"{s['slug']:<14} {s['value']:>8} {sft_eq:<14} {theo_eq:<14}")

    fig = plot_sweep(
        summaries,
        mode=args.mode,
        axis_labels=(args.axis_labels[0], args.axis_labels[1]),
        title=args.title,
        with_theory=args.with_theory,
    )

    stem = args.output_stem or (FIGS_DIR / f"sweep_{args.mode}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"\nwrote {pdf_path}")
    print(f"wrote {png_path}")

    csv_path = args.csv or stem.with_suffix(".csv")
    _write_csv(summaries, csv_path)
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
