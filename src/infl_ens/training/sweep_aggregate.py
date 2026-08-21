"""Aggregate closed-loop sweep runs into summaries and figures."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from infl_ens.training.pool_dynamics import pairwise_spread
from infl_ens.utils.sweep_discovery import (
    RunCell,
    agent_order,
    discover_sigma_seed_history_paths,
    iter_sigma_seed_histories,
    load_history,
    position_tensor,
)
from infl_ens.vis.sweeps import (
    plot_overview,
    plot_series_mean_std,
    plot_trajectory_mean_std,
)


def classify_equilibrium_clusters(
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
    n = positions.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < threshold:
                union(i, j)

    sizes: dict[int, int] = {}
    for i in range(n):
        sizes[find(i)] = sizes.get(find(i), 0) + 1
    return tuple(sorted(sizes.values(), reverse=True))


def margins_from_probe_csv(path: Path) -> dict[int, float]:
    """Cross-batch margin μ(r) = off_diag_mean − diag_mean per round.

    :param path: ``probe.csv`` from :mod:`infl_ens.evaluation.capability_probe`.
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


def mean_std_stack(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
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


def summarise_flat_sweep_run(
    run: dict,
    *,
    cluster_threshold: float,
) -> dict:
    """Compute summary stats for a single flat sweep run.

    :param run: Run descriptor from
        :func:`infl_ens.utils.sweep_discovery.discover_flat_sweep_runs`.
    :type run: dict
    :param cluster_threshold: Threshold for equilibrium-type classification.
    :type cluster_threshold: float
    :returns: Summary with keys ``slug``, ``value``, ``names``,
        ``positions`` (T×N×L), ``equilibrium_type``, ``u_pool_final``,
        ``share_final``, ``u_grid_final``, ``theory_positions`` (or None),
        ``theory_eq_type`` (or None).
    :rtype: dict
    """
    records = load_history(run["history_path"])
    names = agent_order(records)
    pos = position_tensor(records, names)
    last = records[-1]
    eq_type = classify_equilibrium_clusters(pos[-1], threshold=cluster_threshold)
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
        by_name = {a["name"]: a for a in theo["agents"]}
        theo_pos = np.stack(
            [np.asarray(by_name[n]["theory_end"]) for n in names], axis=0,
        )
        theo_u = np.array([by_name[n]["u_pool_theory"] for n in names])
        summary["theory_positions"] = theo_pos
        summary["theory_eq_type"] = classify_equilibrium_clusters(
            theo_pos, threshold=cluster_threshold,
        )
        summary["theory_u_pool"] = theo_u
        summary["sigma"] = float(theo["sigma"])
        summary["sigma_star"] = float(theo["sigma_star"])
    return summary


def write_flat_sweep_csv(summaries: list[dict], path: Path) -> None:
    """Write a one-row-per-run CSV summary of a flat sweep.

    :param summaries: Output of :func:`summarise_flat_sweep_run` per run.
    :type summaries: list[dict]
    :param path: CSV path. Parent directories are created if needed.
    :type path: pathlib.Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
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


def aggregate_group_seed_sweep(
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

        trajs: list[np.ndarray] = []
        names: list[str] = []
        min_t: int | None = None
        for cell in group:
            rec = load_history(cell.run_dir / "history.json")
            if not names:
                names = agent_order(rec)
            pos = position_tensor(rec, names)
            if min_t is None:
                min_t = pos.shape[0]
            else:
                min_t = min(min_t, pos.shape[0])
            trajs.append(pos)

        assert min_t is not None
        trajs = [t[:min_t] for t in trajs]
        pos_mean, pos_std = mean_std_stack(trajs)

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
            n_seeds=n_seeds,
            output_stem=out_dir / "trajectory_mean_std",
        )
        print(f"wrote {out_dir / 'trajectory_mean_std'}.{{pdf,png}}")

        rounds = np.arange(min_t)
        spreads_over_time: list[np.ndarray] = []
        margins_by_seed: list[dict[int, float]] = []

        for cell in group:
            rec = load_history(cell.run_dir / "history.json")
            pos = position_tensor(rec, names)[:min_t]
            spreads = np.array([pairwise_spread(pos[t]) for t in range(min_t)])
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
                "n_rounds": min_t - 1,
                "final_spread": float(spreads[-1]),
                "final_margin": margin_final,
                "mean_theory_gap": gap_mean_seed,
            })

        spread_mean, spread_std = mean_std_stack(spreads_over_time)
        plot_series_mean_std(
            rounds=rounds,
            series_mean=spread_mean,
            series_std=spread_std,
            ylabel="pairwise L2 spread",
            title=f"trait spread vs round — {coord_title} (N={n_seeds} seeds)",
            output_stem=out_dir / "spread_vs_round_mean_std",
            n_seeds=n_seeds,
        )

        if margins_by_seed and any(margins_by_seed):
            common_rounds: list[int] | None = None
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
                m_mean, m_std = mean_std_stack(margin_arrays)
                plot_series_mean_std(
                    rounds=np.array(common_rounds),
                    series_mean=m_mean,
                    series_std=m_std,
                    ylabel=r"margin $\mu(r)$ (off − diag NLL)",
                    title=(
                        f"probe margin vs round — {coord_title} "
                        f"(N={len(margin_arrays)} seeds)"
                    ),
                    output_stem=out_dir / "probe_margin_mean_std",
                    n_seeds=len(margin_arrays),
                )

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


COLLAPSE_SPREAD_THRESH = 0.45
MODE_STEP_DIR_RE = re.compile(r"^mode_(?P<slug>.+)$", re.IGNORECASE)


def aggregate_final_positions(
    entries: list[tuple[float, int, Path]],
) -> dict[float, dict]:
    """Compute mean/std of final positions per sigma.

    :param entries: ``(sigma_fraction, seed, history_path)`` list.
    :type entries: list[tuple[float, int, pathlib.Path]]
    :returns: Nested stats dict keyed by sigma fraction.
    :rtype: dict
    """
    by_sigma: dict[float, list[tuple[int, dict[str, list[float]]]]] = {}
    for sigma, seed, path in entries:
        records = load_history(path)
        positions = records[-1]["positions"]
        by_sigma.setdefault(sigma, []).append((seed, positions))

    result: dict[float, dict] = {}
    for sigma, runs in sorted(by_sigma.items()):
        names = list(runs[0][1].keys())
        n = len(runs)
        spreads: list[float] = []
        per_agent: dict[str, dict] = {}
        for _, pos in runs:
            arr = np.stack([np.asarray(pos[name], dtype=float) for name in names])
            if len(names) >= 2:
                spreads.append(pairwise_spread(arr))
        for name in names:
            arr = np.array([np.asarray(pos[name], dtype=float) for _, pos in runs])
            per_agent[name] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0, ddof=1 if n > 1 else 0).tolist(),
            }
        result[sigma] = {
            "n_runs": n,
            "seeds": [s for s, _ in runs],
            "spread_mean": float(np.mean(spreads)),
            "spread_std": float(np.std(spreads, ddof=1 if n > 1 else 0)),
            "agents": per_agent,
        }
    return result


def print_final_positions_report(
    stats: dict[float, dict],
    axis_labels: tuple[str, str],
) -> None:
    """Print human-readable tables for :func:`aggregate_final_positions`.

    :param stats: Output of :func:`aggregate_final_positions`.
    :type stats: dict
    :param axis_labels: Trait axis names.
    :type axis_labels: tuple[str, str]
    """
    for sigma, block in sorted(stats.items()):
        print(f"\n{'=' * 60}")
        print(f"  sigma_fraction = {sigma:g}   n = {block['n_runs']} seeds")
        print(
            f"  pairwise spread: {block['spread_mean']:.4f} ± "
            f"{block['spread_std']:.4f}"
        )
        print(f"  seeds: {block['seeds']}")
        print(f"{'=' * 60}")
        print(
            f"{'clone':<10} {axis_labels[0]+' (mean±std)':<22} "
            f"{axis_labels[1]+' (mean±std)':<22}"
        )
        print("-" * 60)
        for name, ag in block["agents"].items():
            m0, m1 = ag["mean"]
            s0, s1 = ag["std"]
            print(
                f"{name:<10} {m0:.4f}±{s0:.4f}          "
                f"{m1:.4f}±{s1:.4f}"
            )


def summarize_pairs_near_theory_sweep(
    root: Path,
    *,
    baseline_root: Optional[Path] = None,
) -> tuple[list[dict], Path]:
    """Summarize ``pairs_near_theory`` layout counts per sigma.

    :param root: Sweep root with ``sigma*/seed*/history.json``.
    :type root: pathlib.Path
    :param baseline_root: Optional ``mean_noise`` baseline for comparison.
    :type baseline_root: pathlib.Path | None
    :returns: Per-run rows and path to written ``summary.json``.
    :rtype: tuple[list[dict], pathlib.Path]
    """
    from infl_ens.training.pool_dynamics import classify_layout

    rows: list[dict] = []
    for cell in iter_sigma_seed_histories(root):
        hist = cell["history"]
        names = sorted(hist[0]["positions"].keys())
        p0 = np.stack([np.asarray(hist[0]["positions"][n]) for n in names])
        pf = np.stack([np.asarray(hist[-1]["positions"][n]) for n in names])
        rows.append({
            "sigma": cell["sigma"],
            "seed": cell["seed"],
            "spread0": pairwise_spread(p0),
            "spreadf": pairwise_spread(pf),
            "layout": classify_layout(pf),
            "init_mode": hist[0].get("init_mode", "?"),
        })

    print(f"\n=== pairs_near_theory @ {root} ===")
    for sigma in sorted({r["sigma"] for r in rows}):
        sub = [r for r in rows if r["sigma"] == sigma]
        n22 = sum(1 for r in sub if r["layout"] == "2,2")
        ncol = sum(1 for r in sub if r["layout"] == "collapsed")
        other = len(sub) - n22 - ncol
        print(
            f"\n  sigma={sigma}  (2,2)={n22}/{len(sub)}  "
            f"collapsed={ncol}/{len(sub)}  other={other}"
        )
        print(
            f"    (2,2) seeds:     "
            f"{sorted(r['seed'] for r in sub if r['layout'] == '2,2')}"
        )
        print(
            f"    collapsed seeds: "
            f"{sorted(r['seed'] for r in sub if r['layout'] == 'collapsed')}"
        )
        print(f"  {'seed':>4} {'s0':>8} {'sf':>8} {'layout':>10}")
        for r in sorted(sub, key=lambda x: x["seed"]):
            print(
                f"  {r['seed']:4d} {r['spread0']:8.3f} "
                f"{r['spreadf']:8.3f} {r['layout']:>10}"
            )

    if baseline_root is not None and baseline_root.is_dir():
        print(f"\n=== baseline mean_noise @ {baseline_root} ===")
        from infl_ens.training.pool_dynamics import classify_layout as _classify
        by_sigma: dict[float, list[str]] = {}
        for cell in iter_sigma_seed_histories(baseline_root):
            names = sorted(cell["history"][-1]["positions"].keys())
            pf = np.stack([
                np.asarray(cell["history"][-1]["positions"][n]) for n in names
            ])
            by_sigma.setdefault(cell["sigma"], []).append(_classify(pf))
        for sigma in sorted(by_sigma):
            layouts = by_sigma[sigma]
            n22 = sum(1 for x in layouts if x == "2,2")
            print(
                f"  sigma={sigma}  (2,2)={n22}/{len(layouts)}  "
                f"collapsed={len(layouts) - n22}/{len(layouts)}"
            )

    out = root / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")
    return rows, out
