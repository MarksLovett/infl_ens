"""Roll per-cell final-round eval JSONs into one family x scale NLL table.

Consumes the ``eval_final/eval_results.json`` written by
:mod:`infl_ens.evaluation` for each cell of the model scale-family sweep
(see :mod:`scripts.run_model_sweep`) and produces:

* ``model_sweep_nll.csv`` -- long-form rows
  ``(cell, family, tier, benchmark, mean_nll_over_clones,
  min_nll_over_clones, n_seeds)``.
* ``model_sweep_nll.md`` -- a family x scale pivot (mean NLL over clones,
  averaged over seeds) per benchmark.
* Best-effort grouped bar figure per benchmark under ``--figures-dir`` via
  :func:`infl_ens.vis.benchmark_nll_bar.plot_benchmark_nll_comparison`
  (skipped silently if matplotlib is unavailable).

This lives in ``scripts/`` (AGENTS.md rule: one-off analysis), not the
package: it aggregates heterogeneous base models, which the seed-only
:mod:`infl_ens.evaluation.aggregate` helpers do not cover.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Optional, Sequence

# Cell key -> (family, tier). Kept in sync with scripts/run_model_sweep.sh.
FAMILY_ORDER = ("qwen", "llama", "gemma")
TIER_ORDER = ("1b", "3b", "8b")


def _parse_cell(cell: str) -> tuple[str, str]:
    """Split a ``<family>_<tier>`` cell key into its parts.

    :param cell: Cell directory name, e.g. ``llama_3b``.
    :type cell: str
    :returns: ``(family, tier)``.
    :rtype: tuple[str, str]
    """
    family, _, tier = cell.partition("_")
    return family, tier


def _load_cell_seed(path: Path) -> dict[str, list[float]]:
    """Load one ``eval_results.json`` into per-benchmark clone-NLL lists.

    :param path: Path to an ``eval_results.json`` file.
    :type path: pathlib.Path
    :returns: Mapping ``benchmark -> [mean_nll per clone]``.
    :rtype: dict[str, list[float]]
    """
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    by_bench: dict[str, list[float]] = {}
    for rec in payload.get("results", []):
        bench = str(rec["benchmark"])
        by_bench.setdefault(bench, []).append(float(rec["mean_nll"]))
    return by_bench


def collect_sweep(
    results_root: Path,
    seeds: Sequence[int],
    cells: Optional[Sequence[str]] = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate every cell's eval into per-benchmark statistics.

    For each cell and benchmark, the per-clone NLLs are reduced to a mean
    and a min (best specialist) within a seed, then averaged across seeds.

    :param results_root: Sweep root (``results/model_sweep``).
    :type results_root: pathlib.Path
    :param seeds: Seeds to include.
    :type seeds: Sequence[int]
    :param cells: Optional explicit cell list; defaults to every subdir of
        ``results_root`` that has at least one seed eval.
    :type cells: Sequence[str] | None
    :returns: ``stats[cell][benchmark] = {"mean": float, "min": float,
        "n_seeds": int}``.
    :rtype: dict[str, dict[str, dict[str, float]]]
    """
    if cells is None:
        cells = sorted(p.name for p in results_root.iterdir() if p.is_dir())

    stats: dict[str, dict[str, dict[str, float]]] = {}
    for cell in cells:
        per_bench_mean: dict[str, list[float]] = {}
        per_bench_min: dict[str, list[float]] = {}
        for seed in seeds:
            eval_json = results_root / cell / f"seed{seed}" / "eval_final" / "eval_results.json"
            if not eval_json.is_file():
                continue
            by_bench = _load_cell_seed(eval_json)
            for bench, clone_nlls in by_bench.items():
                if not clone_nlls:
                    continue
                per_bench_mean.setdefault(bench, []).append(fmean(clone_nlls))
                per_bench_min.setdefault(bench, []).append(min(clone_nlls))
        if not per_bench_mean:
            continue
        stats[cell] = {
            bench: {
                "mean": fmean(per_bench_mean[bench]),
                "min": fmean(per_bench_min[bench]),
                "n_seeds": float(len(per_bench_mean[bench])),
            }
            for bench in per_bench_mean
        }
    return stats


def _sorted_cells(cells: Sequence[str]) -> list[str]:
    """Order cells by family then tier, unknown keys last (alphabetical).

    :param cells: Cell keys.
    :type cells: Sequence[str]
    :returns: Ordered cell keys.
    :rtype: list[str]
    """
    def key(cell: str) -> tuple[int, int, str]:
        fam, tier = _parse_cell(cell)
        fi = FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else len(FAMILY_ORDER)
        ti = TIER_ORDER.index(tier) if tier in TIER_ORDER else len(TIER_ORDER)
        return (fi, ti, cell)

    return sorted(cells, key=key)


def write_csv(stats: dict[str, dict[str, dict[str, float]]], out_path: Path) -> None:
    """Write the long-form CSV.

    :param stats: Aggregated statistics from :func:`collect_sweep`.
    :type stats: dict
    :param out_path: Destination ``.csv`` path.
    :type out_path: pathlib.Path
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["cell", "family", "tier", "benchmark",
             "mean_nll_over_clones", "min_nll_over_clones", "n_seeds"]
        )
        for cell in _sorted_cells(list(stats)):
            family, tier = _parse_cell(cell)
            for bench in sorted(stats[cell]):
                s = stats[cell][bench]
                writer.writerow(
                    [cell, family, tier, bench,
                     f"{s['mean']:.6f}", f"{s['min']:.6f}", int(s['n_seeds'])]
                )


def write_markdown(stats: dict[str, dict[str, dict[str, float]]], out_path: Path) -> None:
    """Write a family x scale pivot table (mean NLL over clones) per benchmark.

    :param stats: Aggregated statistics from :func:`collect_sweep`.
    :type stats: dict
    :param out_path: Destination ``.md`` path.
    :type out_path: pathlib.Path
    """
    cells = _sorted_cells(list(stats))
    benches = sorted({b for cell in stats for b in stats[cell]})
    lines: list[str] = ["# Model scale-family sweep -- mean token NLL (lower is better)", ""]
    header = "| cell | family | tier | " + " | ".join(benches) + " |"
    sep = "|" + "---|" * (3 + len(benches))
    lines.extend([header, sep])
    for cell in cells:
        family, tier = _parse_cell(cell)
        row = [cell, family, tier]
        for bench in benches:
            s = stats[cell].get(bench)
            row.append(f"{s['mean']:.4f}" if s else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _maybe_plot(
    stats: dict[str, dict[str, dict[str, float]]],
    figures_dir: Path,
) -> None:
    """Best-effort grouped bar figure (models as series) per benchmark.

    Reuses :func:`infl_ens.vis.benchmark_nll_bar.plot_benchmark_nll_comparison`
    with each model cell treated as a bar series. Silently returns if
    matplotlib (or the vis module) is unavailable.

    :param stats: Aggregated statistics from :func:`collect_sweep`.
    :type stats: dict
    :param figures_dir: Output directory for the figure.
    :type figures_dir: pathlib.Path
    """
    try:
        from infl_ens.vis.benchmark_nll_bar import plot_benchmark_nll_comparison
    except Exception:  # pragma: no cover - optional matplotlib
        print("[summary] matplotlib/vis unavailable; skipping figure")
        return

    cells = _sorted_cells(list(stats))
    benches = sorted({b for cell in stats for b in stats[cell]})
    # adapter_nll[bench][cell] = mean NLL over clones.
    adapter_nll: dict[str, dict[str, float]] = {
        bench: {
            cell: stats[cell][bench]["mean"]
            for cell in cells if bench in stats[cell]
        }
        for bench in benches
    }
    labels = {b: b for b in benches}
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_benchmark_nll_comparison(
        benchmarks=benches,
        benchmark_labels=labels,
        base_nll={b: 0.0 for b in benches},
        adapter_nll=adapter_nll,
        agents=cells,
        include_base=False,
        title="Model scale-family sweep: mean token NLL",
        output_stem=str(figures_dir / "model_sweep_nll"),
    )
    print(f"[summary] wrote figure under {figures_dir}/model_sweep_nll.*")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    :param argv: Argument vector (defaults to ``sys.argv[1:]``).
    :type argv: Sequence[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/model_sweep"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--cells", type=str, nargs="*", default=None,
                        help="Explicit cell keys; default: every subdir with an eval.")
    parser.add_argument("--figures-dir", type=Path,
                        default=Path("scripts/figures/model_sweep"))
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    if not args.results_root.is_dir():
        print(f"error: results root {args.results_root} not found")
        return 2

    stats = collect_sweep(args.results_root, args.seeds, args.cells)
    if not stats:
        print(f"error: no eval_results.json found under {args.results_root}")
        return 1

    csv_path = args.results_root / "model_sweep_nll.csv"
    md_path = args.results_root / "model_sweep_nll.md"
    write_csv(stats, csv_path)
    write_markdown(stats, md_path)
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {md_path}")
    if not args.no_figure:
        _maybe_plot(stats, args.figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
