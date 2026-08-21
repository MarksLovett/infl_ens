"""Aggregate benchmark-eval metrics across training seeds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]


@dataclass(frozen=True)
class AggregatedEvalMetric:
    """Mean NLL for one (agent, benchmark) cell averaged over seeds.

    :param agent: Agent name (e.g. ``clone-0``).
    :type agent: str
    :param benchmark: Benchmark id (e.g. ``beavertails``).
    :type benchmark: str
    :param axis_name: Trait axis (e.g. ``harm``).
    :type axis_name: str
    :param mean_nll: Mean of per-seed ``mean_nll`` values.
    :type mean_nll: float
    :param std_nll: Sample std of per-seed ``mean_nll`` values.
    :type std_nll: float
    :param n_seeds: Number of seeds contributing.
    :type n_seeds: int
    :param per_seed: Mapping ``seed -> mean_nll`` for audit.
    :type per_seed: dict[int, float]
    """

    agent: str
    benchmark: str
    axis_name: str
    mean_nll: float
    std_nll: float
    n_seeds: int
    per_seed: dict[int, float]


def load_eval_report(path: PathLike) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load one ``eval_results.json`` file.

    :param path: Path to the JSON report.
    :type path: str | pathlib.Path
    :returns: Tuple ``(meta, results)``.
    :rtype: tuple[dict, list[dict]]
    :raises FileNotFoundError: If ``path`` is missing.
    :raises ValueError: If the file has no ``results`` list.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{p} has no 'results' list")
    return dict(payload.get("meta", {})), results


def aggregate_eval_across_seeds(
    reports_by_seed: dict[int, PathLike],
    *,
    agents: Optional[Sequence[str]] = None,
    benchmarks: Optional[Sequence[str]] = None,
) -> list[AggregatedEvalMetric]:
    """Average ``mean_nll`` across seeds for each (agent, benchmark) pair.

    :param reports_by_seed: Mapping ``seed -> path`` to ``eval_results.json``.
    :type reports_by_seed: dict[int, str | pathlib.Path]
    :param agents: Optional agent filter.
    :type agents: Sequence[str] | None
    :param benchmarks: Optional benchmark-name filter.
    :type benchmarks: Sequence[str] | None
    :returns: One aggregated record per (agent, benchmark) present in
        every loaded seed (partial seeds are skipped for that cell).
    :rtype: list[AggregatedEvalMetric]
    :raises ValueError: If ``reports_by_seed`` is empty.
    """
    if not reports_by_seed:
        raise ValueError("reports_by_seed must be non-empty")

    want_agents = set(agents) if agents is not None else None
    want_bench = set(benchmarks) if benchmarks is not None else None

    # cell_key -> seed -> mean_nll
    cells: dict[tuple[str, str, str], dict[int, float]] = {}
    axis_by_cell: dict[tuple[str, str], str] = {}

    for seed, path in sorted(reports_by_seed.items()):
        _, rows = load_eval_report(path)
        for row in rows:
            agent = str(row["agent"])
            bench = str(row["benchmark"])
            if want_agents is not None and agent not in want_agents:
                continue
            if want_bench is not None and bench not in want_bench:
                continue
            key = (agent, bench)
            axis_by_cell[key] = str(row.get("axis_name", ""))
            cell = (agent, bench, axis_by_cell[key])
            cells.setdefault(cell, {})[int(seed)] = float(row["mean_nll"])

    out: list[AggregatedEvalMetric] = []
    for (agent, bench, axis), by_seed in sorted(cells.items()):
        if not by_seed:
            continue
        vals = np.asarray(list(by_seed.values()), dtype=float)
        out.append(
            AggregatedEvalMetric(
                agent=agent,
                benchmark=bench,
                axis_name=axis,
                mean_nll=float(vals.mean()),
                std_nll=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                n_seeds=len(vals),
                per_seed=dict(by_seed),
            )
        )
    return out


def discover_seed_eval_reports(
    root: PathLike,
    *,
    pattern: str = "eval_results.json",
) -> dict[int, Path]:
    """Find per-seed eval reports under ``root/seed*/``.

    Expects layout ``<root>/seed<N>/<pattern>`` or
    ``<root>/seed<N>_final/<pattern>`` is **not** supported — pass a list
    of explicit paths instead. Also accepts flat siblings named
    ``eval_pairs_near_eq_r40_seed0_final`` via ``glob``.

    :param root: Parent directory containing ``seed*`` subdirs with reports.
    :type root: str | pathlib.Path
    :param pattern: Report filename.
    :type pattern: str
    :returns: ``seed -> report path`` for every match found.
    :rtype: dict[int, pathlib.Path]
    """
    root = Path(root)
    found: dict[int, Path] = {}
    if not root.is_dir():
        return found
    import re
    seed_re = re.compile(r"^seed(\d+)$", re.IGNORECASE)
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = seed_re.match(child.name)
        if not m:
            continue
        report = child / pattern
        if report.is_file():
            found[int(m.group(1))] = report
    return found


def write_aggregated_eval_report(
    metrics: Sequence[AggregatedEvalMetric],
    output_dir: PathLike,
    *,
    meta: Optional[dict[str, Any]] = None,
) -> Path:
    """Write ``eval_aggregate.json`` with mean ± std over seeds.

    :param metrics: Aggregated cells.
    :type metrics: Sequence[AggregatedEvalMetric]
    :param output_dir: Output directory.
    :type output_dir: str | pathlib.Path
    :param meta: Optional metadata block.
    :type meta: dict | None
    :returns: Path to the written JSON file.
    :rtype: pathlib.Path
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
        "aggregated": [
            {
                "agent": m.agent,
                "benchmark": m.benchmark,
                "axis_name": m.axis_name,
                "mean_nll": m.mean_nll,
                "std_nll": m.std_nll,
                "n_seeds": m.n_seeds,
                "per_seed": {str(k): v for k, v in sorted(m.per_seed.items())},
            }
            for m in metrics
        ],
    }
    path = out / "eval_aggregate.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


@dataclass(frozen=True)
class EvalMatrix:
    """Mean NLL matrix with agents as rows and benchmarks as columns.

    :param row_labels: Agent names (rows).
    :type row_labels: tuple[str, ...]
    :param col_labels: Benchmark names (columns).
    :type col_labels: tuple[str, ...]
    :param means: Mean NLL array, shape ``(n_rows, n_cols)``.
    :type means: numpy.ndarray
    :param stds: Per-cell std over seeds, same shape as ``means``.
    :type stds: numpy.ndarray
    :param axis_by_column: Benchmark id → axis name (e.g. ``harm``).
    :type axis_by_column: dict[str, str]
    """

    row_labels: tuple[str, ...]
    col_labels: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    axis_by_column: dict[str, str]


def load_aggregated_report(path: PathLike) -> tuple[dict[str, Any], list[AggregatedEvalMetric]]:
    """Load ``eval_aggregate.json`` into :class:`AggregatedEvalMetric` records.

    :param path: Path to an aggregate JSON file.
    :type path: str | pathlib.Path
    :returns: Tuple ``(meta, metrics)``.
    :rtype: tuple[dict, list[AggregatedEvalMetric]]
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = payload.get("aggregated", [])
    metrics: list[AggregatedEvalMetric] = []
    for row in rows:
        per_seed = {int(k): float(v) for k, v in row.get("per_seed", {}).items()}
        metrics.append(
            AggregatedEvalMetric(
                agent=str(row["agent"]),
                benchmark=str(row["benchmark"]),
                axis_name=str(row.get("axis_name", "")),
                mean_nll=float(row["mean_nll"]),
                std_nll=float(row.get("std_nll", 0.0)),
                n_seeds=int(row.get("n_seeds", len(per_seed))),
                per_seed=per_seed,
            )
        )
    return dict(payload.get("meta", {})), metrics


def build_eval_matrix(
    metrics: Sequence[AggregatedEvalMetric],
    *,
    row_order: Optional[Sequence[str]] = None,
    col_order: Optional[Sequence[str]] = None,
) -> EvalMatrix:
    """Build an agent × benchmark mean-NLL matrix from aggregated metrics.

    :param metrics: Aggregated per-(agent, benchmark) records.
    :type metrics: Sequence[AggregatedEvalMetric]
    :param row_order: Optional explicit agent ordering.
    :type row_order: Sequence[str] | None
    :param col_order: Optional explicit benchmark ordering.
    :type col_order: Sequence[str] | None
    :returns: Populated matrix container.
    :rtype: EvalMatrix
    :raises ValueError: If ``metrics`` is empty.
    """
    if not metrics:
        raise ValueError("metrics must be non-empty")

    agents = list(row_order) if row_order is not None else sorted({m.agent for m in metrics})
    benches = list(col_order) if col_order is not None else sorted({m.benchmark for m in metrics})
    lookup = {(m.agent, m.benchmark): m for m in metrics}
    axis_by_column = {m.benchmark: m.axis_name for m in metrics}

    means = np.full((len(agents), len(benches)), np.nan, dtype=float)
    stds = np.full((len(agents), len(benches)), np.nan, dtype=float)
    for i, agent in enumerate(agents):
        for j, bench in enumerate(benches):
            cell = lookup.get((agent, bench))
            if cell is None:
                continue
            means[i, j] = cell.mean_nll
            stds[i, j] = cell.std_nll

    return EvalMatrix(
        row_labels=tuple(agents),
        col_labels=tuple(benches),
        means=means,
        stds=stds,
        axis_by_column=axis_by_column,
    )


def format_eval_matrix_csv(
    matrix: EvalMatrix,
    *,
    include_std: bool = True,
    float_fmt: str = "{:.4f}",
) -> str:
    """Format the matrix as CSV (mean only, or ``mean±std`` strings).

    :param matrix: Matrix to format.
    :type matrix: EvalMatrix
    :param include_std: If ``True``, each cell is ``mean±std``; else mean only.
    :type include_std: bool
    :param float_fmt: Format string for floating-point values.
    :type float_fmt: str
    :returns: CSV text with a header row and agent column.
    :rtype: str
    """
    header = ["agent"] + list(matrix.col_labels)
    lines = [",".join(header)]
    for i, agent in enumerate(matrix.row_labels):
        cells = [agent]
        for j in range(len(matrix.col_labels)):
            m, s = matrix.means[i, j], matrix.stds[i, j]
            if np.isnan(m):
                cells.append("")
            elif include_std:
                cells.append(f"{float_fmt.format(m)}±{float_fmt.format(s)}")
            else:
                cells.append(float_fmt.format(m))
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def format_eval_matrix_markdown(
    matrix: EvalMatrix,
    *,
    include_std: bool = True,
    float_fmt: str = "{:.4f}",
) -> str:
    """Format the matrix as a GitHub-flavoured Markdown table.

    :param matrix: Matrix to format.
    :type matrix: EvalMatrix
    :param include_std: If ``True``, cells show ``mean ± std``.
    :type include_std: bool
    :param float_fmt: Float format for cell values.
    :type float_fmt: str
    :returns: Markdown table text.
    :rtype: str
    """
    cols = list(matrix.col_labels)
    header = "| agent | " + " | ".join(cols) + " |"
    sep = "|---|" + "|".join(["---:" for _ in cols]) + "|"
    lines = [header, sep]
    for i, agent in enumerate(matrix.row_labels):
        row_cells = [agent]
        for j in range(len(cols)):
            m, s = matrix.means[i, j], matrix.stds[i, j]
            if np.isnan(m):
                row_cells.append("—")
            elif include_std:
                row_cells.append(f"{float_fmt.format(m)} ± {float_fmt.format(s)}")
            else:
                row_cells.append(float_fmt.format(m))
        lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(lines) + "\n"


def format_eval_matrix_latex(
    matrix: EvalMatrix,
    *,
    include_std: bool = True,
    float_fmt: str = "{:.4f}",
    caption: str = "Mean token NLL (lower is better)",
    label: str = "tab:eval_nll",
) -> str:
    """Format the matrix as a LaTeX ``tabular`` wrapped in ``table``.

    :param matrix: Matrix to format.
    :type matrix: EvalMatrix
    :param include_std: If ``True``, cells use ``mean $\\pm$ std``.
    :type include_std: bool
    :param float_fmt: Float format for values.
    :type float_fmt: str
    :param caption: Table caption.
    :type caption: str
    :param label: LaTeX label.
    :type label: str
    :returns: LaTeX source.
    :rtype: str
    """
    ncol = len(matrix.col_labels)
    col_spec = "l" + "r" * ncol
    header_cols = " & ".join(
        f"\\textbf{{{b}}}" for b in matrix.col_labels
    )
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"Agent & {header_cols} \\\\",
        "\\midrule",
    ]
    for i, agent in enumerate(matrix.row_labels):
        cells = [agent.replace("-", "\\text{-}")]
        for j in range(ncol):
            m, s = matrix.means[i, j], matrix.stds[i, j]
            if np.isnan(m):
                cells.append("---")
            elif include_std:
                cells.append(
                    f"{float_fmt.format(m)} $\\pm$ {float_fmt.format(s)}"
                )
            else:
                cells.append(float_fmt.format(m))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines) + "\n"


def matrix_to_json_dict(matrix: EvalMatrix) -> dict[str, Any]:
    """Serialize a matrix for programmatic use (e.g. plotting scripts).

    :param matrix: Matrix to serialize.
    :type matrix: EvalMatrix
    :returns: JSON-serializable dict with ``rows``, ``columns``, ``means``,
        ``stds``, and ``axis_by_column``.
    :rtype: dict
    """
    return {
        "rows": list(matrix.row_labels),
        "columns": list(matrix.col_labels),
        "axis_by_column": dict(matrix.axis_by_column),
        "means": matrix.means.tolist(),
        "stds": matrix.stds.tolist(),
    }


def write_eval_matrix_outputs(
    matrix: EvalMatrix,
    output_dir: PathLike,
    *,
    stem: str = "eval_matrix",
    include_std: bool = True,
) -> dict[str, Path]:
    """Write CSV, Markdown, LaTeX, and JSON matrix files.

    :param matrix: Matrix to export.
    :type matrix: EvalMatrix
    :param output_dir: Destination directory.
    :type output_dir: str | pathlib.Path
    :param stem: Filename stem (extensions added per format).
    :type stem: str
    :param include_std: Whether formatted cells include ± std.
    :type include_std: bool
    :returns: Mapping format name → written path.
    :rtype: dict[str, pathlib.Path]
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    csv_path = out / f"{stem}.csv"
    csv_path.write_text(
        format_eval_matrix_csv(matrix, include_std=include_std),
        encoding="utf-8",
    )
    paths["csv"] = csv_path

    md_path = out / f"{stem}.md"
    md_path.write_text(
        format_eval_matrix_markdown(matrix, include_std=include_std),
        encoding="utf-8",
    )
    paths["markdown"] = md_path

    tex_path = out / f"{stem}.tex"
    tex_path.write_text(
        format_eval_matrix_latex(matrix, include_std=include_std),
        encoding="utf-8",
    )
    paths["latex"] = tex_path

    json_path = out / f"{stem}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(matrix_to_json_dict(matrix), fh, indent=2)
    paths["json"] = json_path

    return paths
