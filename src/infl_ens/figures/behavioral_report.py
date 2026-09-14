"""Pure table and plot builders for behavioral safety evaluations."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from typing import Any


def behavioral_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten a behavioral summary into one row per metric.

    :param summary: Schema-versioned behavioral summary.
    :type summary: Mapping[str, Any]
    :returns: Long-form rows with target, protocol, suite, metric and value.
    :rtype: list[dict[str, Any]]
    :raises ValueError: If the summary schema is unsupported.
    """
    if int(summary.get("schema_version", -1)) != 1:
        raise ValueError("behavioral report requires summary schema version 1")
    rows: list[dict[str, Any]] = []
    for item in summary.get("rows", []):
        result = item.get("aggregate", {})
        graders = ",".join(str(value) for value in result.get("graders", []))
        fidelity = ",".join(str(value) for value in result.get("fidelity", []))
        scopes = [("all", result.get("overall", {}))]
        scopes.extend(sorted((result.get("by_category") or {}).items()))
        for category, aggregate in scopes:
            counts = {
                "n": int(aggregate.get("n", 0)),
                "n_scored": int(aggregate.get("n_scored", 0)),
                "n_errors": int(aggregate.get("n_errors", 0)),
            }
            for metric, value in aggregate.items():
                if (
                    metric in counts
                    or metric.endswith("_n")
                    or not isinstance(value, (int, float))
                ):
                    continue
                rows.append({
                    "target": str(item["target"]),
                    "target_kind": str(item["target_kind"]),
                    "protocol": str(item["protocol"]),
                    "suite": str(item["suite"]),
                    "graders": graders,
                    "fidelity": fidelity,
                    "category": str(category),
                    "metric": str(metric),
                    "value": float(value),
                    **counts,
                })
    return rows


def behavioral_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render long-form behavioral rows as CSV.

    :param rows: Rows returned by :func:`behavioral_rows`.
    :type rows: Sequence[Mapping[str, Any]]
    :returns: CSV text.
    :rtype: str
    """
    fields = [
        "target", "target_kind", "protocol", "suite", "graders", "fidelity",
        "category", "metric", "value", "n", "n_scored", "n_errors",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({name: row.get(name) for name in fields} for row in rows)
    return stream.getvalue()


def behavioral_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render a readable long-form behavioral results table.

    :param rows: Rows returned by :func:`behavioral_rows`.
    :type rows: Sequence[Mapping[str, Any]]
    :returns: Markdown report.
    :rtype: str
    """
    lines = [
        "# Behavioral safety evaluation",
        "",
        "All values below are means over the scored cases. Grader fidelity and per-case "
        "errors remain recorded in each score artifact; proxy metrics must not be presented "
        "as publication-grade benchmark scores.",
        "",
        "| Target | Protocol | Suite | Fidelity | Category | Metric | Value | Scored | Errors |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target']} | {row['protocol']} | {row['suite']} | {row['fidelity']} "
            f"| {row['category']} "
            f"| {row['metric']} "
            f"| {float(row['value']):.4f} | {row['n_scored']} | {row['n_errors']} |",
        )
    return "\n".join(lines) + "\n"


def plot_safety_utility(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Plot harmful-compliance safety against benign-prompt utility.

    The plot uses the first available harmful-compliance metric from HarmBench
    or StrongREJECT and XSTest calibrated behavior as the utility coordinate.

    :param rows: Rows returned by :func:`behavioral_rows`.
    :type rows: Sequence[Mapping[str, Any]]
    :returns: Matplotlib figure.
    :rtype: matplotlib.figure.Figure
    :raises ValueError: If no target has both coordinates.
    """
    import matplotlib.pyplot as plt

    by_key: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    for row in rows:
        if str(row.get("category", "all")) != "all":
            continue
        key = (str(row["target"]), str(row["protocol"]))
        by_key.setdefault(key, {})[(str(row["suite"]), str(row["metric"]))] = float(
            row["value"],
        )
    points: list[tuple[str, str, float, float]] = []
    harm_metrics = (
        ("harmbench", "harmful_compliance_proxy"),
        ("strongreject", "harmful_compliance_proxy"),
        ("harmbench", "score"),
        ("strongreject", "score"),
    )
    for (target, protocol), metrics in by_key.items():
        harm = next((metrics[key] for key in harm_metrics if key in metrics), None)
        utility = metrics.get(("xstest", "calibrated_behavior"))
        if harm is not None and utility is not None:
            points.append((target, protocol, harm, utility))
    if not points:
        raise ValueError("safety-utility plot needs a harm metric and XSTest calibrated behavior")
    fig, axis = plt.subplots(figsize=(7.0, 5.0))
    for target, protocol, harm, utility in points:
        axis.scatter(harm, utility, s=45)
        axis.annotate(
            f"{target} ({protocol})",
            (harm, utility),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Harmful compliance (lower is better)")
    axis.set_ylabel("XSTest calibrated behavior (higher is better)")
    axis.set_title("Safety-utility trade-off")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    return fig


__all__ = [
    "behavioral_csv",
    "behavioral_markdown",
    "behavioral_rows",
    "plot_safety_utility",
]
