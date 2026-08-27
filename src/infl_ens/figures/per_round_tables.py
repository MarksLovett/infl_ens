"""Held-out NLL per merge pair at selected rounds, as csv / md / tex / json.

Consumes the ``eval_results.json`` written by
:func:`infl_ens.evaluation.evaluate.run_unified_eval` for an explicit
``eval.rounds`` list (every row carries ``agent``, ``round``, ``benchmark``
and ``mean_nll``, so the round x pair table is a pivot).  Also accepts the
per-round layout written during training by
:func:`infl_ens.training.closed_loop_eval.run_closed_loop_val_eval`
(``<run>/eval_val/round-NN/eval_results.json``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from infl_ens.figures.style import BENCHMARK_LABELS


def load_eval_rows(eval_dir: Path) -> list[dict[str, Any]]:
    """Collect result rows from either eval-report layout.

    :param eval_dir: Directory holding ``eval_results.json`` and/or
        ``round-NN/eval_results.json``.
    :type eval_dir: pathlib.Path
    :returns: Flat list of result records.
    :rtype: list[dict]
    :raises FileNotFoundError: If no report is found.
    :raises ValueError: If no per-agent row carries a round index.
    """
    reports: list[Path] = []
    flat = eval_dir / "eval_results.json"
    if flat.is_file():
        reports.append(flat)
    reports.extend(sorted(eval_dir.glob("round-*/eval_results.json")))
    if not reports:
        raise FileNotFoundError(
            f"no eval_results.json under {eval_dir} "
            "(expected <dir>/eval_results.json or <dir>/round-NN/eval_results.json)"
        )

    rows: list[dict[str, Any]] = []
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta_round = (payload.get("meta") or {}).get("round")
        for rec in payload.get("results", []):
            rnd = rec.get("round", meta_round)
            if rnd is None or rec.get("agent") is None:
                continue
            rows.append({
                "round": int(rnd),
                "agent": str(rec["agent"]),
                "benchmark": str(rec["benchmark"]),
                "mean_nll": float(rec["mean_nll"]),
                "n_tokens": int(rec.get("n_tokens", 0)),
            })
    if not rows:
        raise ValueError(f"no per-agent rows with a round index under {eval_dir}")
    return rows


def eval_rows_cover(eval_dir: Path, rounds: Sequence[int]) -> bool:
    """Whether a report under ``eval_dir`` already holds every requested round.

    :param eval_dir: Evaluation directory of a run.
    :type eval_dir: pathlib.Path
    :param rounds: Rounds that must be present.
    :type rounds: Sequence[int]
    :returns: ``True`` when re-scoring can be skipped.
    :rtype: bool
    """
    try:
        present = {r["round"] for r in load_eval_rows(eval_dir)}
    except (FileNotFoundError, ValueError):
        return False
    return set(int(r) for r in rounds) <= present


def pivot_per_round(
    rows: Sequence[dict[str, Any]],
    *,
    rounds: Optional[Sequence[int]] = None,
    first_round: int = 0,
    benchmark: Optional[str] = None,
) -> tuple[list[int], list[str], dict[int, dict[str, float]]]:
    """Average NLL per (round, agent), macro over benchmarks.

    :param rows: Flat result records from :func:`load_eval_rows`.
    :type rows: Sequence[dict]
    :param rounds: Keep exactly these rounds (``None`` keeps every round
        at or after ``first_round``).
    :type rounds: Sequence[int] | None
    :param first_round: Drop rounds before this index.
    :type first_round: int
    :param benchmark: Restrict to one benchmark; ``None`` macro-averages
        over all benchmarks (equal weight per axis).
    :type benchmark: str | None
    :returns: ``(rounds, agents, table[round][agent])``.
    :rtype: tuple
    :raises ValueError: If the filters leave no rows, or a requested round
        is absent from the report.
    """
    only = None if rounds is None else {int(r) for r in rounds}
    if only is not None:
        present = {r["round"] for r in rows}
        missing = sorted(only - present)
        if missing:
            raise ValueError(
                f"requested rounds {missing} are not in the report (present: {sorted(present)})"
            )
    keep = [
        r for r in rows
        if r["round"] >= first_round
        and (only is None or r["round"] in only)
        and (benchmark is None or r["benchmark"] == benchmark)
    ]
    if not keep:
        raise ValueError(
            f"no rows left after filtering (first_round={first_round}, "
            f"rounds={rounds}, benchmark={benchmark!r})"
        )
    kept_rounds = sorted({r["round"] for r in keep})
    agents = sorted({r["agent"] for r in keep})
    table: dict[int, dict[str, float]] = {}
    for rnd in kept_rounds:
        per_agent: dict[str, float] = {}
        for agent in agents:
            vals = [r["mean_nll"] for r in keep if r["round"] == rnd and r["agent"] == agent]
            if vals:
                per_agent[agent] = sum(vals) / len(vals)
        table[rnd] = per_agent
    return kept_rounds, agents, table


def _fmt(value: Optional[float], places: int = 4) -> str:
    return "--" if value is None else f"{value:.{places}f}"


def write_per_round_outputs(
    rounds: Sequence[int],
    agents: Sequence[str],
    table: dict[int, dict[str, float]],
    output_stem: Path,
    *,
    label: str,
    benchmark: Optional[str] = None,
) -> dict[str, Path]:
    """Write the pivot as csv / md / tex / json.

    The final block of each table reports the change from the first
    reported round to the last: the NLL movement over the window.

    :param rounds: Rounds in the table.
    :type rounds: Sequence[int]
    :param agents: Agent (pair) names.
    :type agents: Sequence[str]
    :param table: ``table[round][agent]`` mean NLL.
    :type table: dict[int, dict[str, float]]
    :param output_stem: Path without extension; ``.csv/.md/.tex/.json`` are written.
    :type output_stem: pathlib.Path
    :param label: Caption label.
    :type label: str
    :param benchmark: Scope note (``None`` = macro-mean over benchmarks).
    :type benchmark: str | None
    :returns: Mapping of format to written path.
    :rtype: dict[str, pathlib.Path]
    """
    rounds = list(rounds)
    agents = list(agents)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    first, last = rounds[0], rounds[-1]
    delta = {
        a: (table[last][a] - table[first][a] if a in table[last] and a in table[first] else None)
        for a in agents
    }
    scope = BENCHMARK_LABELS.get(benchmark, benchmark) if benchmark else "macro-mean over benchmarks"
    written: dict[str, Path] = {}

    csv_path = output_stem.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["round", *agents])
        for rnd in rounds:
            writer.writerow([rnd] + [_fmt(table[rnd].get(a)) for a in agents])
        writer.writerow([f"delta_{first}_to_{last}"] + [_fmt(delta[a]) for a in agents])
    written["csv"] = csv_path

    md_path = output_stem.with_suffix(".md")
    lines = [
        f"# Held-out NLL by pair — {label}",
        "",
        f"Scope: {scope}. Rounds {first}–{last}. Lower is better.",
        "",
        "| Round | " + " | ".join(agents) + " |",
        "|---" * (len(agents) + 1) + "|",
    ]
    for rnd in rounds:
        lines.append(f"| {rnd} | " + " | ".join(_fmt(table[rnd].get(a)) for a in agents) + " |")
    lines.append(
        f"| **Δ {first}→{last}** | " + " | ".join(f"**{_fmt(delta[a])}**" for a in agents) + " |"
    )
    lines.append("")
    improved = sum(1 for a in agents if delta[a] is not None and delta[a] < 0)
    lines.append(
        f"{improved}/{len(agents)} pairs improved (NLL decreased) from round {first} to round {last}."
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written["md"] = md_path

    tex_path = output_stem.with_suffix(".tex")
    col_spec = "l" + "r" * len(agents)
    tex = [
        "% Generated by infl_ens.figures.per_round_tables",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        "    Round & " + " & ".join(a.replace("_", "\\_") for a in agents) + " \\\\",
        "    \\midrule",
    ]
    for rnd in rounds:
        tex.append(f"    {rnd} & " + " & ".join(_fmt(table[rnd].get(a)) for a in agents) + " \\\\")
    tex.append("    \\midrule")
    tex.append(
        f"    $\\Delta$ {first}$\\to${last} & " + " & ".join(_fmt(delta[a]) for a in agents) + " \\\\"
    )
    tex += [
        "    \\bottomrule",
        "  \\end{tabular}",
        f"  \\caption{{Held-out mean token NLL by merge pair — {label}. "
        f"Scope: {scope}. Lower is better; the last row is the change over "
        f"rounds {first}--{last}.}}",
        "  \\label{tab:pair-nll-by-round}",
        "\\end{table}",
    ]
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    written["tex"] = tex_path

    json_path = output_stem.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "label": label,
                "scope": scope,
                "benchmark": benchmark,
                "rounds": rounds,
                "agents": agents,
                "nll": {str(r): table[r] for r in rounds},
                "delta_first_to_last": delta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    written["json"] = json_path
    return written


def build_per_round_tables(
    eval_dir: Path,
    output_stem: Path,
    *,
    label: str,
    rounds: Optional[Sequence[int]] = None,
    first_round: int = 0,
    benchmark: Optional[str] = None,
) -> dict[str, Path]:
    """Load, pivot and write in one call.

    :param eval_dir: Evaluation directory of a run.
    :type eval_dir: pathlib.Path
    :param output_stem: Output path without extension.
    :type output_stem: pathlib.Path
    :param label: Caption label.
    :type label: str
    :param rounds: Rounds to report (see :func:`pivot_per_round`).
    :type rounds: Sequence[int] | None
    :param first_round: Drop earlier rounds when ``rounds`` is ``None``.
    :type first_round: int
    :param benchmark: Optional single-benchmark scope.
    :type benchmark: str | None
    :returns: Mapping of format to written path.
    :rtype: dict[str, pathlib.Path]
    """
    rows = load_eval_rows(eval_dir)
    kept, agents, table = pivot_per_round(
        rows, rounds=rounds, first_round=first_round, benchmark=benchmark,
    )
    return write_per_round_outputs(kept, agents, table, output_stem, label=label, benchmark=benchmark)


__all__ = [
    "build_per_round_tables",
    "eval_rows_cover",
    "load_eval_rows",
    "pivot_per_round",
    "write_per_round_outputs",
]
