#!/usr/bin/env python3
"""Per-round held-out NLL for each merge pair, from a multi-round eval report.

Consumes the ``eval_results.json`` written by the unified evaluation path
(:func:`infl_ens.evaluation.evaluate.run_unified_eval`) when it is given an
explicit ``eval.rounds`` list, e.g.::

    python -m infl_ens.evaluation --config <arm>.yaml -- \\
        eval.partitions='["val"]' eval.rounds='[4,5,6,7,8,9,10,11]'

Every row of that report already carries ``agent``, ``round``, ``benchmark``
and ``mean_nll``, so the round x pair table is a pivot. Also accepts the
per-round layout written during training by
:func:`infl_ens.training.closed_loop_eval.run_closed_loop_val_eval`
(``<run>/eval_val/round-NN/eval_results.json``) -- pass ``--eval-dir`` and both
shapes are discovered.

Writes ``<stem>.csv``, ``.md`` and ``.tex`` (a booktabs table ready to
``\\input``), plus a ``.json`` with the raw pivot.

Example::

    python scripts/build_per_round_pair_nll_table.py \\
        --eval-dir results/seven_axis_soft_topk3_pairs/seed0/eval_val \\
        --output-stem results/seven_axis_soft_topk3_pairs/seed0/tables/pair_nll_by_round \\
        --label "Soft top-3 pairs (val)" --first-round 4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Display order / labels for the seven safety axes.
BENCHMARK_LABELS: dict[str, str] = {
    "beavertails": "Harm",
    "halueval": "Hallucination",
    "jbb_behaviors": "Jailbreak",
    "ai4privacy": "Privacy",
    "orbench": "Over-refusal",
    "prompt_injection": "Injection",
    "do_not_answer": "Policy",
}


def _load_rows(eval_dir: Path) -> list[dict[str, Any]]:
    """Collect result rows from either eval-report layout.

    :param eval_dir: Directory holding ``eval_results.json`` and/or
        ``round-NN/eval_results.json``.
    :type eval_dir: pathlib.Path
    :returns: Flat list of result records.
    :rtype: list[dict]
    :raises FileNotFoundError: If no report is found.
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


def _pivot(
    rows: list[dict[str, Any]],
    *,
    first_round: int,
    benchmark: Optional[str],
    only_rounds: Optional[list[int]] = None,
) -> tuple[list[int], list[str], dict[int, dict[str, float]]]:
    """Average NLL per (round, agent), macro over benchmarks.

    :param rows: Flat result records.
    :type rows: list[dict]
    :param first_round: Drop rounds before this index.
    :type first_round: int
    :param benchmark: Restrict to one benchmark; ``None`` macro-averages
        over all benchmarks (equal weight per axis).
    :type benchmark: str | None
    :param only_rounds: Keep exactly these rounds. Lets a report that
        happens to hold a wider sweep be reduced to the two rounds an
        early-vs-late comparison needs, without re-scoring anything.
    :type only_rounds: list[int] | None
    :returns: ``(rounds, agents, table[round][agent])``.
    :rtype: tuple
    :raises ValueError: If the filters leave no rows, or a requested round
        is absent from the report.
    """
    if only_rounds is not None:
        present = {r["round"] for r in rows}
        missing = [r for r in only_rounds if r not in present]
        if missing:
            raise ValueError(
                f"requested rounds {missing} are not in the report "
                f"(present: {sorted(present)})"
            )
    keep = [
        r for r in rows
        if r["round"] >= first_round
        and (only_rounds is None or r["round"] in only_rounds)
        and (benchmark is None or r["benchmark"] == benchmark)
    ]
    if not keep:
        raise ValueError(
            f"no rows left after filtering (first_round={first_round}, "
            f"rounds={only_rounds}, benchmark={benchmark!r})"
        )
    rounds = sorted({r["round"] for r in keep})
    agents = sorted({r["agent"] for r in keep})
    table: dict[int, dict[str, float]] = {}
    for rnd in rounds:
        per_agent: dict[str, float] = {}
        for agent in agents:
            vals = [
                r["mean_nll"] for r in keep
                if r["round"] == rnd and r["agent"] == agent
            ]
            if vals:
                per_agent[agent] = sum(vals) / len(vals)
        table[rnd] = per_agent
    return rounds, agents, table


def _fmt(value: Optional[float], places: int = 4) -> str:
    """Fixed-width number or an em dash when missing."""
    return "--" if value is None else f"{value:.{places}f}"


def write_outputs(
    rounds: list[int],
    agents: list[str],
    table: dict[int, dict[str, float]],
    output_stem: Path,
    *,
    label: str,
    benchmark: Optional[str],
) -> dict[str, Path]:
    """Write the pivot as csv / md / tex / json.

    The final block of each table reports the change from the first
    reported round to the last -- the "NLL update" over the window.

    :returns: Mapping of format to written path.
    :rtype: dict[str, pathlib.Path]
    """
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    first, last = rounds[0], rounds[-1]
    delta = {
        a: (
            table[last][a] - table[first][a]
            if a in table[last] and a in table[first] else None
        )
        for a in agents
    }
    scope = (
        BENCHMARK_LABELS.get(benchmark, benchmark)
        if benchmark else "macro-mean over benchmarks"
    )
    written: dict[str, Path] = {}

    csv_path = output_stem.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["round"] + agents)
        for rnd in rounds:
            writer.writerow(
                [rnd] + [_fmt(table[rnd].get(a)) for a in agents]
            )
        writer.writerow([f"delta_{first}_to_{last}"] + [_fmt(delta[a]) for a in agents])
    written["csv"] = csv_path

    md_path = output_stem.with_suffix(".md")
    lines = [
        f"# Per-round held-out NLL by pair — {label}",
        "",
        f"Scope: {scope}. Rounds {first}–{last}. Lower is better.",
        "",
        "| Round | " + " | ".join(agents) + " |",
        "|---" * (len(agents) + 1) + "|",
    ]
    for rnd in rounds:
        lines.append(
            f"| {rnd} | " + " | ".join(_fmt(table[rnd].get(a)) for a in agents) + " |"
        )
    lines.append(
        f"| **Δ {first}→{last}** | "
        + " | ".join(f"**{_fmt(delta[a])}**" for a in agents) + " |"
    )
    lines.append("")
    improved = sum(1 for a in agents if delta[a] is not None and delta[a] < 0)
    lines.append(
        f"{improved}/{len(agents)} pairs improved (NLL decreased) "
        f"from round {first} to round {last}."
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written["md"] = md_path

    tex_path = output_stem.with_suffix(".tex")
    col_spec = "l" + "r" * len(agents)
    tex = [
        "% Generated by scripts/build_per_round_pair_nll_table.py",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        "    Round & " + " & ".join(a.replace("_", "\\_") for a in agents) + " \\\\",
        "    \\midrule",
    ]
    for rnd in rounds:
        tex.append(
            f"    {rnd} & "
            + " & ".join(_fmt(table[rnd].get(a)) for a in agents) + " \\\\"
        )
    tex.append("    \\midrule")
    tex.append(
        f"    $\\Delta$ {first}$\\to${last} & "
        + " & ".join(_fmt(delta[a]) for a in agents) + " \\\\"
    )
    tex += [
        "    \\bottomrule",
        "  \\end{tabular}",
        f"  \\caption{{Per-round held-out mean token NLL by merge pair — {label}. "
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


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir", required=True,
        help="Directory with eval_results.json (or round-NN/eval_results.json).",
    )
    parser.add_argument(
        "--output-stem", required=True,
        help="Output path without extension; .csv/.md/.tex/.json are written.",
    )
    parser.add_argument("--label", default="run", help="Caption label.")
    parser.add_argument(
        "--first-round", type=int, default=4,
        help="First round to report (default: 4).",
    )
    parser.add_argument(
        "--benchmark", default=None,
        help="Restrict to one benchmark; default macro-averages over all.",
    )
    parser.add_argument(
        "--rounds", default=None,
        help="Comma-separated rounds to keep, e.g. '4,11' for an "
             "early-vs-late comparison. Default: every round present.",
    )
    args = parser.parse_args()
    only_rounds = (
        [int(x) for x in args.rounds.split(",") if x.strip()]
        if args.rounds else None
    )

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_absolute():
        eval_dir = ROOT / eval_dir
    stem = Path(args.output_stem)
    if not stem.is_absolute():
        stem = ROOT / stem

    rows = _load_rows(eval_dir)
    rounds, agents, table = _pivot(
        rows, first_round=args.first_round, benchmark=args.benchmark,
        only_rounds=only_rounds,
    )
    written = write_outputs(
        rounds, agents, table, stem,
        label=args.label, benchmark=args.benchmark,
    )
    print(json.dumps({k: str(v) for k, v in written.items()}, indent=2))
    print(f"rounds {rounds[0]}..{rounds[-1]} x {len(agents)} agents")


if __name__ == "__main__":
    main()
