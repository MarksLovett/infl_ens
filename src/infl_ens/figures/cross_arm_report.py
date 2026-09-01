"""Cross-arm analysis: what the per-arm figures cannot show.

1. **Data matching.** Are the specialist arms trained on the same round
   batches?  If yes, one pooled generalist is a fair comparator for all of
   them.  Checked by comparing each round's prompt SET (soft rounds are
   logged verbatim as ``batch_prompts``; hard rounds are the disjoint union
   of the per-agent lists).
2. **Routing headline.** Oracle / pooled / learned per arm, with both gaps
   (``learned - pooled`` = value of specialisation, ``oracle - learned`` =
   headroom left in the router).
3. **Pair stability.** Final and worst within-pair L2 per arm, the audit
   of the claim that co-located clones taking independent steps stay
   together.
4. **Learning curve.** NLL movement from the first reported round to the
   last, per arm, from :mod:`infl_ens.figures.per_round_tables`.

The pure builders take already-loaded artifacts; :func:`write_cross_arm_report`
does the disk I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence


def load_history(run_dir: Path) -> list[dict[str, Any]]:
    """Read ``<run_dir>/history.json``.

    :param run_dir: Closed-loop run root.
    :type run_dir: pathlib.Path
    :returns: Per-round records.
    :rtype: list[dict]
    :raises FileNotFoundError: If the history is missing.
    :raises ValueError: If it holds no rounds.
    """
    path = run_dir / "history.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"{path} holds no rounds")
    return records


def load_routing_report(run_dir: Path) -> Optional[dict[str, Any]]:
    """Load the routing diagnostic JSON of a run, if present.

    :param run_dir: Closed-loop run root.
    :type run_dir: pathlib.Path
    :returns: Parsed report or ``None``.
    :rtype: dict | None
    """
    for name in ("routing_ensemble_diagnostics.json", "routing_weight_comparison.json"):
        path = run_dir / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def load_per_round_table(run_dir: Path) -> Optional[dict[str, Any]]:
    """Load the per-round pair NLL pivot of a run, if generated.

    :param run_dir: Closed-loop run root.
    :type run_dir: pathlib.Path
    :returns: Parsed pivot or ``None``.
    :rtype: dict | None
    """
    path = run_dir / "tables" / "pair_nll_by_round.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def round_prompt_sets(records: Sequence[dict[str, Any]]) -> list[set[str]]:
    """Prompt set trained on in each round, routing-mode agnostic.

    :param records: History records.
    :type records: Sequence[dict]
    :returns: One set per round.
    :rtype: list[set[str]]
    """
    out: list[set[str]] = []
    for rec in records:
        batch = rec.get("batch_prompts")
        if batch:
            out.append({str(p) for p in batch})
            continue
        prompts: set[str] = set()
        for lst in (rec.get("agent_prompts") or {}).values():
            prompts.update(str(p) for p in lst)
        out.append(prompts)
    return out


def within_pair_distances(records: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    """Per-pair within-group L2 across rounds, from logged geometry.

    :param records: History records.
    :type records: Sequence[dict]
    :returns: Group name to one distance per round (empty when the run
        logged no ``agent_geometry``).
    :rtype: dict[str, list[float]]
    """
    series: dict[str, list[float]] = {}
    for rec in records:
        geom = (rec.get("agent_geometry") or {}).get("within_merge_l2") or {}
        for name, dist in geom.items():
            series.setdefault(str(name), []).append(float(dist))
    return series


def data_matching(
    histories: Sequence[tuple[str, Sequence[dict[str, Any]]]],
) -> dict[str, Any]:
    """Check that every arm trained on the same round batches as the first.

    :param histories: ``(label, records)`` per arm.
    :type histories: Sequence[tuple[str, Sequence[dict]]]
    :returns: JSON-safe summary with ``all_identical``.
    :rtype: dict
    """
    if len(histories) < 2:
        return {"n_arms": len(histories), "all_identical": True, "pairs": []}
    ref_label, ref_records = histories[0]
    ref_sets = round_prompt_sets(ref_records)
    pairs = []
    all_same = True
    for label, records in histories[1:]:
        sets = round_prompt_sets(records)
        n = min(len(ref_sets), len(sets))
        identical = [ref_sets[i] == sets[i] for i in range(n)]
        jaccard = [
            (len(ref_sets[i] & sets[i]) / len(ref_sets[i] | sets[i]))
            if (ref_sets[i] | sets[i]) else 1.0
            for i in range(n)
        ]
        same = bool(n) and all(identical)
        all_same = all_same and same
        pairs.append({
            "arm_a": ref_label,
            "arm_b": label,
            "n_rounds_compared": n,
            "rounds_identical": sum(identical),
            "min_jaccard": min(jaccard) if jaccard else None,
            "all_identical": same,
        })
    return {"n_arms": len(histories), "all_identical": all_same, "pairs": pairs}


def _fmt(x: Optional[float], places: int = 4) -> str:
    return "--" if x is None else f"{x:.{places}f}"


def build_cross_arm_report(
    arms: Sequence[tuple[str, Path]],
    *,
    generalist_run_dir: Optional[Path] = None,
) -> tuple[dict[str, Any], str]:
    """Assemble the cross-arm report from the arms' run directories.

    :param arms: ``(label, run_dir)`` per specialist arm, in display order.
    :type arms: Sequence[tuple[str, pathlib.Path]]
    :param generalist_run_dir: Optional pooled-generalist run, recorded
        for provenance.
    :type generalist_run_dir: pathlib.Path | None
    :returns: ``(report_json, report_markdown)``.
    :rtype: tuple[dict, str]
    """
    histories = [(label, load_history(run)) for label, run in arms]
    report: dict[str, Any] = {"arms": {}, "data_matching": {}, "notes": []}
    md: list[str] = ["# Cross-arm analysis", ""]

    md.append("## 1. Data matching (is one generalist fair to every arm?)")
    md.append("")
    matching = data_matching(histories)
    report["data_matching"] = matching
    if matching["pairs"]:
        for pair in matching["pairs"]:
            md.append(
                f"- **{pair['arm_a']}** vs **{pair['arm_b']}**: "
                f"{pair['rounds_identical']}/{pair['n_rounds_compared']} rounds identical "
                f"(min Jaccard {_fmt(pair['min_jaccard'])})."
            )
        md.append("")
        md.append(
            "**One pooled generalist is data-matched to every arm.**"
            if matching["all_identical"] else
            "**The arms do NOT all share round batches; a shared generalist is "
            "not data-matched to every arm.**"
        )
        if not matching["all_identical"]:
            report["notes"].append(
                "Round batches differ between arms; a shared generalist is not data-matched."
            )
    else:
        md.append("_Only one arm supplied; nothing to match._")
    md.append("")

    md.append("## 2. Routing headline (held-out flat pool)")
    md.append("")
    md.append(
        "| Arm | Oracle | Pooled generalist | Learned specialists | "
        "Learned − Pooled | Oracle − Learned |"
    )
    md.append("|---|---|---|---|---|---|")
    for label, run in arms:
        rep = load_routing_report(run)
        if rep is None:
            md.append(f"| {label} | -- | -- | -- | -- | -- |")
            report["arms"].setdefault(label, {})["routing"] = None
            report["notes"].append(f"{label}: no routing diagnostic JSON found.")
            continue
        flat = rep["flat"]
        oracle = float(flat["oracle_routing_nll"])
        pooled = float(flat["pooled_nll"])
        learned = float(flat["learned_routing_expected_nll"])
        report["arms"].setdefault(label, {})["routing"] = {
            "oracle": oracle, "pooled": pooled, "learned": learned,
            "learned_minus_pooled": learned - pooled,
            "oracle_minus_learned": oracle - learned,
            "n_prompts": flat.get("n_prompts"), "round": flat.get("round"),
        }
        md.append(
            f"| {label} | {_fmt(oracle)} | {_fmt(pooled)} | {_fmt(learned)} | "
            f"{learned - pooled:+.4f} | {oracle - learned:+.4f} |"
        )
    md.append("")
    md.append(
        "_Negative `Learned − Pooled` means routed specialists beat the "
        "data-matched generalist. `Oracle − Learned` is the headroom a "
        "perfect router would still recover from this same set of adapters._"
    )
    md.append("")

    md.append("## 3. Pair stability (within-pair L2)")
    md.append("")
    md.append("| Arm | Pairs | Final max | Final mean | Worst over run |")
    md.append("|---|---|---|---|---|")
    for label, records in histories:
        series = within_pair_distances(records)
        if not series:
            md.append(f"| {label} | -- | -- | -- | -- |")
            continue
        finals = [v[-1] for v in series.values() if v]
        worst = max((max(v) for v in series.values() if v), default=None)
        report["arms"].setdefault(label, {})["within_pair"] = {
            "final_max": max(finals) if finals else None,
            "final_mean": (sum(finals) / len(finals)) if finals else None,
            "worst_over_run": worst,
            "per_pair_final": {k: (v[-1] if v else None) for k, v in series.items()},
        }
        md.append(
            f"| {label} | {len(series)} | {max(finals):.3e} | "
            f"{sum(finals) / len(finals):.3e} | {worst:.3e} |"
        )
    md.append("")
    md.append(
        "_Co-location is predicted, not enforced: every clone takes its own "
        "step. A value of 0 means partners stayed together on their own; a "
        "growing value is real separation (expected under hard routing, where "
        "partners draw different prompt subsets)._"
    )
    md.append("")

    md.append("## 4. Held-out NLL movement (first reported round → last)")
    md.append("")
    md.append("| Arm | Rounds | Pairs improved | Mean Δ NLL | Best pair | Worst pair |")
    md.append("|---|---|---|---|---|---|")
    for label, run in arms:
        pr = load_per_round_table(run)
        deltas = (
            {k: v for k, v in pr["delta_first_to_last"].items() if v is not None}
            if pr else {}
        )
        if not deltas:
            md.append(f"| {label} | -- | -- | -- | -- | -- |")
            continue
        improved = sum(1 for v in deltas.values() if v < 0)
        best = min(deltas.items(), key=lambda kv: kv[1])
        worst = max(deltas.items(), key=lambda kv: kv[1])
        report["arms"].setdefault(label, {})["per_round"] = {
            "rounds": pr["rounds"], "delta_first_to_last": deltas,
            "n_improved": improved, "n_pairs": len(deltas),
        }
        md.append(
            f"| {label} | {pr['rounds'][0]}–{pr['rounds'][-1]} | "
            f"{improved}/{len(deltas)} | "
            f"{sum(deltas.values()) / len(deltas):+.4f} | "
            f"{best[0]} ({best[1]:+.4f}) | {worst[0]} ({worst[1]:+.4f}) |"
        )
    md.append("")

    if generalist_run_dir is not None:
        report["generalist_run_dir"] = str(generalist_run_dir)
        md.append(f"Generalist run: `{generalist_run_dir}`")
        md.append("")
    if report["notes"]:
        md.append("## Notes")
        md.append("")
        md += [f"- {n}" for n in report["notes"]]
        md.append("")
    return report, "\n".join(md) + "\n"


def write_cross_arm_report(
    arms: Sequence[tuple[str, Path]],
    output_dir: Path,
    *,
    generalist_run_dir: Optional[Path] = None,
) -> list[Path]:
    """Build the report and write ``cross_analysis.{md,json}``.

    :param arms: ``(label, run_dir)`` per specialist arm.
    :type arms: Sequence[tuple[str, pathlib.Path]]
    :param output_dir: Destination directory.
    :type output_dir: pathlib.Path
    :param generalist_run_dir: Optional pooled-generalist run.
    :type generalist_run_dir: pathlib.Path | None
    :returns: Written paths.
    :rtype: list[pathlib.Path]
    """
    report, md = build_cross_arm_report(arms, generalist_run_dir=generalist_run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "cross_analysis.md"
    json_path = output_dir / "cross_analysis.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return [md_path, json_path]


__all__ = [
    "build_cross_arm_report",
    "data_matching",
    "load_history",
    "load_per_round_table",
    "load_routing_report",
    "round_prompt_sets",
    "within_pair_distances",
    "write_cross_arm_report",
]
