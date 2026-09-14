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
from collections import Counter
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


def round_record_sequences(records: Sequence[dict[str, Any]]) -> list[tuple[str, ...]]:
    """Return duplicate-preserving record sequences for data-match audits.

    :param records: History records.
    :type records: Sequence[dict[str, Any]]
    :returns: One ordered record-ID sequence per round.
    :rtype: list[tuple[str, ...]]
    """
    out: list[tuple[str, ...]] = []
    for record in records:
        record_ids = record.get("batch_record_ids")
        if record_ids is not None:
            out.append(tuple(str(value) for value in record_ids))
            continue
        prompts = list(record.get("batch_prompts") or [])
        responses = list(record.get("batch_responses") or [""] * len(prompts))
        if prompts:
            out.append(tuple(
                f"{prompt}\0{response}"
                for prompt, response in zip(prompts, responses)
            ))
            continue
        flattened: list[str] = []
        agent_prompts = record.get("agent_prompts") or {}
        agent_responses = record.get("agent_responses") or {}
        for name in sorted(agent_prompts):
            p_values = list(agent_prompts[name])
            r_values = list(agent_responses.get(name) or [""] * len(p_values))
            flattened.extend(
                f"{prompt}\0{response}" for prompt, response in zip(p_values, r_values)
            )
        out.append(tuple(flattened))
    return out


def _multiset_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a = Counter(left)
    b = Counter(right)
    union = sum((a | b).values())
    return sum((a & b).values()) / union if union else 1.0


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


def resource_ledger(records: Sequence[dict[str, Any]]) -> dict[str, float | int | None]:
    """Aggregate comparable training-resource fields from history.

    :param records: Run history.
    :type records: Sequence[dict[str, Any]]
    :returns: Stored/trainable parameters, exposures, memory, forwards, and GPU-hours.
    :rtype: dict[str, float | int | None]
    """
    total_tokens = 0
    total_wall = 0.0
    total_forwards = 0
    peak_memory = 0
    last_units: list[dict[str, Any]] = []
    for record in records:
        raw = record.get("resource_accounting") or {}
        units = (
            [dict(value) for value in raw.values() if isinstance(value, dict)]
            if isinstance(raw, dict) and any(isinstance(value, dict) for value in raw.values())
            else [dict(raw)] if isinstance(raw, dict) and raw else []
        )
        if units:
            last_units = units
        total_tokens += sum(int(unit.get("token_exposures", 0)) for unit in units)
        total_wall += sum(float(unit.get("wall_seconds", 0.0)) for unit in units)
        total_forwards += sum(int(unit.get("model_forwards", 0)) for unit in units)
        peak_memory = max(
            [peak_memory, *(int(unit.get("peak_memory_bytes", 0)) for unit in units)]
        )
    parameters = sum(int(unit.get("trainable_parameters", 0)) for unit in last_units)
    stored_rank = sum(
        int(unit.get("stored_lora_rank", unit.get("active_lora_rank", 0)))
        for unit in last_units
    )
    active_rank = max(
        (int(unit.get("active_lora_rank", 0)) for unit in last_units),
        default=0,
    )
    return {
        "stored_trainable_parameters": parameters or None,
        "stored_lora_rank": stored_rank or None,
        "active_lora_rank": active_rank or None,
        "token_exposures": total_tokens or None,
        "peak_memory_bytes": peak_memory or None,
        "model_forwards": total_forwards or None,
        "gpu_hours": total_wall / 3600.0 if total_wall else None,
    }


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
    ref_sequences = round_record_sequences(ref_records)
    pairs = []
    all_same = True
    for label, records in histories[1:]:
        sequences = round_record_sequences(records)
        n = min(len(ref_sequences), len(sequences))
        identical = [
            ref_sequences[i] == sequences[i]
            or Counter(ref_sequences[i]) == Counter(sequences[i])
            for i in range(n)
        ]
        jaccard = [
            _multiset_jaccard(ref_sequences[i], sequences[i])
            for i in range(n)
        ]
        same = bool(n) and len(ref_sequences) == len(sequences) and all(identical)
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
    generalist_runs: Optional[Sequence[tuple[str, Path]]] = None,
) -> tuple[dict[str, Any], str]:
    """Assemble the cross-arm report from the arms' run directories.

    :param arms: ``(label, run_dir)`` per specialist arm, in display order.
    :type arms: Sequence[tuple[str, pathlib.Path]]
    :param generalist_run_dir: Optional pooled-generalist run, recorded
        for provenance. Retained for compatibility with older callers.
    :type generalist_run_dir: pathlib.Path | None
    :param generalist_runs: Named pooled-reference runs.
    :type generalist_runs: Sequence[tuple[str, pathlib.Path]] | None
    :returns: ``(report_json, report_markdown)``.
    :rtype: tuple[dict, str]
    """
    histories = [(label, load_history(run)) for label, run in arms]
    named_generalists = list(generalist_runs or [])
    if not named_generalists and generalist_run_dir is not None:
        named_generalists = [(generalist_run_dir.name, generalist_run_dir)]
    generalist_histories = [
        (label, load_history(run))
        for label, run in named_generalists
        if (run / "history.json").is_file()
    ]
    report: dict[str, Any] = {"arms": {}, "data_matching": {}, "notes": []}
    md: list[str] = ["# Cross-arm analysis", ""]

    md.append("## 1. Data matching (is one generalist fair to every arm?)")
    md.append("")

    training_histories = [
        (label, records)
        for label, records in histories
        if any(
            record.get("batch_record_ids") or record.get("batch_prompts")
            for record in records
        )
    ]
    matching = data_matching(training_histories)
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
        learned = float(
            flat.get("learned_model_nll", flat["learned_routing_expected_nll"])
        )
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

    md.append("### Router matrix")
    md.append("")
    md.append(
        "| Arm | Router | Test expected NLL | Low-support NLL | "
        "Sequence mixture | Oracle regret | Utilization | Effective experts |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    first_report: Optional[dict[str, Any]] = None
    for label, run in arms:
        rep = load_routing_report(run)
        if rep is None:
            continue
        if first_report is None:
            first_report = rep
        low_routers = (
            ((rep.get("slices") or {}).get("low_support_id") or {}).get("routers") or {}
        )
        learned_model = rep.get("learned_model") or {}
        if learned_model:
            low_slice = (rep.get("slices") or {}).get("low_support_id") or {}
            learned_value = learned_model.get("mean_nll")
            oracle_value = (rep.get("oracle") or {}).get("mean_nll")
            regret = (
                float(learned_value) - float(oracle_value)
                if learned_value is not None and oracle_value is not None
                else None
            )
            md.append(
                f"| {label} | deployed learned mixture | {_fmt(learned_value)} | "
                f"{_fmt(low_slice.get('learned_model_nll'))} | -- | "
                f"{_fmt(regret)} | -- | -- |"
            )
        report["arms"].setdefault(label, {})["routers"] = rep.get("routers") or {}
        for router_name, values in (rep.get("routers") or {}).items():
            low = low_routers.get(router_name) or {}
            utilization = values.get("utilization") or []
            used_fraction = (
                sum(float(value) > 0.0 for value in utilization) / len(utilization)
                if utilization
                else None
            )
            md.append(
                f"| {label} | {router_name} | {_fmt(values.get('expected_nll'))} | "
                f"{_fmt(low.get('expected_nll'))} | "
                f"{_fmt(values.get('sequence_mixture_nll'))} | "
                f"{_fmt(values.get('oracle_regret'))} | "
                f"{_fmt(used_fraction, 2)} | "
                f"{_fmt(values.get('effective_experts'), 2)} |"
            )
    if first_report is not None:
        low_references = (
            ((first_report.get("slices") or {}).get("low_support_id") or {}).get("references")
            or {}
        )
        for reference_name, values in (first_report.get("references") or {}).items():
            low = low_references.get(reference_name) or {}
            md.append(
                f"| {reference_name} | single adapter | {_fmt(values.get('mean_nll'))} | "
                f"{_fmt(low.get('mean_nll'))} | -- | -- | 1.00 | 1.00 |"
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

    md.append("## 5. Resource accounting")
    md.append("")
    md.append(
        "| Arm | Stored trainable params | Stored rank | Active rank | Token exposures | "
        "Model forwards | Peak GiB | GPU-hours |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, records in [*histories, *generalist_histories]:
        ledger = resource_ledger(records)
        report["arms"].setdefault(label, {})["resources"] = ledger
        peak = ledger["peak_memory_bytes"]
        peak_gib = None if peak is None else float(peak) / (1024 ** 3)
        md.append(
            f"| {label} | {ledger['stored_trainable_parameters'] or '--'} | "
            f"{ledger['stored_lora_rank'] or '--'} | "
            f"{ledger['active_lora_rank'] or '--'} | "
            f"{ledger['token_exposures'] or '--'} | "
            f"{ledger['model_forwards'] or '--'} | {_fmt(peak_gib, 2)} | "
            f"{_fmt(ledger['gpu_hours'], 2)} |"
        )
    md.append("")

    if named_generalists:
        report["generalist_run_dirs"] = {
            label: str(run) for label, run in named_generalists
        }
        report["generalist_run_dir"] = str(named_generalists[0][1])
        md.append(
            "Generalist runs: "
            + ", ".join(f"**{label}** (`{run}`)" for label, run in named_generalists)
        )
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
    generalist_runs: Optional[Sequence[tuple[str, Path]]] = None,
) -> list[Path]:
    """Build the report and write ``cross_analysis.{md,json}``.

    :param arms: ``(label, run_dir)`` per specialist arm.
    :type arms: Sequence[tuple[str, pathlib.Path]]
    :param output_dir: Destination directory.
    :type output_dir: pathlib.Path
    :param generalist_run_dir: Optional pooled-generalist run.
    :type generalist_run_dir: pathlib.Path | None
    :param generalist_runs: Named pooled-reference runs.
    :type generalist_runs: Sequence[tuple[str, pathlib.Path]] | None
    :returns: Written paths.
    :rtype: list[pathlib.Path]
    """
    report, md = build_cross_arm_report(
        arms,
        generalist_run_dir=generalist_run_dir,
        generalist_runs=generalist_runs,
    )
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
    "round_record_sequences",
    "resource_ledger",
    "within_pair_distances",
    "write_cross_arm_report",
]
