#!/usr/bin/env python3
"""Cross-arm analysis for the three-arm seven-axis comparison.

Ties together artifacts that are otherwise reported per arm and answers the
questions the individual figures cannot:

1. **Data matching.** Are the two specialist arms trained on the same round
   batches? If yes, one pooled generalist is a fair comparator for both.
   Checked by comparing each round's prompt SET (soft rounds are logged
   verbatim as ``batch_prompts``; hard rounds are the disjoint union of the
   per-agent lists).
2. **Routing headline.** Oracle / pooled / learned per arm, with both gaps
   (``learned - pooled`` = value of specialisation, ``oracle - learned`` =
   headroom left in the router).
3. **Pair stability.** Final and worst within-pair L2 per arm — the audit of
   the theory's claim that co-located clones taking independent steps stay
   together.
4. **Learning curve.** Per-round NLL movement from the first reported round
   to the last, per arm, from the tables produced by
   :mod:`scripts.build_per_round_pair_nll_table`.

Writes ``cross_analysis.md`` and ``cross_analysis.json`` into ``--output-dir``.

Example::

    python scripts/cross_analyze_three_arms.py \\
        --arm "Soft top-3=results/seven_axis_soft_topk3_pairs/seed0" \\
        --arm "Hard (SFT)=results/seven_axis_hard_pairs_matched/seed0" \\
        --output-dir scripts/figures/three_arm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]


def _load_history(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "history.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def round_prompt_sets(records: list[dict[str, Any]]) -> list[set[str]]:
    """Prompt set trained on in each round, routing-mode agnostic.

    Soft rounds log the batch verbatim; hard rounds partition it across
    agents, so the union recovers it.

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


def within_pair_distances(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Per-pair within-group L2 across rounds, from logged geometry.

    :returns: Mapping of group name to one distance per round (may be empty
        when the run logged no ``agent_geometry``).
    :rtype: dict[str, list[float]]
    """
    series: dict[str, list[float]] = {}
    for rec in records:
        geom = (rec.get("agent_geometry") or {}).get("within_merge_l2") or {}
        for name, dist in geom.items():
            series.setdefault(str(name), []).append(float(dist))
    return series


def _routing(run_dir: Path) -> Optional[dict[str, Any]]:
    """Load whichever routing diagnostic JSON is present."""
    for name in (
        "routing_ensemble_diagnostics.json",
        "routing_weight_comparison.json",
    ):
        path = run_dir / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _per_round(run_dir: Path) -> Optional[dict[str, Any]]:
    """Load the per-round pair NLL pivot if it was generated."""
    path = run_dir / "tables" / "pair_nll_by_round.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _fmt(x: Optional[float], places: int = 4) -> str:
    return "--" if x is None else f"{x:.{places}f}"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", action="append", required=True, metavar="LABEL=RUN_DIR",
        help="Specialist arm label and run directory; repeat per arm.",
    )
    parser.add_argument("--generalist-run-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    arms: list[tuple[str, Path]] = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm must be LABEL=RUN_DIR, got {spec!r}")
        label, raw = spec.split("=", 1)
        path = Path(raw)
        arms.append((label, path if path.is_absolute() else ROOT / path))

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = {label: _load_history(run) for label, run in arms}
    report: dict[str, Any] = {"arms": {}, "data_matching": {}, "notes": []}
    md: list[str] = ["# Three-arm cross analysis", ""]

    # --- 1. data matching -------------------------------------------------
    md.append("## 1. Data matching (is one generalist fair to both arms?)")
    md.append("")
    if len(arms) >= 2:
        (label_a, _), (label_b, _) = arms[0], arms[1]
        sets_a = round_prompt_sets(histories[label_a])
        sets_b = round_prompt_sets(histories[label_b])
        n = min(len(sets_a), len(sets_b))
        identical = [sets_a[i] == sets_b[i] for i in range(n)]
        jaccard = [
            (len(sets_a[i] & sets_b[i]) / len(sets_a[i] | sets_b[i]))
            if (sets_a[i] | sets_b[i]) else 1.0
            for i in range(n)
        ]
        all_same = bool(n) and all(identical)
        report["data_matching"] = {
            "arm_a": label_a, "arm_b": label_b, "n_rounds_compared": n,
            "rounds_identical": sum(identical),
            "min_jaccard": min(jaccard) if jaccard else None,
            "all_identical": all_same,
        }
        md.append(
            f"Compared {n} rounds of **{label_a}** against **{label_b}**: "
            f"{sum(identical)}/{n} rounds have an identical prompt set "
            f"(min Jaccard {min(jaccard):.4f})." if jaccard else "No rounds to compare."
        )
        md.append("")
        md.append(
            "**One pooled generalist is data-matched to both arms.**"
            if all_same else
            "**The arms do NOT share round batches — each arm needs its own "
            "pooled replay before the generalist comparison is fair.**"
        )
        if not all_same:
            report["notes"].append(
                "Round batches differ between arms; a shared generalist is not "
                "data-matched."
            )
    else:
        md.append("_Only one arm supplied; nothing to match._")
    md.append("")

    # --- 2. routing headline ---------------------------------------------
    md.append("## 2. Routing headline (held-out flat pool)")
    md.append("")
    md.append(
        "| Arm | Oracle | Pooled generalist | Learned specialists | "
        "Learned − Pooled | Oracle − Learned |"
    )
    md.append("|---|---|---|---|---|---|")
    for label, run in arms:
        rep = _routing(run)
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

    # --- 3. pair stability ------------------------------------------------
    md.append("## 3. Pair stability (within-pair L2)")
    md.append("")
    md.append("| Arm | Pairs | Final max | Final mean | Worst over run |")
    md.append("|---|---|---|---|---|")
    for label, _run in arms:
        series = within_pair_distances(histories[label])
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

    # --- 4. learning curve ------------------------------------------------
    md.append("## 4. Per-round movement (first reported round → last)")
    md.append("")
    md.append("| Arm | Rounds | Pairs improved | Mean Δ NLL | Best pair | Worst pair |")
    md.append("|---|---|---|---|---|---|")
    for label, run in arms:
        pr = _per_round(run)
        if pr is None:
            md.append(f"| {label} | -- | -- | -- | -- | -- |")
            continue
        deltas = {k: v for k, v in pr["delta_first_to_last"].items() if v is not None}
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

    if args.generalist_run_dir:
        report["generalist_run_dir"] = args.generalist_run_dir
        md.append(f"Generalist run: `{args.generalist_run_dir}`")
        md.append("")

    if report["notes"]:
        md.append("## Notes")
        md.append("")
        md += [f"- {n}" for n in report["notes"]]
        md.append("")

    (out_dir / "cross_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "cross_analysis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(f"wrote {out_dir / 'cross_analysis.md'}")
    print(f"wrote {out_dir / 'cross_analysis.json'}")


if __name__ == "__main__":
    main()
