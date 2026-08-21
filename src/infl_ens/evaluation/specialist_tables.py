"""Own-axis specialist vs pooled-baseline comparison tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

BENCHMARK_ORDER: tuple[str, ...] = (
    "beavertails",
    "halueval",
    "ai4privacy",
    "orbench",
    "prompt_injection",
    "do_not_answer",
)

BENCHMARK_LABELS: dict[str, str] = {
    "beavertails": "Harm",
    "halueval": "Hallucination",
    "ai4privacy": "Privacy",
    "orbench": "Over-refusal",
    "prompt_injection": "Injection",
    "do_not_answer": "Policy",
}

# Own-axis specialist per benchmark.
AXIS_SPECIALIST: dict[str, str] = {
    "beavertails": "merge-harm",
    "halueval": "merge-hallucination",
    "ai4privacy": "merge-privacy",
    "orbench": "merge-overrefusal",
    "prompt_injection": "merge-injection",
    "do_not_answer": "merge-policy",
}

POOLED_BASELINE_AGENT = "pooled-baseline"
MERGE_GENERALIST_AGENT = "merge-generalist"

SEVEN_AXIS_BENCHMARK_ORDER: tuple[str, ...] = (
    "beavertails",
    "halueval",
    "jbb_behaviors",
    "ai4privacy",
    "orbench",
    "prompt_injection",
    "do_not_answer",
)

SEVEN_AXIS_SPECIALIST: dict[str, str] = {
    **AXIS_SPECIALIST,
    "jbb_behaviors": "merge-jailbreak",
}


@dataclass(frozen=True)
class SpecialistComparisonRow:
    """One benchmark row in a specialist-vs-pooled table.

    :param benchmark: Benchmark identifier.
    :type benchmark: str
    :param axis_label: Human-readable axis name.
    :type axis_label: str
    :param specialist: Merge adapter for the axis.
    :type specialist: str
    :param specialist_nll: Mean NLL of the specialist.
    :type specialist_nll: float
    :param baseline_nll: Mean NLL of ``pooled-baseline``.
    :type baseline_nll: float
    :param delta: ``specialist_nll - baseline_nll`` (negative = specialist
        better).
    :type delta: float
    :param specialist_wins: Whether specialist beats pooled baseline on NLL.
    :type specialist_wins: bool
    """

    benchmark: str
    axis_label: str
    specialist: str
    specialist_nll: float
    baseline_nll: float
    delta: float
    specialist_wins: bool


def load_eval_scores(path: str | Path) -> dict[str, dict[str, float]]:
    """Load ``mean_nll`` indexed by benchmark then agent.

    :param path: Path to ``eval_results.json``.
    :type path: str | pathlib.Path
    :returns: Nested score mapping.
    :rtype: dict[str, dict[str, float]]
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    bench_keys = set(BENCHMARK_ORDER) | set(SEVEN_AXIS_BENCHMARK_ORDER)
    out: dict[str, dict[str, float]] = {b: {} for b in bench_keys}
    for row in payload.get("results", []):
        bench = row.get("benchmark")
        agent = row.get("agent")
        if bench in out and agent:
            out[bench][agent] = float(row["mean_nll"])
    return out


def _resolve_agent_score(
    scores_bench: Mapping[str, float],
    agent: str,
) -> float:
    """Look up an agent score with merge-name aliases.

    :param scores_bench: Agent scores for one benchmark.
    :type scores_bench: Mapping[str, float]
    :param agent: Requested agent name.
    :type agent: str
    :returns: Mean NLL for the agent.
    :rtype: float
    :raises KeyError: If no matching score exists.
    """
    if agent in scores_bench:
        return float(scores_bench[agent])
    if agent == "merge-jailbreak" and MERGE_GENERALIST_AGENT in scores_bench:
        return float(scores_bench[MERGE_GENERALIST_AGENT])
    raise KeyError(agent)


def merge_eval_scores(
    specialist_scores: Mapping[str, Mapping[str, float]],
    baseline_scores: Mapping[str, Mapping[str, float]],
    *,
    baseline_agent: str = POOLED_BASELINE_AGENT,
) -> dict[str, dict[str, float]]:
    """Combine specialist merge scores with pooled-baseline scores.

    :param specialist_scores: Scores from the merge closed-loop eval JSON.
    :type specialist_scores: Mapping[str, Mapping[str, float]]
    :param baseline_scores: Scores from the pooled-baseline eval JSON.
    :type baseline_scores: Mapping[str, Mapping[str, float]]
    :param baseline_agent: Agent name for the full-data baseline.
    :type baseline_agent: str
    :returns: ``benchmark -> agent -> mean_nll`` with both sources merged.
    :rtype: dict[str, dict[str, float]]
    """
    merged: dict[str, dict[str, float]] = {
        b: dict(specialist_scores.get(b, {})) for b in BENCHMARK_ORDER
    }
    for bench in BENCHMARK_ORDER:
        if baseline_agent in baseline_scores.get(bench, {}):
            merged[bench][baseline_agent] = baseline_scores[bench][baseline_agent]
    return merged


def build_specialist_vs_pooled_table(
    scores: Mapping[str, Mapping[str, float]],
    *,
    baseline_agent: str = POOLED_BASELINE_AGENT,
) -> list[SpecialistComparisonRow]:
    """Build per-benchmark own-axis specialist vs pooled-baseline rows.

    :param scores: ``benchmark -> agent -> mean_nll``.
    :type scores: Mapping[str, Mapping[str, float]]
    :param baseline_agent: Pooled baseline agent name.
    :type baseline_agent: str
    :returns: One row per benchmark with both scores present.
    :rtype: list[SpecialistComparisonRow]
    :raises KeyError: If baseline or a specialist score is missing.
    """
    rows: list[SpecialistComparisonRow] = []
    for bench in BENCHMARK_ORDER:
        specialist = AXIS_SPECIALIST[bench]
        spec_nll = _resolve_agent_score(scores[bench], specialist)
        base_nll = _resolve_agent_score(scores[bench], baseline_agent)
        delta = spec_nll - base_nll
        rows.append(
            SpecialistComparisonRow(
                benchmark=bench,
                axis_label=BENCHMARK_LABELS[bench],
                specialist=specialist,
                specialist_nll=spec_nll,
                baseline_nll=base_nll,
                delta=delta,
                specialist_wins=delta < 0.0,
            )
        )
    return rows


# Backward-compatible alias.
build_specialist_vs_generalist_table = build_specialist_vs_pooled_table


def format_comparison_markdown(
    rows: Sequence[SpecialistComparisonRow],
    *,
    title: str,
) -> str:
    """Render a markdown table from comparison rows.

    :param rows: Table rows.
    :type rows: Sequence[SpecialistComparisonRow]
    :param title: Section heading.
    :type title: str
    :returns: Markdown text.
    :rtype: str
    """
    lines = [
        f"## {title}",
        "",
        "| Axis | Specialist | Spec NLL | Pooled NLL | Δ (spec−pool) | Spec wins |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        win = "✓" if row.specialist_wins else ""
        lines.append(
            f"| {row.axis_label} | `{row.specialist}` | "
            f"{row.specialist_nll:.4f} | {row.baseline_nll:.4f} | "
            f"{row.delta:+.4f} | {win} |"
        )
    n_wins = sum(1 for r in rows if r.specialist_wins)
    lines.extend([
        "",
        f"**Specialists beat pooled baseline on {n_wins}/{len(rows)} axes** "
        "(lower NLL is better).",
        "",
    ])
    return "\n".join(lines)


def load_routing_headline(path: str | Path) -> dict[str, float]:
    """Load flat route-then-score headline metrics from routing JSON.

    :param path: Path to ``routing_ensemble_diagnostics.json``.
    :type path: str | pathlib.Path
    :returns: Flat metric mapping.
    :rtype: dict[str, float]
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    flat = payload.get("flat", payload)
    return {
        "pooled_nll": float(flat["pooled_nll"]),
        "learned_routing_nll": float(
            flat.get("learned_routing_expected_nll", flat["learned_routing_nll"]),
        ),
        "learned_routing_argmax_nll": float(
            flat.get("learned_routing_argmax_nll", flat.get("learned_routing_nll", 0)),
        ),
        "learned_routing_sampled_nll": float(
            flat.get("learned_routing_sampled_nll", flat.get("learned_routing_nll", 0)),
        ),
        "oracle_routing_nll": float(flat["oracle_routing_nll"]),
        "routing_agreement_argmax": float(
            flat.get("routing_agreement_argmax", flat.get("routing_agreement", 0)),
        ),
        "n_prompts": int(flat.get("n_prompts", 0)),
    }


def format_routing_headline_markdown(
    scores: Mapping[str, float],
    *,
    title: str = "Test partition (withheld flat pool)",
) -> str:
    """Render flat route-then-score headline markdown.

  Expected proportional :math:`G` routing is the primary learned metric;
  argmax and sampled proportional are shown for comparison. Oracle routing
  is the specialization-worth ceiling.

    :param scores: Output of :func:`load_routing_headline`.
    :type scores: Mapping[str, float]
    :param title: Section title.
    :type title: str
    :returns: Markdown text.
    :rtype: str
    """
    pooled = scores["pooled_nll"]
    learned = scores["learned_routing_nll"]
    argmax = scores.get("learned_routing_argmax_nll", learned)
    sampled = scores.get("learned_routing_sampled_nll", learned)
    oracle = scores["oracle_routing_nll"]
    lines = [
        f"## {title} — **headline**",
        "",
        "| Metric | Mean NLL | Δ vs pooled |",
        "|---|---:|---:|",
        f"| Pooled baseline | {pooled:.4f} | — |",
        f"| **Learned routing (expected G)** | **{learned:.4f}** | "
        f"{learned - pooled:+.4f} |",
        f"| Learned routing (sampled G) | {sampled:.4f} | {sampled - pooled:+.4f} |",
        f"| Learned routing (argmax G) | {argmax:.4f} | {argmax - pooled:+.4f} |",
        f"| Oracle routing (ceiling) | {oracle:.4f} | {oracle - pooled:+.4f} |",
        "",
        f"Flat pool: **{scores.get('n_prompts', 0)}** prompts. "
        "Per-benchmark specialist-vs-pooled table below is **diagnostic only**.",
        "",
    ]
    return "\n".join(lines)


def write_specialist_comparison_tables(
    merge_train_eval_path: str | Path,
    merge_test_eval_path: str | Path,
    baseline_train_eval_path: str | Path,
    baseline_test_eval_path: str | Path,
    output_dir: str | Path,
    *,
    routing_test_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write train/test specialist-vs-pooled-baseline markdown tables.

    When ``routing_test_path`` is set, prepend a flat route-then-score
    headline (expected proportional :math:`G`) and demote per-benchmark
    tables to diagnostic sections.

    :param merge_train_eval_path: Merge-run eval JSON on the train partition.
    :type merge_train_eval_path: str | pathlib.Path
    :param merge_test_eval_path: Merge-run eval JSON on the test partition.
    :type merge_test_eval_path: str | pathlib.Path
    :param baseline_train_eval_path: Pooled-baseline eval JSON on train.
    :type baseline_train_eval_path: str | pathlib.Path
    :param baseline_test_eval_path: Pooled-baseline eval JSON on test.
    :type baseline_test_eval_path: str | pathlib.Path
    :param output_dir: Directory for markdown and JSON summaries.
    :type output_dir: str | pathlib.Path
    :param routing_test_path: Optional flat routing diagnostics JSON on test.
    :type routing_test_path: str | pathlib.Path | None
    :returns: Summary dict with win counts and row payloads.
    :rtype: dict
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_scores = merge_eval_scores(
        load_eval_scores(merge_train_eval_path),
        load_eval_scores(baseline_train_eval_path),
    )
    test_scores = merge_eval_scores(
        load_eval_scores(merge_test_eval_path),
        load_eval_scores(baseline_test_eval_path),
    )
    train_rows = build_specialist_vs_pooled_table(train_scores)
    test_rows = build_specialist_vs_pooled_table(test_scores)

    train_md = format_comparison_markdown(
        train_rows, title="Train partition (cap 1k / benchmark) — diagnostic",
    )
    test_headline = ""
    routing_summary: dict[str, Any] | None = None
    if routing_test_path is not None:
        routing_scores = load_routing_headline(routing_test_path)
        test_headline = format_routing_headline_markdown(routing_scores) + "\n"
        routing_summary = routing_scores
    test_md = test_headline + format_comparison_markdown(
        test_rows,
        title="Test partition (cap 1k / benchmark) — diagnostic",
    )

    (out / "specialist_vs_pooled_train.md").write_text(
        train_md, encoding="utf-8",
    )
    (out / "specialist_vs_pooled_test.md").write_text(
        test_md, encoding="utf-8",
    )
    # Legacy filenames for scripts that still look for generalist tables.
    (out / "specialist_vs_generalist_train.md").write_text(
        train_md, encoding="utf-8",
    )
    (out / "specialist_vs_generalist_test.md").write_text(
        test_md, encoding="utf-8",
    )

    summary = {
        "baseline_agent": POOLED_BASELINE_AGENT,
        "train_wins": sum(1 for r in train_rows if r.specialist_wins),
        "test_wins": sum(1 for r in test_rows if r.specialist_wins),
        "n_axes": len(BENCHMARK_ORDER),
        "train_rows": [row.__dict__ for row in train_rows],
        "test_rows": [row.__dict__ for row in test_rows],
        "routing_headline": routing_summary,
    }
    (out / "specialist_vs_pooled_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    (out / "specialist_vs_generalist_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    return summary
