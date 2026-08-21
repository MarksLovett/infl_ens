"""Plot seven-axis merge-specialist vs pooled-baseline benchmark NLL figure.

Reads ``eval_results.json`` from the pair-merge run and baseline replay,
writes matplotlib PDF/PNG and a standalone pgfplots ``.tex`` under
``scripts/figures/``.

Example (on doob after posttrain eval)::

    python scripts/plot_seven_axis_eval_figure.py \\
        --merge-eval results/seven_axis_pair_merge_r40/seed0/eval_final_round/eval_results.json \\
        --baseline-eval results/seven_axis_baseline_replay_r40/seed0/eval_final_round/eval_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from infl_ens.vis.benchmark_nll_bar import plot_benchmark_nll_comparison

BENCHMARK_ORDER = [
    "beavertails",
    "halueval",
    "toxicchat",
    "ai4privacy",
    "orbench",
    "prompt_injection",
    "do_not_answer",
]
BENCHMARK_LABELS = {
    "beavertails": "Harm",
    "halueval": "Hallucination",
    "toxicchat": "Jailbreak",
    "ai4privacy": "Privacy",
    "orbench": "Over-refusal",
    "prompt_injection": "Injection",
    "do_not_answer": "Policy",
}
AGENT_ORDER = [
    "merge-harm",
    "merge-hallucination",
    "merge-privacy",
    "merge-injection",
    "merge-overrefusal",
    "merge-policy",
    "merge-generalist",
    "pooled-baseline",
]
AGENT_COLORS = {
    "merge-harm": "blue!65!black",
    "merge-hallucination": "red!75!black",
    "merge-privacy": "green!55!black",
    "merge-injection": "orange!80!black",
    "merge-overrefusal": "purple!70!black",
    "merge-policy": "brown!70!black",
    "merge-generalist": "cyan!60!black",
    "pooled-baseline": "black!70",
}
LEGEND_LABELS = {
    "merge-harm": "Harm pair",
    "merge-hallucination": "Hallucination pair",
    "merge-privacy": "Privacy pair",
    "merge-injection": "Injection pair",
    "merge-overrefusal": "Over-refusal pair",
    "merge-policy": "Policy pair",
    "merge-generalist": "Generalist pair",
    "pooled-baseline": "Pooled baseline",
}


def _load_eval(path: Path) -> dict[str, dict[str, float]]:
    """Load ``eval_results.json`` into benchmark -> agent -> mean NLL.

    :param path: Path to evaluation JSON.
    :type path: pathlib.Path
    :returns: Nested score mapping.
    :rtype: dict[str, dict[str, float]]
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {b: {} for b in BENCHMARK_ORDER}
    for row in payload["results"]:
        bench = row["benchmark"]
        agent = row["agent"]
        if bench in out:
            out[bench][agent] = float(row["mean_nll"])
    return out


def _merge_scores(
    merge_scores: dict[str, dict[str, float]],
    baseline_scores: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Combine merge and baseline adapter scores per benchmark.

    :param merge_scores: Merge-run eval mapping.
    :type merge_scores: dict
    :param baseline_scores: Baseline-run eval mapping.
    :type baseline_scores: dict
    :returns: Unified adapter NLL table.
    :rtype: dict[str, dict[str, float]]
    """
    combined: dict[str, dict[str, float]] = {b: {} for b in BENCHMARK_ORDER}
    for bench in BENCHMARK_ORDER:
        for agent in AGENT_ORDER[:-1]:
            if agent in merge_scores[bench]:
                combined[bench][agent] = merge_scores[bench][agent]
        if "pooled-baseline" in baseline_scores[bench]:
            combined[bench]["pooled-baseline"] = baseline_scores[bench]["pooled-baseline"]
    return combined


def _load_base_nll(path: Path | None) -> dict[str, float]:
    """Load optional base-model NLL keyed by benchmark.

    :param path: ``base_eval.json`` or similar.
    :type path: pathlib.Path | None
    :returns: Benchmark -> mean NLL (empty if path missing).
    :rtype: dict[str, float]
    """
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results") or payload.get("benchmarks") or []
    out: dict[str, float] = {}
    for row in rows:
        bench = row.get("benchmark")
        if bench in BENCHMARK_ORDER:
            out[bench] = float(row["mean_nll"])
    return out


def _pgfplots_body(
    base_nll: Mapping[str, float],
    adapter_mean: Mapping[str, Mapping[str, float]],
    *,
    agents: list[str],
    title: str,
    include_base: bool,
) -> str:
    """Build pgfplots LaTeX source for the seven-axis bar chart.

    :param base_nll: Optional base-model scores.
    :type base_nll: Mapping[str, float]
    :param adapter_mean: Benchmark -> agent -> mean NLL.
    :type adapter_mean: Mapping[str, Mapping[str, float]]
    :param agents: Adapter series in plot order.
    :type agents: list[str]
    :param title: Figure title.
    :type title: str
    :param include_base: Whether to draw the base-model series.
    :type include_base: bool
    :returns: Full standalone LaTeX document.
    :rtype: str
    """
    xlabels = [BENCHMARK_LABELS[b] for b in BENCHMARK_ORDER]
    plots: list[str] = []
    if include_base:
        coords_base = " ".join(
            f"({BENCHMARK_LABELS[b]},{base_nll[b]:.3f})"
            for b in BENCHMARK_ORDER
            if b in base_nll
        )
        plots.extend([
            f"\\addplot+[fill=gray!50, draw=black!50] coordinates {{{coords_base}}};",
            r"\addlegendentry{Base}",
        ])
    for agent in agents:
        pts = " ".join(
            f"({BENCHMARK_LABELS[b]},{adapter_mean[b][agent]:.3f})"
            for b in BENCHMARK_ORDER
            if agent in adapter_mean[b]
        )
        color = AGENT_COLORS.get(agent, "gray!60")
        label = LEGEND_LABELS.get(agent, agent)
        plots.append(
            f"\\addplot+[fill={color}, draw=black!40] coordinates {{{pts}}};"
        )
        plots.append(f"\\addlegendentry{{{label}}}")

    body = "\n".join(plots)
    return f"""\\documentclass[tikz,border=3pt]{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\begin{{document}}
\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=14cm, height=7cm,
  ybar, bar width=5pt,
  enlarge x limits=0.08,
  ymin=0, ymax=3.5,
  ylabel={{Mean token NLL (lower is better)}},
  symbolic x coords={{{','.join(xlabels)}}},
  xtick=data,
  x tick label style={{font=\\scriptsize, align=center}},
  legend style={{font=\\scriptsize}},
  legend columns=2,
  ymajorgrids=true,
  grid style={{dashed, gray!35}},
  title={{{title}}},
]
{body}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{document}}
"""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merge-eval",
        default="results/seven_axis_pair_merge_r40/seed0/eval_final_round/eval_results.json",
    )
    parser.add_argument(
        "--baseline-eval",
        default="results/seven_axis_baseline_replay_r40/seed0/eval_final_round/eval_results.json",
    )
    parser.add_argument(
        "--base-eval-json",
        default=None,
        help="Optional base-model eval JSON (all seven benchmarks).",
    )
    parser.add_argument(
        "--output-stem",
        default="scripts/figures/seven_axis_merge_vs_baseline",
    )
    parser.add_argument(
        "--round-label",
        default="round 39",
    )
    args = parser.parse_args(argv)

    merge_scores = _load_eval(Path(args.merge_eval))
    baseline_scores = _load_eval(Path(args.baseline_eval))
    adapter_mean = _merge_scores(merge_scores, baseline_scores)
    base_nll = _load_base_nll(
        Path(args.base_eval_json) if args.base_eval_json else None,
    )
    include_base = all(b in base_nll for b in BENCHMARK_ORDER)
    agents = [a for a in AGENT_ORDER if any(a in adapter_mean[b] for b in BENCHMARK_ORDER)]
    title = (
        rf"Seven-axis pair-merge specialists vs pooled baseline "
        rf"({args.round_label}, seed 0)"
    )

    out_stem = Path(args.output_stem)
    plot_benchmark_nll_comparison(
        BENCHMARK_ORDER,
        BENCHMARK_LABELS,
        base_nll,
        adapter_mean,
        adapter_std=None,
        agents=agents,
        include_base=include_base,
        title=title,
        output_stem=out_stem,
    )

    tex_path = out_stem.with_suffix(".tex")
    tex_path.write_text(
        _pgfplots_body(
            base_nll,
            adapter_mean,
            agents=agents,
            title=title,
            include_base=include_base,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_stem}.pdf")
    print(f"wrote {out_stem}.png")
    print(f"wrote {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
