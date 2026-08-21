"""Plot benchmark NLL comparison (10-seed specialists/generalist vs base).

Writes a grouped benchmark-NLL figure, defaulting to
``scripts/figures/ai4privacy_fixed_specialists_vs_base_generalist.{pdf,png,tex}``.

Example::

    python scripts/plot_ai4privacy_fixed_vs_base_figure.py \\
        --sweep-root results/ai4privacy_fixed_theory_specialists_r40
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from infl_ens.vis.benchmark_nll_bar import plot_benchmark_nll_comparison

BENCHMARK_ORDER = ["beavertails", "halueval", "toxicchat", "ai4privacy"]
BENCHMARK_LABELS = {
    "beavertails": "Harm\n(BeaverTails)",
    "halueval": "Hallucination\n(HaluEval)",
    "toxicchat": "Jailbreak\n(ToxicChat)",
    "ai4privacy": "Privacy\n(AI4Privacy)",
}
AGENTS = ["clone-0", "clone-1", "clone-2", "clone-3"]


def _series_agents(
    include_generalist: bool,
    adapter_mean: dict[str, dict[str, float]],
    requested_agents: list[str] | None = None,
) -> list[str]:
    """Return plot series in display order.

    :param include_generalist: Whether to append the pooled generalist.
    :type include_generalist: bool
    :param adapter_mean: Benchmark -> agent -> NLL mapping.
    :type adapter_mean: dict
    :param requested_agents: Optional explicit agent order.
    :type requested_agents: list[str] | None
    :returns: Ordered adapter series.
    :rtype: list[str]
    """
    if requested_agents is not None:
        agents = list(requested_agents)
    else:
        seen = {
            agent
            for bench_scores in adapter_mean.values()
            for agent in bench_scores
            if agent != "generalist"
        }
        agents = [agent for agent in AGENTS if agent in seen]
        agents.extend(sorted(seen - set(agents)))
    if include_generalist:
        agents.append("generalist")
    return agents


def _load_aggregate(sweep_root: Path) -> tuple[dict[str, float], dict, dict]:
    """Load base NLL and per-agent mean/std from a sweep directory.

    :param sweep_root: Run root with ``seed*/eval_final_round/``.
    :type sweep_root: pathlib.Path
    :returns: ``(base_nll, adapter_mean, adapter_std)`` keyed by benchmark then agent.
    :rtype: tuple[dict, dict, dict]
    """
    compare_path = sweep_root / "compare_vs_base.json"
    base_path = sweep_root / "base_eval_matched.json"

    if compare_path.exists():
        compare = json.loads(compare_path.read_text(encoding="utf-8"))
        base_nll = {k: float(v) for k, v in compare["base"].items()}
        adapter_mean = compare["adapters"]
    elif base_path.exists():
        base_nll = {
            r["benchmark"]: float(r["mean_nll"])
            for r in json.loads(base_path.read_text(encoding="utf-8"))["results"]
        }
        adapter_mean = {}
    else:
        raise FileNotFoundError(
            f"Need {compare_path} or {base_path}; run compare_ai4privacy_fixed_vs_base.py first.",
        )

    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for seed_dir in sorted(sweep_root.glob("seed*")):
        eval_path = seed_dir / "eval_final_round" / "eval_results.json"
        if not eval_path.exists():
            continue
        for row in json.loads(eval_path.read_text(encoding="utf-8"))["results"]:
            by_key[(row["benchmark"], row["agent"])].append(float(row["mean_nll"]))

    adapter_std: dict[str, dict[str, float]] = {b: {} for b in BENCHMARK_ORDER}
    if not adapter_mean:
        adapter_mean = {b: {} for b in BENCHMARK_ORDER}
    for (bench, agent), vals in sorted(by_key.items()):
        adapter_mean.setdefault(bench, {})[agent] = statistics.mean(vals)
        adapter_std[bench][agent] = (
            statistics.pstdev(vals) if len(vals) > 1 else 0.0
        )

    return base_nll, adapter_mean, adapter_std


def _merge_generalist(
    adapter_mean: dict[str, dict[str, float]],
    adapter_std: dict[str, dict[str, float]],
    generalist_root: Path,
) -> int:
    """Merge pooled generalist seed evals into the adapter aggregates.

    :param adapter_mean: Mutable benchmark -> agent -> mean NLL mapping.
    :type adapter_mean: dict
    :param adapter_std: Mutable benchmark -> agent -> std NLL mapping.
    :type adapter_std: dict
    :param generalist_root: Root with ``seed*/eval_final_round/eval_results.json``.
    :type generalist_root: pathlib.Path
    :returns: Number of generalist seed eval files consumed.
    :rtype: int
    """
    eval_paths = sorted(generalist_root.glob("seed*/eval_final_round/eval_results.json"))
    by_bench: dict[str, list[float]] = defaultdict(list)
    for eval_path in eval_paths:
        for row in json.loads(eval_path.read_text(encoding="utf-8"))["results"]:
            if row.get("agent") == "generalist":
                by_bench[row["benchmark"]].append(float(row["mean_nll"]))

    for bench, vals in by_bench.items():
        adapter_mean.setdefault(bench, {})["generalist"] = statistics.mean(vals)
        adapter_std.setdefault(bench, {})["generalist"] = (
            statistics.pstdev(vals) if len(vals) > 1 else 0.0
        )
    return len(eval_paths)


def _pgfplots_body(
    base_nll: dict[str, float],
    adapter_mean: dict[str, dict[str, float]],
    adapter_std: dict[str, dict[str, float]],
    n_seeds: int,
    *,
    agents: list[str],
    title: str,
) -> str:
    """Build pgfplots LaTeX source."""
    xlabels = [
        "Harm",
        "Hallucination",
        "Jailbreak",
        "Privacy",
    ]
    coords_base = " ".join(f"({xlabels[i]},{base_nll[b]:.3f})" for i, b in enumerate(BENCHMARK_ORDER))

    plots = [
        f"\\addplot+[fill=gray!50, draw=black!50] coordinates {{{coords_base}}};",
        r"\addlegendentry{Base}",
    ]
    colors = {
        "clone-0": "blue!65!black",
        "clone-1": "red!75!black",
        "clone-2": "red!55",
        "clone-3": "blue!45",
        "clone-4": "green!55!black",
        "clone-5": "green!35",
        "generalist": "purple!70!black",
    }
    for agent in agents:
        pts = " ".join(
            f"({xlabels[i]},{adapter_mean[b][agent]:.3f})"
            for i, b in enumerate(BENCHMARK_ORDER)
        )
        errs = ") (".join(f"{adapter_std[b][agent]:.3f}" for b in BENCHMARK_ORDER)
        plots.append(
            f"\\addplot+[fill={colors.get(agent, 'gray!60')}, draw=black!40, "
            f"error bars/.cd, y explicit, "
            f"error values={{({errs})}}] coordinates {{{pts}}};"
        )
        plots.append(f"\\addlegendentry{{{agent}}}")

    body = "\n".join(plots)
    return f"""\\documentclass[tikz,border=3pt]{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\begin{{document}}
\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=12.5cm, height=6.5cm,
  ybar, bar width=7pt,
  enlarge x limits=0.15,
  ymin=0, ymax=4.0,
  ylabel={{Mean token NLL (lower is better)}},
  symbolic x coords={{{','.join(xlabels)}}},
  xtick=data,
  legend style={{font=\\small}},
  legend columns=3,
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
        "--sweep-root",
        default="results/ai4privacy_fixed_theory_specialists_r40",
        help="Sweep directory with eval JSON files.",
    )
    parser.add_argument(
        "--generalist-root",
        default="results/ai4privacy_fixed_theory_generalist_r40",
        help="Generalist sweep directory with eval JSON files.",
    )
    parser.add_argument(
        "--output-stem",
        default="scripts/figures/ai4privacy_fixed_specialists_vs_base_generalist",
        help="Output path without extension.",
    )
    parser.add_argument(
        "--agents",
        default=None,
        help="Comma-separated adapter order. Defaults to agents found in eval files.",
    )
    parser.add_argument(
        "--round-label",
        default="round 39",
        help="Human-readable round label for the figure title.",
    )
    args = parser.parse_args(argv)

    sweep_root = Path(args.sweep_root)
    base_nll, adapter_mean, adapter_std = _load_aggregate(sweep_root)
    n_seeds = len(list(sweep_root.glob("seed*/eval_final_round/eval_results.json")))
    generalist_root = Path(args.generalist_root)
    n_generalist = 0
    if generalist_root.exists():
        n_generalist = _merge_generalist(
            adapter_mean,
            adapter_std,
            generalist_root,
        )
    requested_agents = (
        [part.strip() for part in args.agents.split(",") if part.strip()]
        if args.agents
        else None
    )
    agents = _series_agents(
        include_generalist=n_generalist > 0,
        adapter_mean=adapter_mean,
        requested_agents=requested_agents,
    )
    title = (
        rf"Fixed-theory specialists vs base vs generalist "
        rf"({args.round_label}, {n_seeds} specialist seeds"
        + (rf", {n_generalist} generalist seeds" if n_generalist else "")
        + ")"
    )

    plot_benchmark_nll_comparison(
        BENCHMARK_ORDER,
        BENCHMARK_LABELS,
        base_nll,
        adapter_mean,
        adapter_std,
        agents=agents,
        title=title,
        output_stem=args.output_stem,
    )

    out_stem = Path(args.output_stem)

    tex_path = out_stem.with_suffix(".tex")
    tex_path.write_text(
        _pgfplots_body(
            base_nll,
            adapter_mean,
            adapter_std,
            n_seeds,
            agents=agents,
            title=title,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_stem}.pdf")
    print(f"wrote {out_stem}.png")
    print(f"wrote {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
