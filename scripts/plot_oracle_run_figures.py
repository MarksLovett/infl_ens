#!/usr/bin/env python3
"""Generate all diagnostic plots for the seven-axis oracle / pair-merge run.

Produces under ``scripts/figures/oracle_run/``:

* ``specialist_pair_nll`` — every merge specialist × benchmark (test partition)
* ``specialist_vs_pooled`` — dedicated specialist vs pooled Δ bars
* ``final_positions`` — 7-D positions projected onto each axis pair, colored by merge
* ``data_split`` — train / val / test counts used by the run
* ``oracle_support`` — oracle winner counts by merge × benchmark
* ``merge_prompt_counts`` — final-round routed prompts per merge LoRA

Example::

    python scripts/plot_oracle_run_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from infl_ens.vis.save import save_figure

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
AXIS_NAMES = [
    "harm",
    "hallucination",
    "jailbreak",
    "privacy",
    "overrefusal",
    "injection",
    "policy",
]
MERGE_ORDER = [
    "merge-harm",
    "merge-hallucination",
    "merge-privacy",
    "merge-injection",
    "merge-overrefusal",
    "merge-policy",
    "merge-generalist",
]
MERGE_COLORS = {
    "merge-harm": "#1f4e79",
    "merge-hallucination": "#c0392b",
    "merge-privacy": "#1e8449",
    "merge-injection": "#d35400",
    "merge-overrefusal": "#6c3483",
    "merge-policy": "#7b241c",
    "merge-generalist": "#148f77",
}
MERGE_LABELS = {
    "merge-harm": "Harm pair",
    "merge-hallucination": "Hallucination pair",
    "merge-privacy": "Privacy pair",
    "merge-injection": "Injection pair",
    "merge-overrefusal": "Over-refusal pair",
    "merge-policy": "Policy pair",
    "merge-generalist": "Generalist / jailbreak pair",
}
# Fixed merge groups from seven_axis_pair_merge_split.yaml
MERGE_GROUPS = {
    "merge-overrefusal": ["clone-0", "clone-13"],
    "merge-generalist": ["clone-1", "clone-9"],  # trained as merge-jailbreak
    "merge-policy": ["clone-2", "clone-8"],
    "merge-harm": ["clone-3", "clone-6"],
    "merge-injection": ["clone-4", "clone-10"],
    "merge-hallucination": ["clone-5", "clone-11"],
    "merge-privacy": ["clone-7", "clone-12"],
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )


def _load_eval_matrix(path: Path) -> dict[str, dict[str, float]]:
    """Load eval JSON into ``benchmark -> agent -> mean_nll``.

    :param path: ``eval_results.json`` path.
    :type path: pathlib.Path
    :returns: Nested NLL table.
    :rtype: dict[str, dict[str, float]]
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {b: {} for b in BENCHMARK_ORDER}
    for row in payload["results"]:
        b = row["benchmark"]
        if b in out:
            out[b][row["agent"]] = float(row["mean_nll"])
    return out


def _pooled_from_summary(path: Path) -> dict[str, float]:
    """Read pooled-baseline test NLLs from specialist summary JSON.

    :param path: ``specialist_vs_pooled_summary.json``.
    :type path: pathlib.Path
    :returns: Benchmark -> pooled NLL.
    :rtype: dict[str, float]
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["benchmark"]: float(row["baseline_nll"])
        for row in payload.get("test_rows", [])
    }


def plot_specialist_pair_nll(
    scores: dict[str, dict[str, float]],
    pooled: dict[str, float],
    *,
    output_stem: Path,
    title: str,
) -> Figure:
    """Grouped bars: all merge specialists (+ pooled) × benchmarks.

    :param scores: Benchmark -> agent -> NLL.
    :type scores: dict
    :param pooled: Benchmark -> pooled NLL.
    :type pooled: dict
    :param output_stem: Save stem.
    :type output_stem: pathlib.Path
    :param title: Figure title.
    :type title: str
    :returns: Figure.
    :rtype: matplotlib.figure.Figure
    """
    _setup_style()
    agents = [a for a in MERGE_ORDER if any(a in scores[b] for b in BENCHMARK_ORDER)]
    if pooled:
        agents = agents + ["pooled-baseline"]
    x = np.arange(len(BENCHMARK_ORDER))
    width = 0.8 / max(len(agents), 1)
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    for i, agent in enumerate(agents):
        if agent == "pooled-baseline":
            vals = [pooled.get(b, np.nan) for b in BENCHMARK_ORDER]
            color = "#444444"
            label = "Pooled baseline"
        else:
            vals = [scores[b].get(agent, np.nan) for b in BENCHMARK_ORDER]
            color = MERGE_COLORS[agent]
            label = MERGE_LABELS[agent]
        ax.bar(
            x + (i - (len(agents) - 1) / 2) * width,
            vals,
            width=width * 0.95,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_LABELS[b] for b in BENCHMARK_ORDER])
    ax.set_ylabel(r"Mean token NLL $\downarrow$")
    ax.set_ylim(0, 3.6)
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=True, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_stem)
    return fig


def plot_specialist_vs_pooled(
    summary_path: Path,
    *,
    output_stem: Path,
) -> Figure:
    """Specialist-on-own-axis vs pooled, with Δ annotations.

    :param summary_path: ``specialist_vs_pooled_summary.json``.
    :type summary_path: pathlib.Path
    :param output_stem: Save stem.
    :type output_stem: pathlib.Path
    :returns: Figure.
    :rtype: matplotlib.figure.Figure
    """
    _setup_style()
    rows = json.loads(summary_path.read_text(encoding="utf-8"))["test_rows"]
    labels = [r["axis_label"] for r in rows]
    spec = [float(r["specialist_nll"]) for r in rows]
    pool = [float(r["baseline_nll"]) for r in rows]
    deltas = [float(r["delta"]) for r in rows]
    x = np.arange(len(rows))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - width / 2, spec, width, color="#1f4e79", edgecolor="black",
           linewidth=0.4, label="Dedicated specialist pair")
    ax.bar(x + width / 2, pool, width, color="#777777", edgecolor="black",
           linewidth=0.4, label="Pooled baseline")
    for i, d in enumerate(deltas):
        y = max(spec[i], pool[i]) + 0.04
        ax.text(
            i, y, f"Δ={d:+.3f}",
            ha="center", va="bottom", fontsize=8,
            color="#1e8449" if d < 0 else "#c0392b",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Mean token NLL $\downarrow$")
    ax.set_ylim(0, max(max(spec), max(pool)) * 1.25)
    ax.set_title("Oracle-run specialist pair vs pooled (test, own axis)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, output_stem)
    return fig


def plot_final_positions(
    positions: dict[str, list[float]],
    *,
    output_stem: Path,
) -> Figure:
    """Pairwise axis projections of final clone positions, colored by merge.

    :param positions: Clone name -> 7-D position.
    :type positions: dict
    :param output_stem: Save stem.
    :type output_stem: pathlib.Path
    :returns: Figure.
    :rtype: matplotlib.figure.Figure
    """
    _setup_style()
    clone_to_merge = {
        c: m for m, members in MERGE_GROUPS.items() for c in members
    }
    pairs = [(i, j) for i in range(7) for j in range(i + 1, 7)]
    n = len(pairs)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
    axes_flat = np.asarray(axes).ravel()
    for ax, (i, j) in zip(axes_flat, pairs):
        for clone, pos in sorted(positions.items()):
            merge = clone_to_merge.get(clone, "merge-generalist")
            ax.scatter(
                pos[i], pos[j],
                s=55, c=MERGE_COLORS[merge],
                edgecolors="black", linewidths=0.4, zorder=3,
            )
        # connect merge pairs
        for merge, members in MERGE_GROUPS.items():
            if all(m in positions for m in members):
                p0 = positions[members[0]]
                p1 = positions[members[1]]
                ax.plot(
                    [p0[i], p1[i]], [p0[j], p1[j]],
                    color=MERGE_COLORS[merge], alpha=0.45, lw=1.2, zorder=2,
                )
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(AXIS_NAMES[i], fontsize=8)
        ax.set_ylabel(AXIS_NAMES[j], fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.3)
    for ax in axes_flat[n:]:
        ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MERGE_COLORS[m],
               markeredgecolor="black", markersize=8, label=MERGE_LABELS[m])
        for m in MERGE_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        "Oracle-run final positions (round 5) — pairs linked, colored by merge LoRA",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    save_figure(fig, output_stem)
    return fig


def plot_position_radar(
    positions: dict[str, list[float]],
    *,
    output_stem: Path,
) -> Figure:
    """One radar per merge pair (mean of the two co-located clones).

    :param positions: Clone -> 7-D vector.
    :type positions: dict
    :param output_stem: Save stem.
    :type output_stem: pathlib.Path
    :returns: Figure.
    :rtype: matplotlib.figure.Figure
    """
    _setup_style()
    angles = np.linspace(0, 2 * np.pi, len(AXIS_NAMES), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    fig, axes = plt.subplots(
        2, 4, figsize=(14, 7), subplot_kw={"projection": "polar"},
    )
    axes_flat = axes.ravel()
    for ax, merge in zip(axes_flat, MERGE_ORDER):
        members = MERGE_GROUPS[merge]
        vec = np.mean([positions[m] for m in members], axis=0)
        vals = np.concatenate([vec, vec[:1]])
        ax.plot(angles, vals, color=MERGE_COLORS[merge], lw=2)
        ax.fill(angles, vals, color=MERGE_COLORS[merge], alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(AXIS_NAMES, fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.set_title(MERGE_LABELS[merge], fontsize=10, pad=12)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=6)
    axes_flat[-1].axis("off")
    fig.suptitle("Merge-pair trait profiles (mean of paired clones, round 5)")
    fig.tight_layout()
    save_figure(fig, output_stem)
    return fig


def plot_data_split(meta: dict[str, Any], *, output_stem: Path) -> Figure:
    """Train / val / test counts for the oracle-run data split.

    :param meta: ``data_split.json`` ``meta`` block (or full file with meta).
    :type meta: dict
    :param output_stem: Save stem.
    :type output_stem: pathlib.Path
    :returns: Figure.
    :rtype: matplotlib.figure.Figure
    """
    _setup_style()
    per = meta.get("per_benchmark") or meta.get("meta", {}).get("per_benchmark")
    if per is None and "meta" in meta:
        per = meta["meta"]["per_benchmark"]
        meta = meta["meta"]
    benches = [b for b in BENCHMARK_ORDER if b in per]
    train = [per[b]["n_train"] for b in benches]
    val = [per[b]["n_val"] for b in benches]
    test = [per[b]["n_test"] for b in benches]
    x = np.arange(len(benches))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, train, width, label="Train", color="#1f4e79", edgecolor="black", lw=0.3)
    ax.bar(x, val, width, label="Val", color="#7f8c8d", edgecolor="black", lw=0.3)
    ax.bar(x + width, test, width, label="Test", color="#148f77", edgecolor="black", lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_LABELS[b] for b in benches])
    ax.set_ylabel("Examples")
    ax.set_title(
        f"Oracle-run data split "
        f"(train={meta.get('n_train')}, val={meta.get('n_val')}, "
        f"test={meta.get('n_test')}; batch={meta.get('batch_size')}, "
        f"rounds={meta.get('n_rounds')})"
    )
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_stem)
    return fig


def plot_oracle_support(
    routing: dict[str, Any],
    *,
    output_stem: Path,
) -> Figure:
    """Heatmap of oracle winner counts: merge × benchmark.

    :param routing: ``routing_weight_comparison.json``.
    :type routing: dict
    :param output_stem: Save stem.
    :type output_stem: pathlib.Path
    :returns: Figure.
    :rtype: matplotlib.figure.Figure
    """
    _setup_style()
    support = routing["merge_support_argmax"]
    # routing uses jbb_behaviors; map toxicchat slot to jbb if present
    route_benches = [
        "beavertails", "halueval", "jbb_behaviors", "ai4privacy",
        "orbench", "prompt_injection", "do_not_answer",
    ]
    route_labels = [
        "Harm", "Hallucination", "Jailbreak*", "Privacy",
        "Over-refusal", "Injection", "Policy",
    ]
    merges = [m for m in MERGE_ORDER if m in support]
    mat = np.zeros((len(merges), len(route_benches)))
    for i, m in enumerate(merges):
        by_b = support[m].get("by_benchmark", {})
        for j, b in enumerate(route_benches):
            mat[i, j] = float(by_b.get(b, 0))
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(route_labels)))
    ax.set_xticklabels(route_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(merges)))
    ax.set_yticklabels([MERGE_LABELS[m] for m in merges])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center",
                    fontsize=8, color="black" if mat[i, j] < mat.max() * 0.6 else "white")
    ax.set_title("Oracle winners (argmin merge NLL) by benchmark")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Prompt count")
    fig.text(
        0.5, 0.01,
        "*Jailbreak column is JBB-Behaviors (n=40) in the flat routing diagnostic.",
        ha="center", fontsize=8, style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output_stem)
    return fig


def plot_merge_prompt_counts(
    counts: dict[str, int],
    *,
    output_stem: Path,
) -> Figure:
    """Final-round routed prompt counts per merge LoRA.

    :param counts: Merge name -> prompt count.
    :type counts: dict
    :param output_stem: Save stem.
    :type output_stem: pathlib.Path
    :returns: Figure.
    :rtype: matplotlib.figure.Figure
    """
    _setup_style()
    merges = [m for m in MERGE_ORDER if m in counts]
    vals = [counts[m] for m in merges]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        np.arange(len(merges)), vals,
        color=[MERGE_COLORS[m] for m in merges],
        edgecolor="black", linewidth=0.4,
    )
    ax.set_xticks(np.arange(len(merges)))
    ax.set_xticklabels([MERGE_LABELS[m] for m in merges], rotation=20, ha="right")
    ax.set_ylabel("Routed prompts (final round)")
    ax.set_title("Oracle-run merge LoRA training mass (round 5)")
    for i, v in enumerate(vals):
        ax.text(i, v + 8, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_stem)
    return fig


def write_nll_tex(
    scores: dict[str, dict[str, float]],
    pooled: dict[str, float],
    path: Path,
    *,
    title: str,
) -> None:
    """Standalone pgfplots TeX for specialist-pair NLL bars.

    :param scores: Benchmark -> agent -> NLL.
    :type scores: dict
    :param pooled: Benchmark -> pooled NLL.
    :type pooled: dict
    :param path: Output ``.tex`` path.
    :type path: pathlib.Path
    :param title: Axis title.
    :type title: str
    """
    pgf_colors = {
        "merge-harm": "blue!65!black",
        "merge-hallucination": "red!75!black",
        "merge-privacy": "green!55!black",
        "merge-injection": "orange!80!black",
        "merge-overrefusal": "purple!70!black",
        "merge-policy": "brown!70!black",
        "merge-generalist": "cyan!60!black",
        "pooled-baseline": "black!70",
    }
    xcoords = ",".join(BENCHMARK_LABELS[b] for b in BENCHMARK_ORDER)
    plots: list[str] = []
    agents = [a for a in MERGE_ORDER if any(a in scores[b] for b in BENCHMARK_ORDER)]
    for agent in agents:
        pts = " ".join(
            f"({BENCHMARK_LABELS[b]},{scores[b][agent]:.3f})"
            for b in BENCHMARK_ORDER if agent in scores[b]
        )
        plots.append(
            f"\\addplot+[fill={pgf_colors[agent]}, draw=black!40] coordinates {{{pts}}};"
        )
        plots.append(f"\\addlegendentry{{{MERGE_LABELS[agent]}}}")
    if pooled:
        pts = " ".join(
            f"({BENCHMARK_LABELS[b]},{pooled[b]:.3f})"
            for b in BENCHMARK_ORDER if b in pooled
        )
        plots.append(
            f"\\addplot+[fill={pgf_colors['pooled-baseline']}, draw=black!40] "
            f"coordinates {{{pts}}};"
        )
        plots.append(r"\addlegendentry{Pooled baseline}")
    body = "\n".join(plots)
    tex = f"""\\documentclass[tikz,border=3pt]{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\begin{{document}}
\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=14cm, height=7cm,
  ybar, bar width=5pt,
  enlarge x limits=0.08,
  ymin=0, ymax=3.6,
  ylabel={{Mean token NLL (lower is better)}},
  symbolic x coords={{{xcoords}}},
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")


def main() -> int:
    """CLI entry point.

    :returns: Exit code.
    :rtype: int
    """
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=root / "results/seven_axis_pair_merge_split/seed0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "scripts/figures/oracle_run",
    )
    args = parser.parse_args()
    run = args.run_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    eval_path = run / "eval_test_eval_results.json"
    if not eval_path.is_file():
        eval_path = run / "eval_test" / "eval_results.json"
    scores = _load_eval_matrix(eval_path)
    pooled = _pooled_from_summary(run / "tables" / "specialist_vs_pooled_summary.json")

    title = "Seven-axis oracle run — specialist pair NLL (test, round 5)"
    plot_specialist_pair_nll(
        scores, pooled,
        output_stem=out / "specialist_pair_nll",
        title=title,
    )
    write_nll_tex(scores, pooled, out / "specialist_pair_nll.tex", title=title)

    plot_specialist_vs_pooled(
        run / "tables" / "specialist_vs_pooled_summary.json",
        output_stem=out / "specialist_vs_pooled",
    )

    traj = json.loads((run / "positions_trajectory.json").read_text(encoding="utf-8"))
    last = traj["rounds"][-1]
    positions = {k: list(map(float, v)) for k, v in last["positions"].items()}
    plot_final_positions(positions, output_stem=out / "final_positions")
    plot_position_radar(positions, output_stem=out / "merge_pair_profiles")

    # data split meta — may live only on doob; fall back to embedded summary
    split_path = run / "data_split.json"
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        plot_data_split(split, output_stem=out / "data_split")
    else:
        # reconstruct from known meta printed earlier / embed minimal
        meta = {
            "n_train": 20172,
            "n_val": 2882,
            "n_test": 5763,
            "batch_size": 3362,
            "n_rounds": 6,
            "per_benchmark": {
                "beavertails": {"n_train": 3500, "n_val": 500, "n_test": 1000},
                "halueval": {"n_train": 3500, "n_val": 500, "n_test": 1000},
                "toxicchat": {"n_train": 3500, "n_val": 500, "n_test": 1000},
                "ai4privacy": {"n_train": 3500, "n_val": 500, "n_test": 1000},
                "orbench": {"n_train": 2208, "n_val": 316, "n_test": 631},
                "prompt_injection": {"n_train": 464, "n_val": 66, "n_test": 132},
                "do_not_answer": {"n_train": 3500, "n_val": 500, "n_test": 1000},
            },
        }
        plot_data_split(meta, output_stem=out / "data_split")

    routing = json.loads(
        (run / "routing_weight_comparison.json").read_text(encoding="utf-8"),
    )
    plot_oracle_support(routing, output_stem=out / "oracle_support")

    # final-round merge prompt counts from trajectory if present, else hardcoded last dump
    counts = {
        "merge-overrefusal": 409,
        "merge-injection": 329,
        "merge-policy": 671,
        "merge-harm": 744,
        "merge-generalist": 263,
        "merge-hallucination": 489,
        "merge-privacy": 457,
    }
    counts_path = run / "merge_prompt_counts_round5.json"
    if counts_path.is_file():
        counts = json.loads(counts_path.read_text(encoding="utf-8"))
    plot_merge_prompt_counts(counts, output_stem=out / "merge_prompt_counts")

    print(f"wrote figures under {out}")
    for p in sorted(out.glob("*")):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
