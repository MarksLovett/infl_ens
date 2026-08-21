#!/usr/bin/env python3
"""Write seven-axis eval pgfplots bar chart (.tex only, no matplotlib)."""

from __future__ import annotations

import json
from pathlib import Path

BENCHMARK_ORDER = [
    "beavertails", "halueval", "toxicchat", "ai4privacy",
    "orbench", "prompt_injection", "do_not_answer",
]
LABELS = {
    "beavertails": "Harm",
    "halueval": "Hallucination",
    "toxicchat": "Jailbreak",
    "ai4privacy": "Privacy",
    "orbench": "Over-refusal",
    "prompt_injection": "Injection",
    "do_not_answer": "Policy",
}
AGENTS = [
    "merge-harm", "merge-hallucination", "merge-privacy", "merge-injection",
    "merge-overrefusal", "merge-policy", "merge-generalist", "pooled-baseline",
]
COLORS = {
    "merge-harm": "blue!65!black",
    "merge-hallucination": "red!75!black",
    "merge-privacy": "green!55!black",
    "merge-injection": "orange!80!black",
    "merge-overrefusal": "purple!70!black",
    "merge-policy": "brown!70!black",
    "merge-generalist": "cyan!60!black",
    "pooled-baseline": "black!70",
}
NAMES = {
    "merge-harm": "Harm pair",
    "merge-hallucination": "Hallucination pair",
    "merge-privacy": "Privacy pair",
    "merge-injection": "Injection pair",
    "merge-overrefusal": "Over-refusal pair",
    "merge-policy": "Policy pair",
    "merge-generalist": "Generalist pair",
    "pooled-baseline": "Pooled baseline",
}


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    out = {b: {} for b in BENCHMARK_ORDER}
    for row in json.loads(path.read_text(encoding="utf-8"))["results"]:
        if row["benchmark"] in out:
            out[row["benchmark"]][row["agent"]] = float(row["mean_nll"])
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    merge = load_scores(
        root / "results/seven_axis_pair_merge_r40/seed0/eval_final_round/eval_results.json",
    )
    base = load_scores(
        root / "results/seven_axis_baseline_replay_r40/seed0/eval_final_round/eval_results.json",
    )
    scores = {b: {} for b in BENCHMARK_ORDER}
    for b in BENCHMARK_ORDER:
        for a in AGENTS[:-1]:
            if a in merge[b]:
                scores[b][a] = merge[b][a]
        if "pooled-baseline" in base[b]:
            scores[b]["pooled-baseline"] = base[b]["pooled-baseline"]

    plots = []
    for agent in AGENTS:
        pts = " ".join(
            f"({LABELS[b]},{scores[b][agent]:.3f})"
            for b in BENCHMARK_ORDER if agent in scores[b]
        )
        plots.append(
            f"\\addplot+[fill={COLORS[agent]}, draw=black!40] coordinates {{{pts}}};"
        )
        plots.append(f"\\addlegendentry{{{NAMES[agent]}}}")

    xcoords = ",".join(LABELS[b] for b in BENCHMARK_ORDER)
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
  ymin=0, ymax=3.5,
  ylabel={{Mean token NLL (lower is better)}},
  symbolic x coords={{{xcoords}}},
  xtick=data,
  x tick label style={{font=\\scriptsize, align=center}},
  legend style={{font=\\scriptsize}},
  legend columns=2,
  ymajorgrids=true,
  grid style={{dashed, gray!35}},
  title={{Seven-axis pair-merge specialists vs pooled baseline (round 39, seed 0)}},
]
{body}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{document}}
"""
    out = root / "scripts/figures/seven_axis_merge_vs_baseline.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
