#!/usr/bin/env python3
"""Write oracle vs pooled-generalist vs learned-specialist routing figure (.tex).

Reads ``routing_weight_comparison.json`` from a seven-axis (or attribution)
route-then-score run and emits a standalone pgfplots figure with:

* a flat-pool headline panel (oracle / pooled / learned), and
* a per-benchmark grouped-bar panel.

Usage::

    python scripts/write_oracle_routing_tex.py
    python scripts/write_oracle_routing_tex.py \\
        --input results/seven_axis_pair_merge_split/seed0/routing_weight_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Display order for the per-benchmark panel (skip tiny JBB probe set).
BENCHMARK_ORDER = [
    "beavertails",
    "halueval",
    "ai4privacy",
    "orbench",
    "prompt_injection",
    "do_not_answer",
]
LABELS = {
    "beavertails": "Harm",
    "halueval": "Hallucination",
    "ai4privacy": "Privacy",
    "orbench": "Over-refusal",
    "prompt_injection": "Injection",
    "do_not_answer": "Policy",
}


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def build_tex(report: dict) -> str:
    """Build standalone pgfplots TeX from a routing comparison report.

    :param report: Loaded ``routing_weight_comparison.json``.
    :type report: dict
    :returns: Standalone LaTeX source.
    :rtype: str
    """
    flat = report["flat"]
    pooled = float(flat["pooled_nll"])
    learned = float(flat["learned_routing_expected_nll"])
    oracle = float(flat["oracle_routing_nll"])
    n_prompts = int(flat.get("n_prompts", 0))
    round_idx = flat.get("round", "?")

    per = report["per_benchmark"]
    benches = [b for b in BENCHMARK_ORDER if b in per]
    xcoords = ",".join(LABELS[b] for b in benches)

    def series(key: str) -> str:
        return " ".join(
            f"({LABELS[b]},{_fmt(float(per[b][key]))})" for b in benches
        )

    oracle_pts = series("oracle_nll")
    pooled_pts = series("pooled_nll")
    learned_pts = series("learned_expected_nll")

    ymax_flat = max(pooled, learned, oracle) * 1.18
    ymax_bench = (
        max(
            float(per[b][k])
            for b in benches
            for k in ("oracle_nll", "pooled_nll", "learned_expected_nll")
        )
        * 1.12
    )
    caption = (
        f"Seven-axis pair-merge routing diagnostic (round {round_idx}): "
        f"oracle {_fmt(oracle)}, pooled {_fmt(pooled)}, learned {_fmt(learned)}."
    )

    return rf"""\documentclass[tikz,border=3pt]{{standalone}}
\usepackage{{pgfplots}}
\usepackage{{amsmath}}
\usetikzlibrary{{calc}}
\pgfplotsset{{compat=1.18}}
\begin{{document}}
\begin{{tikzpicture}}

% ---- Left: flat-pool headline ----
\begin{{axis}}[
  name=flat,
  width=5.2cm, height=6.2cm,
  ybar=0pt,
  bar width=16pt,
  xmin=-0.55, xmax=2.55,
  ymin=1.85, ymax={ymax_flat:.2f},
  ylabel={{Mean token NLL (lower is better)}},
  xtick={{0,1,2}},
  xticklabels={{Oracle,Pooled,Learned}},
  x tick label style={{font=\small, align=center}},
  ymajorgrids=true,
  grid style={{dashed, gray!35}},
  title={{Flat pool ($n={n_prompts}$)}},
  nodes near coords,
  every node near coord/.append style={{font=\scriptsize}},
]
\addplot[fill=teal!70!black, draw=black!40] coordinates {{(0,{_fmt(oracle)})}};
\addplot[fill=black!55, draw=black!40] coordinates {{(1,{_fmt(pooled)})}};
\addplot[fill=blue!65!black, draw=black!40] coordinates {{(2,{_fmt(learned)})}};
\end{{axis}}

% ---- Right: per-benchmark breakdown ----
\begin{{axis}}[
  at={{(flat.east)}},
  anchor=west,
  xshift=1.4cm,
  width=11.5cm, height=6.2cm,
  ybar, bar width=7pt,
  enlarge x limits=0.12,
  ymin=0, ymax={ymax_bench:.2f},
  ylabel={{}},
  symbolic x coords={{{xcoords}}},
  xtick=data,
  x tick label style={{font=\scriptsize, align=center}},
  legend style={{at={{(0.02,0.98)}}, anchor=north west, font=\small}},
  legend columns=1,
  ymajorgrids=true,
  grid style={{dashed, gray!35}},
  title={{Per-benchmark route-then-score}},
]
\addplot+[fill=teal!70!black, draw=black!40] coordinates {{{oracle_pts}}};
\addlegendentry{{Oracle (argmin specialist)}}
\addplot+[fill=black!55, draw=black!40] coordinates {{{pooled_pts}}};
\addlegendentry{{Pooled generalist}}
\addplot+[fill=blue!65!black, draw=black!40] coordinates {{{learned_pts}}};
\addlegendentry{{Learned specialists ($G$)}}
\end{{axis}}

\node[font=\small\itshape, align=center, text=black!70]
  at ($(flat.south)!0.5!(current axis.south)-(0,1.15cm)$)
  {{{caption}}};

\end{{tikzpicture}}
\end{{document}}
"""


def main() -> None:
    """CLI entry point.

    :returns: ``None``.
    :rtype: None
    """
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root
        / "results/seven_axis_pair_merge_split/seed0/routing_weight_comparison.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "scripts/figures/oracle_vs_generalist_vs_specialists.tex",
    )
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    tex = build_tex(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
