#!/usr/bin/env python3
"""Cross-arm bar figure: oracle vs pooled generalist vs each specialist arm (.tex).

Companion to :mod:`scripts.write_oracle_routing_tex`, which draws ONE
specialist arm against the oracle and the generalist. This one overlays
SEVERAL arms so the routing modes can be read against each other and against
the shared bounds.

Each ``--report`` is a routing diagnostic JSON
(``routing_ensemble_diagnostics.json`` or ``routing_weight_comparison.json``,
same schema, produced by :mod:`scripts.routing_ensemble_diagnostics`). Every
report contributes its ``learned_routing_expected_nll``; the oracle and pooled
bars are drawn once per arm too, since the oracle ceiling depends on that
arm's specialists.

Emits a standalone pgfplots document with:

* a flat-pool panel, grouped by arm, and
* a per-benchmark panel of each arm's learned routing, with the generalist
  drawn as a dashed reference line.

Example::

    python scripts/write_arm_comparison_tex.py \\
        --report "Soft top-3=results/seven_axis_soft_topk3_pairs/seed0/routing_ensemble_diagnostics.json" \\
        --report "Hard (SFT)=results/seven_axis_hard_pairs_matched/seed0/routing_ensemble_diagnostics.json" \\
        --output scripts/figures/three_arm/arm_comparison.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

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
# Colour ramp for the specialist arms (oracle/pooled have fixed colours that
# match write_oracle_routing_tex.py, so the figures read as one family).
ARM_COLORS = [
    "blue!65!black",
    "orange!80!black",
    "purple!70!black",
    "green!55!black",
]


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _tex_escape(text: str) -> str:
    """Escape the LaTeX specials that plausibly appear in an arm label."""
    for char, repl in (
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
        ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(char, repl)
    return text


def build_tex(arms: list[tuple[str, dict[str, Any]]]) -> str:
    """Build the standalone cross-arm pgfplots document.

    :param arms: ``(label, report)`` pairs in display order.
    :type arms: list[tuple[str, dict]]
    :returns: Standalone LaTeX source.
    :rtype: str
    :raises ValueError: If no arm has usable per-benchmark data.
    """
    flats = []
    for label, report in arms:
        flat = report["flat"]
        flats.append({
            "label": label,
            "oracle": float(flat["oracle_routing_nll"]),
            "pooled": float(flat["pooled_nll"]),
            "learned": float(flat["learned_routing_expected_nll"]),
            "n": int(flat.get("n_prompts", 0)),
            "round": flat.get("round", "?"),
        })

    benches = [
        b for b in BENCHMARK_ORDER
        if all(b in report.get("per_benchmark", {}) for _l, report in arms)
    ]
    if not benches:
        raise ValueError("no benchmark is present in every report")
    xcoords = ",".join(LABELS[b] for b in benches)

    arm_xticks = ",".join(str(i) for i in range(len(flats)))
    arm_labels = ",".join(_tex_escape(f["label"]) for f in flats)
    oracle_bars = " ".join(f"({i},{_fmt(f['oracle'])})" for i, f in enumerate(flats))
    pooled_bars = " ".join(f"({i},{_fmt(f['pooled'])})" for i, f in enumerate(flats))
    learned_bars = " ".join(f"({i},{_fmt(f['learned'])})" for i, f in enumerate(flats))

    all_flat = [v for f in flats for v in (f["oracle"], f["pooled"], f["learned"])]
    lo, hi = min(all_flat), max(all_flat)
    span = max(hi - lo, 1e-3)
    ymin_flat = max(0.0, lo - 2.5 * span)
    ymax_flat = hi + 1.9 * span

    # Per-benchmark: one series per arm (learned), generalist as a reference.
    pooled_ref = {
        b: sum(float(r["per_benchmark"][b]["pooled_nll"]) for _l, r in arms)
        / len(arms)
        for b in benches
    }
    # Series order mirrors write_oracle_routing_tex.py: bounds first (oracle,
    # pooled), then one learned series per arm, so the two figures read the
    # same way left to right.
    series_lines = []
    oracle_pts = " ".join(
        f"({LABELS[b]},{_fmt(min(float(r['per_benchmark'][b]['oracle_nll']) for _l, r in arms))})"
        for b in benches
    )
    series_lines.append(
        f"\\addplot+[fill=teal!70!black, draw=black!40] coordinates {{{oracle_pts}}};\n"
        f"\\addlegendentry{{Oracle (best arm)}}"
    )
    pooled_pts = " ".join(f"({LABELS[b]},{_fmt(pooled_ref[b])})" for b in benches)
    series_lines.append(
        f"\\addplot+[fill=black!55, draw=black!40] coordinates {{{pooled_pts}}};\n"
        f"\\addlegendentry{{Pooled generalist}}"
    )
    for idx, (label, report) in enumerate(arms):
        pts = " ".join(
            f"({LABELS[b]},{_fmt(float(report['per_benchmark'][b]['learned_expected_nll']))})"
            for b in benches
        )
        color = ARM_COLORS[idx % len(ARM_COLORS)]
        series_lines.append(
            f"\\addplot+[fill={color}, draw=black!40] coordinates {{{pts}}};\n"
            f"\\addlegendentry{{{_tex_escape(label)} (learned $G$)}}"
        )
    plots = "\n".join(series_lines)

    ymax_bench = max(
        [float(r["per_benchmark"][b][k])
         for _l, r in arms for b in benches
         for k in ("oracle_nll", "pooled_nll", "learned_expected_nll")]
    ) * 1.12

    best = min(flats, key=lambda f: f["learned"])
    caption = (
        f"Cross-arm route-then-score on the held-out pool "
        f"($n={flats[0]['n']}$, round {flats[0]['round']}). "
        f"Best specialist arm: {_tex_escape(best['label'])} at "
        f"{_fmt(best['learned'])} vs pooled {_fmt(best['pooled'])} "
        f"and oracle {_fmt(best['oracle'])}."
    )

    return rf"""\documentclass[tikz,border=3pt]{{standalone}}
\usepackage{{pgfplots}}
\usepackage{{amsmath}}
\usetikzlibrary{{calc}}
\pgfplotsset{{compat=1.18}}
\begin{{document}}
\begin{{tikzpicture}}

% ---- Left: flat-pool headline, grouped by arm ----
\begin{{axis}}[
  name=flat,
  width=6.4cm, height=6.2cm,
  ybar, bar width=9pt,
  enlarge x limits=0.35,
  ymin={ymin_flat:.3f}, ymax={ymax_flat:.3f},
  ylabel={{Mean token NLL (lower is better)}},
  xtick={{{arm_xticks}}},
  xticklabels={{{arm_labels}}},
  x tick label style={{font=\scriptsize, align=center}},
  ymajorgrids=true,
  grid style={{dashed, gray!35}},
  title={{Flat pool}},
  legend style={{at={{(0.5,-0.16)}}, anchor=north, font=\scriptsize, draw=none}},
  legend columns=3,
]
\addplot[fill=teal!70!black, draw=black!40] coordinates {{{oracle_bars}}};
\addlegendentry{{Oracle}}
\addplot[fill=black!55, draw=black!40] coordinates {{{pooled_bars}}};
\addlegendentry{{Pooled}}
\addplot[fill=blue!65!black, draw=black!40] coordinates {{{learned_bars}}};
\addlegendentry{{Learned}}
\end{{axis}}

% ---- Right: per-benchmark, one series per arm ----
\begin{{axis}}[
  at={{(flat.east)}},
  anchor=west,
  xshift=1.5cm,
  width=11.5cm, height=6.2cm,
  ybar, bar width=6pt,
  enlarge x limits=0.12,
  ymin=0, ymax={ymax_bench:.2f},
  symbolic x coords={{{xcoords}}},
  xtick=data,
  x tick label style={{font=\scriptsize, align=center}},
  legend style={{at={{(1.02,0.5)}}, anchor=west, font=\scriptsize}},
  legend columns=1,
  ymajorgrids=true,
  grid style={{dashed, gray!35}},
  title={{Per-benchmark route-then-score}},
]
{plots}
\end{{axis}}

\node[font=\small\itshape, align=center, text=black!70]
  at ($(flat.south)!0.5!(current axis.south)-(0,2.35cm)$)
  {{{caption}}};

\end{{tikzpicture}}
\end{{document}}
"""


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", action="append", required=True, metavar="LABEL=PATH",
        help="Arm label and routing JSON; repeat once per arm.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "scripts/figures/three_arm/arm_comparison.tex",
    )
    args = parser.parse_args()

    arms: list[tuple[str, dict[str, Any]]] = []
    for spec in args.report:
        if "=" not in spec:
            raise SystemExit(f"--report must be LABEL=PATH, got {spec!r}")
        label, raw = spec.split("=", 1)
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        arms.append((label, json.loads(path.read_text(encoding="utf-8"))))

    tex = build_tex(arms)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
