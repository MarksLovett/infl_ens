"""Standalone pgfplots figures from route-then-score reports.

Two writers share one visual family:

- :func:`oracle_routing_tex`: ONE specialist arm against the oracle
  ceiling and the pooled generalist (flat-pool headline + per-benchmark
  panel).
- :func:`arm_comparison_tex`: SEVERAL arms overlaid so the routing modes
  can be read against each other and against the shared bounds.

Both consume the ``routing_ensemble_diagnostics.json`` written by the
pipeline routing stage (:func:`infl_ens.evaluation.routing_eval.report_to_dict`).
:func:`compile_tex` turns the ``.tex`` into a PDF when ``latexmk`` is on
the path.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

from infl_ens.figures.style import BENCHMARK_LABELS, PGF_BENCHMARK_ORDER

#: Colour ramp for specialist arms (oracle/pooled have fixed colours so the
#: single-arm and cross-arm figures read as one family).
ARM_COLORS: tuple[str, ...] = (
    "blue!65!black",
    "orange!80!black",
    "purple!70!black",
    "green!55!black",
    "red!60!black",
    "cyan!60!black",
)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def tex_escape(text: str) -> str:
    """Escape the LaTeX specials that plausibly appear in a label.

    :param text: Raw label.
    :type text: str
    :returns: Escaped label.
    :rtype: str
    """
    for char, repl in (
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
        ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(char, repl)
    return text


def oracle_routing_tex(
    report: dict[str, Any],
    *,
    experiment_label: str = "Specialist arm",
) -> str:
    """Build the single-arm oracle / pooled / learned figure.

    :param report: Loaded routing diagnostic JSON.
    :type report: dict
    :param experiment_label: Concise label identifying the specialist arm.
    :type experiment_label: str
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
    benches = [b for b in PGF_BENCHMARK_ORDER if b in per]
    xcoords = ",".join(BENCHMARK_LABELS[b] for b in benches)

    def series(key: str) -> str:
        return " ".join(
            f"({BENCHMARK_LABELS[b]},{_fmt(float(per[b][key]))})" for b in benches
        )

    oracle_pts = series("oracle_nll")
    pooled_pts = series("pooled_nll")
    learned_pts = series("learned_expected_nll")

    # Derive the flat-panel window from the data rather than a fixed floor.
    lo, hi = min(pooled, learned, oracle), max(pooled, learned, oracle)
    span = max(hi - lo, 1e-3)
    ymin_flat = max(0.0, lo - 2.5 * span)
    ymax_flat = hi + 1.6 * span
    ymax_bench = (
        max(
            float(per[b][k])
            for b in benches
            for k in ("oracle_nll", "pooled_nll", "learned_expected_nll")
        )
        * 1.12
        if benches else 1.0
    )
    caption = (
        f"{tex_escape(experiment_label)} routing diagnostic (round {round_idx}): "
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
  ymin={ymin_flat:.3f}, ymax={ymax_flat:.3f},
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
  legend style={{at={{(1.02,0.5)}}, anchor=west, font=\small}},
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


def arm_comparison_tex(arms: Sequence[tuple[str, dict[str, Any]]]) -> str:
    """Build the cross-arm oracle / pooled / learned figure.

    :param arms: ``(label, report)`` pairs in display order.
    :type arms: Sequence[tuple[str, dict]]
    :returns: Standalone LaTeX source.
    :rtype: str
    :raises ValueError: If no arm is given or no benchmark is present in
        every report.
    """
    if not arms:
        raise ValueError("arm_comparison_tex needs at least one arm")
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
        b for b in PGF_BENCHMARK_ORDER
        if all(b in report.get("per_benchmark", {}) for _l, report in arms)
    ]
    if not benches:
        raise ValueError("no benchmark is present in every report")
    xcoords = ",".join(BENCHMARK_LABELS[b] for b in benches)

    arm_xticks = ",".join(str(i) for i in range(len(flats)))
    arm_labels = ",".join(tex_escape(f["label"]) for f in flats)
    oracle_bars = " ".join(f"({i},{_fmt(f['oracle'])})" for i, f in enumerate(flats))
    pooled_bars = " ".join(f"({i},{_fmt(f['pooled'])})" for i, f in enumerate(flats))
    learned_bars = " ".join(f"({i},{_fmt(f['learned'])})" for i, f in enumerate(flats))

    all_flat = [v for f in flats for v in (f["oracle"], f["pooled"], f["learned"])]
    lo, hi = min(all_flat), max(all_flat)
    span = max(hi - lo, 1e-3)
    ymin_flat = max(0.0, lo - 2.5 * span)
    ymax_flat = hi + 1.9 * span

    pooled_ref = {
        b: sum(float(r["per_benchmark"][b]["pooled_nll"]) for _l, r in arms) / len(arms)
        for b in benches
    }
    series_lines = []
    oracle_pts = " ".join(
        f"({BENCHMARK_LABELS[b]},"
        f"{_fmt(min(float(r['per_benchmark'][b]['oracle_nll']) for _l, r in arms))})"
        for b in benches
    )
    series_lines.append(
        f"\\addplot+[fill=teal!70!black, draw=black!40] coordinates {{{oracle_pts}}};\n"
        f"\\addlegendentry{{Oracle (best arm)}}"
    )
    pooled_pts = " ".join(f"({BENCHMARK_LABELS[b]},{_fmt(pooled_ref[b])})" for b in benches)
    series_lines.append(
        f"\\addplot+[fill=black!55, draw=black!40] coordinates {{{pooled_pts}}};\n"
        f"\\addlegendentry{{Pooled generalist}}"
    )
    for idx, (label, report) in enumerate(arms):
        pts = " ".join(
            f"({BENCHMARK_LABELS[b]},"
            f"{_fmt(float(report['per_benchmark'][b]['learned_expected_nll']))})"
            for b in benches
        )
        color = ARM_COLORS[idx % len(ARM_COLORS)]
        series_lines.append(
            f"\\addplot+[fill={color}, draw=black!40] coordinates {{{pts}}};\n"
            f"\\addlegendentry{{{tex_escape(label)} (learned $G$)}}"
        )
    plots = "\n".join(series_lines)

    ymax_bench = max(
        float(r["per_benchmark"][b][k])
        for _l, r in arms for b in benches
        for k in ("oracle_nll", "pooled_nll", "learned_expected_nll")
    ) * 1.12

    best = min(flats, key=lambda f: f["learned"])
    caption = (
        f"Cross-arm route-then-score on the held-out pool "
        f"($n={flats[0]['n']}$, round {flats[0]['round']}). "
        f"Best specialist arm: {tex_escape(best['label'])} at "
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


def latexmk_available() -> bool:
    """Whether ``latexmk`` is on the path.

    :returns: ``True`` when a TeX toolchain can compile the figures.
    :rtype: bool
    """
    return shutil.which("latexmk") is not None


def compile_tex(path: Path, *, clean: bool = True) -> bool:
    """Compile a standalone ``.tex`` figure to PDF with ``latexmk``.

    :param path: The ``.tex`` file.
    :type path: pathlib.Path
    :param clean: Remove auxiliary files afterwards.
    :type clean: bool
    :returns: ``True`` on success, ``False`` when latexmk is missing or
        fails (the ``.tex`` is left in place either way).
    :rtype: bool
    """
    if not latexmk_available():
        return False
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", path.name]
    try:
        proc = subprocess.run(
            cmd, cwd=path.parent, capture_output=True, text=True, check=False,
        )
    except OSError:
        return False
    if clean:
        subprocess.run(["latexmk", "-c", path.name], cwd=path.parent, capture_output=True, check=False)
    return proc.returncode == 0


__all__ = [
    "ARM_COLORS",
    "arm_comparison_tex",
    "compile_tex",
    "latexmk_available",
    "oracle_routing_tex",
    "tex_escape",
]
