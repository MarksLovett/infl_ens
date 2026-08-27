"""Shared figure style and benchmark display constants.

Every matplotlib figure in :mod:`infl_ens.figures` calls
:func:`apply_paper_style` so the whole set reads as one family; the pgfplots
writers use the same benchmark order and labels.
"""

from __future__ import annotations

#: Benchmark ids in axis (config) order.
BENCHMARK_ORDER: tuple[str, ...] = (
    "beavertails",
    "halueval",
    "jbb_behaviors",
    "ai4privacy",
    "orbench",
    "prompt_injection",
    "do_not_answer",
)

#: Display label per benchmark id.
BENCHMARK_LABELS: dict[str, str] = {
    "beavertails": "Harm",
    "halueval": "Hallucination",
    "jbb_behaviors": "Jailbreak",
    "ai4privacy": "Privacy",
    "orbench": "Over-refusal",
    "prompt_injection": "Injection",
    "do_not_answer": "Policy",
}

#: Benchmarks drawn in the pgfplots per-benchmark panels. JBB-Behaviors is a
#: tiny probe set whose bar would be dominated by noise, so it is skipped.
PGF_BENCHMARK_ORDER: tuple[str, ...] = tuple(b for b in BENCHMARK_ORDER if b != "jbb_behaviors")

#: rcParams shared by every matplotlib figure.
PAPER_RC: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
}


def apply_paper_style() -> None:
    """Apply :data:`PAPER_RC` to matplotlib (Computer Modern math text)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(PAPER_RC)


__all__ = [
    "BENCHMARK_LABELS",
    "BENCHMARK_ORDER",
    "PAPER_RC",
    "PGF_BENCHMARK_ORDER",
    "apply_paper_style",
]
