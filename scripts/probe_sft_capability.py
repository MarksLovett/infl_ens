"""Probe whether closed-loop SFT actually changes model capability.

Example::

    python scripts/probe_sft_capability.py \\
        --run-dir results/safety_truth_n4_r10_strategic \\
        --output-stem scripts/figures/probe_strategic_seed0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import json

from infl_ens.evaluation.capability_probe import (
    cross_batch_margin,
    probe_run,
    write_probe_csv,
)
from infl_ens.vis.capability_probe import plot_probe

FIGS_DIR = Path(__file__).resolve().parent / "figures"


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    :returns: Configured parser.
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="Probe SFT capability change via train-loss curves and cross-perplexity.",
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--base-sft-dir", type=Path, default=None)
    p.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--rounds", type=int, nargs="*", default=None)
    p.add_argument("--max-prompts", type=int, default=64)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--forward-batch-size", type=int, default=8)
    p.add_argument("--output-stem", type=Path, default=None)
    p.add_argument("--title", type=str, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional CLI argument vector.
    :type argv: list[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    args = _build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    base_sft = (args.base_sft_dir or (run_dir / "agents")).resolve()

    records = probe_run(
        run_dir,
        base_sft_dir=base_sft,
        base_model=args.base_model,
        rounds=args.rounds,
        max_prompts_per_batch=args.max_prompts,
        max_seq_length=args.max_seq_length,
        forward_batch_size=args.forward_batch_size,
    )
    if not records:
        print(
            "error: probe produced no records (no per-round adapters found?)",
            file=sys.stderr,
        )
        return 1

    with (run_dir / "history.json").open(encoding="utf-8") as fh:
        history = json.load(fh)
    names = list(history[0]["positions"].keys())
    margins = cross_batch_margin(records)

    print(f"\n{'round':>6} {'diag NLL':>12} {'off NLL':>12} {'margin':>10}")
    print("-" * 44)
    for r in sorted(margins):
        m = margins[r]
        print(
            f"{r:>6} {m['diag_mean']:>12.4f} {m['off_mean']:>12.4f} "
            f"{m['margin']:>10.4f}",
        )
    last_r = max(margins)
    print(f"\nFinal-round margin = {margins[last_r]['margin']:.4f}")
    if margins[last_r]["margin"] < 0.005:
        print("  → SFT-driven specialisation is negligible (< 0.5% of an NLL unit).")
    elif margins[last_r]["margin"] < 0.05:
        print("  → SFT-driven specialisation is small but non-zero.")
    else:
        print("  → SFT-driven specialisation is substantial.")

    stem = args.output_stem or (FIGS_DIR / f"probe_{run_dir.name}")
    csv_path = stem.with_suffix(".csv")
    write_probe_csv(records, csv_path)
    print(f"\nwrote {csv_path}")

    plot_probe(records, history, names, title=args.title, output_stem=stem)
    for ext in ("pdf", "png"):
        print(f"wrote {stem.with_suffix('.' + ext)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
