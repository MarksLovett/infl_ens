"""Measure one LoRA adapter on the safety benchmarks and print a table.

This is the quick-look companion to ``scripts/eval_base_model.py``: rather
than editing an evaluation YAML, point it at a single adapter directory and
it reports mean per-token NLL on each benchmark axis (harm, hallucination,
jailbreak). It is the canonical way to *measure* an adapter on BeaverTails
and HaluEval (and now ToxicChat) from the command line.

The metric is identical to the rest of the eval stack
(:func:`infl_ens.evaluation.metrics.mean_token_nll`), so numbers are
directly comparable to ``eval_base_model.py`` and ``eval_results.json``.

Example::

    python scripts/measure_adapter_on_benchmarks.py \\
        --adapter-dir results/<run>/agents/clone-0 \\
        --beavertails data/beavertails/30k_train.jsonl \\
        --halueval    data/halueval \\
        --toxicchat   data/toxicchat \\
        --max-eval-records 256

Any benchmark whose path is omitted is simply skipped, so this also works
for measuring BeaverTails / HaluEval alone. Per AGENTS.md §3 / §4 rule 1,
this is a thin convenience wrapper; the scoring logic lives in
:mod:`infl_ens.evaluation`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from infl_ens.evaluation.benchmarks import load_benchmark_splits
from infl_ens.evaluation.evaluate import (
    AdapterEvalConfig,
    evaluate_adapter_on_splits,
)


def _build_benchmark_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Assemble the ``benchmarks`` config list from CLI paths.

    Only benchmarks whose path was supplied are included, so the same
    script measures one, two, or all three axes.

    :param args: Parsed CLI arguments.
    :type args: argparse.Namespace
    :returns: List of benchmark config entries for
        :func:`infl_ens.evaluation.benchmarks.load_benchmark_splits`.
    :rtype: list[dict]
    """
    entries: list[dict[str, Any]] = []
    if args.beavertails:
        entries.append(
            {"kind": "beavertails", "path": args.beavertails,
             "max_records": None}
        )
    if args.halueval:
        entries.append(
            {"kind": "halueval", "path": args.halueval,
             "tasks": list(args.halueval_tasks), "max_records": None}
        )
    if args.toxicchat:
        entries.append(
            {"kind": "toxicchat", "path": args.toxicchat,
             "score_mode": args.toxicchat_score_mode,
             "human_annotated_only": args.toxicchat_human_only,
             "max_records": None}
        )
    if args.ai4privacy:
        entries.append(
            {"kind": "ai4privacy", "path": args.ai4privacy,
             "score_mode": args.ai4privacy_score_mode,
             "english_only": True, "max_records": None}
        )
    return entries


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    :param argv: Optional argument vector.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    :raises SystemExit: If no benchmark paths are provided.
    """
    parser = argparse.ArgumentParser(
        description="Measure one adapter on the safety benchmarks.",
    )
    parser.add_argument("--adapter-dir", required=True,
                        help="Path to the LoRA checkpoint directory.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace base-model id.")
    parser.add_argument("--beavertails", default=None,
                        help="BeaverTails JSONL file or directory (harm).")
    parser.add_argument("--halueval", default=None,
                        help="HaluEval file or directory (hallucination).")
    parser.add_argument("--halueval-tasks", nargs="+",
                        default=["qa", "dialogue"],
                        help="HaluEval task subset.")
    parser.add_argument("--toxicchat", default=None,
                        help="ToxicChat CSV file or directory (jailbreak).")
    parser.add_argument("--toxicchat-score-mode", default="jailbreaking",
                        choices=["jailbreaking", "toxicity", "either"],
                        help="ToxicChat axis label.")
    parser.add_argument("--toxicchat-human-only", action="store_true",
                        help="Keep only human-annotated ToxicChat rows.")
    parser.add_argument("--ai4privacy", default=None,
                        help="AI4Privacy JSONL file or directory (privacy).")
    parser.add_argument("--ai4privacy-score-mode", default="density",
                        choices=["density", "binary"],
                        help="AI4Privacy axis scoring mode.")
    parser.add_argument("--max-seq-length", type=int, default=1024,
                        help="Tokenizer truncation length.")
    parser.add_argument("--forward-batch-size", type=int, default=8,
                        help="Inference batch size.")
    parser.add_argument("--max-eval-records", type=int, default=256,
                        help="Per-benchmark subsample cap (None-able via -1).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Subsampling seed.")
    parser.add_argument("--output-dir", default=None,
                        help="Optional dir to also write a JSON report.")
    args = parser.parse_args(argv)

    entries = _build_benchmark_entries(args)
    if not entries:
        parser.error(
            "supply at least one of --beavertails / --halueval / "
            "--toxicchat / --ai4privacy"
        )

    splits = load_benchmark_splits(entries)
    cfg = AdapterEvalConfig(
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        forward_batch_size=args.forward_batch_size,
        max_eval_records=(
            None if args.max_eval_records is not None
            and args.max_eval_records < 0 else args.max_eval_records
        ),
        seed=args.seed,
    )
    results = evaluate_adapter_on_splits(args.adapter_dir, splits, cfg)

    width = max(len(r.axis_name) for r in results) + 2
    print(f"\nadapter: {args.adapter_dir}")
    print(f"{'axis':<{width}}{'benchmark':<14}"
          f"{'mean_nll':>10}{'examples':>10}{'tokens':>10}")
    for r in results:
        print(f"{r.axis_name:<{width}}{r.benchmark:<14}"
              f"{r.mean_nll:>10.4f}{r.n_examples:>10}{r.n_tokens:>10}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "adapter_axis_scores.json"
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "meta": {
                "adapter_dir": args.adapter_dir,
                "base_model": args.base_model,
                "benchmarks": entries,
                "max_eval_records": cfg.max_eval_records,
                "seed": args.seed,
            },
            "results": [asdict(r) for r in results],
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
