"""Evaluate a base (non-adapter) model on benchmark corpora.

Example::

    python scripts/eval_base_model.py \\
        --output-dir results/base_eval_qwen2_5_1_5b \\
        --max-eval-records 128
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from infl_ens.evaluation.base_eval import evaluate_base_model, write_base_eval_report


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate a base model on BeaverTails and HaluEval.",
    )
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--forward-batch-size", type=int, default=8)
    parser.add_argument("--max-eval-records", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/base_model_eval")
    args = parser.parse_args(argv)

    benchmarks = [
        {"kind": "beavertails", "path": "data/beavertails/30k_train.jsonl", "max_records": None},
        {"kind": "halueval", "path": "data/halueval", "tasks": ["qa", "dialogue"], "max_records": None},
    ]
    results, meta = evaluate_base_model(
        benchmarks,
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        forward_batch_size=args.forward_batch_size,
        max_eval_records=args.max_eval_records,
        seed=args.seed,
    )
    out_path = write_base_eval_report(Path(args.output_dir), results, meta)
    print(f"wrote {out_path}")
    for r in results:
        print(
            f"{r.benchmark}: mean_nll={r.mean_nll:.4f} "
            f"({r.n_examples} examples, {r.n_tokens} tokens)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
