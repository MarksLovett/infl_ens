"""Download Do-Not-Answer and benign negatives into ``data/do_not_answer/``.

Do-Not-Answer is hosted on HuggingFace as ``LibrAI/do-not-answer``. This
script writes:

- ``do_not_answer.jsonl`` — refusal-worthy prompts (score ``1.0``).
- ``benign_negatives.jsonl`` — sampled Alpaca instructions (score ``0.0``).

The offline loader in :mod:`infl_ens.data.benchmarks.do_not_answer` mixes
both files when building the policy-violation axis.

Usage::

    python scripts/download_do_not_answer.py --output data/do_not_answer

Per AGENTS.md §3 / §4 rule 1, this is a *one-off download*: it lives in
``scripts/``, not in the package, and is invoked manually.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

DO_NOT_ANSWER_REPO = "LibrAI/do-not-answer"
BENIGN_REPO = "yahma/alpaca-cleaned"


def _download_dna(output_dir: Path) -> int:
    """Write Do-Not-Answer prompts to JSONL.

    :param output_dir: Output directory.
    :type output_dir: pathlib.Path
    :returns: Number of records written.
    :rtype: int
    """
    from datasets import load_dataset

    ds = load_dataset(DO_NOT_ANSWER_REPO, split="train")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "do_not_answer.jsonl"
    n = 0
    with out_file.open("w", encoding="utf-8") as fh:
        for row in ds:
            question = row.get("question") or row.get("instruction")
            if not question:
                continue
            fh.write(
                json.dumps(
                    {
                        "question": question,
                        "risk_area": row.get("risk_area", ""),
                        "types_of_harm": row.get("types_of_harm", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(f"wrote {n} Do-Not-Answer records to {out_file}")
    return n


def _download_benign(output_dir: Path, n_benign: int, seed: int) -> int:
    """Write benign Alpaca instructions to JSONL.

    :param output_dir: Output directory.
    :type output_dir: pathlib.Path
    :param n_benign: Number of benign prompts to sample.
    :type n_benign: int
    :param seed: Shuffle seed.
    :type seed: int
    :returns: Number of records written.
    :rtype: int
    """
    from datasets import load_dataset

    ds = load_dataset(BENIGN_REPO, split="train").shuffle(seed=seed)
    out_file = output_dir / "benign_negatives.jsonl"
    n = 0
    with out_file.open("w", encoding="utf-8") as fh:
        for row in ds:
            if n >= n_benign:
                break
            instruction = row.get("instruction")
            if not instruction:
                continue
            fh.write(
                json.dumps({"instruction": instruction}, ensure_ascii=False)
                + "\n"
            )
            n += 1
    print(f"wrote {n} benign records to {out_file}")
    return n


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    :param argv: Argument vector.
    :type argv: Sequence[str] | None
    :returns: Exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description="Download Do-Not-Answer data.")
    parser.add_argument("--output", type=str, default="data/do_not_answer")
    parser.add_argument(
        "--n-benign",
        type=int,
        default=5000,
        help="Number of benign Alpaca negatives to sample.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out = Path(args.output)
    try:
        _download_dna(out)
        _download_benign(out, args.n_benign, args.seed)
    except ImportError:
        print(
            "error: the `datasets` library is required.\n"
            "Install with: pip install datasets",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"error downloading Do-Not-Answer: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
