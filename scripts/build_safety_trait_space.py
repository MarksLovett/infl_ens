"""Build and cache the BeaverTails + HaluEval trait space.

Convenience wrapper around ``python -m infl_ens.data build-safety-trait-space``
with the defaults used for the safety+truthfulness experiment. Saves both a
pickle (consumed by training) and a JSON summary of axis labels and grid
shape for human inspection.

Usage::

    python scripts/build_safety_trait_space.py \\
        --beavertails data/beavertails/30k_train.jsonl \\
        --halueval    data/halueval \\
        --output      results/safety_truth/trait_space.pkl

Per AGENTS.md §3 / §4 rule 1, this script is a thin convenience wrapper;
the actual logic lives inside ``infl_ens.data``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

# Make src/ importable when invoked directly from the repo root.
ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

from infl_ens.data.__main__ import main as data_main


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    :param argv: Argument vector.
    :type argv: Sequence[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Build the BeaverTails + HaluEval trait space.",
    )
    parser.add_argument("--beavertails", type=str, required=True)
    parser.add_argument("--halueval", type=str, required=True)
    parser.add_argument(
        "--encoder", type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--n-grid", type=int, default=32)
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument(
        "--output", type=str,
        default="results/safety_truth/trait_space.pkl",
    )
    args = parser.parse_args(argv)

    forwarded = [
        "build-safety-trait-space",
        "--beavertails", args.beavertails,
        "--halueval", args.halueval,
        "--encoder", args.encoder,
        "--n-grid", str(args.n_grid),
        "--max-records", str(args.max_records),
        "--output", args.output,
    ]
    rc = data_main(forwarded)
    if rc != 0:
        return rc

    summary_path = Path(args.output).with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump({"output": args.output, "config": vars(args)}, fh, indent=2)
    print(f"summary written to {summary_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
