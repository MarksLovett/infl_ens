"""Download OR-Bench into ``data/orbench/`` as CSV files.

OR-Bench is hosted on HuggingFace as ``orbench-llm/or-bench``. This script
writes one CSV per config (``or-bench-80k.csv``, ``or-bench-hard-1k.csv``,
``or-bench-toxic.csv``) matching the offline loader in
:mod:`infl_ens.data.benchmarks.orbench`.

Usage::

    python scripts/download_orbench.py --output data/orbench

Per AGENTS.md §3 / §4 rule 1, this is a *one-off download*: it lives in
``scripts/``, not in the package, and is invoked manually.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

ORBENCH_REPO = "orbench-llm/or-bench"
ORBENCH_CONFIGS: tuple[str, ...] = (
    "or-bench-80k",
    "or-bench-hard-1k",
    "or-bench-toxic",
)


def _download_config(output_dir: Path, config: str, max_records: int | None) -> int:
    """Download one OR-Bench config to CSV.

    :param output_dir: Output directory.
    :type output_dir: pathlib.Path
    :param config: Dataset config name.
    :type config: str
    :param max_records: Optional record cap.
    :type max_records: int | None
    :returns: Number of records written.
    :rtype: int
    """
    from datasets import load_dataset

    ds = load_dataset(ORBENCH_REPO, config, split="train")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{config}.csv"
    n = 0
    with out_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["prompt", "category"])
        writer.writeheader()
        for row in ds:
            if max_records is not None and n >= max_records:
                break
            writer.writerow(
                {
                    "prompt": row.get("prompt", ""),
                    "category": row.get("category", ""),
                }
            )
            n += 1
    print(f"wrote {n} records to {out_file}")
    return n


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    :param argv: Argument vector.
    :type argv: Sequence[str] | None
    :returns: Exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description="Download OR-Bench.")
    parser.add_argument("--output", type=str, default="data/orbench")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(ORBENCH_CONFIGS),
        help=f"Configs to download (default: {' '.join(ORBENCH_CONFIGS)}).",
    )
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args(argv)
    out = Path(args.output)
    try:
        for config in args.configs:
            _download_config(out, config, args.max_records)
    except ImportError:
        print(
            "error: the `datasets` library is required.\n"
            "Install with: pip install datasets",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"error downloading OR-Bench: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
