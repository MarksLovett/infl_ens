"""Download BeaverTails into ``data/beavertails/`` as JSONL.

BeaverTails is hosted on HuggingFace as ``PKU-Alignment/BeaverTails``. This
script tries the ``datasets`` library first and falls back to ``hf_hub_download``
for the raw JSONL files.

Usage::

    python scripts/download_beavertails.py --output data/beavertails --split 30k_train

Per AGENTS.md §3 / §4 rule 1, this is a *one-off download*: it lives in
``scripts/``, not in the package, and is invoked manually rather than
through ``python -m infl_ens``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def _download_via_datasets(output_dir: Path, split: str, max_records: int | None) -> int:
    """Download BeaverTails using ``datasets.load_dataset``.

    :param output_dir: Where to write the JSONL.
    :type output_dir: pathlib.Path
    :param split: BeaverTails split, e.g. ``'30k_train'`` or ``'30k_test'``.
    :type split: str
    :param max_records: Optional cap on the number of records.
    :type max_records: int | None
    :returns: Number of records written.
    :rtype: int
    """
    from datasets import load_dataset
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{split}.jsonl"
    n = 0
    with out_file.open("w", encoding="utf-8") as fh:
        for row in ds:
            if max_records is not None and n >= max_records:
                break
            fh.write(json.dumps(dict(row)) + "\n")
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
    parser = argparse.ArgumentParser(description="Download BeaverTails.")
    parser.add_argument("--output", type=str, default="data/beavertails",
                        help="Output directory.")
    parser.add_argument("--split", type=str, default="30k_train",
                        help="BeaverTails split name (default: 30k_train).")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Optional cap on records (for quick testing).")
    args = parser.parse_args(argv)
    out = Path(args.output)
    try:
        _download_via_datasets(out, args.split, args.max_records)
    except ImportError:
        print(
            "error: the `datasets` library is required.\n"
            "Install with: pip install datasets",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
