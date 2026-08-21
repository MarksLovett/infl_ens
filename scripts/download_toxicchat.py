"""Download ToxicChat into ``data/toxicchat/`` as CSV.

ToxicChat is hosted on HuggingFace as ``lmsys/toxic-chat`` (config
``toxicchat0124``). This script tries the ``datasets`` library first and
writes one CSV per split, matching the column layout the offline loader in
:mod:`infl_ens.data.benchmarks.toxicchat` expects
(``conv_id, user_input, model_output, human_annotation, toxicity,
jailbreaking, openai_moderation``).

Usage::

    python scripts/download_toxicchat.py --output data/toxicchat --splits train test

Per AGENTS.md §3 / §4 rule 1, this is a *one-off download*: it lives in
``scripts/``, not in the package, and is invoked manually rather than
through ``python -m infl_ens``.

.. note::

   ToxicChat is released under ``cc-by-nc-4.0`` (non-commercial) and is a
   gated dataset on HuggingFace. You may need to accept the dataset terms
   and authenticate (``huggingface-cli login``) before this download
   succeeds.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

#: HuggingFace dataset id and default config.
TOXICCHAT_REPO = "lmsys/toxic-chat"
TOXICCHAT_CONFIG = "toxicchat0124"

#: Column order written to disk; matches the upstream CSV schema and the
#: offline loader's expectations.
TOXICCHAT_COLUMNS: tuple[str, ...] = (
    "conv_id",
    "user_input",
    "model_output",
    "human_annotation",
    "toxicity",
    "jailbreaking",
    "openai_moderation",
)


def _download_split_via_datasets(
    output_dir: Path,
    split: str,
    config: str,
    max_records: int | None,
) -> int:
    """Download one ToxicChat split using ``datasets.load_dataset``.

    :param output_dir: Where to write the CSV.
    :type output_dir: pathlib.Path
    :param split: Split name, e.g. ``'train'`` or ``'test'``.
    :type split: str
    :param config: Dataset config name, e.g. ``'toxicchat0124'``.
    :type config: str
    :param max_records: Optional cap on the number of records.
    :type max_records: int | None
    :returns: Number of records written.
    :rtype: int
    """
    from datasets import load_dataset

    ds = load_dataset(TOXICCHAT_REPO, config, split=split)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"toxic-chat_annotation_{split}.csv"
    n = 0
    with out_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(TOXICCHAT_COLUMNS))
        writer.writeheader()
        for row in ds:
            if max_records is not None and n >= max_records:
                break
            writer.writerow({k: row.get(k, "") for k in TOXICCHAT_COLUMNS})
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
    parser = argparse.ArgumentParser(description="Download ToxicChat.")
    parser.add_argument("--output", type=str, default="data/toxicchat",
                        help="Output directory.")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        help="Splits to download (default: train test).")
    parser.add_argument("--config", type=str, default=TOXICCHAT_CONFIG,
                        help=f"Dataset config (default: {TOXICCHAT_CONFIG}).")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Optional cap on records per split.")
    args = parser.parse_args(argv)
    out = Path(args.output)
    try:
        for split in args.splits:
            _download_split_via_datasets(
                out, split, args.config, args.max_records,
            )
    except ImportError:
        print(
            "error: the `datasets` library is required.\n"
            "Install with: pip install datasets",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # pragma: no cover - network / gating errors
        print(f"error downloading ToxicChat: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
