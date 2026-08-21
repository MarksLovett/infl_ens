"""Download JailbreakBench JBB-Behaviors into ``data/jbb_behaviors/``.

JBB-Behaviors is hosted on HuggingFace as ``JailbreakBench/JBB-Behaviors``
(config ``behaviors``). This script writes ``harmful_behaviors.csv`` and
``benign_behaviors.csv`` matching the offline loader in
:mod:`infl_ens.data.benchmarks.jbb_behaviors`.

Usage::

    python scripts/download_jbb_behaviors.py --output data/jbb_behaviors
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

JBB_REPO = "JailbreakBench/JBB-Behaviors"
JBB_CONFIG = "behaviors"
HARMFUL_LOCAL = "harmful_behaviors.csv"
BENIGN_LOCAL = "benign_behaviors.csv"


def _write_split(rows: object, path: Path) -> None:
    """Write a HuggingFace split to CSV."""
    fieldnames = list(rows.column_names)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"wrote {path} ({len(rows)} rows)")


def _download_with_datasets(output: Path) -> None:
    """Fetch JBB CSVs via the HuggingFace ``datasets`` library."""
    from datasets import load_dataset

    output.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(JBB_REPO, JBB_CONFIG)
    _write_split(ds["harmful"], output / HARMFUL_LOCAL)
    _write_split(ds["benign"], output / BENIGN_LOCAL)


def _download_with_hf_hub(output: Path) -> None:
    """Fetch JBB CSVs via ``huggingface_hub``."""
    from huggingface_hub import hf_hub_download

    output.mkdir(parents=True, exist_ok=True)
    for remote, local in (
        ("data/harmful-behaviors.csv", HARMFUL_LOCAL),
        ("data/benign-behaviors.csv", BENIGN_LOCAL),
    ):
        cached = hf_hub_download(
            repo_id=JBB_REPO,
            filename=remote,
            repo_type="dataset",
        )
        dest = output / local
        dest.write_bytes(Path(cached).read_bytes())
        print(f"wrote {dest}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/jbb_behaviors"),
        help="Output directory for JBB CSV files.",
    )
    args = parser.parse_args(argv)

    try:
        _download_with_datasets(args.output)
    except Exception as ds_err:
        print(f"datasets download failed ({ds_err}); trying hf_hub...", file=sys.stderr)
        try:
            _download_with_hf_hub(args.output)
        except Exception as hub_err:
            print(f"hf_hub_download failed: {hub_err}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
