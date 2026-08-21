"""Download prompt-injection data into ``data/prompt_injection/``.

Supports:

- ``threat_matrix`` (default): ``neuralchemy/prompt-injection-Threat-Matrix``
  binary config — 32k rows, capped to ``--max-records`` (default 5000).
- ``deepset``: legacy ``deepset/prompt-injections`` (~662 rows).

Writes ``prompt_injection.jsonl`` with ``text`` and ``label`` fields for the
offline loader in :mod:`infl_ens.data.benchmarks.prompt_injection`.

Usage::

    python scripts/download_prompt_injection.py --output data/prompt_injection
    python scripts/download_prompt_injection.py --source deepset

Per AGENTS.md §3 / §4 rule 1, this is a *one-off download*: it lives in
``scripts/``, not in the package, and is invoked manually.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

THREAT_MATRIX_REPO = "neuralchemy/prompt-injection-Threat-Matrix"
THREAT_MATRIX_CONFIG = "binary"
DEESET_REPO = "deepset/prompt-injections"


def _row_to_record(row: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    """Normalize one HF row to ``{text, label, ...}``.

    :param row: Raw dataset row.
    :type row: dict
    :param source: Dataset identifier for provenance.
    :type source: str
    :returns: JSON-serializable record or ``None`` if unparsable.
    :rtype: dict | None
    """
    text = row.get("text") or row.get("prompt")
    if not text:
        return None
    label = row.get("label")
    if label is None:
        label = row.get("binary_label")
    if label is None:
        label = row.get("injection")
    if label is None:
        return None
    if isinstance(label, str):
        positive = label.strip().lower() in ("1", "true", "yes", "injection", "malicious", "jailbreak")
        label_int = 1 if positive else 0
    else:
        label_int = 1 if int(label) == 1 else 0
    return {
        "text": str(text),
        "label": label_int,
        "source": source,
    }


def _download_threat_matrix(
    output_dir: Path,
    *,
    max_records: int,
    seed: int,
) -> int:
    """Download Neuralchemy Threat Matrix (binary config).

    :param output_dir: Output directory.
    :type output_dir: pathlib.Path
    :param max_records: Record cap after shuffling all splits.
    :type max_records: int
    :param seed: Shuffle seed for reproducible subsampling.
    :type seed: int
    :returns: Number of records written.
    :rtype: int
    """
    from datasets import concatenate_datasets, load_dataset

    ds = load_dataset(THREAT_MATRIX_REPO, THREAT_MATRIX_CONFIG)
    parts = [ds[split_name] for split_name in ds]
    combined = concatenate_datasets(parts)
    indices = list(range(len(combined)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_records < len(indices):
        indices = indices[:max_records]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "prompt_injection.jsonl"
    n = 0
    with out_file.open("w", encoding="utf-8") as fh:
        for idx in indices:
            row = combined[int(idx)]
            rec = _row_to_record(row, source=THREAT_MATRIX_REPO)
            if rec is None:
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    manifest = {
        "source": THREAT_MATRIX_REPO,
        "config": THREAT_MATRIX_CONFIG,
        "n_records": n,
        "max_records": max_records,
        "seed": seed,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {n} records to {out_file}")
    return n


def _download_deepset(output_dir: Path, max_records: int | None) -> int:
    """Download ``deepset/prompt-injections``.

    :param output_dir: Output directory.
    :type output_dir: pathlib.Path
    :param max_records: Optional record cap.
    :type max_records: int | None
    :returns: Number of records written.
    :rtype: int
    """
    from datasets import load_dataset

    ds = load_dataset(DEESET_REPO)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "prompt_injection.jsonl"
    n = 0
    with out_file.open("w", encoding="utf-8") as fh:
        for split_name in ds:
            for row in ds[split_name]:
                if max_records is not None and n >= max_records:
                    break
                rec = _row_to_record(row, source=DEESET_REPO)
                if rec is None:
                    continue
                rec["hf_split"] = split_name
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            if max_records is not None and n >= max_records:
                break
    manifest = {
        "source": DEESET_REPO,
        "n_records": n,
        "max_records": max_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {n} records to {out_file}")
    return n


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    :param argv: Argument vector.
    :type argv: Sequence[str] | None
    :returns: Exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description="Download prompt-injection data.")
    parser.add_argument("--output", type=str, default="data/prompt_injection")
    parser.add_argument(
        "--source",
        choices=("threat_matrix", "deepset"),
        default="threat_matrix",
        help="HF dataset source (default: threat_matrix).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=5000,
        help="Cap records (default 5000 for threat_matrix; use 0 for all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed when subsampling threat_matrix.",
    )
    args = parser.parse_args(argv)
    max_records = None if args.max_records <= 0 else args.max_records
    try:
        if args.source == "threat_matrix":
            cap = max_records if max_records is not None else 5000
            _download_threat_matrix(
                Path(args.output),
                max_records=cap,
                seed=args.seed,
            )
        else:
            _download_deepset(Path(args.output), max_records)
    except ImportError:
        print(
            "error: the `datasets` library is required.\n"
            "Install with: pip install datasets",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"error downloading prompt-injection data: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
