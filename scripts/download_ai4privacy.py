"""Download AI4Privacy PII-masking data into ``data/ai4privacy/``.

AI4Privacy ``pii-masking-200k`` is hosted on HuggingFace as
``ai4privacy/pii-masking-200k`` (``apache-2.0``) and ships per-language
JSONL shards (``english_pii_43k.jsonl``, ``french_pii_62k.jsonl``, ...).
This script tries the ``datasets`` library first and writes one JSONL file
that the offline loader in :mod:`infl_ens.data.benchmarks.ai4privacy`
consumes, preserving the ``source_text`` / ``target_text`` /
``privacy_mask`` fields the density scorer needs.

Usage::

    python scripts/download_ai4privacy.py --output data/ai4privacy --max-records 50000

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

#: HuggingFace dataset id.
AI4PRIVACY_REPO = "ai4privacy/pii-masking-200k"

#: Fields the offline loader expects to find in each written record. The
#: native HF export uses ``source_text`` / ``target_text``; the loader also
#: accepts the Isotonic-mirror aliases, but we normalise to the native names
#: on download so the on-disk file is canonical.
_PREFERRED_FIELDS: tuple[str, ...] = (
    "source_text",
    "target_text",
    "privacy_mask",
    "span_labels",
    "language",
)


def _row_to_record(row: dict) -> dict:
    """Normalise one dataset row to the canonical on-disk record.

    Accepts both native (``source_text``/``target_text``) and mirror
    (``unmasked_text``/``masked_text``) key names and emits native names.

    :param row: Raw dataset row.
    :type row: dict
    :returns: Record dict containing the preferred fields when present.
    :rtype: dict
    """
    out: dict = {}
    src = row.get("source_text") or row.get("unmasked_text")
    tgt = row.get("target_text") or row.get("masked_text")
    if src is not None:
        out["source_text"] = src
    if tgt is not None:
        out["target_text"] = tgt
    for key in ("privacy_mask", "span_labels", "language"):
        if key in row and row[key] is not None:
            out[key] = row[key]
    return out


def _download_via_datasets(
    output_dir: Path,
    language: str,
    max_records: int | None,
) -> int:
    """Download AI4Privacy using ``datasets.load_dataset``.

    :param output_dir: Where to write the JSONL.
    :type output_dir: pathlib.Path
    :param language: Language code to keep (filtered on the ``language``
        column when present), e.g. ``'en'``. Use ``'all'`` to keep every
        language.
    :type language: str
    :param max_records: Optional cap on the number of records written.
    :type max_records: int | None
    :returns: Number of records written.
    :rtype: int
    """
    from datasets import load_dataset

    ds = load_dataset(AI4PRIVACY_REPO, split="train")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "en" if language == "en" else language
    out_file = output_dir / f"english_pii.jsonl" if language == "en" \
        else output_dir / f"{suffix}_pii.jsonl"
    n = 0
    with out_file.open("w", encoding="utf-8") as fh:
        for row in ds:
            if language != "all":
                lang = str(row.get("language", "")).lower()
                # Some shards encode language as 'English' rather than 'en'.
                if lang and lang not in (language, "english"
                                         if language == "en" else language):
                    continue
            rec = _row_to_record(dict(row))
            if "source_text" not in rec:
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if max_records is not None and n >= max_records:
                break
    print(f"wrote {n} records to {out_file}")
    return n


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    :param argv: Argument vector.
    :type argv: Sequence[str] | None
    :returns: Exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description="Download AI4Privacy PII data.")
    parser.add_argument("--output", type=str, default="data/ai4privacy",
                        help="Output directory.")
    parser.add_argument("--language", type=str, default="en",
                        help="Language to keep ('en' default, 'all' for every "
                             "language).")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Optional cap on records written.")
    args = parser.parse_args(argv)
    out = Path(args.output)
    try:
        _download_via_datasets(out, args.language, args.max_records)
    except ImportError:
        print(
            "error: the `datasets` library is required.\n"
            "Install with: pip install datasets",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # pragma: no cover - network errors
        print(f"error downloading AI4Privacy: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
