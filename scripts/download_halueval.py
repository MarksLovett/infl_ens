"""Download HaluEval task files into ``data/halueval/``.

HaluEval is released on GitHub at ``RUCAIBox/HaluEval`` with task-specific
JSON files (``qa_data.json``, ``dialogue_data.json``,
``summarization_data.json``, ``general_data.json``). This script fetches
the requested files via HTTPS.

Usage::

    python scripts/download_halueval.py --output data/halueval --tasks qa dialogue

Per AGENTS.md §3 / §4 rule 1, this is a *one-off download*: it lives in
``scripts/``, not in the package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence
from urllib.request import urlopen

#: Base URL for the HaluEval data files on the official GitHub repository.
HALUEVAL_BASE = (
    "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/"
)

#: Task identifier → filename mapping used upstream.
TASK_FILES: dict[str, str] = {
    "qa": "qa_data.json",
    "dialogue": "dialogue_data.json",
    "summarization": "summarization_data.json",
    "general": "general_data.json",
}


def _download(task: str, output_dir: Path) -> Path:
    """Fetch one HaluEval task file via HTTPS.

    :param task: Task identifier from :data:`TASK_FILES`.
    :type task: str
    :param output_dir: Output directory.
    :type output_dir: pathlib.Path
    :returns: Path to the downloaded file.
    :rtype: pathlib.Path
    :raises KeyError: If ``task`` is not a recognised task identifier.
    """
    fname = TASK_FILES[task]
    url = HALUEVAL_BASE + fname
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / fname
    with urlopen(url) as resp, dest.open("wb") as fh:
        fh.write(resp.read())
    print(f"downloaded {url} → {dest}")
    return dest


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    :param argv: Argument vector.
    :type argv: Sequence[str] | None
    :returns: Exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description="Download HaluEval task files.")
    parser.add_argument("--output", type=str, default="data/halueval")
    parser.add_argument("--tasks", nargs="+", default=["qa"],
                        choices=list(TASK_FILES.keys()),
                        help="Which HaluEval tasks to download.")
    args = parser.parse_args(argv)
    out = Path(args.output)
    for task in args.tasks:
        try:
            _download(task, out)
        except Exception as exc:  # pragma: no cover - network errors
            print(f"error downloading {task}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
