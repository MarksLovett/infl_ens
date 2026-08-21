#!/usr/bin/env python3
"""Delete intermediate per-round LoRA checkpoints, keeping only the final round.

Usage::

    python scripts/prune_final_round_adapters.py results/
    python scripts/prune_final_round_adapters.py results/ --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from infl_ens.utils.checkpoints import prune_intermediate_adapters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path, help="Path to results/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.results_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    stats = prune_intermediate_adapters(root, dry_run=args.dry_run)
    print(f"agents scanned       : {stats['agents_scanned']}")
    print(f"round dirs removed   : {stats['round_dirs_removed']}")
    print(f"adapters before/after: {stats['adapters_before']} / {stats['adapters_after']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
