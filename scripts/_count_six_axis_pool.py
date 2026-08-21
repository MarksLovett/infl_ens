"""Print per-benchmark prompt counts for a six-axis router config."""

from __future__ import annotations

import argparse
from pathlib import Path

from infl_ens.training.__main__ import _load_splits, _load_yaml


def main() -> None:
    """Print axis counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/benchmark/router/six_axis_safety.yaml",
    )
    args = parser.parse_args()
    splits = _load_splits(_load_yaml(Path(args.config)))
    total = sum(s.n for s in splits)
    print(f"total_pool={total}")
    for s in splits:
        share = 100.0 * s.n / total
        print(f"{s.axis_name:18s} {s.name:18s} n={s.n:5d} share={share:5.1f}%")


if __name__ == "__main__":
    main()
