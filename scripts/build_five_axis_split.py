#!/usr/bin/env python3
"""Build 70/10/20 split manifest for the five-axis collapse config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infl_ens.data.splits import (  # noqa: E402
    build_split_manifest,
    choose_exact_train_coverage,
    save_split_manifest,
)
from infl_ens.training.__main__ import _load_splits, _load_yaml  # noqa: E402


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmark/router/seven_axis_collapse_dead_axes.yaml",
    )
    parser.add_argument(
        "--output",
        default="data/splits/five_axis_seed0.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override data_split.seed / top-level seed",
    )
    args = parser.parse_args()

    cfg = _load_yaml(ROOT / args.config)
    splits = _load_splits(cfg)
    ds = dict(cfg.get("data_split") or {})
    seed = int(args.seed if args.seed is not None else ds.get("seed", cfg.get("seed", 0)))
    manifest = build_split_manifest(
        splits,
        train_frac=float(ds.get("train_frac", 0.7)),
        val_frac=float(ds.get("val_frac", 0.1)),
        test_frac=float(ds.get("test_frac", 0.2)),
        seed=seed,
    )
    batch_size, n_rounds = choose_exact_train_coverage(
        manifest.n_train,
        preferred_batch_sizes=tuple(
            int(x) for x in ds.get(
                "preferred_batch_sizes",
                [3500, 2450, 1225, 980, 700, 350],
            )
        ),
        min_rounds=int(ds.get("min_rounds", 5)),
        max_rounds=int(ds.get("max_rounds", 50)),
        target_n_rounds=ds.get("target_n_rounds"),
    )
    manifest = build_split_manifest(
        splits,
        train_frac=float(ds.get("train_frac", 0.7)),
        val_frac=float(ds.get("val_frac", 0.1)),
        test_frac=float(ds.get("test_frac", 0.2)),
        seed=seed,
        meta={
            "config": args.config,
            "n_train": manifest.n_train,
            "n_val": manifest.n_val,
            "n_test": manifest.n_test,
            "batch_size": batch_size,
            "n_rounds": n_rounds,
            "per_benchmark": {
                b.name: {
                    "n_train": len(b.train),
                    "n_val": len(b.val),
                    "n_test": len(b.test),
                    "n_total": len(b.train) + len(b.val) + len(b.test),
                }
                for b in manifest.benchmarks
            },
        },
    )
    out = ROOT / args.output
    save_split_manifest(manifest, out)
    print(json.dumps({"manifest": str(out), "batch_size": batch_size, "n_rounds": n_rounds}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
