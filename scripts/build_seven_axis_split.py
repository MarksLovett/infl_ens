#!/usr/bin/env python3
"""Build and persist the seven-axis 70/10/20 stratified data split."""

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


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmark/router/seven_axis_pair_merge_split.yaml",
        help="Training config whose benchmarks block defines the pool.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Manifest output path (default: data_split.manifest in config).",
    )
    args = parser.parse_args()

    cfg = _load_yaml(ROOT / args.config)
    splits = _load_splits(cfg)
    ds = dict(cfg.get("data_split") or {})
    seed = int(ds.get("seed", cfg.get("seed", 0)))
    manifest = build_split_manifest(
        splits,
        train_frac=float(ds.get("train_frac", 0.7)),
        val_frac=float(ds.get("val_frac", 0.1)),
        test_frac=float(ds.get("test_frac", 0.2)),
        seed=seed,
    )
    target_n_rounds = ds.get("target_n_rounds")
    batch_size, n_rounds = choose_exact_train_coverage(
        manifest.n_train,
        preferred_batch_sizes=tuple(
            int(x) for x in ds.get(
                "preferred_batch_sizes",
                [4900, 2450, 1225, 980, 700, 350],
            )
        ),
        target_n_rounds=(
            int(target_n_rounds) if target_n_rounds is not None else None
        ),
        min_rounds=int(ds.get("min_rounds", 5)),
        max_rounds=int(ds.get("max_rounds", 50)),
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
    out = Path(args.output or ds.get("manifest", "data/splits/seven_axis_seed0.json"))
    if not out.is_absolute():
        out = ROOT / out
    path = save_split_manifest(manifest, out)
    print(json.dumps(manifest.meta, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
