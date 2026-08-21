"""Resolve train/val/test partitions for closed-loop training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from infl_ens.data.benchmarks import BenchmarkSplit
from infl_ens.data.splits import (
    DataSplitManifest,
    apply_manifest_partition,
    choose_exact_train_coverage,
    flatten_partition_prompts,
    load_split_manifest,
    save_split_manifest,
    build_split_manifest,
)


def resolve_closed_loop_data_split(
    cfg: dict[str, Any],
    splits: list[BenchmarkSplit],
    *,
    repo_root: Path,
) -> tuple[
    DataSplitManifest,
    list[str],
    list[str],
    list[str],
    list[str],
    int,
    int,
]:
    """Load or build a split manifest and derive closed-loop batch plan.

    :param cfg: Top-level training config with optional ``data_split`` block.
    :type cfg: dict
    :param splits: Full loaded benchmark splits.
    :type splits: list[BenchmarkSplit]
    :param repo_root: Repository root for resolving relative manifest paths.
    :type repo_root: pathlib.Path
    :returns: ``(manifest, train_prompts, train_responses, pool_prompts,
        pool_responses, batch_size, n_rounds)``.
    :rtype: tuple
    """
    ds = dict(cfg.get("data_split") or {})
    if not ds:
        raise ValueError(
            "data_split block is required for partitioned closed-loop runs"
        )

    manifest_path = ds.get("manifest")
    if manifest_path:
        manifest = load_split_manifest(repo_root / manifest_path)
    else:
        manifest = build_split_manifest(
            splits,
            train_frac=float(ds.get("train_frac", 0.7)),
            val_frac=float(ds.get("val_frac", 0.1)),
            test_frac=float(ds.get("test_frac", 0.2)),
            seed=int(ds.get("seed", cfg.get("seed", 0))),
        )
        out_path = ds.get(
            "write_manifest",
            "data/splits/six_axis_seed0.json",
        )
        save_split_manifest(manifest, repo_root / out_path)
        manifest = load_split_manifest(repo_root / out_path)

    train_prompts, train_responses, _ = flatten_partition_prompts(
        splits, manifest, "train",
    )
    pool_prompts, pool_responses, _ = flatten_partition_prompts(
        splits, manifest, "train_val",
    )

    cl = cfg.get("closed_loop", {})
    if bool(ds.get("cover_train_exactly", True)):
        preferred = tuple(
            int(x) for x in ds.get(
                "preferred_batch_sizes",
                [4900, 2450, 1225, 980, 700, 350],
            )
        )
        target_n_rounds = ds.get("target_n_rounds")
        batch_size, n_rounds = choose_exact_train_coverage(
            len(train_prompts),
            preferred_batch_sizes=preferred,
            target_n_rounds=(
                int(target_n_rounds) if target_n_rounds is not None else None
            ),
            min_rounds=int(ds.get("min_rounds", 5)),
            max_rounds=int(ds.get("max_rounds", 50)),
        )
    else:
        batch_size = int(cl.get("batch_size", 256))
        n_rounds = int(cl.get("n_rounds", 5))

    return (
        manifest,
        train_prompts,
        train_responses,
        pool_prompts,
        pool_responses,
        batch_size,
        n_rounds,
    )


def shuffled_train_batch_indices(
    train_n: int,
    batch_size: int,
    n_rounds: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Build without-replacement batch index lists covering train exactly once.

    When ``batch_size * n_rounds != train_n``, the first
    ``train_n % n_rounds`` rounds receive one extra example each so every
  row is used exactly once.

    :param train_n: Number of training rows.
    :type train_n: int
    :param batch_size: Nominal queries per round (minimum when remainder
        is distributed).
    :type batch_size: int
    :param n_rounds: Number of closed-loop rounds.
    :type n_rounds: int
    :param rng: NumPy RNG for shuffling.
    :type rng: numpy.random.Generator
    :returns: List of length ``n_rounds`` with int index arrays.
    :rtype: list[numpy.ndarray]
    :raises ValueError: If ``n_rounds <= 0`` or sizes do not sum to
        ``train_n``.
    """
    if n_rounds <= 0:
        raise ValueError(f"n_rounds must be positive, got {n_rounds}")
    base = train_n // n_rounds
    extra = train_n % n_rounds
    if batch_size != base and batch_size * n_rounds not in (train_n, train_n - extra):
        # ``batch_size`` is the nominal floor from choose_exact_train_coverage.
        pass
    order = rng.permutation(train_n)
    batches: list[np.ndarray] = []
    start = 0
    for r in range(n_rounds):
        size = base + (1 if r < extra else 0)
        batches.append(order[start:start + size])
        start += size
    if start != train_n:
        raise ValueError(
            f"batch indices cover {start} rows, expected {train_n}"
        )
    return batches


def partitioned_splits_for_eval(
    splits: list[BenchmarkSplit],
    manifest: DataSplitManifest,
    partition: str,
) -> list[BenchmarkSplit]:
    """Return benchmark splits restricted to a manifest partition.

    :param splits: Full loaded splits.
    :type splits: list[BenchmarkSplit]
    :param manifest: Split manifest.
    :type manifest: DataSplitManifest
    :param partition: ``train``, ``val``, ``test``, or ``train_val``.
    :type partition: str
    :returns: Partitioned splits in loader order.
    :rtype: list[BenchmarkSplit]
    """
    return apply_manifest_partition(splits, manifest, partition)  # type: ignore[arg-type]
