"""Tests for stratified benchmark splits and exact train coverage."""

from __future__ import annotations

import numpy as np

from infl_ens.data.benchmarks import BenchmarkSplit
from infl_ens.data.splits import (
    build_split_manifest,
    choose_exact_train_coverage,
    stratified_split_one,
)
from infl_ens.training.data_split import shuffled_train_batch_indices


def _toy_split(name: str, n: int = 100) -> BenchmarkSplit:
  return BenchmarkSplit(
      name=name,
      prompts=[f"{name}-{i}" for i in range(n)],
      scores=np.linspace(0.0, 1.0, n),
      axis_name=name,
      responses=[f"resp-{i}" for i in range(n)],
  )


def test_stratified_split_one_partitions_all_rows() -> None:
    split = _toy_split("beavertails", 1000)
    part = stratified_split_one(split, seed=0)
    used = set(part.train) | set(part.val) | set(part.test)
    assert used == set(range(1000))
    assert len(part.train) + len(part.val) + len(part.test) == 1000
    assert len(part.train) == 700
    assert len(part.val) == 100
    assert len(part.test) == 200


def test_build_split_manifest_seven_benchmarks() -> None:
    splits = [_toy_split(f"bench-{i}", 5000) for i in range(7)]
    manifest = build_split_manifest(splits, seed=0)
    assert manifest.n_train == 7 * 3500
    assert manifest.n_val == 7 * 500
    assert manifest.n_test == 7 * 1000


def test_choose_exact_train_coverage_24500() -> None:
    batch_size, n_rounds = choose_exact_train_coverage(24500)
    assert batch_size * n_rounds == 24500


def test_choose_exact_train_coverage_target_rounds() -> None:
    batch_size, n_rounds = choose_exact_train_coverage(
        20172, target_n_rounds=12,
    )
    assert n_rounds == 12
    assert batch_size * n_rounds == 20172


def test_shuffled_train_batch_indices_cover_once() -> None:
    rng = np.random.default_rng(0)
    batches = shuffled_train_batch_indices(20, 4, 5, rng)
    flat = np.concatenate(batches)
    assert flat.shape == (20,)
    assert len(np.unique(flat)) == 20


def test_shuffled_train_batch_indices_remainder() -> None:
    rng = np.random.default_rng(0)
    batches = shuffled_train_batch_indices(20172, 840, 24, rng)
    sizes = [len(b) for b in batches]
    assert len(batches) == 24
    assert sum(sizes) == 20172
    assert sizes.count(841) == 12
    assert sizes.count(840) == 12
