"""Stratified train/validation/test splits for benchmark corpora.

Each benchmark is partitioned independently so withheld and validation
sets have equal per-benchmark representation. Split indices are persisted
with the RNG seed for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from infl_ens.data.benchmarks import BenchmarkSplit

PartitionName = Literal["train", "val", "test", "train_val"]
VALID_PARTITIONS = frozenset({"train", "val", "test", "train_val"})


@dataclass(frozen=True)
class BenchmarkPartitionIndices:
    """Row indices for one benchmark split.

    :param name: Benchmark identifier matching :class:`BenchmarkSplit.name`.
    :type name: str
    :param train: Training row indices.
    :type train: tuple[int, ...]
    :param val: Validation row indices.
    :type val: tuple[int, ...]
    :param test: Withheld test row indices.
    :type test: tuple[int, ...]
    """

    name: str
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    def select(self, partition: PartitionName) -> tuple[int, ...]:
        """Return indices for ``partition``.

        :param partition: One of ``train``, ``val``, ``test``, or
            ``train_val`` (train concatenated with val).
        :type partition: str
        :returns: Row indices into the full loaded benchmark.
        :rtype: tuple[int, ...]
        :raises ValueError: If ``partition`` is unknown.
        """
        if partition == "train":
            return self.train
        if partition == "val":
            return self.val
        if partition == "test":
            return self.test
        if partition == "train_val":
            return self.train + self.val
        raise ValueError(
            f"partition must be one of {sorted(VALID_PARTITIONS)}, "
            f"got {partition!r}"
        )


@dataclass(frozen=True)
class DataSplitManifest:
    """Persisted stratified split over multiple benchmarks.

    :param seed: RNG seed used to shuffle each benchmark before
        partitioning.
    :type seed: int
    :param train_frac: Fraction assigned to training.
    :type train_frac: float
    :param val_frac: Fraction assigned to validation.
    :type val_frac: float
    :param test_frac: Fraction assigned to withheld test.
    :type test_frac: float
    :param benchmarks: Per-benchmark index partitions.
    :type benchmarks: tuple[BenchmarkPartitionIndices, ...]
    :param meta: Optional free-form metadata (counts, batch plan).
    :type meta: dict[str, Any]
    """

    seed: int
    train_frac: float
    val_frac: float
    test_frac: float
    benchmarks: tuple[BenchmarkPartitionIndices, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        """Total training rows across benchmarks.

        :returns: Sum of training partition sizes.
        :rtype: int
        """
        return sum(len(b.train) for b in self.benchmarks)

    @property
    def n_val(self) -> int:
        """Total validation rows across benchmarks.

        :returns: Sum of validation partition sizes.
        :rtype: int
        """
        return sum(len(b.val) for b in self.benchmarks)

    @property
    def n_test(self) -> int:
        """Total withheld test rows across benchmarks.

        :returns: Sum of test partition sizes.
        :rtype: int
        """
        return sum(len(b.test) for b in self.benchmarks)

    @property
    def n_train_val(self) -> int:
        """Total rows in the global routing pool (train + val).

        :returns: ``n_train + n_val``.
        :rtype: int
        """
        return self.n_train + self.n_val

    def partition_for(self, name: str) -> BenchmarkPartitionIndices:
        """Look up partition indices by benchmark name.

        :param name: Benchmark identifier.
        :type name: str
        :returns: Partition record.
        :rtype: BenchmarkPartitionIndices
        :raises KeyError: If ``name`` is absent from the manifest.
        """
        for part in self.benchmarks:
            if part.name == name:
                return part
        raise KeyError(f"benchmark {name!r} not in split manifest")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible mapping.

        :returns: Manifest dictionary.
        :rtype: dict
        """
        return {
            "seed": self.seed,
            "train_frac": self.train_frac,
            "val_frac": self.val_frac,
            "test_frac": self.test_frac,
            "meta": dict(self.meta),
            "benchmarks": {
                b.name: {
                    "train": list(b.train),
                    "val": list(b.val),
                    "test": list(b.test),
                }
                for b in self.benchmarks
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataSplitManifest":
        """Load from a serialized mapping.

        :param payload: JSON object produced by :meth:`to_dict`.
        :type payload: dict
        :returns: Parsed manifest.
        :rtype: DataSplitManifest
        """
        benchmarks = tuple(
            BenchmarkPartitionIndices(
                name=name,
                train=tuple(int(i) for i in part["train"]),
                val=tuple(int(i) for i in part["val"]),
                test=tuple(int(i) for i in part["test"]),
            )
            for name, part in payload["benchmarks"].items()
        )
        return cls(
            seed=int(payload["seed"]),
            train_frac=float(payload["train_frac"]),
            val_frac=float(payload["val_frac"]),
            test_frac=float(payload["test_frac"]),
            benchmarks=benchmarks,
            meta=dict(payload.get("meta") or {}),
        )


def stratified_split_one(
    split: BenchmarkSplit,
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    seed: int = 0,
) -> BenchmarkPartitionIndices:
    """Partition one benchmark into train/val/test strata.

    Rows are shuffled with ``seed``, then contiguous slices are taken in
    proportion ``train_frac``, ``val_frac``, ``test_frac``. Remainder rows
    are assigned to training so every row is used exactly once.

    :param split: Loaded benchmark corpus.
    :type split: BenchmarkSplit
    :param train_frac: Training fraction in ``(0, 1)``.
    :type train_frac: float
    :param val_frac: Validation fraction in ``(0, 1)``.
    :type val_frac: float
    :param test_frac: Test fraction in ``(0, 1)``.
    :type test_frac: float
    :param seed: RNG seed for the within-benchmark shuffle.
    :type seed: int
    :returns: Integer row indices for each partition.
    :rtype: BenchmarkPartitionIndices
    :raises ValueError: If fractions do not sum to ~1 or split is empty.
    """
    total_frac = train_frac + val_frac + test_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(
            f"fractions must sum to 1, got {total_frac:.6f}"
        )
    n = split.n
    if n == 0:
        raise ValueError(f"benchmark {split.name!r} has no rows")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = int(round(test_frac * n))
    assigned = n_train + n_val + n_test
    if assigned > n:
        overflow = assigned - n
        n_train = max(0, n_train - overflow)
        assigned = n_train + n_val + n_test
    if assigned < n:
        n_train += n - assigned

    train_idx = tuple(int(i) for i in order[:n_train])
    val_idx = tuple(int(i) for i in order[n_train:n_train + n_val])
    test_idx = tuple(int(i) for i in order[n_train + n_val:n_train + n_val + n_test])
    return BenchmarkPartitionIndices(
        name=split.name,
        train=train_idx,
        val=val_idx,
        test=test_idx,
    )


def build_split_manifest(
    splits: Sequence[BenchmarkSplit],
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    seed: int = 0,
    meta: dict[str, Any] | None = None,
) -> DataSplitManifest:
    """Build a stratified manifest over all benchmarks.

    Each benchmark uses ``seed + hash(name)`` so partitions are
    independent but reproducible.

    :param splits: Loaded benchmark splits.
    :type splits: Sequence[BenchmarkSplit]
    :param train_frac: Training fraction.
    :type train_frac: float
    :param val_frac: Validation fraction.
    :type val_frac: float
    :param test_frac: Test fraction.
    :type test_frac: float
    :param seed: Base RNG seed recorded in the manifest.
    :type seed: int
    :param meta: Optional metadata merged into the manifest.
    :type meta: dict | None
    :returns: Combined split manifest.
    :rtype: DataSplitManifest
    """
    parts: list[BenchmarkPartitionIndices] = []
    for split in splits:
        bench_seed = seed + (hash(split.name) & 0xFFFF)
        parts.append(
            stratified_split_one(
                split,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                seed=bench_seed,
            )
        )
    return DataSplitManifest(
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        benchmarks=tuple(parts),
        meta=dict(meta or {}),
    )


def save_split_manifest(manifest: DataSplitManifest, path: str | Path) -> Path:
    """Write a split manifest to JSON.

    :param manifest: Split manifest.
    :type manifest: DataSplitManifest
    :param path: Output file path.
    :type path: str | pathlib.Path
    :returns: Resolved output path.
    :rtype: pathlib.Path
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(manifest.to_dict(), fh, indent=2)
    return out.resolve()


def load_split_manifest(path: str | Path) -> DataSplitManifest:
    """Load a split manifest from JSON.

    :param path: Manifest file path.
    :type path: str | pathlib.Path
    :returns: Parsed manifest.
    :rtype: DataSplitManifest
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        return DataSplitManifest.from_dict(json.load(fh))


def apply_manifest_partition(
    splits: Sequence[BenchmarkSplit],
    manifest: DataSplitManifest,
    partition: PartitionName,
) -> list[BenchmarkSplit]:
    """Select rows from each benchmark according to ``partition``.

    :param splits: Full loaded benchmark splits (same order/names as
        when the manifest was built).
    :type splits: Sequence[BenchmarkSplit]
    :param manifest: Persisted split manifest.
    :type manifest: DataSplitManifest
    :param partition: ``train``, ``val``, ``test``, or ``train_val``.
    :type partition: str
    :returns: Partitioned benchmark splits.
    :rtype: list[BenchmarkSplit]
    :raises ValueError: If a split name is missing from the manifest.
    """
    if partition not in VALID_PARTITIONS:
        raise ValueError(
            f"partition must be one of {sorted(VALID_PARTITIONS)}, "
            f"got {partition!r}"
        )
    out: list[BenchmarkSplit] = []
    for split in splits:
        part = manifest.partition_for(split.name)
        idx = part.select(partition)
        out.append(split.take(idx) if idx else split.take([]))
    return out


def flatten_partition_prompts(
    splits: Sequence[BenchmarkSplit],
    manifest: DataSplitManifest,
    partition: PartitionName,
) -> tuple[list[str], list[str], list[str]]:
    """Flatten prompts, responses, and benchmark labels for a partition.

    :param splits: Full loaded benchmark splits.
    :type splits: Sequence[BenchmarkSplit]
    :param manifest: Split manifest.
    :type manifest: DataSplitManifest
    :param partition: Partition to flatten.
    :type partition: str
    :returns: ``(prompts, responses, benchmark_names)`` with one entry
        per row; ``benchmark_names[i]`` identifies the source benchmark
        of ``prompts[i]``.
    :rtype: tuple[list[str], list[str], list[str]]
    """
    prompts: list[str] = []
    responses: list[str] = []
    bench_names: list[str] = []
    for split in splits:
        part = manifest.partition_for(split.name)
        idx = part.select(partition)
        resp = split.responses or [""] * split.n
        for i in idx:
            prompts.append(split.prompts[i])
            responses.append(resp[i])
            bench_names.append(split.name)
    return prompts, responses, bench_names


def choose_exact_train_coverage(
    train_n: int,
    *,
    preferred_batch_sizes: Sequence[int] = (4900, 2450, 1225, 980, 700, 350),
    target_n_rounds: int | None = None,
    min_rounds: int = 5,
    max_rounds: int = 50,
) -> tuple[int, int]:
    """Choose ``(batch_size, n_rounds)`` with ``batch_size * n_rounds == train_n``.

    If ``target_n_rounds`` is set, returns ``(train_n // target_n_rounds,
    target_n_rounds)`` when that division is exact.

    Otherwise scans ``preferred_batch_sizes`` from largest to smallest and
    returns the first pair whose round count lies in ``[min_rounds,
    max_rounds]``.

    :param train_n: Number of training rows.
    :type train_n: int
    :param preferred_batch_sizes: Candidate batch sizes (descending priority).
    :type preferred_batch_sizes: Sequence[int]
    :param target_n_rounds: Optional explicit round count (smaller batches).
    :type target_n_rounds: int | None
    :param min_rounds: Minimum acceptable number of rounds.
    :type min_rounds: int
    :param max_rounds: Maximum acceptable number of rounds.
    :type max_rounds: int
    :returns: ``(batch_size, n_rounds)``.
    :rtype: tuple[int, int]
    :raises ValueError: If no exact factorization is found.
    """
    if train_n <= 0:
        raise ValueError(f"train_n must be positive, got {train_n}")

    if target_n_rounds is not None:
        if not (min_rounds <= target_n_rounds <= max_rounds):
            raise ValueError(
                f"target_n_rounds={target_n_rounds} outside "
                f"[{min_rounds}, {max_rounds}]"
            )
        batch_size = train_n // target_n_rounds
        if batch_size < 1:
            raise ValueError(
                f"target_n_rounds={target_n_rounds} too large for "
                f"train_n={train_n}"
            )
        return batch_size, target_n_rounds

    divisors = sorted(
        {d for d in range(1, train_n + 1) if train_n % d == 0},
        reverse=True,
    )
    preferred = list(preferred_batch_sizes)
    candidates: list[int] = []
    for b in preferred:
        if b in divisors:
            candidates.append(b)
    for b in divisors:
        if b not in candidates:
            candidates.append(b)

    for batch_size in candidates:
        n_rounds = train_n // batch_size
        if min_rounds <= n_rounds <= max_rounds:
            return batch_size, n_rounds

    raise ValueError(
        f"no (batch_size, n_rounds) with batch_size*n_rounds={train_n} "
        f"and {min_rounds} <= n_rounds <= {max_rounds}; "
        f"try adjusting max_records or preferred_batch_sizes"
    )
