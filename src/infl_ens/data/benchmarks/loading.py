"""Load benchmark splits from a config ``benchmarks`` list.

Training, evaluation and the pipeline all describe their corpora with the
same YAML shape (a list of ``{kind, path, ...}`` entries; see
:data:`infl_ens.config.BENCHMARK_ENTRY_KEYS`), so one loader serves them
all: the trait-space axes used in training are exactly the held-out
corpora scored at evaluation time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from infl_ens.data.benchmarks.ai4privacy import load_ai4privacy
from infl_ens.data.benchmarks.base import BenchmarkSplit
from infl_ens.data.benchmarks.beavertails import load_beavertails
from infl_ens.data.benchmarks.do_not_answer import load_do_not_answer
from infl_ens.data.benchmarks.halueval import load_halueval
from infl_ens.data.benchmarks.jbb_behaviors import load_jbb_behaviors
from infl_ens.data.benchmarks.orbench import load_orbench
from infl_ens.data.benchmarks.prompt_injection import load_prompt_injection
from infl_ens.data.splits import apply_manifest_partition, load_split_manifest

#: Supported ``kind`` values in a benchmarks config entry.
BENCHMARK_KINDS: frozenset[str] = frozenset(
    {
        "beavertails",
        "halueval",
        "jbb_behaviors",
        "ai4privacy",
        "orbench",
        "prompt_injection",
        "do_not_answer",
    }
)


def load_benchmark_split(entry: dict[str, Any]) -> BenchmarkSplit:
    """Load one benchmark split from a config entry.

    :param entry: Mapping with at least ``kind`` and ``path``. Per-kind
        optional keys:

        - ``beavertails``: ``categories``, ``max_records``.
        - ``halueval``: ``tasks``, ``max_records``.
        - ``jbb_behaviors``: ``include_benign``, ``max_records``.
        - ``ai4privacy``: ``score_mode``, ``english_only``, ``max_records``.
        - ``orbench``: ``configs``, ``max_records``.
        - ``prompt_injection``: ``max_records``.
        - ``do_not_answer``: ``benign_path``, ``include_benign``,
          ``max_records``.
    :type entry: dict
    :returns: The loaded split.
    :rtype: BenchmarkSplit
    :raises ValueError: If the entry has an unknown ``kind``.
    :raises FileNotFoundError: Propagated from the underlying loader when
        the data path is missing.
    """
    kind = entry["kind"]
    path = entry["path"]
    max_records = entry.get("max_records")
    if kind == "beavertails":
        return load_beavertails(
            path,
            categories=entry.get("categories"),
            max_records=max_records,
        )
    if kind == "halueval":
        return load_halueval(
            path,
            tasks=entry.get("tasks"),
            max_records=max_records,
        )
    if kind == "jbb_behaviors":
        return load_jbb_behaviors(
            path,
            include_benign=bool(entry.get("include_benign", True)),
            max_records=max_records,
        )
    if kind == "ai4privacy":
        return load_ai4privacy(
            path,
            score_mode=entry.get("score_mode", "density"),
            english_only=bool(entry.get("english_only", True)),
            max_records=max_records,
        )
    if kind == "orbench":
        return load_orbench(
            path,
            configs=tuple(entry["configs"]) if entry.get("configs") else None,
            max_records=max_records,
        )
    if kind == "prompt_injection":
        return load_prompt_injection(path, max_records=max_records)
    if kind == "do_not_answer":
        return load_do_not_answer(
            path,
            benign_path=entry.get("benign_path"),
            include_benign=bool(entry.get("include_benign", True)),
            max_records=max_records,
        )
    raise ValueError(
        f"unknown benchmark kind {kind!r}; expected one of {sorted(BENCHMARK_KINDS)}"
    )


def load_benchmark_splits(
    entries: Sequence[dict[str, Any]],
) -> list[BenchmarkSplit]:
    """Load every benchmark split described by a config ``benchmarks`` block.

    :param entries: Config entries (see :func:`load_benchmark_split`).
    :type entries: Sequence[dict]
    :returns: Loaded splits in the same order as ``entries``.
    :rtype: list[BenchmarkSplit]
    """
    return [load_benchmark_split(entry) for entry in entries]


def load_benchmark_splits_with_partition(
    entries: Sequence[dict[str, Any]],
    *,
    manifest_path: str | Path,
    partition: str,
    repo_root: str | Path | None = None,
) -> list[BenchmarkSplit]:
    """Load benchmarks and restrict rows to a persisted split partition.

    :param entries: Benchmark config entries (same shape as
        :func:`load_benchmark_splits`).
    :type entries: Sequence[dict]
    :param manifest_path: Path to a :class:`DataSplitManifest` JSON file.
    :type manifest_path: str | pathlib.Path
    :param partition: ``train``, ``val``, ``test``, or ``train_val``.
    :type partition: str
    :param repo_root: Optional root for resolving relative ``manifest_path``.
    :type repo_root: str | pathlib.Path | None
    :returns: Partitioned benchmark splits.
    :rtype: list[BenchmarkSplit]
    """
    splits = load_benchmark_splits(entries)
    path = Path(manifest_path)
    if repo_root is not None and not path.is_absolute():
        path = Path(repo_root) / path
    manifest = load_split_manifest(path)
    return apply_manifest_partition(splits, manifest, partition)  # type: ignore[arg-type]


def subsample_split(
    split: BenchmarkSplit,
    max_records: int,
    *,
    seed: int = 0,
) -> BenchmarkSplit:
    """Return a random sub-split with at most ``max_records`` rows.

    :param split: Full benchmark split.
    :type split: BenchmarkSplit
    :param max_records: Maximum number of rows to keep.
    :type max_records: int
    :param seed: RNG seed for reproducible subsampling.
    :type seed: int
    :returns: Sub-split (unchanged if ``split.n <= max_records``).
    :rtype: BenchmarkSplit
    """
    if split.n <= max_records:
        return split
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = rng.choice(split.n, size=max_records, replace=False)
    return split.take(idx.tolist())


__all__ = [
    "BENCHMARK_KINDS",
    "load_benchmark_split",
    "load_benchmark_splits",
    "load_benchmark_splits_with_partition",
    "subsample_split",
]
