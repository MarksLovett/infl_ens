"""Load safety benchmark splits for adapter evaluation.

Supports BeaverTails (harm), HaluEval (hallucination), ToxicChat
(jailbreak), AI4Privacy (privacy density), OR-Bench (over-refusal),
prompt-injection validation (injection), and Do-Not-Answer
(policy violation). Evaluation reuses the same
YAML ``benchmarks`` list shape as :mod:`infl_ens.training.__main__` so one
config can name both the trait-space axes used in training and the held-out
corpora used at eval time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from infl_ens.data.benchmarks import (
    BenchmarkSplit,
    load_ai4privacy,
    load_beavertails,
    load_do_not_answer,
    load_halueval,
    load_jbb_behaviors,
    load_orbench,
    load_prompt_injection,
    load_toxicchat,
)
from infl_ens.data.splits import apply_manifest_partition, load_split_manifest

#: Supported ``kind`` values in a benchmarks config entry.
BENCHMARK_KINDS: frozenset[str] = frozenset(
    {
        "beavertails",
        "halueval",
        "jbb_behaviors",
        "toxicchat",
        "ai4privacy",
        "orbench",
        "prompt_injection",
        "do_not_answer",
    }
)


def load_benchmark_splits(
    entries: Sequence[dict[str, Any]],
) -> list[BenchmarkSplit]:
    """Load every benchmark split described by a config ``benchmarks`` block.

    :param entries: Sequence of mappings, each with at least ``kind`` and
        ``path``. Per-kind optional keys:

        - ``beavertails``: ``categories``, ``max_records``.
        - ``halueval``: ``tasks``, ``max_records``.
        - ``toxicchat``: ``score_mode``, ``human_annotated_only``,
          ``max_records``.
        - ``ai4privacy``: ``score_mode``, ``english_only``, ``max_records``.
        - ``orbench``: ``configs``, ``max_records``.
        - ``prompt_injection``: ``max_records``.
        - ``do_not_answer``: ``benign_path``, ``include_benign``,
          ``max_records``.
    :type entries: Sequence[dict]
    :returns: Loaded splits in the same order as ``entries``.
    :rtype: list[BenchmarkSplit]
    :raises ValueError: If an entry has an unknown ``kind``.
    :raises FileNotFoundError: Propagated from the underlying loaders when
        data paths are missing.
    """
    splits: list[BenchmarkSplit] = []
    for entry in entries:
        kind = entry["kind"]
        path = entry["path"]
        max_records = entry.get("max_records")
        if kind == "beavertails":
            splits.append(
                load_beavertails(
                    path,
                    categories=entry.get("categories"),
                    max_records=max_records,
                )
            )
        elif kind == "halueval":
            splits.append(
                load_halueval(
                    path,
                    tasks=entry.get("tasks"),
                    max_records=max_records,
                )
            )
        elif kind == "jbb_behaviors":
            splits.append(
                load_jbb_behaviors(
                    path,
                    include_benign=bool(entry.get("include_benign", True)),
                    max_records=max_records,
                )
            )
        elif kind == "toxicchat":
            splits.append(
                load_toxicchat(
                    path,
                    score_mode=entry.get("score_mode", "jailbreaking"),
                    human_annotated_only=bool(
                        entry.get("human_annotated_only", False)
                    ),
                    max_records=max_records,
                )
            )
        elif kind == "ai4privacy":
            splits.append(
                load_ai4privacy(
                    path,
                    score_mode=entry.get("score_mode", "density"),
                    english_only=bool(entry.get("english_only", True)),
                    max_records=max_records,
                )
            )
        elif kind == "orbench":
            splits.append(
                load_orbench(
                    path,
                    configs=tuple(entry["configs"]) if entry.get("configs") else None,
                    max_records=max_records,
                )
            )
        elif kind == "prompt_injection":
            splits.append(
                load_prompt_injection(
                    path,
                    max_records=max_records,
                )
            )
        elif kind == "do_not_answer":
            splits.append(
                load_do_not_answer(
                    path,
                    benign_path=entry.get("benign_path"),
                    include_benign=bool(entry.get("include_benign", True)),
                    max_records=max_records,
                )
            )
        else:
            raise ValueError(
                f"unknown benchmark kind {kind!r}; "
                f"expected one of {sorted(BENCHMARK_KINDS)}"
            )
    return splits


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
