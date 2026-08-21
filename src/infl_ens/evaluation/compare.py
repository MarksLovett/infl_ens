"""Compare saved LoRA adapters on safety benchmarks and merge-by-corner roles.

Provides shared helpers for baseline-vs-specialist eval, multi-model sweeps,
and proximity-merge corner aggregation used by thin scripts under
``scripts/``.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from infl_ens.evaluation.adapters import (
    load_adapter_model,
    load_base_causal_lm,
    resolve_adapter_dir,
)
from infl_ens.evaluation.benchmarks import load_benchmark_splits, subsample_split
from infl_ens.evaluation.metrics import format_chat_example, mean_token_nll, split_to_texts
from infl_ens.training.baseline_replay import load_closed_loop_history, pooled_batch_from_round

DEFAULT_SAFETY_BENCHMARKS: list[dict[str, Any]] = [
    {"kind": "beavertails", "path": "data/beavertails/30k_train.jsonl"},
    {
        "kind": "halueval",
        "path": "data/halueval",
        "tasks": ["qa", "dialogue"],
    },
]

_MERGE_DIR_RE = re.compile(r"^merge-(.+)$")


@dataclass(frozen=True)
class ModelScore:
    """Benchmark NLL for one checkpoint.

    :param label: Model label (e.g. ``pooled-baseline``).
    :type label: str
    :param benchmark: Benchmark id.
    :type benchmark: str
    :param mean_nll: Mean per-token NLL.
    :type mean_nll: float
    :param n_examples: Number of scored examples.
    :type n_examples: int
    """

    label: str
    benchmark: str
    mean_nll: float
    n_examples: int


def resolve_adapter_at(run_dir: Path, label: str, round_idx: int) -> Path:
    """Resolve ``<run_dir>/agents/<label>/round-NN``.

    :param run_dir: Run root.
    :type run_dir: pathlib.Path
    :param label: Agent subdirectory name.
    :type label: str
    :param round_idx: Round index.
    :type round_idx: int
    :returns: Adapter directory path.
    :rtype: pathlib.Path
    :raises FileNotFoundError: If no adapter exists.
    """
    p = run_dir / "agents" / label / f"round-{round_idx:02d}"
    return resolve_adapter_dir(p)


def eval_adapter(
    adapter_dir: Path,
    label: str,
    splits,
    *,
    base_model: str,
    max_eval_records: int,
    seed: int,
    max_seq_length: int,
    forward_batch_size: int,
    base_model_obj=None,
    tokenizer=None,
    device=None,
) -> tuple[list[ModelScore], object, object, object]:
    """Score one adapter on all benchmark splits.

    :param adapter_dir: LoRA checkpoint path.
    :type adapter_dir: pathlib.Path
    :param label: Row label for outputs.
    :type label: str
    :param splits: Benchmark splits.
    :type splits: list
    :param base_model: HuggingFace base model id.
    :type base_model: str
    :param max_eval_records: Subsample cap per benchmark.
    :type max_eval_records: int
    :param seed: Subsample RNG seed.
    :type seed: int
    :param max_seq_length: Token cap.
    :type max_seq_length: int
    :param forward_batch_size: Inference batch size.
    :type forward_batch_size: int
    :param base_model_obj: Optional cached base model.
    :type base_model_obj: Any
    :param tokenizer: Optional cached tokenizer.
    :type tokenizer: Any
    :param device: Optional torch device.
    :type device: Any
    :returns: Scores plus ``(base_model, tokenizer, device)`` for reuse.
    :rtype: tuple
    """
    owns = base_model_obj is None
    if owns:
        base_model_obj, tokenizer, device = load_base_causal_lm(base_model)
    model = load_adapter_model(base_model_obj, adapter_dir)
    scores: list[ModelScore] = []
    try:
        for split in splits:
            eval_split = subsample_split(split, max_eval_records, seed=seed)
            texts = split_to_texts(eval_split)
            mean_nll, _, n_ex = mean_token_nll(
                model,
                tokenizer,
                texts,
                max_length=max_seq_length,
                batch_size=forward_batch_size,
                device=device,
            )
            scores.append(
                ModelScore(
                    label=label,
                    benchmark=eval_split.name,
                    mean_nll=mean_nll,
                    n_examples=n_ex,
                )
            )
    finally:
        import torch

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return scores, base_model_obj, tokenizer, device


def _load_base_scores(path: Path) -> list[ModelScore]:
    """Load frozen base-model NLL from ``base_eval.json``.

    Accepts top-level keys ``benchmarks``, ``scores``, or ``results``.

    :param path: JSON written by base-model eval.
    :type path: pathlib.Path
    :returns: Benchmark scores with label ``base``.
    :rtype: list[ModelScore]
    """
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    rows = data.get("benchmarks", data.get("scores", data.get("results", [])))
    out: list[ModelScore] = []
    for row in rows:
        out.append(
            ModelScore(
                label="base",
                benchmark=str(row["benchmark"]),
                mean_nll=float(row["mean_nll"]),
                n_examples=int(row.get("n_examples", row.get("n", 0))),
            )
        )
    return out


def compare_baseline_vs_specialists(
    baseline_dir: Path,
    specialist_dir: Path,
    *,
    round_idx: int = 39,
    baseline_name: str = "pooled-baseline",
    specialists: Sequence[str] = ("clone-0", "clone-1", "clone-2", "clone-3"),
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_eval_records: int = 128,
    seed: int = 0,
    max_seq_length: int = 1024,
    forward_batch_size: int = 8,
    benchmarks: Optional[Sequence[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Evaluate baseline and specialist adapters; optional cross-NLL on pooled batch.

    :param baseline_dir: Baseline replay run directory.
    :type baseline_dir: pathlib.Path
    :param specialist_dir: Closed-loop specialist run directory.
    :type specialist_dir: pathlib.Path
    :param round_idx: Adapter round index.
    :type round_idx: int
    :param baseline_name: Baseline agent subdirectory name.
    :type baseline_name: str
    :param specialists: Specialist agent names.
    :type specialists: Sequence[str]
    :param base_model: HuggingFace base model id.
    :type base_model: str
    :param max_eval_records: Subsample cap per benchmark.
    :type max_eval_records: int
    :param seed: Subsample RNG seed.
    :type seed: int
    :param max_seq_length: Token cap.
    :type max_seq_length: int
    :param forward_batch_size: Inference batch size.
    :type forward_batch_size: int
    :param benchmarks: Optional benchmark config entries.
    :type benchmarks: Sequence[dict] | None
    :returns: Payload with ``benchmark_scores`` and ``cross_nll_on_pooled_round_batch``.
    :rtype: dict
    """
    bench_entries = list(benchmarks or DEFAULT_SAFETY_BENCHMARKS)
    splits = load_benchmark_splits(bench_entries)
    specialist_list = list(specialists)

    baseline_adapter = resolve_adapter_at(baseline_dir, baseline_name, round_idx)
    all_scores: list[ModelScore] = []
    base, tok, dev = None, None, None

    sc, base, tok, dev = eval_adapter(
        baseline_adapter,
        baseline_name,
        splits,
        base_model=base_model,
        max_eval_records=max_eval_records,
        seed=seed,
        max_seq_length=max_seq_length,
        forward_batch_size=forward_batch_size,
    )
    all_scores.extend(sc)

    for name in specialist_list:
        ad = resolve_adapter_at(specialist_dir, name, round_idx)
        sc, base, tok, dev = eval_adapter(
            ad,
            name,
            splits,
            base_model=base_model,
            max_eval_records=max_eval_records,
            seed=seed,
            max_seq_length=max_seq_length,
            forward_batch_size=forward_batch_size,
            base_model_obj=base,
            tokenizer=tok,
            device=dev,
        )
        all_scores.extend(sc)

    cross_rows: list[dict[str, Any]] = []
    hist_path = specialist_dir / "history.json"
    if hist_path.is_file():
        records = load_closed_loop_history(hist_path)
        rec = next((r for r in records if int(r["round"]) == round_idx), None)
        if rec is not None:
            prompts, responses = pooled_batch_from_round(rec)
            texts = [format_chat_example(p, r) for p, r in zip(prompts, responses)]
            models = [(baseline_name, baseline_adapter)] + [
                (n, resolve_adapter_at(specialist_dir, n, round_idx))
                for n in specialist_list
            ]
            base, tok, dev = load_base_causal_lm(base_model)
            for label, adir in models:
                model = load_adapter_model(base, adir)
                nll, _, n_ex = mean_token_nll(
                    model,
                    tok,
                    texts,
                    max_length=max_seq_length,
                    batch_size=forward_batch_size,
                    device=dev,
                )
                cross_rows.append({
                    "label": label,
                    "round": round_idx,
                    "mean_nll_on_pooled_batch": nll,
                    "n_examples": n_ex,
                })
                del model

    return {
        "baseline_dir": str(baseline_dir.resolve()),
        "specialist_dir": str(specialist_dir.resolve()),
        "round": round_idx,
        "benchmark_scores": [asdict(s) for s in all_scores],
        "cross_nll_on_pooled_round_batch": cross_rows,
    }


def compare_all_models(
    specialist_dir: Path,
    baseline_dir: Path,
    merge_dir: Path,
    *,
    base_eval_json: Optional[Path] = None,
    round_idx: int = 39,
    baseline_name: str = "pooled-baseline",
    merge_names: Optional[Sequence[str]] = None,
    specialists: Sequence[str] = ("clone-0", "clone-1", "clone-2", "clone-3"),
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_eval_records: int = 128,
    seed: int = 0,
    max_seq_length: int = 1024,
    forward_batch_size: int = 8,
    benchmarks: Optional[Sequence[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Compare base, pooled baseline, specialists, and merge adapters.

    :param specialist_dir: Specialist closed-loop run directory.
    :type specialist_dir: pathlib.Path
    :param baseline_dir: Baseline replay run directory.
    :type baseline_dir: pathlib.Path
    :param merge_dir: Pair-merge run directory.
    :type merge_dir: pathlib.Path
    :param base_eval_json: Optional frozen base-model eval JSON.
    :type base_eval_json: pathlib.Path | None
    :param round_idx: Adapter round index.
    :type round_idx: int
    :param baseline_name: Baseline agent subdirectory name.
    :type baseline_name: str
    :param merge_names: Merge trainer names; auto-discover ``merge-*`` if ``None``.
    :type merge_names: Sequence[str] | None
    :param specialists: Specialist agent names.
    :type specialists: Sequence[str]
    :param base_model: HuggingFace base model id.
    :type base_model: str
    :param max_eval_records: Subsample cap per benchmark.
    :type max_eval_records: int
    :param seed: Subsample RNG seed.
    :type seed: int
    :param max_seq_length: Token cap.
    :type max_seq_length: int
    :param forward_batch_size: Inference batch size.
    :type forward_batch_size: int
    :param benchmarks: Optional benchmark config entries.
    :type benchmarks: Sequence[dict] | None
    :returns: Comparison payload with ``benchmark_scores``.
    :rtype: dict
    """
    bench_entries = list(benchmarks or DEFAULT_SAFETY_BENCHMARKS)
    splits = load_benchmark_splits(bench_entries)
    specialist_list = list(specialists)

    if merge_names is not None:
        resolved_merge_names = list(merge_names)
    else:
        agents_dir = merge_dir / "agents"
        resolved_merge_names = sorted(
            p.name
            for p in agents_dir.iterdir()
            if p.is_dir() and p.name.startswith("merge-")
        )

    all_scores: list[ModelScore] = []
    base_eval_path = Path(base_eval_json) if base_eval_json is not None else None
    if base_eval_path is not None and base_eval_path.is_file():
        all_scores.extend(_load_base_scores(base_eval_path))

    base, tok, dev = None, None, None
    models_to_eval = (
        [(baseline_name, resolve_adapter_at(baseline_dir, baseline_name, round_idx))]
        + [(n, resolve_adapter_at(specialist_dir, n, round_idx)) for n in specialist_list]
        + [(n, resolve_adapter_at(merge_dir, n, round_idx)) for n in resolved_merge_names]
    )
    for label, adapter in models_to_eval:
        sc, base, tok, dev = eval_adapter(
            adapter,
            label,
            splits,
            base_model=base_model,
            max_eval_records=max_eval_records,
            seed=seed,
            max_seq_length=max_seq_length,
            forward_batch_size=forward_batch_size,
            base_model_obj=base,
            tokenizer=tok,
            device=dev,
        )
        all_scores.extend(sc)

    return {
        "specialist_dir": str(specialist_dir.resolve()),
        "baseline_dir": str(baseline_dir.resolve()),
        "merge_dir": str(merge_dir.resolve()),
        "base_eval_json": (
            str(base_eval_path.resolve())
            if base_eval_path is not None and base_eval_path.is_file()
            else None
        ),
        "round": round_idx,
        "benchmark_scores": [asdict(s) for s in all_scores],
    }


def parse_merge_members(train_name: str) -> list[str]:
    """Parse ``merge-clone-0-clone-2`` into member clone names.

    :param train_name: Adapter subdirectory name.
    :type train_name: str
    :returns: Sorted member names.
    :rtype: list[str]
    :raises ValueError: If ``train_name`` is not a merge adapter name.
    """
    m = _MERGE_DIR_RE.match(train_name)
    if not m:
        raise ValueError(f"not a merge adapter name: {train_name!r}")
    body = m.group(1)
    if not body.startswith("clone-"):
        raise ValueError(f"unexpected merge name format: {train_name!r}")
    parts = body.split("-clone-")
    first = parts[0]
    if not first.startswith("clone-"):
        raise ValueError(train_name)
    members = [first] + [f"clone-{p}" for p in parts[1:]]
    return sorted(set(members))


def corner_centroid(
    positions: dict[str, list[float]],
    members: list[str],
) -> np.ndarray:
    """Mean final-round position of merge members.

    :param positions: ``agent_name -> [harm, halluc, ...]``.
    :type positions: dict
    :param members: Router names in the merge group.
    :type members: list[str]
    :returns: Centroid vector.
    :rtype: numpy.ndarray
    """
    pts = [np.asarray(positions[m], dtype=float) for m in members]
    return np.stack(pts, axis=0).mean(axis=0)


def assign_corner_roles(
    merge_entries: list[tuple[str, list[str], np.ndarray]],
) -> dict[str, str]:
    """Label each merge adapter by corner role from harm-axis ordering.

    Two merges map to ``corner-low-harm`` and ``corner-high-harm``. One merge
    maps to ``corner-single``. More than two map to ``corner-0``, ``corner-1``,
    ... sorted by harm coordinate.

    :param merge_entries: ``(train_name, members, centroid)`` list.
    :type merge_entries: list
    :returns: ``train_name -> corner_role``.
    :rtype: dict[str, str]
    """
    if not merge_entries:
        return {}
    sorted_entries = sorted(merge_entries, key=lambda x: float(x[2][0]))
    n = len(sorted_entries)
    if n == 1:
        roles = ["corner-single"]
    elif n == 2:
        roles = ["corner-low-harm", "corner-high-harm"]
    else:
        roles = [f"corner-{i}" for i in range(n)]
    return {name: role for (name, _, _), role in zip(sorted_entries, roles)}


@dataclass(frozen=True)
class CornerMergeRecord:
    """One merge adapter scored on benchmarks for a single seed.

    :param seed: Run seed index.
    :type seed: int
    :param train_name: LoRA folder name (e.g. ``merge-clone-0-clone-2``).
    :type train_name: str
    :param members: Router clones merged for training.
    :type members: list[str]
    :param corner_role: Role label (``corner-low-harm``, etc.).
    :type corner_role: str
    :param centroid: Mean member position at final round.
    :type centroid: list[float]
    :param scores: Benchmark NLL rows.
    :type scores: list[ModelScore]
    """

    seed: int
    train_name: str
    members: list[str]
    corner_role: str
    centroid: list[float]
    scores: list[ModelScore]


def process_merge_seed(
    run_dir: Path,
    *,
    round_idx: int,
    do_eval: bool,
    base_model: str,
    max_eval_records: int,
    eval_seed: int,
    max_seq_length: int,
    forward_batch_size: int,
    benchmarks: Optional[Sequence[dict[str, Any]]] = None,
) -> list[CornerMergeRecord]:
    """Assign corner roles and load or compute benchmark scores for one seed.

    :param run_dir: ``.../seed{N}`` directory.
    :type run_dir: pathlib.Path
    :param round_idx: Adapter round index.
    :type round_idx: int
    :param do_eval: If True, run NLL eval when compare JSON missing.
    :type do_eval: bool
    :param base_model: HuggingFace model id.
    :type base_model: str
    :param max_eval_records: Subsample cap per benchmark.
    :type max_eval_records: int
    :param eval_seed: Subsample RNG seed.
    :type eval_seed: int
    :param max_seq_length: Token cap.
    :type max_seq_length: int
    :param forward_batch_size: Inference batch size.
    :type forward_batch_size: int
    :param benchmarks: Optional benchmark config entries.
    :type benchmarks: Sequence[dict] | None
    :returns: Corner-labeled records for this seed.
    :rtype: list[CornerMergeRecord]
    :raises FileNotFoundError: If history or scores are missing.
    """
    hist_path = run_dir / "history.json"
    if not hist_path.is_file():
        raise FileNotFoundError(hist_path)
    history = json.loads(hist_path.read_text(encoding="utf-8"))
    final = history[-1]
    positions = final["positions"]

    agents_dir = run_dir / "agents"
    merge_names = sorted(
        p.name for p in agents_dir.iterdir()
        if p.is_dir() and p.name.startswith("merge-")
    )
    entries: list[tuple[str, list[str], np.ndarray]] = []
    for name in merge_names:
        members = parse_merge_members(name)
        centroid = corner_centroid(positions, members)
        entries.append((name, members, centroid))
    role_map = assign_corner_roles(entries)

    compare_path = run_dir / f"compare_all_round{round_idx:02d}.json"
    cached: dict[str, list[ModelScore]] = {}
    if compare_path.is_file():
        data = json.loads(compare_path.read_text(encoding="utf-8"))
        for row in data.get("benchmark_scores", []):
            if row["label"].startswith("merge-"):
                cached.setdefault(row["label"], []).append(
                    ModelScore(
                        label=row["label"],
                        benchmark=row["benchmark"],
                        mean_nll=float(row["mean_nll"]),
                        n_examples=int(row["n_examples"]),
                    ),
                )

    records: list[CornerMergeRecord] = []
    splits = None
    base, tok, dev = None, None, None

    for name, members, centroid in entries:
        role = role_map[name]
        if name in cached and len(cached[name]) >= 1:
            scores = [
                ModelScore(
                    label=role,
                    benchmark=s.benchmark,
                    mean_nll=s.mean_nll,
                    n_examples=s.n_examples,
                )
                for s in cached[name]
            ]
        elif do_eval:
            if splits is None:
                splits = load_benchmark_splits(list(benchmarks or DEFAULT_SAFETY_BENCHMARKS))
            adapter = resolve_adapter_at(run_dir, name, round_idx)
            scores, base, tok, dev = eval_adapter(
                adapter,
                role,
                splits,
                base_model=base_model,
                max_eval_records=max_eval_records,
                seed=eval_seed,
                max_seq_length=max_seq_length,
                forward_batch_size=forward_batch_size,
                base_model_obj=base,
                tokenizer=tok,
                device=dev,
            )
        else:
            raise FileNotFoundError(
                f"No scores for {name} in {compare_path} and do_eval=False",
            )

        m = re.search(r"seed(\d+)", str(run_dir))
        seed_idx = int(m.group(1)) if m else 0
        records.append(
            CornerMergeRecord(
                seed=seed_idx,
                train_name=name,
                members=members,
                corner_role=role,
                centroid=centroid.tolist(),
                scores=scores,
            ),
        )
    return records


def aggregate_merge_by_corner(
    records: list[CornerMergeRecord],
) -> dict[str, Any]:
    """Mean ± std NLL per ``corner_role`` and benchmark across seeds.

    :param records: All per-seed corner records.
    :type records: list[CornerMergeRecord]
    :returns: Aggregate report dict with ``per_seed`` and ``aggregate_by_corner``.
    :rtype: dict
    """
    by_role_bench: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_seed_rows: list[dict[str, Any]] = []

    for rec in records:
        per_seed_rows.append({
            "seed": rec.seed,
            "train_name": rec.train_name,
            "members": rec.members,
            "corner_role": rec.corner_role,
            "centroid": rec.centroid,
            "benchmark_scores": [asdict(s) for s in rec.scores],
        })
        for sc in rec.scores:
            by_role_bench[(rec.corner_role, sc.benchmark)].append(sc.mean_nll)

    agg_rows = []
    for (role, bench), vals in sorted(by_role_bench.items()):
        agg_rows.append({
            "corner_role": role,
            "benchmark": bench,
            "mean_nll": statistics.mean(vals),
            "std_nll": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
        })
    return {
        "per_seed": per_seed_rows,
        "aggregate_by_corner": agg_rows,
    }


def aggregate_compare_reports(
    paths: Sequence[Path],
) -> dict[str, Any]:
    """Mean ± std NLL per model label and benchmark across seed reports.

    :param paths: ``compare_all_round*.json`` files (one per seed).
    :type paths: Sequence[pathlib.Path]
    :returns: Aggregate report with ``n_files`` and ``rows``.
    :rtype: dict
    """
    by_label_bench: dict[tuple[str, str], list[float]] = defaultdict(list)
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            data = json.load(fh)
        for row in data.get("benchmark_scores", []):
            by_label_bench[(row["label"], row["benchmark"])].append(
                float(row["mean_nll"]),
            )

    rows = []
    for (label, bench), vals in sorted(by_label_bench.items()):
        rows.append({
            "label": label,
            "benchmark": bench,
            "mean_nll": statistics.mean(vals),
            "std_nll": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
        })
    return {"n_files": len(paths), "rows": rows}


def print_aggregate_compare_table(report: dict[str, Any]) -> None:
    """Print a benchmark × model table from :func:`aggregate_compare_reports`.

    :param report: Aggregate report dict.
    :type report: dict
    """
    rows = report["rows"]
    benches = sorted({r["benchmark"] for r in rows})
    labels = sorted({r["label"] for r in rows}, key=lambda x: (
        0 if x == "base" else 1 if x == "pooled-baseline" else 2,
        x,
    ))
    print(f"{'model':<16}" + "".join(f"{b:>18}" for b in benches))
    for lab in labels:
        line = f"{lab:<16}"
        for b in benches:
            cell = next(
                (r for r in rows if r["label"] == lab and r["benchmark"] == b),
                None,
            )
            if cell is None:
                line += f"{'—':>18}"
            else:
                line += f"{cell['mean_nll']:8.4f}±{cell['std_nll']:<8.4f}"
        print(line)
