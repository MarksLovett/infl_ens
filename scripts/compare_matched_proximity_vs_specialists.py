"""Benchmark comparison: proximity merge vs specialists from the SAME run.

Evaluates ``clone-*`` and ``merge-*`` adapters under one run directory,
labels merge adapters by corner role, and aggregates specialist clones
that occupy each corner at round 39.

Example::

    python scripts/compare_matched_proximity_vs_specialists.py \\
        --run-root results/proximity_plus_specialists_r40/seed0 \\
        --round 39

Aggregate all seeds::

    python scripts/compare_matched_proximity_vs_specialists.py \\
        --run-root results/proximity_plus_specialists_r40 \\
        --round 39 \\
        --all-seeds \\
        --output results/proximity_plus_specialists_corner_compare.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from infl_ens.evaluation.compare import (
    ModelScore,
    eval_adapter,
    process_merge_seed,
    resolve_adapter_at,
)
from infl_ens.utils.agent_init import harm_pair_indices


def _specialist_corner_roles(
    positions: dict[str, list[float]],
    names: list[str],
) -> dict[str, str]:
    """Map each clone to ``corner-low-harm`` or ``corner-high-harm``.

    :param positions: Final-round positions from ``history.json``.
    :type positions: dict
    :param names: Clone names.
    :type names: list[str]
    :returns: ``clone-name -> corner_role``.
    :rtype: dict[str, str]
    """
    P = np.stack([np.asarray(positions[n], dtype=float) for n in names])
    low_idx, high_idx = harm_pair_indices(P)
    low_names = {names[int(i)] for i in low_idx}
    role_map: dict[str, str] = {}
    for n in names:
        role_map[n] = (
            "corner-low-harm" if n in low_names else "corner-high-harm"
        )
    return role_map


def eval_matched_run(
    run_dir: Path,
    *,
    round_idx: int,
    do_eval: bool,
    base_model: str,
    max_eval_records: int,
    eval_seed: int,
    max_seq_length: int,
    forward_batch_size: int,
) -> dict[str, Any]:
    """Score specialists and merge adapters from one matched run.

    :param run_dir: ``seedN`` directory or sweep root (with ``--all-seeds``).
    :type run_dir: pathlib.Path
    :returns: Report dict for one or many seeds.
    :rtype: dict
    """
    merge_records = process_merge_seed(
        run_dir,
        round_idx=round_idx,
        do_eval=do_eval,
        base_model=base_model,
        max_eval_records=max_eval_records,
        eval_seed=eval_seed,
        max_seq_length=max_seq_length,
        forward_batch_size=forward_batch_size,
    )

    hist = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    final = hist[-1]
    positions = final["positions"]
    names = ["clone-0", "clone-1", "clone-2", "clone-3"]
    clone_roles = _specialist_corner_roles(positions, names)

    from infl_ens.evaluation.benchmarks import load_benchmark_splits

    benchmarks = [
        {"kind": "beavertails", "path": "data/beavertails/30k_train.jsonl"},
        {"kind": "halueval", "path": "data/halueval", "tasks": ["qa", "dialogue"]},
    ]
    splits = load_benchmark_splits(benchmarks)

    clone_scores: list[ModelScore] = []
    base, tok, dev = None, None, None
    for name in names:
        adapter = resolve_adapter_at(run_dir, name, round_idx)
        sc, base, tok, dev = eval_adapter(
            adapter,
            name,
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
        clone_scores.extend(sc)

    m = re.search(r"seed(\d+)", str(run_dir))
    seed_idx = int(m.group(1)) if m else 0

    return {
        "seed": seed_idx,
        "run_dir": str(run_dir.resolve()),
        "round": round_idx,
        "init_mode": (hist[0].get("theory_init") or {}).get("init_mode"),
        "clone_corner_roles": clone_roles,
        "clone_scores": [asdict(s) for s in clone_scores],
        "merge_by_corner": [asdict(s) for r in merge_records for s in r.scores],
        "merge_meta": [
            {
                "train_name": r.train_name,
                "members": r.members,
                "corner_role": r.corner_role,
                "centroid": r.centroid,
            }
            for r in merge_records
        ],
    }


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Build mean ± std tables across seeds.

    :param reports: Per-seed report dicts.
    :type reports: list
    :returns: Aggregate dict.
    :rtype: dict
    """
    by_label: dict[tuple[str, str], list[float]] = defaultdict(list)
    corner_clone: dict[tuple[str, str], list[float]] = defaultdict(list)

    for rep in reports:
        roles = rep["clone_corner_roles"]
        for row in rep["clone_scores"]:
            lab = row["label"]
            bench = row["benchmark"]
            by_label[(lab, bench)].append(float(row["mean_nll"]))
            corner_clone[(roles[lab], bench)].append(float(row["mean_nll"]))
        for row in rep["merge_by_corner"]:
            by_label[(row["label"], row["benchmark"])].append(float(row["mean_nll"]))

    rows = []
    for (lab, bench), vals in sorted(by_label.items()):
        rows.append({
            "label": lab,
            "benchmark": bench,
            "mean_nll": statistics.mean(vals),
            "std_nll": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
        })

    corner_rows = []
    for (role, bench), vals in sorted(corner_clone.items()):
        corner_rows.append({
            "corner_role": role,
            "benchmark": bench,
            "mean_nll_specialists_at_corner": statistics.mean(vals),
            "std_nll": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n_clone_scores": len(vals),
        })

    return {
        "n_seeds": len(reports),
        "aggregate": rows,
        "specialists_mean_by_corner": corner_rows,
    }


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    :param argv: Optional argument vector.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--round", type=int, default=39)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-eval-records", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--forward-batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    root = Path(args.run_root)
    if args.all_seeds:
        seed_dirs = sorted(root.glob("seed*"))
    else:
        seed_dirs = [root]

    reports = [
        eval_matched_run(
            sd,
            round_idx=args.round,
            do_eval=args.eval,
            base_model=args.base_model,
            max_eval_records=args.max_eval_records,
            eval_seed=args.eval_seed,
            max_seq_length=args.max_seq_length,
            forward_batch_size=args.forward_batch_size,
        )
        for sd in seed_dirs
    ]

    out_report: dict[str, Any] = {
        "per_seed": reports,
        "summary": aggregate_reports(reports),
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(out_report, fh, indent=2)

    agg = out_report["summary"]["aggregate"]
    benches = sorted({r["benchmark"] for r in agg})
    print(f"\n=== matched run: merge vs specialists ({len(reports)} seeds) ===\n")
    labels = sorted({r["label"] for r in agg}, key=lambda x: (
        0 if x.startswith("corner-") else 1 if x.startswith("merge-") else 2,
        x,
    ))
    print(f"{'model':<28}" + "".join(f"{b:>14}" for b in benches))
    for lab in labels:
        row = f"{lab:<28}"
        for b in benches:
            cell = next((r for r in agg if r["label"] == lab and r["benchmark"] == b), None)
            row += f"{cell['mean_nll']:14.4f}" if cell else f"{'—':>14}"
        print(row)

    print("\n=== mean specialist NLL at each corner (across clones in that corner) ===")
    for cr in out_report["summary"]["specialists_mean_by_corner"]:
        print(
            f"  {cr['corner_role']} {cr['benchmark']}: "
            f"{cr['mean_nll_specialists_at_corner']:.4f} ± {cr['std_nll']:.4f} "
            f"(n={cr['n_clone_scores']})",
        )

    if args.output:
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
