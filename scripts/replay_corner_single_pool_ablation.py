"""Replay + eval: single pooled LoRA per corner vs merge vs specialists.

One LoRA per corner trains on the **same merged batch** proximity merge
used (union of the two co-located routers' routed prompts each round).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from infl_ens.training.corner_pooled_replay import (
    HIGH_TRAINER,
    LOW_TRAINER,
    replay_corner_pooled_single_trainer,
)


def _load_sft_cfg(seed: int) -> tuple:
    import numpy as np

    from infl_ens.training.sft_training import SFTTrainingConfig

    cfg_path = Path(
        "configs/benchmark/router/safety_truth_n4_r40_proximity_plus_specialists_cum.yaml",
    )
    with cfg_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    sft_dict = dict(cfg["closed_loop"]["sft"])
    sft_dict["seed"] = seed
    dummy_project = lambda texts: np.zeros((len(list(texts)), 2))
    return SFTTrainingConfig(**sft_dict), dummy_project


def run_replay(history: Path, output_dir: Path, seed: int) -> None:
    """Replay corner-pooled single-trainer SFT.

    :param history: ``history.json`` path.
    :type history: pathlib.Path
    :param output_dir: Agent output root.
    :type output_dir: pathlib.Path
    :param seed: SFT seed.
    :type seed: int
    """
    sft_cfg, project = _load_sft_cfg(seed)
    replay_corner_pooled_single_trainer(
        history,
        sft_cfg=sft_cfg,
        project=project,
        output_dir=output_dir,
        loss_reweight="position_only",
    )


def run_compare(source_root: Path, round_idx: int, out_path: Path) -> None:
    """Evaluate single-pool, merge, and clone adapters; print table.

    :param source_root: ``proximity_plus_specialists_r40`` root.
    :type source_root: pathlib.Path
    :param round_idx: Adapter round.
    :type round_idx: int
    :param out_path: JSON output path.
    :type out_path: pathlib.Path
    """
    from infl_ens.evaluation.benchmarks import load_benchmark_splits
    from infl_ens.evaluation.compare import eval_adapter, resolve_adapter_at

    benchmarks = [
        {"kind": "beavertails", "path": "data/beavertails/30k_train.jsonl"},
        {"kind": "halueval", "path": "data/halueval", "tasks": ["qa", "dialogue"]},
    ]
    splits = load_benchmark_splits(benchmarks)
    base_model = "Qwen/Qwen2.5-1.5B-Instruct"

    per_seed: list[dict] = []
    agg: dict[tuple[str, str], list[float]] = defaultdict(list)

    for seed_dir in sorted(source_root.glob("seed*")):
        agents = seed_dir / "agents"
        labels = [LOW_TRAINER, HIGH_TRAINER]
        for p in agents.iterdir():
            if p.is_dir() and p.name.startswith("merge-"):
                labels.append(p.name)
        for c in ("clone-0", "clone-1", "clone-2", "clone-3"):
            if (agents / c / f"round-{round_idx:02d}").exists():
                labels.append(c)

        base, tok, dev = None, None, None
        rows: list[dict] = []
        for label in labels:
            try:
                ad = resolve_adapter_at(seed_dir, label, round_idx)
            except FileNotFoundError:
                continue
            sc, base, tok, dev = eval_adapter(
                ad, label, splits,
                base_model=base_model,
                max_eval_records=128,
                seed=0,
                max_seq_length=1024,
                forward_batch_size=8,
                base_model_obj=base,
                tokenizer=tok,
                device=dev,
            )
            for s in sc:
                rows.append({
                    "seed": int(seed_dir.name.replace("seed", "")),
                    "label": s.label,
                    "benchmark": s.benchmark,
                    "mean_nll": s.mean_nll,
                })
                agg[(s.label, s.benchmark)].append(s.mean_nll)
        per_seed.append({"seed_dir": str(seed_dir), "scores": rows})

    summary = [
        {
            "label": lab,
            "benchmark": bench,
            "mean_nll": statistics.mean(vals),
            "std_nll": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
        }
        for (lab, bench), vals in sorted(agg.items())
    ]

    report = {"per_seed": per_seed, "summary": summary, "round": round_idx}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    benches = sorted({s["benchmark"] for s in summary})
    print(f"\n=== ablation: single-pool vs merge vs specialists (round {round_idx}) ===\n")
    all_labels = {s["label"] for s in summary}
    ordered: list[str] = []
    for lab in [LOW_TRAINER, HIGH_TRAINER]:
        if lab in all_labels:
            ordered.append(lab)
    ordered.extend(sorted(x for x in all_labels if x.startswith("merge-")))
    for i in range(4):
        c = f"clone-{i}"
        if c in all_labels:
            ordered.append(c)
    print(f"{'model':<28}" + "".join(f"{b:>12}" for b in benches))
    for lab in ordered:
        line = f"{lab:<28}"
        for b in benches:
            cell = next((s for s in summary if s["label"] == lab and s["benchmark"] == b), None)
            line += f"{cell['mean_nll']:12.4f}" if cell else f"{'—':>12}"
        print(line)
    print(f"\nwrote {out_path}")


def main() -> int:
    """CLI entry point.

    :returns: Exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--source-root", type=str, default="results/proximity_plus_specialists_r40")
    parser.add_argument("--round", type=int, default=39)
    parser.add_argument("--output", type=str, default="results/corner_single_pool_ablation_compare.json")
    args = parser.parse_args()

    if args.compare_only:
        run_compare(Path(args.source_root), args.round, Path(args.output))
        return 0

    if not args.history or not args.output_dir:
        raise SystemExit("--history and --output-dir required for replay")
    run_replay(Path(args.history), Path(args.output_dir), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
