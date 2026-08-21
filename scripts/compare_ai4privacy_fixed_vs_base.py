#!/usr/bin/env python3
"""Compare ai4privacy_fixed_theory_specialists_r40 (all seeds) vs base model."""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from infl_ens.evaluation.adapters import load_base_causal_lm
from infl_ens.evaluation.benchmarks import load_benchmark_splits, subsample_split
from infl_ens.evaluation.metrics import mean_token_nll, split_to_texts

BENCHMARKS = [
    {
        "kind": "beavertails",
        "path": "data/beavertails/30k_train.jsonl",
        "max_records": 5000,
    },
    {
        "kind": "halueval",
        "path": "data/halueval",
        "tasks": ["qa", "dialogue"],
        "max_records": 5000,
    },
    {
        "kind": "toxicchat",
        "path": "data/toxicchat",
        "score_mode": "jailbreaking",
        "human_annotated_only": False,
        "max_records": 5000,
    },
    {
        "kind": "ai4privacy",
        "path": "data/ai4privacy",
        "score_mode": "density",
        "english_only": True,
        "max_records": 5000,
    },
]

EVAL_CFG = {
    "max_seq_length": 1024,
    "forward_batch_size": 8,
    "max_eval_records": 256,
    "seed": 0,
}
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SWEEP_ROOT = Path("results/ai4privacy_fixed_theory_specialists_r40")


@dataclass(frozen=True)
class BenchRow:
    """One benchmark row for reporting."""

    benchmark: str
    axis: str
    mean_nll: float
    std_nll: float | None
    n: int


def run_base_eval(out_path: Path) -> dict[str, float]:
    """Score base model on all benchmarks; return benchmark -> mean_nll."""
    splits = load_benchmark_splits(BENCHMARKS)
    model, tokenizer, device = load_base_causal_lm(BASE_MODEL)
    rows: list[dict[str, object]] = []
    scores: dict[str, float] = {}
    for split in splits:
        eval_split = subsample_split(
            split,
            int(EVAL_CFG["max_eval_records"]),
            seed=int(EVAL_CFG["seed"]),
        )
        texts = split_to_texts(eval_split)
        mean_nll, n_tokens, n_examples = mean_token_nll(
            model,
            tokenizer,
            texts,
            max_length=int(EVAL_CFG["max_seq_length"]),
            batch_size=int(EVAL_CFG["forward_batch_size"]),
            device=device,
        )
        scores[eval_split.name] = mean_nll
        rows.append(
            {
                "benchmark": eval_split.name,
                "axis_name": eval_split.axis_name,
                "mean_nll": mean_nll,
                "n_examples": n_examples,
                "n_tokens": n_tokens,
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "base_model": BASE_MODEL,
            **EVAL_CFG,
            "benchmarks": BENCHMARKS,
        },
        "results": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return scores


def load_base_scores(path: Path) -> dict[str, float]:
    """Load benchmark -> mean_nll from a base_eval-style JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["benchmark"]: float(r["mean_nll"]) for r in data["results"]}


def aggregate_adapters() -> dict[tuple[str, str], list[float]]:
    """Collect per-(agent, benchmark) NLL across seeds."""
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for seed in range(10):
        p = SWEEP_ROOT / f"seed{seed}" / "eval_final_round" / "eval_results.json"
        if not p.exists():
            continue
        for row in json.loads(p.read_text(encoding="utf-8"))["results"]:
            by_key[(row["agent"], row["benchmark"])].append(float(row["mean_nll"]))
    return by_key


def main() -> int:
    """CLI entry."""
    base_path = SWEEP_ROOT / "base_eval_matched.json"
    if "--run-base" in sys.argv or not base_path.exists():
        print(f"Running base eval -> {base_path}")
        base_scores = run_base_eval(base_path)
    else:
        print(f"Loading base eval from {base_path}")
        base_scores = load_base_scores(base_path)

    by_agent = aggregate_adapters()
    benchmarks = sorted({b for _, b in by_agent})
    agents = sorted({a for a, _ in by_agent})

    print("\nMean NLL (lower = better). Adapters: mean +/- std over 10 seeds, round 39.")
    print(f"Base: single {BASE_MODEL}, {EVAL_CFG['max_eval_records']} examples/benchmark\n")

    header = f"{'benchmark':<14}" + "".join(f"{a:>12}" for a in agents) + f"{'base':>12}"
    print(header)
    print("-" * len(header))

    summary: dict[str, object] = {
        "base": {b: base_scores[b] for b in benchmarks},
        "adapters": {},
    }

    for bench in benchmarks:
        cells = []
        agent_means: dict[str, float] = {}
        for agent in agents:
            vals = by_agent.get((agent, bench), [])
            if not vals:
                cells.append(f"{'n/a':>12}")
                continue
            m = statistics.mean(vals)
            s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            agent_means[agent] = m
            cells.append(f"{m:6.3f}{s:5.3f}"[:12].rjust(12))
        base_cell = f"{base_scores.get(bench, float('nan')):12.3f}"
        print(f"{bench:<14}" + "".join(cells) + base_cell)
        summary["adapters"][bench] = agent_means

    print("\nBest adapter per benchmark (vs base):")
    for bench in benchmarks:
        base_nll = base_scores[bench]
        best_agent = min(agents, key=lambda a: statistics.mean(by_agent[(a, bench)]))
        best_nll = statistics.mean(by_agent[(best_agent, bench)])
        delta = best_nll - base_nll
        print(
            f"  {bench:<12} {best_agent} {best_nll:.3f}  "
            f"(base {base_nll:.3f}, delta {delta:+.3f})"
        )

    out_json = SWEEP_ROOT / "compare_vs_base.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
