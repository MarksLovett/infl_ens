import json
from pathlib import Path

root = Path("/home/mlovett/infl_ens/results")
merge_dir = root / "seven_axis_pair_merge_split/seed0"
base_dir = root / "seven_axis_baseline_replay_split/seed0"
benchmarks = [
    "beavertails", "halueval", "toxicchat", "ai4privacy",
    "orbench", "prompt_injection", "do_not_answer",
]
labels = {
    "beavertails": "Harm",
    "halueval": "Hallucination",
    "toxicchat": "Jailbreak",
    "ai4privacy": "Privacy",
    "orbench": "Over-refusal",
    "prompt_injection": "Injection",
    "do_not_answer": "Policy",
}

for split in ["eval_train", "eval_test"]:
    merge_path = merge_dir / split / "eval_results.json"
    base_path = base_dir / split / "eval_results.json"
    if not merge_path.is_file():
        continue
    merge_rows = {
        r["benchmark"]: r["mean_nll"]
        for r in json.loads(merge_path.read_text())["results"]
        if r.get("agent") == "merge-generalist"
    }
    if not base_path.is_file():
        print(f"=== {split}: merge-generalist only (pooled eval missing) ===")
        for b in benchmarks:
            print(f"  {labels[b]:<14} {merge_rows[b]:.4f}")
        print()
        continue
    base_rows = {
        r["benchmark"]: r["mean_nll"]
        for r in json.loads(base_path.read_text())["results"]
        if r.get("agent") == "pooled-baseline"
    }
    print(f"=== {split}: merge-generalist vs pooled-baseline ===")
    print(f"{'Axis':<14} {'Gen pair':>10} {'Pooled':>10} {'Δ(gen-pool)':>12} {'Pooled wins':>12}")
    pooled_wins = 0
    for b in benchmarks:
        g = merge_rows[b]
        p = base_rows[b]
        d = g - p
        win = d > 0  # lower NLL better, pooled wins if gen > pooled
        pooled_wins += int(win)
        mark = "✓" if win else ""
        print(f"{labels[b]:<14} {g:10.4f} {p:10.4f} {d:+12.4f} {mark:>12}")
    print(f"Pooled baseline beats merge-generalist on {pooled_wins}/{len(benchmarks)} axes\n")
