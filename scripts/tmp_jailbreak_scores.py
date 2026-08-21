import json
from pathlib import Path

root = Path("/home/mlovett/infl_ens/results/seven_axis_pair_merge_split/seed0")
agents_order = [
    "merge-harm", "merge-hallucination", "merge-privacy", "merge-injection",
    "merge-overrefusal", "merge-policy", "merge-generalist",
]
for split in ["eval_train", "eval_test"]:
    rows = {
        r["agent"]: r["mean_nll"]
        for r in json.loads((root / split / "eval_results.json").read_text())["results"]
        if r["benchmark"] == "toxicchat"
    }
    print(f"=== {split} (toxicchat / jailbreak) ===")
    ranked = sorted(rows.items(), key=lambda x: x[1])
    for a, nll in ranked:
        tag = ""
        if a == "merge-injection":
            tag = " [injection pair]"
        elif a == "merge-generalist":
            tag = " [generalist]"
        print(f"  {a:22s} {nll:.4f}{tag}")
    gen = rows["merge-generalist"]
    print("  vs generalist:")
    for a in agents_order:
        if a == "merge-generalist":
            continue
        d = rows[a] - gen
        print(f"    {a:22s} {d:+.4f}")
