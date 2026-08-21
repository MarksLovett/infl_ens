import json
from collections import Counter
from pathlib import Path

hist = json.loads(Path("results/seven_axis_pair_merge_r40/seed0/history.json").read_text())
# last round sample
r = hist[-1]
batch_n = sum(len(v) for v in r["agent_prompts"].values())
print("round", r["round"], "total routed prompts", batch_n)
for name in sorted(r["agent_prompts"]):
    ps = r["agent_prompts"][name]
    c = Counter(ps)
    dups = sum(1 for k,v in c.items() if v>1)
    dup_ex = sum(v-1 for v in c.values() if v>1)
    print(f"  {name}: n={len(ps)} unique={len(c)} duplicate_instances={dup_ex}")

# merge-harm round 39
from infl_ens.training.merge_training import merge_routed_batch
members = ["clone-3","clone-6"]
mp, _, _ = merge_routed_batch(r["agent_prompts"], r.get("agent_responses",{}), members)
print("merge-harm merged n", len(mp), "unique", len(set(mp)))

# cross-round repeat for merge-harm over all rounds
all_m = []
for rec in hist[1:]:
    p, _, _ = merge_routed_batch(rec["agent_prompts"], rec.get("agent_responses",{}), members)
    all_m.extend(p)
print("merge-harm 40 rounds: total", len(all_m), "unique prompts", len(set(all_m)), "repeat rate", 1-len(set(all_m))/len(all_m))
