"""Print final trait positions from position_fix_comparison runs."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/position_fix_comparison")
for v in ("baseline", "expected_pool", "init_noise_1e-2", "pool_and_noise"):
    f = root / v / "sigma0.25" / "seed0" / "history.json"
    if not f.is_file():
        print(f"{v}: missing")
        continue
    last = json.loads(f.read_text())[-1]
    print(f"=== {v}  round={last['round']}  spread={last.get('pairwise_spread')}")
    print(json.dumps(last["positions"], indent=2))
