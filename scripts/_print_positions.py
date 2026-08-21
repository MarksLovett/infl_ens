"""Print agent positions from history.json or fixed_positions.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else str(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        entry = payload[-1] if len(sys.argv) <= 3 else payload[int(sys.argv[3])]
        pos = entry["positions"]
        print(f"=== {label} (round {entry['round']}) ===")
    else:
        pos = payload.get("positions", payload)
        print(f"=== {label} ===")
    for name in sorted(pos, key=lambda x: int(x.split("-")[1])):
        v = pos[name]
        print(f"  {name}: [{', '.join(f'{x:.3f}' for x in v)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
