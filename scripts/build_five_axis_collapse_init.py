#!/usr/bin/env python3
"""Extract 10-clone / 5-axis init positions for the collapse experiment.

Takes ``six_axis_theory_n12`` positions for ``clone-0`` … ``clone-9`` and
drops the injection coordinate (index 4 in the six-axis trait order:
harm, hallucination, privacy, overrefusal, injection, policy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SIX_AXIS_ORDER = (
    "harm",
    "hallucination",
    "privacy",
    "overrefusal",
    "injection",
    "policy_violation",
)
INJECTION_AXIS_IDX = SIX_AXIS_ORDER.index("injection")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="results/six_axis_theory_n12/fixed_positions.json",
    )
    parser.add_argument(
        "--output",
        default="results/five_axis_collapse_init/fixed_positions.json",
    )
    parser.add_argument("--n-clones", type=int, default=10)
    args = parser.parse_args()

    src = json.loads(Path(args.source).read_text(encoding="utf-8"))
    positions_in = src["positions"]
    out_pos: dict[str, list[float]] = {}
    for i in range(args.n_clones):
        name = f"clone-{i}"
        vec = positions_in[name]
        if len(vec) != len(SIX_AXIS_ORDER):
            raise ValueError(
                f"{name}: expected {len(SIX_AXIS_ORDER)}-D position, got {len(vec)}",
            )
        out_pos[name] = [
            float(v) for j, v in enumerate(vec) if j != INJECTION_AXIS_IDX
        ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": args.source,
        "dropped_axis": "injection",
        "dropped_index": INJECTION_AXIS_IDX,
        "positions": out_pos,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({args.n_clones} clones, L={len(next(iter(out_pos.values())))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
