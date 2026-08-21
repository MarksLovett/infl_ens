#!/usr/bin/env python3
"""Build aligned / misaligned spread init positions from oracle geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MERGE_ORDER = [
    "merge-harm",
    "merge-hallucination",
    "merge-privacy",
    "merge-overrefusal",
    "merge-policy",
]
CLONE_GROUPS: list[tuple[str, str]] = [
    ("clone-0", "clone-1"),
    ("clone-2", "clone-3"),
    ("clone-4", "clone-5"),
    ("clone-6", "clone-7"),
    ("clone-8", "clone-9"),
]


def _centers_for_merge(geometry: dict, merge: str) -> list[list[float]]:
    """Return two k-means centers for a merge."""
    row = geometry["per_merge"][merge]
    centers = row.get("suggested_sub_agent_positions") or []
    if len(centers) != 2:
        raise ValueError(f"{merge}: need exactly 2 sub-agent centers, got {len(centers)}")
    return centers


def build_positions(
    geometry: dict,
    *,
    mode: str,
) -> dict[str, list[float]]:
    """Map clones to positions for aligned or cyclic-misaligned spread."""
    centers_by_merge = {
        m: _centers_for_merge(geometry, m) for m in MERGE_ORDER
    }
    if mode == "aligned":
        assignment = centers_by_merge
    elif mode == "misaligned":
        assignment = {
            MERGE_ORDER[i]: centers_by_merge[MERGE_ORDER[(i + 1) % len(MERGE_ORDER)]]
            for i in range(len(MERGE_ORDER))
        }
    else:
        raise ValueError(f"mode must be aligned or misaligned, got {mode!r}")

    positions: dict[str, list[float]] = {}
    for merge, (c0, c1) in zip(MERGE_ORDER, CLONE_GROUPS, strict=True):
        positions[c0] = [float(v) for v in assignment[merge][0]]
        positions[c1] = [float(v) for v in assignment[merge][1]]
    return positions


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry",
        default="results/attribution_2x2/ga_theory_pre/seed0/merge_oracle_geometry.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/within_merge_spread_init",
    )
    args = parser.parse_args()

    geometry = json.loads(Path(args.geometry).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"source_geometry": args.geometry, "arms": {}}
    for mode in ("aligned", "misaligned"):
        positions = build_positions(geometry, mode=mode)
        path = out_dir / f"fixed_positions_{mode}.json"
        payload = {
            "init_mode": f"oracle_k2_{mode}",
            "geometry_source": args.geometry,
            "positions": positions,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        manifest["arms"][mode] = str(path)
        print(f"wrote {path}")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
