#!/usr/bin/env python3
"""Print a one-line-per-round summary of a soft-pair closed-loop run.

Reads the incrementally written ``history.json`` of a
``routing_mode: soft`` + ``sft_merge_groups`` run and reports, per round,
the per-pair prompt counts, the worst within-pair distance (which must stay
at zero: partners share one position by construction) and each pair's mean
share of the batch. Intended for polling a job while it is still running,
which is why it tolerates a partially written history.

Usage::

    python scripts/summarize_soft_pairs_history.py results/<arm>/seed0/history.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_round(record: dict[str, Any]) -> str:
    """Render one round as a single fixed-width line.

    :param record: One entry of ``history.json``.
    :type record: dict
    :returns: Formatted summary line.
    :rtype: str
    """
    counts = record.get("merge_prompt_counts") or {}
    share = record.get("pair_share_batch") or {}
    geometry = record.get("agent_geometry") or {}
    within = geometry.get("within_merge_l2") or {}
    worst = max(within.values()) if within else 0.0
    count_str = ",".join(str(counts[k]) for k in sorted(counts))
    share_str = ",".join(f"{share[k]:.3f}" for k in sorted(share))
    return (
        f"round {int(record.get('round', -1)):>3}  "
        f"prompts=[{count_str}]  "
        f"max_within_pair_l2={worst:.2e}  "
        f"share=[{share_str}]"
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "history",
        type=Path,
        nargs="?",
        default=Path("results/seven_axis_soft_pairs/seed0/history.json"),
        help="Path to a closed-loop history.json.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="Show only the last N rounds (0 = all).",
    )
    args = parser.parse_args()

    if not args.history.is_file():
        print(f"no history yet at {args.history}")
        return
    try:
        history = json.loads(args.history.read_text(encoding="utf-8"))
    except ValueError:
        print(f"{args.history} is mid-write; try again in a moment")
        return
    if not history:
        print(f"{args.history} is empty")
        return

    first = history[0]
    mode = first.get("routing_mode", "hard")
    units = first.get("soft_routing_units", "agents")
    print(
        f"routing_mode={mode} units={units} "
        f"soft_top_k={first.get('soft_top_k')} "
        f"soft_loss={first.get('soft_loss')} "
        f"position_update={first.get('position_update')} "
        f"rounds={len(history)}",
    )
    members = first.get("pair_members") or {}
    for name in sorted(members):
        print(f"  {name}: {members[name]}")
    rounds = history[-args.tail:] if args.tail else history
    for record in rounds:
        print(summarize_round(record))


if __name__ == "__main__":
    main()
