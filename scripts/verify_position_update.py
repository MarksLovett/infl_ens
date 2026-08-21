"""CLI: verify closed-loop position updates in history.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from infl_ens.training.history_audit import (
    build_trait_space_from_config,
    load_config,
    verify_history,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verify position updates in a closed-loop history.json.",
    )
    p.add_argument("history", type=Path, help="Path to history.json")
    p.add_argument(
        "--config", type=Path, required=True,
        help="YAML config used for the run (trait space rebuild).",
    )
    p.add_argument(
        "--repo-root", type=Path, default=_REPO_ROOT,
        help="Repo root for benchmark paths.",
    )
    p.add_argument(
        "--blend", type=float, default=None,
        help="EMA blend; default read from config closed_loop.blend.",
    )
    p.add_argument(
        "--rounds", type=int, nargs="*", default=None,
        help="Round indices to check (default: all >= 1).",
    )
    p.add_argument(
        "--json-out", type=Path, default=None,
        help="Optional path to write JSON summary.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)
    space = build_trait_space_from_config(cfg, args.repo_root)
    blend = float(
        args.blend
        if args.blend is not None
        else cfg.get("closed_loop", {}).get("blend", 0.5)
    )

    summary = verify_history(
        args.history, space, blend=blend, rounds=args.rounds,
    )

    print(f"history       : {summary['history']}")
    print(f"loss_reweight : {summary['loss_reweight']}")
    print(f"blend         : {summary['blend']}")
    print(
        f"verdict       : {summary['verdict']}  "
        f"(weighted wins {summary['weighted_wins']}/"
        f"{summary['total_compared']}, "
        f"unweighted wins {summary['unweighted_wins']}/"
        f"{summary['total_compared']})"
    )
    print()
    for rd in summary["rounds"]:
        print(f"Round {rd['round']}:")
        for name, info in rd["agents"].items():
            if info.get("status") == "no_prompts":
                print(f"  {name}: (no prompts)")
                continue
            if info.get("matches") in ("weighted", "unweighted"):
                print(
                    f"  {name}: err_w={info['err_weighted']:.2e}  "
                    f"err_uw={info['err_unweighted']:.2e}  "
                    f"-> {info['matches']}"
                )
            else:
                print(f"  {name}: err_uw={info.get('err_unweighted', float('nan')):.2e}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
