"""Verify closed-loop position updates against weighted/unweighted centroids.

Reads ``history.json`` from a closed-loop run and checks, per agent per
round, whether the recorded position delta matches:

- **weighted** centroid with ``(1-G)`` scores from ``agent_sample_weights``
- **unweighted** centroid of ``agent_prompts``

The pre-fix ``position_only`` bug applied weights only via ``sample_weights``
(so ``one_minus_G`` matched weighted, ``position_only`` matched unweighted).

Run on doob::

    python scripts/verify_position_update.py \\
        results/position_only_long_round_sweep/r10/history.json \\
        --config configs/benchmark/router/safety_truth_n4_r10_position_only_long_cum.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infl_ens.data.benchmarks import (  # noqa: E402
    build_safety_trait_space,
    load_beavertails,
    load_halueval,
)
from infl_ens.data.encoders import SentenceTransformerEncoder  # noqa: E402
from infl_ens.data.trait_space import TraitSpace, position_from_corpus  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_splits(cfg: dict[str, Any], repo_root: Path) -> list:
    splits = []
    for entry in cfg.get("benchmarks", []):
        kind = entry["kind"]
        path = repo_root / entry["path"]
        if kind == "beavertails":
            splits.append(load_beavertails(
                path, max_records=entry.get("max_records"),
            ))
        elif kind == "halueval":
            splits.append(load_halueval(
                path,
                tasks=entry.get("tasks"),
                max_records=entry.get("max_records"),
            ))
        else:
            raise ValueError(f"unknown benchmark kind: {kind!r}")
    return splits


def _build_trait_space(cfg: dict[str, Any], repo_root: Path) -> TraitSpace:
    splits = _load_splits(cfg, repo_root)
    ts_cfg = cfg.get("trait_space", {})
    encoder = SentenceTransformerEncoder(
        model_name=ts_cfg.get(
            "encoder", "sentence-transformers/all-MiniLM-L6-v2",
        ),
    )
    return build_safety_trait_space(
        splits, encoder,
        n_grid=int(ts_cfg.get("n_grid", 32)),
        kde_bandwidth=ts_cfg.get("kde_bandwidth"),
        threshold=float(ts_cfg.get("threshold", 0.5)),
    )


def verify_history(
    history_path: Path,
    space: TraitSpace,
    *,
    blend: float,
    rounds: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Check position deltas against weighted and unweighted centroid predictions.

    :param history_path: Path to ``history.json``.
    :type history_path: pathlib.Path
    :param space: Trait space for projection.
    :type space: TraitSpace
    :param blend: EMA blend coefficient from the run config.
    :type blend: float
    :param rounds: Round indices to check; default all rounds ``>= 1``.
    :type rounds: Sequence[int] | None
    :returns: Summary dict with per-round verdicts.
    :rtype: dict
    """
    with history_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    if not records:
        raise ValueError(f"{history_path} is empty")

    agent_names = list(records[0]["positions"].keys())
    prev_pos = {
        n: np.asarray(records[0]["positions"][n], dtype=float)
        for n in agent_names
    }
    round_indices = (
        list(rounds)
        if rounds is not None
        else list(range(1, len(records)))
    )

    summary: dict[str, Any] = {
        "history": str(history_path),
        "loss_reweight": records[0].get("loss_reweight"),
        "n_rounds": len(records),
        "blend": blend,
        "rounds": [],
    }

    n_weighted = 0
    n_unweighted = 0
    n_total = 0

    for r in round_indices:
        if r >= len(records):
            continue
        rec = records[r]
        round_info: dict[str, Any] = {"round": r, "agents": {}}
        for name in agent_names:
            prompts = rec.get("agent_prompts", {}).get(name, [])
            weights = rec.get("agent_sample_weights", {}).get(name, [])
            pos_before = prev_pos[name]
            pos_after = np.asarray(rec["positions"][name], dtype=float)
            actual_delta = pos_after - pos_before

            if not prompts:
                round_info["agents"][name] = {"status": "no_prompts"}
                continue

            uw_centroid = position_from_corpus(prompts, space.project)
            drift_uw = blend * (uw_centroid - pos_before)
            err_uw = float(np.linalg.norm(actual_delta - drift_uw))

            if weights and len(weights) == len(prompts):
                w_centroid = position_from_corpus(
                    prompts, space.project, scores=weights,
                )
                drift_w = blend * (w_centroid - pos_before)
                err_w = float(np.linalg.norm(actual_delta - drift_w))
                winner = "weighted" if err_w < err_uw else "unweighted"
                if winner == "weighted":
                    n_weighted += 1
                else:
                    n_unweighted += 1
                n_total += 1
                round_info["agents"][name] = {
                    "err_weighted": err_w,
                    "err_unweighted": err_uw,
                    "matches": winner,
                }
            else:
                round_info["agents"][name] = {
                    "err_unweighted": err_uw,
                    "matches": "unweighted_only",
                }

        summary["rounds"].append(round_info)
        prev_pos = {
            n: np.asarray(rec["positions"][n], dtype=float)
            for n in agent_names
        }

    summary["weighted_wins"] = n_weighted
    summary["unweighted_wins"] = n_unweighted
    summary["total_compared"] = n_total
    summary["verdict"] = (
        "WEIGHTED_CENTROID" if n_weighted > n_unweighted
        else "UNWEIGHTED_CENTROID" if n_unweighted > n_weighted
        else "TIE"
    )
    return summary


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
        "--repo-root", type=Path, default=ROOT,
        help="Repo root for benchmark paths (default: auto).",
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
    args = _build_parser().parse_args(argv)
    cfg = _load_yaml(args.config)
    space = _build_trait_space(cfg, args.repo_root)
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
            if "matches" in info and info["matches"] in ("weighted", "unweighted"):
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
