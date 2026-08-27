"""Audit closed-loop position updates against centroid predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from infl_ens.data.benchmarks import build_safety_trait_space
from infl_ens.data.trait_space_cache import make_trait_space_encoder
from infl_ens.data.trait_space import TraitSpace, position_from_corpus
from infl_ens.evaluation.benchmarks import load_benchmark_splits


def build_trait_space_from_config(cfg: dict[str, Any], repo_root: Path) -> TraitSpace:
    """Rebuild the trait space described in a training YAML config.

    :param cfg: Parsed training config.
    :type cfg: dict
    :param repo_root: Repository root for benchmark paths.
    :type repo_root: pathlib.Path
    :returns: Trait space for projection.
    :rtype: TraitSpace
    """
    entries: list[dict[str, Any]] = []
    for entry in cfg.get("benchmarks", []):
        resolved = dict(entry)
        resolved["path"] = str(repo_root / entry["path"])
        entries.append(resolved)
    splits = load_benchmark_splits(entries)
    ts_cfg = cfg.get("trait_space", {})
    encoder = make_trait_space_encoder(cfg)
    return build_safety_trait_space(
        splits,
        encoder,
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
        "position_update": records[0].get("position_update"),
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


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML training config.

    :param path: Config path.
    :type path: pathlib.Path
    :returns: Parsed config dict.
    :rtype: dict
    """
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
