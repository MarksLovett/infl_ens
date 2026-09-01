"""Task registry for ``python -m infl_ens.training``.

- ``closed_loop``: :func:`infl_ens.training.closed_loop.run_closed_loop`.
- ``baseline_replay``: :func:`run_baseline_replay`, one pooled cumulative
  LoRA trained on the union of the per-round routed batches logged in an
  existing ``history.json`` (see :mod:`infl_ens.training.baseline_replay`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from infl_ens.config import resolve_sft_block
from infl_ens.training.closed_loop import run_closed_loop
from infl_ens.training.setup import load_splits, make_trait_space


def run_baseline_replay(cfg: dict[str, Any]) -> int:
    """Replay pooled baseline SFT from a closed-loop ``history.json``.

    :param cfg: Configuration with ``history_path``, ``output_dir``,
        ``benchmarks``, ``trait_space``, ``sft``, and ``baseline_replay``.
    :type cfg: dict
    :returns: Exit code.
    :rtype: int
    """
    from infl_ens.training.baseline_replay import (
        make_pooled_baseline_agent,
        replay_pooled_baseline_sft,
    )
    from infl_ens.training.sft_training import SFTTrainingConfig

    hist_path = Path(cfg["history_path"])
    if not hist_path.is_file():
        raise FileNotFoundError(hist_path)

    splits = load_splits(cfg)
    space = make_trait_space(cfg, splits)
    br = cfg.get("baseline_replay", {})
    agent_name = str(br.get("agent_name", "pooled-baseline"))
    agent = make_pooled_baseline_agent(space, name=agent_name)

    sft_cfg_dict = resolve_sft_block(cfg)
    sft_cfg = SFTTrainingConfig(**sft_cfg_dict)
    rounds = br.get("rounds")

    summaries = replay_pooled_baseline_sft(
        hist_path,
        agent,
        sft_cfg,
        project=space.project,
        output_dir=Path(sft_cfg.output_dir),
        rounds=rounds,
        save_per_round=bool(br.get("save_per_round", True)),
    )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "replay_summary.json"
    history_path = out_dir / "history.json"
    baseline_history = [
        {
            "round": s["round"],
            "positions": {agent_name: s["position"]},
            "batch_centroid": s["batch_centroid"],
            "cumulative_centroid": s["cumulative_centroid"],
            "n_train": s["n_train"],
            "cumulative_n_train": s["cumulative_n_train"],
            "output_dir": s["output_dir"],
        }
        for s in summaries
    ]
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "history_path": str(hist_path.resolve()),
                "agent": agent_name,
                "n_rounds": len(summaries),
                "rounds": summaries,
                "centroid_history_path": str(history_path),
            },
            fh,
            indent=2,
        )
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(baseline_history, fh, indent=2)
    print(
        f"baseline replay done: {len(summaries)} rounds, wrote "
        f"{summary_path} and {history_path}"
    )
    return 0


#: ``task`` value -> runner. Every runner takes the resolved config and
#: returns a process exit code.
TASKS: dict[str, Callable[[dict[str, Any]], int]] = {
    "closed_loop": run_closed_loop,
    "baseline_replay": run_baseline_replay,
}

__all__ = ["TASKS", "run_baseline_replay", "run_closed_loop"]
