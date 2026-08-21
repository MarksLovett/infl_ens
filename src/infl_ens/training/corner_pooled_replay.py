"""Replay corner-pooled SFT: one LoRA per corner on the merged pair batch.

Uses the same per-round ``agent_prompts`` as a closed-loop run where merge
training saw the union of two co-located routers. Trains **one** cumulative
LoRA per harm corner (primary trainer only), for comparison with merge
(two names, one LoRA per pair label) and per-clone specialists (split batch).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from infl_ens.inflgame.router.agents import RouterAgent
from infl_ens.training.baseline_replay import load_closed_loop_history
from infl_ens.training.merge_training import merge_routed_batch
from infl_ens.utils.agent_init import harm_pair_indices

PathLike = Union[str, Path]

LOW_TRAINER = "single-pool-low"
HIGH_TRAINER = "single-pool-high"


def corner_pairs_from_round(
    record: dict[str, Any],
    router_names: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Split routers into low- and high-harm pairs from round positions.

    :param record: One ``history.json`` round record.
    :type record: dict
    :param router_names: Clone names in row order.
    :type router_names: Sequence[str]
    :returns: ``(low_harm_members, high_harm_members)`` sorted names.
    :rtype: tuple[list[str], list[str]]
    """
    positions = np.stack(
        [np.asarray(record["positions"][n], dtype=float) for n in router_names],
        axis=0,
    )
    low_idx, high_idx = harm_pair_indices(positions)
    low_names = sorted(router_names[int(i)] for i in low_idx)
    high_names = sorted(router_names[int(i)] for i in high_idx)
    return low_names, high_names


def replay_corner_pooled_single_trainer(
    history_path: PathLike,
    *,
    sft_cfg: Any,
    project: Any,
    output_dir: PathLike,
    router_names: Sequence[str] | None = None,
    rounds: Optional[Sequence[int]] = None,
    save_per_round: bool = True,
    loss_reweight: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Train one LoRA per corner on merged pair batches from logged history.

    :param history_path: Closed-loop ``history.json`` (e.g. proximity+specialists).
    :type history_path: str | pathlib.Path
    :param sft_cfg: SFT configuration (cumulative LoRA recommended).
    :type sft_cfg: SFTTrainingConfig
    :param project: Trait-space projector (unused; position updates skipped).
    :type project: Callable
    :param output_dir: Root for ``single-pool-low`` / ``single-pool-high``.
    :type output_dir: str | pathlib.Path
    :param router_names: Router agent names; default four clones.
    :type router_names: Sequence[str] | None
    :param rounds: Optional round indices to replay.
    :type rounds: Sequence[int] | None
    :param save_per_round: Write ``round-NN`` adapter checkpoints.
    :type save_per_round: bool
    :param loss_reweight: If ``one_minus_G``, use logged ``agent_sample_weights``.
    :type loss_reweight: str | None
    :returns: Per-round training summaries.
    :rtype: list[dict]
    """
    from infl_ens.training.sft_training import sft_train_agent

    names = list(router_names or ["clone-0", "clone-1", "clone-2", "clone-3"])
    records = load_closed_loop_history(history_path)
    by_round = {int(r["round"]): r for r in records}
    target_rounds = (
        sorted(by_round.keys()) if rounds is None
        else sorted(int(r) for r in rounds)
    )

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    agent_low = RouterAgent(name=LOW_TRAINER, position=np.zeros(2))
    agent_high = RouterAgent(name=HIGH_TRAINER, position=np.zeros(2))

    summaries: list[dict[str, Any]] = []
    for r in target_rounds:
        rec = by_round[r]
        low_m, high_m = corner_pairs_from_round(rec, names)
        agent_prompts = rec["agent_prompts"]
        agent_responses = rec.get("agent_responses", {})
        weights = rec.get("agent_sample_weights", {})

        for train_agent, members, role in (
            (agent_low, low_m, "low"),
            (agent_high, high_m, "high"),
        ):
            mp, mr, mw = merge_routed_batch(
                agent_prompts,
                agent_responses,
                members,
                agent_sample_weights=weights if weights else None,
                loss_reweight=loss_reweight,
            )
            if not mp:
                continue
            sw = mw if loss_reweight == "one_minus_G" else None
            out_override = (
                str(out_root / train_agent.name / f"round-{r:02d}")
                if save_per_round else None
            )
            result = sft_train_agent(
                train_agent,
                prompts=mp,
                responses=mr if any(mr) else None,
                cfg=sft_cfg,
                eval_prompts=mp,
                project=project,
                out_dir_override=out_override,
                sample_weights=sw,
                skip_position_update=True,
            )
            summaries.append({
                "round": r,
                "corner": role,
                "train_name": train_agent.name,
                "members": members,
                "n_train": result["n_train"],
                "train_loss": result.get("train_loss"),
                "output_dir": result["output_dir"],
            })
    return summaries
