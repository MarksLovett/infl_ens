"""Replay closed-loop SFT batches on a single pooled baseline adapter.

Specialists in the influencer game each train on their routed subset of
each round's minibatch. This module trains **one** cumulative LoRA on the
**union** of all agents' routed examples for that round (the full routed
batch), using batches logged in ``history.json``.

That gives a fair data-volume match: specialists split the batch four ways;
the pooled baseline sees every example the specialists saw combined.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from infl_ens.data.trait_space import TraitSpace
from infl_ens.inflgame.router.agents import RouterAgent

PathLike = Union[str, Path]


def load_closed_loop_history(path: PathLike) -> list[dict[str, Any]]:
    """Load a closed-loop ``history.json`` file.

    :param path: Path to ``history.json``.
    :type path: str | pathlib.Path
    :returns: Per-round records.
    :rtype: list[dict]
    :raises ValueError: If the file is empty or missing required fields.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    if not records:
        raise ValueError(f"{p} is empty")
    if "agent_prompts" not in records[0]:
        raise ValueError(
            f"{p} missing agent_prompts; re-run closed loop with logging enabled"
        )
    return records


def pooled_batch_from_round(
    record: dict[str, Any],
    *,
    dedupe: Optional[bool] = None,
) -> tuple[list[str], list[str | None]]:
    """Reconstruct the full routed minibatch for one round.

    Under hard routing each prompt is routed to exactly one specialist, so
    concatenating all ``agent_prompts`` lists recovers the routed batch (up
    to ordering). Under soft routing with ``soft_top_k > 1`` a prompt appears
    in several agents' lists, and a naive concatenation would train the
    pooled baseline on that prompt several times per round — inflating its
    data volume relative to the specialists it is meant to match. Such
    rounds are therefore de-duplicated back to the underlying batch. The
    decision keys only on ``routing_mode`` / ``soft_top_k``: the soft
    ``soft_loss`` and ``position_update`` arms all log the same batch
    structure, so they replay identically.

    :param record: One round dict from ``history.json``.
    :type record: dict
    :param dedupe: Force de-duplication on or off. ``None`` (default)
        de-duplicates exactly when the round logged soft routing with
        ``soft_top_k > 1``.
    :type dedupe: bool | None
    :returns: ``(prompts, responses)`` with aligned lengths.
    :rtype: tuple[list[str], list[str | None]]
    """
    agent_prompts: dict[str, list[str]] = record["agent_prompts"]
    agent_responses: dict[str, list[str]] = record.get("agent_responses", {})

    if dedupe is None:
        dedupe = (
            str(record.get("routing_mode", "hard")) == "soft"
            and int(record.get("soft_top_k") or 1) > 1
        )

    if dedupe:
        # The round batch verbatim is the most faithful source; fall back to
        # per-agent batch indices, then to first-occurrence by prompt text.
        batch_prompts = record.get("batch_prompts")
        if batch_prompts:
            batch_responses = list(
                record.get("batch_responses") or [None] * len(batch_prompts)
            )
            return (
                [str(p) for p in batch_prompts],
                [r if r else None for r in batch_responses],
            )
        agent_indices = record.get("agent_batch_indices") or {}
        aligned = bool(agent_indices) and all(
            len(agent_indices.get(name, [])) == len(agent_prompts[name])
            for name in agent_prompts
        )
        if aligned:
            by_index: dict[int, tuple[str, str | None]] = {}
            for name in sorted(agent_prompts.keys()):
                p_list = list(agent_prompts[name])
                r_list = list(agent_responses.get(name, []))
                for i, p in enumerate(p_list):
                    r = r_list[i] if i < len(r_list) and r_list[i] else None
                    by_index.setdefault(int(agent_indices[name][i]), (p, r))
            ordered = [by_index[k] for k in sorted(by_index)]
            return [p for p, _ in ordered], [r for _, r in ordered]

        seen: set[str] = set()
        prompts: list[str] = []
        responses: list[str | None] = []
        for name in sorted(agent_prompts.keys()):
            p_list = list(agent_prompts[name])
            r_list = list(agent_responses.get(name, []))
            for i, p in enumerate(p_list):
                if p in seen:
                    continue
                seen.add(p)
                prompts.append(p)
                responses.append(
                    r_list[i] if i < len(r_list) and r_list[i] else None
                )
        return prompts, responses

    prompts = []
    responses = []
    for name in sorted(agent_prompts.keys()):
        p_list = list(agent_prompts[name])
        r_list = list(agent_responses.get(name, []))
        if r_list and len(r_list) == len(p_list):
            for p, r in zip(p_list, r_list):
                prompts.append(p)
                responses.append(r if r else None)
        else:
            prompts.extend(p_list)
            responses.extend([None] * len(p_list))
    return prompts, responses


def replay_pooled_baseline_sft(
    history_path: PathLike,
    agent: RouterAgent,
    sft_cfg: Any,
    *,
    project: Any,
    output_dir: PathLike,
    rounds: Optional[Sequence[int]] = None,
    save_per_round: bool = True,
    skip_empty_rounds: bool = True,
) -> list[dict[str, Any]]:
    """Train one pooled baseline adapter round-by-round from logged history.

    :param history_path: Source ``history.json`` from a closed-loop run.
    :type history_path: str | pathlib.Path
    :param agent: Mutable :class:`RouterAgent` (typically ``pooled-baseline``).
    :type agent: RouterAgent
    :param sft_cfg: :class:`~infl_ens.training.sft_training.SFTTrainingConfig`.
    :type sft_cfg: SFTTrainingConfig
    :param project: Trait-space projector (only used if position updates run).
    :type project: Callable
    :param output_dir: Root directory for adapters
        (``<output_dir>/pooled-baseline/round-NN``).
    :type output_dir: str | pathlib.Path
    :param rounds: Optional subset of round indices. ``None`` uses every
        round in the history file in order.
    :type rounds: Sequence[int] | None
    :param save_per_round: If ``True``, write ``round-NN`` subdirectories.
    :type save_per_round: bool
    :param skip_empty_rounds: Skip rounds where no prompts were routed.
    :type skip_empty_rounds: bool
    :returns: Per-round summary dicts including train metadata plus pooled
        batch and cumulative centroids in trait space.
    :rtype: list[dict]
    """
    from infl_ens.training.sft_training import sft_train_agent

    records = load_closed_loop_history(history_path)
    by_round = {int(r["round"]): r for r in records}
    target_rounds = (
        sorted(by_round.keys()) if rounds is None
        else sorted(int(r) for r in rounds)
    )

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    cumulative_sum: Optional[np.ndarray] = None
    cumulative_n = 0

    for r in target_rounds:
        rec = by_round.get(r)
        if rec is None:
            continue
        prompts, responses = pooled_batch_from_round(rec)
        if not prompts:
            if skip_empty_rounds:
                continue
            raise ValueError(f"round {r} has no routed prompts")
        coords = np.asarray(project(prompts), dtype=float)
        if coords.ndim != 2 or coords.shape[0] != len(prompts):
            raise ValueError(
                f"project returned shape {coords.shape}; expected "
                f"({len(prompts)}, n_axes)"
            )
        batch_centroid = coords.mean(axis=0)
        batch_sum = coords.sum(axis=0)
        if cumulative_sum is None:
            cumulative_sum = batch_sum.copy()
        else:
            cumulative_sum = cumulative_sum + batch_sum
        cumulative_n += len(prompts)
        cumulative_centroid = cumulative_sum / float(cumulative_n)
        agent.position = cumulative_centroid.copy()

        out_override = (
            str(out_root / agent.name / f"round-{r:02d}")
            if save_per_round else None
        )
        result = sft_train_agent(
            agent,
            prompts=prompts,
            responses=responses,
            cfg=sft_cfg,
            eval_prompts=prompts,
            project=project,
            blend=1.0,
            out_dir_override=out_override,
            skip_position_update=True,
        )
        summaries.append({
            "round": r,
            "n_train": result["n_train"],
            "train_loss": result.get("train_loss"),
            "output_dir": result["output_dir"],
            "loaded_prior_lora": result.get("loaded_prior_lora"),
            "batch_centroid": batch_centroid.tolist(),
            "cumulative_n_train": cumulative_n,
            "cumulative_centroid": cumulative_centroid.tolist(),
            "position": agent.position.tolist(),
        })
    return summaries


def make_pooled_baseline_agent(space: TraitSpace, name: str = "pooled-baseline") -> RouterAgent:
    """Create a placeholder agent for pooled baseline SFT.

    :param space: Trait space (position is set to the resource mean).
    :type space: TraitSpace
    :param name: Agent identifier.
    :type name: str
    :returns: Router agent with fixed position at :math:`\\mathbb{E}_B[b]`.
    :rtype: RouterAgent
    """
    return RouterAgent(name=name, position=np.asarray(space.mean, dtype=float))
