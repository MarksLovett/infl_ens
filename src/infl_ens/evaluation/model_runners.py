"""Resolve and load behavioral targets from experiment arms."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from infl_ens.config import resolve_sft_block
from infl_ens.evaluation.adapters import discover_adapters, load_base_causal_lm
from infl_ens.experiment import ArmSpec, ExperimentConfig


@dataclass(frozen=True)
class BehavioralTarget:
    """One model configuration evaluated by the behavioral stage.

    :param name: Unique experiment-level target name.
    :type name: str
    :param kind: ``base``, ``adapter``, ``routed`` or ``mixture_lora``.
    :type kind: str
    :param base_model: Hugging Face base-model identifier.
    :type base_model: str
    :param arm_name: Source experiment arm, or ``None`` for the base model.
    :type arm_name: str | None
    :param run_dir: Source run directory, if applicable.
    :type run_dir: pathlib.Path | None
    :param round_idx: Final training round, if applicable.
    :type round_idx: int | None
    :param checkpoint_paths: Ordered adapter/mixture checkpoints.
    :type checkpoint_paths: tuple[pathlib.Path, ...]
    :param expert_names: Names aligned with checkpoint paths.
    :type expert_names: tuple[str, ...]
    :param config: Resolved arm configuration.
    :type config: Mapping[str, Any]
    """

    name: str
    kind: str
    base_model: str
    arm_name: str | None = None
    run_dir: Path | None = None
    round_idx: int | None = None
    checkpoint_paths: tuple[Path, ...] = ()
    expert_names: tuple[str, ...] = ()
    config: Mapping[str, Any] | None = None

    @property
    def artifact_name(self) -> str:
        """Return a filesystem-safe target name.

        :returns: Sanitized target identifier.
        :rtype: str
        """
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name)


def _resolved_config(arm: ArmSpec) -> dict[str, Any]:
    from infl_ens.config import load_config

    path = arm.run_dir / "resolved_config.yaml"
    return load_config(path, validate=False) if path.is_file() else arm.load()


def _last_round(arm: ArmSpec) -> int:
    history = json.loads((arm.run_dir / "history.json").read_text(encoding="utf-8"))
    if not history:
        raise ValueError(f"{arm.name}: history is empty")
    return int(history[-1]["round"])


def _mixture_target(arm: ArmSpec, cfg: Mapping[str, Any], round_idx: int) -> BehavioralTarget:
    mix = cfg.get("mixture_lora") or {}
    model_name = (
        "ewora-dense"
        if str(mix.get("mode", "dense")) == "dense"
        else "trait-gated-topk"
    )
    checkpoint = arm.run_dir / "agents" / model_name / f"round-{round_idx:02d}"
    if not (checkpoint / "mixture_config.json").is_file():
        raise FileNotFoundError(f"missing mixture checkpoint {checkpoint}")
    metadata = json.loads((checkpoint / "mixture_config.json").read_text(encoding="utf-8"))
    names = tuple(f"component-{index}" for index in range(int(metadata["n_experts"])))
    return BehavioralTarget(
        name=arm.name,
        kind="mixture_lora",
        base_model=str(metadata["base_model"]),
        arm_name=arm.name,
        run_dir=arm.run_dir,
        round_idx=round_idx,
        checkpoint_paths=(checkpoint,),
        expert_names=names,
        config=dict(cfg),
    )


def _adapter_targets(
    arm: ArmSpec,
    cfg: Mapping[str, Any],
    round_idx: int,
) -> list[BehavioralTarget]:
    refs = discover_adapters(arm.run_dir, rounds=[round_idx])
    if not refs:
        raise FileNotFoundError(f"{arm.name}: no adapters found at round {round_idx}")
    base_model = str(resolve_sft_block(dict(cfg))["base_model"])
    if str(cfg.get("task")) == "adapter_merge":
        return [
            BehavioralTarget(
                name=f"{arm.name}:{ref.agent}",
                kind="adapter",
                base_model=base_model,
                arm_name=arm.name,
                run_dir=arm.run_dir,
                round_idx=round_idx,
                checkpoint_paths=(ref.path,),
                expert_names=(ref.agent,),
                config=dict(cfg),
            )
            for ref in refs
        ]
    if arm.role == "generalist" or len(refs) == 1:
        ref = refs[0]
        return [BehavioralTarget(
            name=arm.name,
            kind="adapter",
            base_model=base_model,
            arm_name=arm.name,
            run_dir=arm.run_dir,
            round_idx=round_idx,
            checkpoint_paths=(ref.path,),
            expert_names=(ref.agent,),
            config=dict(cfg),
        )]

    from infl_ens.evaluation.routing_eval import parse_merge_groups, resolve_merge_adapters

    _clone_to_merge, config_names = parse_merge_groups(cfg.get("closed_loop") or {})
    if not config_names:
        names = [ref.agent for ref in refs]
        paths = [ref.path for ref in refs]
    else:
        names, _mapping = resolve_merge_adapters(arm.run_dir, round_idx, config_names)
        paths = [
            arm.run_dir / "agents" / name / f"round-{round_idx:02d}"
            for name in names
        ]
    return [BehavioralTarget(
        name=arm.name,
        kind="routed",
        base_model=base_model,
        arm_name=arm.name,
        run_dir=arm.run_dir,
        round_idx=round_idx,
        checkpoint_paths=tuple(paths),
        expert_names=tuple(names),
        config=dict(cfg),
    )]


def resolve_behavioral_targets(
    experiment: ExperimentConfig,
    *,
    selected_arms: Sequence[str] | None = None,
) -> list[BehavioralTarget]:
    """Resolve base and final-checkpoint targets from an experiment.

    :param experiment: Parsed experiment with behavioral settings.
    :type experiment: ExperimentConfig
    :param selected_arms: Optional command-line arm restriction.
    :type selected_arms: Sequence[str] | None
    :returns: Targets in stable experiment order.
    :rtype: list[BehavioralTarget]
    """
    settings = experiment.behavioral_eval
    if settings is None:
        return []
    if settings.target_arms == ("all",):
        wanted = {arm.name for arm in experiment.arms}
    else:
        wanted = set(settings.target_arms)
    if selected_arms:
        wanted &= set(selected_arms)
    configs = [_resolved_config(arm) for arm in experiment.arms if arm.name in wanted]
    if not configs:
        raise ValueError("behavioral evaluation selected no experiment arms")
    base_models = {str(resolve_sft_block(config)["base_model"]) for config in configs}
    if len(base_models) != 1:
        raise ValueError(
            "behavioral evaluation requires one shared base model; selected arms use "
            f"{sorted(base_models)}",
        )
    base_model = next(iter(base_models))
    targets: list[BehavioralTarget] = []
    if settings.include_base_model:
        targets.append(BehavioralTarget(name="base_model", kind="base", base_model=base_model))
    for arm in experiment.arms:
        if arm.name not in wanted:
            continue
        cfg = _resolved_config(arm)
        round_idx = _last_round(arm)
        if str(cfg.get("task")) == "mixture_lora":
            targets.append(_mixture_target(arm, cfg, round_idx))
        else:
            targets.extend(_adapter_targets(arm, cfg, round_idx))
    return targets


def load_multi_adapter_model(target: BehavioralTarget) -> tuple[Any, Any, Any, dict[str, str]]:
    """Load one PEFT model containing every adapter of a routed target.

    :param target: Routed target descriptor.
    :type target: BehavioralTarget
    :returns: ``(model, tokenizer, device, display_to_internal_name)``.
    :rtype: tuple[Any, Any, Any, dict[str, str]]
    :raises ValueError: If target is not routed.
    """
    if target.kind != "routed":
        raise ValueError("load_multi_adapter_model requires a routed target")
    from peft import PeftModel

    base, tokenizer, device = load_base_causal_lm(target.base_model)
    mapping = {
        name: f"behavioral_expert_{index}"
        for index, name in enumerate(target.expert_names)
    }
    first_name = target.expert_names[0]
    model = PeftModel.from_pretrained(
        base,
        str(target.checkpoint_paths[0]),
        adapter_name=mapping[first_name],
    )
    for name, path in zip(target.expert_names[1:], target.checkpoint_paths[1:]):
        model.load_adapter(str(path), adapter_name=mapping[name])
    model.to(device)
    model.eval()
    return model, tokenizer, device, mapping


__all__ = [
    "BehavioralTarget",
    "load_multi_adapter_model",
    "resolve_behavioral_targets",
]
