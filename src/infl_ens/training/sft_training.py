"""LoRA SFT trainer for router agents.

In the closed loop described in :mod:`closed_loop_demo`, each
:class:`infl_ens.inflgame.router.RouterAgent` is a small language model
(default: ``Qwen/Qwen2.5-1.5B-Instruct``) that is fine-tuned on the
queries it receives. This module implements that fine-tuning step using
PEFT/LoRA, then re-estimates the agent's trait-space position from a
held-out eval corpus.

Design points
-------------

- The trainer accepts queries plus *optional* per-query target responses.
  For the benchmarks shipped here (BeaverTails, HaluEval) targets are
  available as ``BenchmarkSplit.responses``; for a generic closed loop
  you can pass ``None`` and run plain causal LM next-token prediction on
  the prompt itself (useful for "domain priming").
- LoRA adapters are saved under ``results/<run_id>/agents/<name>/`` so the
  same base weights are shared across all agents, drastically cutting
  per-agent disk.
- After training, the agent's :meth:`RouterAgent.update_position_from_corpus`
  is called with a caller-supplied eval corpus and projector. The trainer
  does not embed prompts itself; that responsibility stays with the
  trait-space encoder (AGENTS.md §3: data subpackage owns projection).

Heavy dependencies (``torch``, ``transformers``, ``peft``, ``trl``,
``datasets``) are imported lazily so this module can be inspected without
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from infl_ens.inflgame.router.agents import RouterAgent


@dataclass
class SFTTrainingConfig:
    """Hyperparameters for one LoRA SFT round of one agent.

    :param base_model: HuggingFace repo id of the base model.
    :type base_model: str
    :param output_dir: Per-agent output directory. LoRA adapters and the
        tokenizer config are written here.
    :type output_dir: str
    :param max_seq_length: Token cap per training example.
    :type max_seq_length: int
    :param per_device_batch_size: Batch size per device.
    :type per_device_batch_size: int
    :param gradient_accumulation_steps: Number of micro-batches per
        optimiser step.
    :type gradient_accumulation_steps: int
    :param num_train_epochs: Epochs over the training corpus.
    :type num_train_epochs: float
    :param learning_rate: Peak learning rate.
    :type learning_rate: float
    :param lora_r: LoRA rank.
    :type lora_r: int
    :param lora_alpha: LoRA alpha scaling.
    :type lora_alpha: int
    :param lora_dropout: LoRA dropout probability.
    :type lora_dropout: float
    :param lora_target_modules: Modules to attach LoRA adapters to.
        Defaults to all attention projections in the Qwen2.5 family.
    :type lora_target_modules: tuple[str, ...]
    :param bf16: Whether to train in bfloat16 (recommended on A100; falls
        back to fp16 on A40 if the device does not support bf16).
    :type bf16: bool
    :param gradient_checkpointing: Whether to enable activation
        checkpointing (saves memory at moderate compute cost).
    :type gradient_checkpointing: bool
    :param logging_steps: How often (in optimiser steps) to log training
        loss. With small per-round batches (e.g. 4 optimiser steps), the
        default value of ``10`` produces no per-step records — only a
        final summary. Set to ``1`` for full per-step capture, which is
        what :mod:`scripts.probe_sft_capability`'s Tier 1 panel needs.
    :type logging_steps: int
    :param cumulative_lora: If ``True``, when ``agent.metadata['lora_dir']``
        points to an existing adapter from a previous round, that adapter
        is loaded and **kept training** rather than being replaced by a
        fresh LoRA. The agent's capability therefore accumulates across
        rounds. Default ``False`` preserves the original "independent
        rounds, fresh base model each call" semantics.

        Notes:

        - Only applies when a prior ``lora_dir`` exists on the agent's
          metadata. The first round of any agent always trains a fresh
          LoRA regardless of this flag.
        - The flag is independent of ``closed_loop.save_per_round`` —
          cumulative training can be combined with either flat or
          per-round adapter archiving.
        - Catastrophic forgetting becomes possible across rounds; pair
          with the capability probe to detect it (look for diagonal NLL
          dropping while off-diagonal stays flat or rises).
    :type cumulative_lora: bool
    :param seed: RNG seed.
    :type seed: int
    """

    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "results/sft"
    max_seq_length: int = 1024
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = field(default_factory=lambda: (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ))
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 1
    cumulative_lora: bool = False
    seed: int = 0


def _format_chat(prompt: str, response: Optional[str]) -> str:
    """Format one (prompt, response) example as a chat-style string.

    Used as the default formatting hook. For Qwen2.5-Instruct this is
    accepted by ``SFTTrainer`` directly; for other model families pass a
    custom ``formatting_func`` to :func:`sft_train_agent`.

    :param prompt: User prompt.
    :type prompt: str
    :param response: Optional assistant response.
    :type response: str | None
    :returns: A single training string.
    :rtype: str
    """
    if response:
        return (
            "<|im_start|>user\n" + prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n" + response + "<|im_end|>"
        )
    return "<|im_start|>user\n" + prompt + "<|im_end|>"


def sft_train_agent(
    agent: RouterAgent,
    prompts: Sequence[str],
    responses: Optional[Sequence[Optional[str]]],
    cfg: SFTTrainingConfig,
    *,
    eval_prompts: Sequence[str],
    project: Callable[[Sequence[str]], np.ndarray],
    blend: float = 1.0,
    formatting_func: Optional[Callable[[str, Optional[str]], str]] = None,
    out_dir_override: Optional[str] = None,
) -> dict[str, Any]:
    """Run one LoRA SFT round for a single agent and refresh its position.

    :param agent: Agent to train (mutated in place: ``agent.position`` is
        re-estimated, ``agent.metadata['lora_dir']`` is set).
    :type agent: RouterAgent
    :param prompts: Training prompts for this agent (from the router's
        per-round routing decisions).
    :type prompts: Sequence[str]
    :param responses: Optional target responses, same length as
        ``prompts``. Items may be ``None`` for prompt-only training.
    :type responses: Sequence[str | None] | None
    :param cfg: Training configuration.
    :type cfg: SFTTrainingConfig
    :param eval_prompts: Held-out prompts used to re-estimate the agent's
        trait-space position after training.
    :type eval_prompts: Sequence[str]
    :param project: Trait-space projector (from
        :class:`infl_ens.data.trait_space.TraitSpace.project`).
    :type project: Callable[[Sequence[str]], numpy.ndarray]
    :param blend: EMA coefficient for the position update, in ``[0, 1]``.
    :type blend: float
    :param formatting_func: Optional ``(prompt, response) -> str`` hook
        overriding the default Qwen2.5 chat template.
    :type formatting_func: Callable[[str, str | None], str] | None
    :param out_dir_override: Optional explicit output directory for the
        adapter. When ``None``, defaults to ``Path(cfg.output_dir) /
        agent.name`` (current behaviour). When provided, the adapter is
        saved exactly at the given path — useful for per-round adapter
        archiving (e.g. ``agents/clone-0/round-03``) so a capability probe
        can reload each round's checkpoint independently.
    :type out_dir_override: str | None
    :returns: Dictionary with keys ``output_dir`` (path to LoRA adapter),
        ``n_train`` (training examples), ``train_loss`` (final train loss
        if available), and ``log_history`` (per-step log records emitted by
        the SFT trainer; usable for plotting per-round loss curves).
    :rtype: dict
    :raises ImportError: If ``transformers``/``peft``/``trl``/``datasets``
        are not installed.
    :raises ValueError: If ``prompts`` is empty or shape-mismatched with
        ``responses``.
    """
    if not prompts:
        raise ValueError("prompts must be non-empty")
    if responses is not None and len(responses) != len(prompts):
        raise ValueError(
            f"responses length {len(responses)} != prompts length {len(prompts)}"
        )

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTTrainer
        # trl >= 0.12: SFTConfig replaces TrainingArguments for SFT and
        # absorbs dataset_text_field / max_seq_length. Older trl falls back
        # to the transformers TrainingArguments class.
        try:
            from trl import SFTConfig as _SFTConfig
        except ImportError:  # pragma: no cover - old trl
            from transformers import TrainingArguments as _SFTConfig
    except ImportError as exc:  # pragma: no cover - environment-level
        raise ImportError(
            "sft_train_agent requires transformers, peft, trl, datasets, and torch; "
            "install with `pip install transformers peft trl datasets accelerate`."
        ) from exc

    # Seed everything; fall back if utils.seeding is unavailable.
    try:
        from infl_ens.utils.seeding import seed_all
        seed_all(cfg.seed)
    except ImportError:  # pragma: no cover
        import random
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    fmt = formatting_func or _format_chat
    rs = responses if responses is not None else [None] * len(prompts)
    texts = [fmt(p, r) for p, r in zip(prompts, rs)]
    ds = Dataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16 = bool(cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16 else torch.float16
    # transformers >= 5.x uses `dtype`; earlier versions use `torch_dtype`.
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=dtype)
    except TypeError:  # pragma: no cover - old transformers
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype)

    # If cumulative training is requested and the agent already has an
    # adapter from a previous round, load that adapter as a trainable PEFT
    # wrapper. trl's SFTTrainer accepts pre-wrapped PEFT models and will
    # skip its own peft_config initialisation when peft_config is omitted.
    prior_lora: Optional[Path] = None
    if cfg.cumulative_lora:
        prior = agent.metadata.get("lora_dir")
        if prior:
            p = Path(prior)
            # Look for either adapter_model.safetensors (new) or
            # adapter_model.bin (old). Skip silently if the dir was deleted.
            if p.exists() and (
                (p / "adapter_model.safetensors").exists()
                or (p / "adapter_model.bin").exists()
            ):
                prior_lora = p

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    if prior_lora is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, str(prior_lora), is_trainable=True,
        )
        # Suppress SFTTrainer's own PEFT initialisation since the model is
        # already wrapped — passing peft_config alongside a PeftModel
        # double-wraps and breaks gradient flow.
        peft_config_for_trainer: Optional[LoraConfig] = None
    else:
        peft_config_for_trainer = lora_cfg

    if out_dir_override is not None:
        out_dir = Path(out_dir_override)
    else:
        out_dir = Path(cfg.output_dir) / agent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # SFTConfig (trl >= 0.12) absorbs dataset_text_field + max_length /
    # max_seq_length. Field name changed across trl versions; try modern
    # first, fall back gracefully. Older trl with TrainingArguments will
    # ignore these dataset-specific kwargs and let SFTTrainer kwargs route
    # them through (handled below).
    base_args = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        save_strategy="no",  # we save the adapter manually below
        report_to=[],
        seed=cfg.seed,
    )
    try:
        train_args = _SFTConfig(
            dataset_text_field="text",
            max_length=cfg.max_seq_length,
            **base_args,
        )
    except TypeError:
        try:
            train_args = _SFTConfig(
                dataset_text_field="text",
                max_seq_length=cfg.max_seq_length,
                **base_args,
            )
        except TypeError:  # pragma: no cover - very old trl / TrainingArguments
            train_args = _SFTConfig(**base_args)

    # `tokenizer` was renamed to `processing_class` in trl >= 0.12.
    trainer_kwargs: dict = dict(
        model=model,
        args=train_args,
        train_dataset=ds,
    )
    if peft_config_for_trainer is not None:
        trainer_kwargs["peft_config"] = peft_config_for_trainer
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:  # pragma: no cover - old trl
        trainer = SFTTrainer(
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
            **trainer_kwargs,
        )
    result = trainer.train()
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Refresh the agent's trait-space position from the eval corpus.
    agent.update_position_from_corpus(list(eval_prompts), project, blend=blend)
    agent.metadata["lora_dir"] = str(out_dir)
    agent.metadata["base_model"] = cfg.base_model

    # trl exposes per-step records (loss / learning_rate / epoch / step) on
    # trainer.state.log_history; coerce to a plain list of dicts for JSON.
    log_history = list(getattr(getattr(trainer, "state", None),
                                "log_history", []) or [])
    return {
        "output_dir": str(out_dir),
        "n_train": len(prompts),
        "train_loss": float(getattr(result, "training_loss", float("nan"))),
        "log_history": log_history,
        "loaded_prior_lora": str(prior_lora) if prior_lora is not None else None,
    }
