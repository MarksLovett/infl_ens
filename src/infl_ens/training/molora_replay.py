"""Train a MoLoRA model on the routed batches logged by a closed-loop run.

The data-matched counterpart of :mod:`infl_ens.training.baseline_replay`:
where the pooled generalist trains **one** LoRA on the union of every
round's routed batch, this module trains **one MoLoRA model**
(:mod:`infl_ens.training.molora`, :math:`E` gated low-rank experts) on
exactly the same batches in the same order.  Batches come from the source
run's ``history.json`` via
:func:`infl_ens.training.baseline_replay.pooled_batch_from_round`, so soft
top-:math:`k` rounds are de-duplicated back to the underlying minibatch.

Training mirrors :func:`infl_ens.training.sft_training.sft_train_agent`
round for round: a fresh HuggingFace ``Trainer`` (fresh optimiser and LR
schedule) per round, the same chat formatter, the same ``sft``
hyperparameters, and the model kept training across rounds.  No trait
space or agent position is involved; MoLoRA routes internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from infl_ens.training.baseline_replay import (
    load_closed_loop_history,
    pooled_batch_from_round,
)
from infl_ens.training.molora import (
    MoLoRAConfig,
    balance_loss,
    gate_usage,
    inject_molora,
    molora_param_count,
    save_molora,
)

PathLike = Union[str, Path]


class _LMCollator:
    """Pad pre-tokenised examples and build full-sequence LM labels.

    Pad positions are masked to ``-100`` so the loss matches the stock
    unit-weight SFT path used by the specialists and the generalist.

    :param tokenizer: HuggingFace tokenizer with a pad token.
    :type tokenizer: transformers.PreTrainedTokenizerBase
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Any) -> Any:
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"]} for f in features],
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def _build_molora_trainer(trainer_cls: Any, modules: Sequence[Any], balance_coef: float) -> Any:
    """``Trainer`` subclass adding ``balance_coef * balance_loss`` to the LM loss.

    :param trainer_cls: ``transformers.Trainer``.
    :type trainer_cls: type
    :param modules: Injected :class:`MoLoRALinear` wrappers.
    :type modules: Sequence[torch.nn.Module]
    :param balance_coef: Weight of the balance term; ``0`` leaves the loss
        untouched.
    :type balance_coef: float
    :returns: The subclass.
    :rtype: type
    """

    class MoLoRATrainer(trainer_cls):  # type: ignore[misc, valid-type]
        """Trainer whose loss is causal-LM CE plus the gate balance term."""

        def compute_loss(
            self,
            model: Any,
            inputs: Any,
            return_outputs: bool = False,
            **kwargs: Any,
        ) -> Any:
            outputs = model(**inputs)
            loss = outputs.loss
            if balance_coef > 0.0:
                bal = balance_loss(modules)
                if bal is not None:
                    loss = loss + balance_coef * bal.to(loss.dtype)
            return (loss, outputs) if return_outputs else loss

    return MoLoRATrainer


def train_molora_round(
    model: Any,
    tokenizer: Any,
    modules: Sequence[Any],
    molora_cfg: MoLoRAConfig,
    sft_cfg: Any,
    texts: Sequence[str],
    out_dir: PathLike,
) -> dict[str, Any]:
    """One SFT round of an injected MoLoRA model on ``texts``.

    :param model: Base causal LM already passed through
        :func:`infl_ens.training.molora.inject_molora`.
    :type model: transformers.PreTrainedModel
    :param tokenizer: Matching tokenizer.
    :type tokenizer: transformers.PreTrainedTokenizerBase
    :param modules: The injected wrappers (for diagnostics / balance loss).
    :type modules: Sequence[torch.nn.Module]
    :param molora_cfg: Expert / gate hyperparameters.
    :type molora_cfg: MoLoRAConfig
    :param sft_cfg: :class:`~infl_ens.training.sft_training.SFTTrainingConfig`
        (batch size, epochs, LR, precision, checkpointing, seed).
    :type sft_cfg: SFTTrainingConfig
    :param texts: Chat-formatted training strings.
    :type texts: Sequence[str]
    :param out_dir: Where the round's checkpoint is written.
    :type out_dir: str | pathlib.Path
    :returns: ``{"output_dir", "n_train", "train_loss", "gate_usage",
        "log_history"}``.  ``gate_usage`` is the mean gate probability per
        expert over the tokens of the round's last training step (its
        forward pass precedes that step's parameter update).
    :rtype: dict
    :raises ValueError: If ``texts`` is empty.
    """
    if not texts:
        raise ValueError("texts must be non-empty")
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    enc = tokenizer(list(texts), truncation=True, max_length=sft_cfg.max_seq_length)
    ds = Dataset.from_dict({"input_ids": enc["input_ids"]})

    bf16 = bool(sft_cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    fp16 = bool(not bf16 and torch.cuda.is_available())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_args: dict[str, Any] = dict(
        output_dir=str(out),
        per_device_train_batch_size=sft_cfg.per_device_batch_size,
        gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
        num_train_epochs=sft_cfg.num_train_epochs,
        learning_rate=sft_cfg.learning_rate,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=bool(sft_cfg.gradient_checkpointing),
        logging_steps=sft_cfg.logging_steps,
        save_strategy="no",
        report_to=[],
        seed=sft_cfg.seed,
        remove_unused_columns=False,
    )
    if sft_cfg.gradient_checkpointing:
        # The base is frozen, so the checkpointed blocks see no input that
        # requires grad; non-reentrant checkpointing plus input grads keeps
        # the expert gradients flowing (what PEFT does for LoRA).
        base_args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        if hasattr(model, "enable_input_require_grads") and not getattr(
            model, "_molora_input_grads_enabled", False,
        ):
            model.enable_input_require_grads()
            model._molora_input_grads_enabled = True  # one hook, not one per round
        if getattr(model, "config", None) is not None:
            model.config.use_cache = False
    try:
        train_args = TrainingArguments(**base_args)
    except TypeError:  # pragma: no cover - old transformers
        base_args.pop("gradient_checkpointing_kwargs", None)
        train_args = TrainingArguments(**base_args)

    trainer_cls = _build_molora_trainer(Trainer, modules, molora_cfg.balance_coef)
    collator = _LMCollator(tokenizer)
    kwargs: dict[str, Any] = dict(model=model, args=train_args, train_dataset=ds, data_collator=collator)
    try:
        trainer = trainer_cls(processing_class=tokenizer, **kwargs)
    except TypeError:  # pragma: no cover - transformers < 4.46
        trainer = trainer_cls(tokenizer=tokenizer, **kwargs)
    result = trainer.train()
    model.eval()
    save_molora(model, molora_cfg, out, base_model=sft_cfg.base_model)
    tokenizer.save_pretrained(str(out))
    log_history = list(getattr(getattr(trainer, "state", None), "log_history", []) or [])
    usage = gate_usage(modules)
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "output_dir": str(out),
        "n_train": len(texts),
        "train_loss": float(getattr(result, "training_loss", float("nan"))),
        "gate_usage": usage,
        "log_history": log_history,
    }


def replay_molora_sft(
    history_path: PathLike,
    molora_cfg: MoLoRAConfig,
    sft_cfg: Any,
    *,
    output_dir: PathLike,
    agent_name: str = "molora",
    rounds: Optional[Sequence[int]] = None,
    save_per_round: bool = True,
    skip_empty_rounds: bool = True,
    formatting_func: Optional[Callable[[str, Optional[str]], str]] = None,
) -> list[dict[str, Any]]:
    """Train one MoLoRA model round-by-round from a logged closed-loop history.

    :param history_path: Source ``history.json`` of a closed-loop run.
    :type history_path: str | pathlib.Path
    :param molora_cfg: Expert / gate hyperparameters.
    :type molora_cfg: MoLoRAConfig
    :param sft_cfg: :class:`~infl_ens.training.sft_training.SFTTrainingConfig`.
    :type sft_cfg: SFTTrainingConfig
    :param output_dir: Root for checkpoints
        (``<output_dir>/<agent_name>/round-NN``).
    :type output_dir: str | pathlib.Path
    :param agent_name: Checkpoint sub-directory name.
    :type agent_name: str
    :param rounds: Optional subset of round indices; ``None`` replays every
        logged round in order.
    :type rounds: Sequence[int] | None
    :param save_per_round: Write ``round-NN`` sub-directories (``True``) or
        overwrite a single flat directory (``False``).
    :type save_per_round: bool
    :param skip_empty_rounds: Skip rounds with no routed prompts.
    :type skip_empty_rounds: bool
    :param formatting_func: Optional ``(prompt, response) -> str`` override;
        defaults to the base model's chat template via
        :func:`infl_ens.training.sft_training.make_chat_formatter`.
    :type formatting_func: Callable | None
    :returns: Per-round summaries (round, n_train, cumulative_n_train,
        train_loss, gate_usage, output_dir, trainable_params).
    :rtype: list[dict]
    :raises ImportError: If ``torch`` / ``transformers`` / ``datasets`` are
        missing.
    """
    try:
        import random

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-level
        raise ImportError(
            "replay_molora_sft requires torch, transformers and datasets"
        ) from exc
    from infl_ens.training.sft_training import make_chat_formatter

    random.seed(sft_cfg.seed)
    np.random.seed(sft_cfg.seed)
    torch.manual_seed(sft_cfg.seed)

    records = load_closed_loop_history(history_path)
    by_round = {int(r["round"]): r for r in records}
    target_rounds = (
        sorted(by_round.keys()) if rounds is None
        else sorted(int(r) for r in rounds)
    )

    tokenizer = AutoTokenizer.from_pretrained(sft_cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fmt = formatting_func or make_chat_formatter(tokenizer)

    bf16 = bool(sft_cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    try:
        model = AutoModelForCausalLM.from_pretrained(sft_cfg.base_model, dtype=dtype)
    except TypeError:  # pragma: no cover - old transformers
        model = AutoModelForCausalLM.from_pretrained(sft_cfg.base_model, torch_dtype=dtype)
    if torch.cuda.is_available():
        model.to("cuda")
    modules = inject_molora(model, molora_cfg)
    counts = molora_param_count(model)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
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
        texts = [fmt(p, resp) for p, resp in zip(prompts, responses)]
        cumulative_n += len(prompts)
        out_dir = out_root / agent_name / (f"round-{r:02d}" if save_per_round else "")
        result = train_molora_round(
            model, tokenizer, modules, molora_cfg, sft_cfg, texts, out_dir,
        )
        summaries.append({
            "round": r,
            "n_train": result["n_train"],
            "cumulative_n_train": cumulative_n,
            "train_loss": result["train_loss"],
            "gate_usage": result["gate_usage"],
            "output_dir": result["output_dir"],
            "trainable_params": counts,
        })
    return summaries


__all__ = ["replay_molora_sft", "train_molora_round"]
