"""Deterministic text generation for single and routed behavioral models."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from infl_ens.data.behavioral import BehavioralCase
from infl_ens.evaluation.behavioral_artifact import GenerationRecord
from infl_ens.experiment import BehavioralGenerationSettings


def seed_generation(seed: int) -> None:
    """Seed Python, NumPy and PyTorch before any generation RNG use.

    :param seed: Shared generation seed.
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - lightweight installations
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def render_chat(tokenizer: Any, case: BehavioralCase) -> str:
    """Render a case with the target tokenizer's chat template.

    :param tokenizer: Hugging Face tokenizer.
    :type tokenizer: Any
    :param case: Model-visible behavioral case.
    :type case: BehavioralCase
    :returns: Generation-ready prompt string.
    :rtype: str
    """
    messages = [message.to_dict() for message in case.messages]
    if hasattr(tokenizer, "apply_chat_template"):
        return str(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ))
    return "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    ) + "\nassistant:"


def mixture_log_probabilities(logits: Any, weights: Any) -> Any:
    """Combine expert next-token distributions with prompt-level weights.

    :param logits: Expert logits with shape ``(K, M, V)``.
    :type logits: torch.Tensor
    :param weights: Nonnegative row weights with shape ``(M, K)``.
    :type weights: torch.Tensor
    :returns: Mixture log probabilities with shape ``(M, V)``.
    :rtype: torch.Tensor
    :raises ValueError: If shapes or weights are invalid.
    """
    import torch

    if logits.ndim != 3 or weights.ndim != 2:
        raise ValueError("logits must be (K,M,V) and weights must be (M,K)")
    if logits.shape[0] != weights.shape[1] or logits.shape[1] != weights.shape[0]:
        raise ValueError("expert/batch dimensions do not align")
    if not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("mixture weights must be finite and nonnegative")
    row_sum = weights.sum(dim=1, keepdim=True)
    if bool((row_sum <= 0).any()):
        raise ValueError("every mixture-weight row must have positive mass")
    normalized = weights / row_sum
    log_weights = torch.where(
        normalized > 0,
        normalized.log(),
        torch.full_like(normalized, -torch.inf),
    )
    expert_log_probs = torch.log_softmax(logits, dim=-1)
    return torch.logsumexp(
        expert_log_probs + log_weights.T[:, :, None], dim=0,
    )


def _input_limit(tokenizer: Any, settings: BehavioralGenerationSettings) -> int | None:
    if settings.max_input_tokens is not None:
        return settings.max_input_tokens
    model_limit = int(getattr(tokenizer, "model_max_length", 0) or 0)
    return model_limit if 0 < model_limit < 1_000_000 else None


def _tokenize_prompts(
    tokenizer: Any,
    prompts: Sequence[str],
    settings: BehavioralGenerationSettings,
) -> tuple[dict[str, Any], list[int]]:
    lengths = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in prompts
    ]
    limit = _input_limit(tokenizer, settings)
    if limit is not None and any(length > limit for length in lengths):
        too_long = max(lengths)
        if settings.fail_on_truncation:
            raise ValueError(
                f"behavioral prompt has {too_long} tokens, over max_input_tokens={limit}",
            )
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=limit is not None,
        max_length=limit,
    )
    actual_lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
    return dict(encoded), actual_lengths


def generate_standard_model(
    model: Any,
    tokenizer: Any,
    cases: Sequence[BehavioralCase],
    *,
    target: str,
    protocol: str,
    settings: BehavioralGenerationSettings,
    trait_coordinates: np.ndarray | None = None,
    router_weights: np.ndarray | None = None,
    expert_names: Sequence[str] = (),
) -> list[GenerationRecord]:
    """Generate with a normal Hugging Face ``generate`` implementation.

    :param model: Causal language model.
    :type model: Any
    :param tokenizer: Matching tokenizer.
    :type tokenizer: Any
    :param cases: Ordered behavioral cases.
    :type cases: Sequence[BehavioralCase]
    :param target: Result target name.
    :type target: str
    :param protocol: Result protocol name.
    :type protocol: str
    :param settings: Decoding settings.
    :type settings: BehavioralGenerationSettings
    :param trait_coordinates: Optional ``(M,L)`` routing coordinates.
    :type trait_coordinates: numpy.ndarray | None
    :param router_weights: Optional ``(M,K)`` recorded weights.
    :type router_weights: numpy.ndarray | None
    :param expert_names: Names aligned with router columns.
    :type expert_names: Sequence[str]
    :returns: Generated records in case order.
    :rtype: list[GenerationRecord]
    """
    import torch

    seed_generation(settings.seed)
    device = next(model.parameters()).device
    records: list[GenerationRecord] = []
    for start in range(0, len(cases), settings.batch_size):
        batch = list(cases[start:start + settings.batch_size])
        prompts = [render_chat(tokenizer, case) for case in batch]
        encoded, input_lengths = _tokenize_prompts(tokenizer, prompts, settings)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        kwargs: dict[str, Any] = {
            "do_sample": settings.do_sample,
            "max_new_tokens": settings.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if settings.do_sample:
            kwargs["temperature"] = settings.temperature
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, **kwargs)
        elapsed = time.perf_counter() - started
        prompt_width = int(encoded["input_ids"].shape[1])
        for local, case in enumerate(batch):
            new_ids = generated[local, prompt_width:].detach().cpu().tolist()
            eos = tokenizer.eos_token_id
            finish = "eos" if eos is not None and eos in new_ids else "length"
            if eos is not None and eos in new_ids:
                new_ids = new_ids[:new_ids.index(eos)]
            row = start + local
            weights = (
                {
                    str(name): float(router_weights[row, index])
                    for index, name in enumerate(expert_names)
                }
                if router_weights is not None else {}
            )
            selected = tuple(
                name for name, value in weights.items() if value > 0.0
            )
            records.append(GenerationRecord(
                case_id=case.case_id,
                suite=case.suite,
                target=target,
                protocol=protocol,
                messages=tuple(message.to_dict() for message in case.messages),
                response=tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
                token_ids=tuple(int(value) for value in new_ids),
                router_weights=weights,
                selected_experts=selected,
                trait_coordinates=(
                    tuple(float(value) for value in trait_coordinates[row])
                    if trait_coordinates is not None else ()
                ),
                finish_reason=finish,
                input_tokens=int(input_lengths[local]),
                output_tokens=len(new_ids),
                model_forwards=len(new_ids) + int(finish == "eos"),
                latency_seconds=float(elapsed / len(batch)),
            ))
    return records


def generate_trait_mixture_model(
    model: Any,
    tokenizer: Any,
    cases: Sequence[BehavioralCase],
    standardized_coordinates: np.ndarray,
    *,
    recorded_coordinates: np.ndarray,
    target: str,
    protocol: str,
    settings: BehavioralGenerationSettings,
    hard_argmax: bool,
    expert_names: Sequence[str],
    top_k: int | None,
) -> list[GenerationRecord]:
    """Autoregressively decode a learned trait-gated LoRA mixture.

    Gate weights are computed once from the prompt coordinates and held fixed
    for all generated tokens. ``hard_argmax`` replaces the deployed gate by a
    one-hot vector without retraining.

    :param model: Model returned by ``load_mixture_checkpoint``.
    :type model: Any
    :param tokenizer: Matching tokenizer.
    :type tokenizer: Any
    :param cases: Ordered behavioral cases.
    :type cases: Sequence[BehavioralCase]
    :param standardized_coordinates: Gate inputs, shape ``(M,L)``.
    :type standardized_coordinates: numpy.ndarray
    :param recorded_coordinates: Unstandardized coordinates, shape ``(M,L)``.
    :type recorded_coordinates: numpy.ndarray
    :param target: Result target name.
    :type target: str
    :param protocol: Result protocol.
    :type protocol: str
    :param settings: Decoding settings.
    :type settings: BehavioralGenerationSettings
    :param hard_argmax: Force one component when true.
    :type hard_argmax: bool
    :param expert_names: Component names.
    :type expert_names: Sequence[str]
    :param top_k: Deployed gate width, or ``None`` for a dense gate.
    :type top_k: int | None
    :returns: Generated records.
    :rtype: list[GenerationRecord]
    """
    import torch

    from infl_ens.training.mixture_lora import gate_weights

    seed_generation(settings.seed)
    device = next(model.parameters()).device
    records: list[GenerationRecord] = []
    for index, case in enumerate(cases):
        prompt = render_chat(tokenizer, case)
        encoded, lengths = _tokenize_prompts(tokenizer, [prompt], settings)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        traits = torch.as_tensor(
            standardized_coordinates[index:index + 1], device=device,
        ).float()
        with torch.inference_mode():
            deployed, _dense = gate_weights(
                model.gate(traits), top_k=top_k,
            )
        if hard_argmax:
            hard = torch.zeros_like(deployed)
            hard.scatter_(1, deployed.argmax(dim=1, keepdim=True), 1.0)
            deployed = hard
        generated: list[int] = []
        past = None
        started = time.perf_counter()
        finish = "length"
        with torch.inference_mode():
            for _step in range(settings.max_new_tokens):
                step_ids = input_ids if past is None else input_ids[:, -1:]
                output = model(
                    trait_coords=traits,
                    gate_override=deployed,
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                logits = output.logits[:, -1, :]
                if settings.do_sample:
                    probabilities = torch.softmax(logits / settings.temperature, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                token = int(next_token.item())
                if token == tokenizer.eos_token_id:
                    finish = "eos"
                    break
                generated.append(token)
                input_ids = torch.cat((input_ids, next_token), dim=1)
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_token)), dim=1,
                )
                past = output.past_key_values
        weights_np = deployed[0].detach().float().cpu().numpy()
        weights = {
            str(name): float(weights_np[position])
            for position, name in enumerate(expert_names)
        }
        records.append(GenerationRecord(
            case_id=case.case_id,
            suite=case.suite,
            target=target,
            protocol=protocol,
            messages=tuple(message.to_dict() for message in case.messages),
            response=tokenizer.decode(generated, skip_special_tokens=True).strip(),
            token_ids=tuple(generated),
            router_weights=weights,
            selected_experts=tuple(name for name, value in weights.items() if value > 0.0),
            trait_coordinates=tuple(float(value) for value in recorded_coordinates[index]),
            finish_reason=finish,
            input_tokens=lengths[0],
            output_tokens=len(generated),
            model_forwards=(len(generated) + (finish == "eos")),
            latency_seconds=float(time.perf_counter() - started),
        ))
    return records


def generate_probability_mixture(
    model: Any,
    tokenizer: Any,
    cases: Sequence[BehavioralCase],
    weights: np.ndarray,
    expert_names: Sequence[str],
    set_expert: Callable[[str], None],
    *,
    coordinates: np.ndarray,
    target: str,
    protocol: str,
    settings: BehavioralGenerationSettings,
) -> list[GenerationRecord]:
    """Decode independent adapters as a token-probability ensemble.

    Separate KV caches are maintained for every active expert because cached
    hidden states depend on the selected LoRA adapter.

    :param model: Multi-adapter PEFT causal LM.
    :type model: Any
    :param tokenizer: Matching tokenizer.
    :type tokenizer: Any
    :param cases: Ordered behavioral cases.
    :type cases: Sequence[BehavioralCase]
    :param weights: Prompt-level row-stochastic weights ``(M,K)``.
    :type weights: numpy.ndarray
    :param expert_names: Adapter names aligned with columns.
    :type expert_names: Sequence[str]
    :param set_expert: Callback activating one adapter.
    :type set_expert: Callable[[str], None]
    :param coordinates: Frozen trait coordinates ``(M,L)``.
    :type coordinates: numpy.ndarray
    :param target: Result target name.
    :type target: str
    :param protocol: Result protocol name.
    :type protocol: str
    :param settings: Decoding settings.
    :type settings: BehavioralGenerationSettings
    :returns: Generated records.
    :rtype: list[GenerationRecord]
    """
    import torch

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(cases), len(expert_names)):
        raise ValueError("weights shape does not match cases and experts")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    row_sums = weights.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("every weights row must have positive mass")
    weights = weights / row_sums
    seed_generation(settings.seed)
    device = next(model.parameters()).device
    records: list[GenerationRecord] = []
    for case_index, case in enumerate(cases):
        prompt = render_chat(tokenizer, case)
        encoded, lengths = _tokenize_prompts(tokenizer, [prompt], settings)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        active = np.flatnonzero(weights[case_index] > 0.0)
        active_names = [str(expert_names[int(value)]) for value in active]
        active_weights = torch.as_tensor(
            weights[case_index, active][None, :], device=device, dtype=torch.float32,
        )
        past_by_expert: dict[str, Any] = {name: None for name in active_names}
        generated: list[int] = []
        started = time.perf_counter()
        finish = "length"
        with torch.inference_mode():
            for _step in range(settings.max_new_tokens):
                expert_logits: list[Any] = []
                for name in active_names:
                    set_expert(name)
                    past = past_by_expert[name]
                    step_ids = input_ids if past is None else input_ids[:, -1:]
                    output = model(
                        input_ids=step_ids,
                        attention_mask=attention_mask,
                        past_key_values=past,
                        use_cache=True,
                    )
                    expert_logits.append(output.logits[:, -1, :])
                    past_by_expert[name] = output.past_key_values
                mixed = mixture_log_probabilities(
                    torch.stack(expert_logits, dim=0), active_weights,
                )
                if settings.do_sample:
                    probabilities = torch.softmax(mixed / settings.temperature, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                else:
                    next_token = mixed.argmax(dim=-1, keepdim=True)
                token = int(next_token.item())
                if token == tokenizer.eos_token_id:
                    finish = "eos"
                    break
                generated.append(token)
                input_ids = torch.cat((input_ids, next_token), dim=1)
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_token)), dim=1,
                )
        record_weights = {
            str(name): float(weights[case_index, column])
            for column, name in enumerate(expert_names)
        }
        records.append(GenerationRecord(
            case_id=case.case_id,
            suite=case.suite,
            target=target,
            protocol=protocol,
            messages=tuple(message.to_dict() for message in case.messages),
            response=tokenizer.decode(generated, skip_special_tokens=True).strip(),
            token_ids=tuple(generated),
            router_weights=record_weights,
            selected_experts=tuple(active_names),
            trait_coordinates=tuple(float(value) for value in coordinates[case_index]),
            finish_reason=finish,
            input_tokens=lengths[0],
            output_tokens=len(generated),
            model_forwards=(len(generated) + (finish == "eos")) * len(active_names),
            latency_seconds=float(time.perf_counter() - started),
        ))
    return records


__all__ = [
    "generate_probability_mixture",
    "generate_standard_model",
    "generate_trait_mixture_model",
    "mixture_log_probabilities",
    "render_chat",
    "seed_generation",
]
