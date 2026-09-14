"""Jointly trained prompt-gated mixtures of low-rank adapter components.

This module implements the external neural baselines without claiming a
token-level Switch/GShard reproduction.  A linear gate consumes the same
fixed trait coordinates used by the game and combines seven rank-16 LoRA
components either densely (EWoRA-style) or through a top-k gate
(MoLoRA-inspired trait-gated LoRA-MoE).
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from infl_ens.data.splits import flatten_partition_records
from infl_ens.evaluation.metrics import format_chat_example
from infl_ens.training.data_split import (
    resolve_closed_loop_data_split,
    shuffled_train_batch_indices,
)
from infl_ens.training.setup import (
    load_splits,
    make_trait_space,
    write_history,
    write_resolved_config,
)


def mixture_lora_delta(
    inputs: Any,
    lora_a: Any,
    lora_b: Any,
    gates: Any,
    *,
    scale: float,
) -> Any:
    """Compute a batch-specific weighted LoRA delta.

    :param inputs: Tensor shape ``(B, ..., I)``.
    :type inputs: torch.Tensor
    :param lora_a: Expert A factors, shape ``(K, R, I)``.
    :type lora_a: torch.Tensor
    :param lora_b: Expert B factors, shape ``(K, O, R)``.
    :type lora_b: torch.Tensor
    :param gates: Expert weights per batch row, shape ``(B, K)``.
    :type gates: torch.Tensor
    :param scale: LoRA scaling multiplier.
    :type scale: float
    :returns: Tensor shape ``(B, ..., O)``.
    :rtype: torch.Tensor
    """
    import torch

    if inputs.ndim < 2:
        raise ValueError("inputs must have a batch and feature dimension")
    flat = inputs.to(dtype=lora_a.dtype).reshape(inputs.shape[0], -1, inputs.shape[-1])
    gates = gates.to(dtype=lora_a.dtype)
    hidden = torch.einsum("bti,kri->bktr", flat, lora_a)
    expert_delta = torch.einsum("bktr,kor->bkto", hidden, lora_b)
    mixed = torch.einsum("bk,bkto->bto", gates, expert_delta)
    return (float(scale) * mixed).reshape(*inputs.shape[:-1], lora_b.shape[1])


def gate_weights(logits: Any, *, top_k: int | None = None) -> tuple[Any, Any]:
    """Return deployed and dense softmax gate probabilities.

    :param logits: Gate logits, shape ``(B, K)``.
    :type logits: torch.Tensor
    :param top_k: Experts retained per prompt; ``None`` keeps all.
    :type top_k: int | None
    :returns: ``(deployed_weights, dense_probabilities)``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    import torch

    dense = torch.softmax(logits, dim=-1)
    if top_k is None or top_k >= dense.shape[-1]:
        return dense, dense
    if top_k < 1:
        raise ValueError("top_k must be positive")
    values, indices = torch.topk(dense, top_k, dim=-1)
    sparse = torch.zeros_like(dense).scatter(-1, indices, values)
    sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(sparse.dtype).tiny
    )
    return sparse, dense


def load_balance_loss(probabilities: Any) -> Any:
    """Squared-utilization auxiliary loss with optimum at uniform load.

    :param probabilities: Dense gate probabilities, shape ``(B, K)``.
    :type probabilities: torch.Tensor
    :returns: Scalar loss equal to one at perfectly uniform mean load.
    :rtype: torch.Tensor
    """
    mean_load = probabilities.mean(dim=0)
    return probabilities.shape[1] * (mean_load * mean_load).sum()


def build_mixture_lora_model(
    base_model: Any,
    *,
    trait_dim: int,
    n_experts: int,
    expert_rank: int,
    lora_alpha: float,
    target_modules: Sequence[str],
    top_k: int | None,
    lora_dropout: float = 0.0,
) -> Any:
    """Freeze and wrap selected linear layers with gated LoRA components.

    :param base_model: Hugging Face causal language model.
    :type base_model: torch.nn.Module
    :param trait_dim: Number of gate input features.
    :type trait_dim: int
    :param n_experts: Number of LoRA components.
    :type n_experts: int
    :param expert_rank: Rank of each component.
    :type expert_rank: int
    :param lora_alpha: LoRA alpha applied per component.
    :type lora_alpha: float
    :param target_modules: Leaf module names to replace.
    :type target_modules: Sequence[str]
    :param top_k: Sparse expert count, or ``None`` for dense aggregation.
    :type top_k: int | None
    :param lora_dropout: Dropout probability on each component's input.
    :type lora_dropout: float
    :returns: A model accepting ``trait_coords`` in ``forward``.
    :rtype: torch.nn.Module
    :raises ValueError: If no target linear module is found.
    """
    import torch
    from torch import nn

    class MixtureLoRALinear(nn.Module):
        """Linear layer augmented by context-weighted LoRA components."""

        def __init__(self, base: nn.Linear) -> None:
            super().__init__()
            self.base = base
            self.scale = float(lora_alpha) / float(expert_rank)
            self.dropout = nn.Dropout(float(lora_dropout))
            self.lora_a = nn.Parameter(
                torch.empty(n_experts, expert_rank, base.in_features)
            )
            self.lora_b = nn.Parameter(
                torch.zeros(n_experts, base.out_features, expert_rank)
            )
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self.current_gates: Any | None = None

        def forward(self, inputs: Any) -> Any:
            """Apply the frozen base layer and its weighted LoRA update.

            :param inputs: Input activations with batch dimension first.
            :type inputs: Any
            :returns: Base output plus the context-weighted low-rank delta.
            :rtype: Any
            :raises RuntimeError: If the enclosing model did not install gate
                                  weights before the layer invocation.
            """
            output = self.base(inputs)
            if self.current_gates is None:
                raise RuntimeError("mixture LoRA layer called without gate weights")
            delta = mixture_lora_delta(
                self.dropout(inputs),
                self.lora_a,
                self.lora_b,
                self.current_gates,
                scale=self.scale,
            )
            return output + delta.to(output.dtype)

    for parameter in base_model.parameters():
        parameter.requires_grad = False
    targets = set(target_modules)
    wrapped: list[MixtureLoRALinear] = []
    for qualified_name, module in list(base_model.named_modules()):
        if not qualified_name or qualified_name.rsplit(".", 1)[-1] not in targets:
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent_name, _, leaf = qualified_name.rpartition(".")
        parent = base_model.get_submodule(parent_name) if parent_name else base_model
        replacement = MixtureLoRALinear(module)
        setattr(parent, leaf, replacement)
        wrapped.append(replacement)
    if not wrapped:
        raise ValueError(f"no linear target modules found among {sorted(targets)}")

    class TraitGatedLoraModel(nn.Module):
        """Frozen causal LM with a shared trait gate over LoRA components."""

        def __init__(self) -> None:
            super().__init__()
            self.base_model = base_model
            self.gate = nn.Linear(trait_dim, n_experts)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            self.last_dense_probabilities: Any | None = None

        def forward(
            self,
            *,
            trait_coords: Any,
            gate_override: Any | None = None,
            **kwargs: Any,
        ) -> Any:
            """Run a gated mixture-LoRA causal-language-model forward pass.

            :param trait_coords: Standardized trait coordinates for each sequence.
            :type trait_coords: Any
            :param gate_override: Optional precomputed deployed gate weights.
            :type gate_override: Any | None
            :param kwargs: Keyword arguments forwarded to the base model.
            :type kwargs: Any
            :returns: The base model's causal language-model output.
            :rtype: Any
            """
            deployed, dense = gate_weights(self.gate(trait_coords.float()), top_k=top_k)
            if gate_override is not None:
                deployed = gate_override.to(device=deployed.device, dtype=deployed.dtype)
            self.last_dense_probabilities = dense
            for layer in wrapped:
                layer.current_gates = deployed
            try:
                return self.base_model(**kwargs)
            finally:
                for layer in wrapped:
                    layer.current_gates = None

    return TraitGatedLoraModel()


def _mixture_state_dict(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith("gate.") or ".lora_a" in name or ".lora_b" in name
    }


def save_mixture_checkpoint(
    model: Any,
    tokenizer: Any,
    path: Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Save only learned gate/LoRA tensors plus reconstruction metadata.

    :param model: Gated LoRA model.
    :type model: torch.nn.Module
    :param tokenizer: Matching tokenizer.
    :type tokenizer: transformers.PreTrainedTokenizerBase
    :param path: Checkpoint directory.
    :type path: pathlib.Path
    :param metadata: JSON-safe reconstruction metadata.
    :type metadata: Mapping[str, Any]
    :returns: Checkpoint directory.
    :rtype: pathlib.Path
    """
    import torch

    path.mkdir(parents=True, exist_ok=True)
    torch.save(_mixture_state_dict(model), path / "mixture_lora.pt")
    (path / "mixture_config.json").write_text(
        json.dumps(dict(metadata), indent=2), encoding="utf-8",
    )
    tokenizer.save_pretrained(str(path))
    return path


def load_mixture_checkpoint(path: str | Path, *, device: Any | None = None) -> tuple[Any, Any]:
    """Reconstruct a learned-gate LoRA mixture checkpoint.

    :param path: Directory written by :func:`save_mixture_checkpoint`.
    :type path: str | pathlib.Path
    :param device: Optional torch device.
    :type device: torch.device | None
    :returns: ``(model, tokenizer)``.
    :rtype: tuple[torch.nn.Module, transformers.PreTrainedTokenizerBase]
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(path)
    metadata = json.loads(
        (root / "mixture_config.json").read_text(encoding="utf-8")
    )
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if target_device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if target_device.type == "cuda" else torch.float32
    )
    base = AutoModelForCausalLM.from_pretrained(metadata["base_model"], torch_dtype=dtype)
    model = build_mixture_lora_model(
        base,
        trait_dim=int(metadata["trait_dim"]),
        n_experts=int(metadata["n_experts"]),
        expert_rank=int(metadata["expert_rank"]),
        lora_alpha=float(metadata["lora_alpha"]),
        target_modules=metadata["target_modules"],
        top_k=metadata.get("top_k"),
        lora_dropout=float(metadata.get("lora_dropout", 0.0)),
    )
    state = torch.load(root / "mixture_lora.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(target_device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(root)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_mixture_lora(cfg: dict[str, Any]) -> int:
    """Train a dense or top-k trait-gated LoRA mixture round by round.

    :param cfg: Resolved ``mixture_lora`` run configuration.
    :type cfg: dict[str, Any]
    :returns: Process exit code.
    :rtype: int
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from infl_ens.config import resolve_sft_block

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    splits = load_splits(cfg)
    space = make_trait_space(cfg, splits)
    repo_root = Path(cfg.get("repo_root", Path(__file__).resolve().parents[3]))
    if cfg.get("data_split"):
        manifest, prompts, responses, _pool_p, _pool_r, batch_size, n_rounds = (
            resolve_closed_loop_data_split(cfg, splits, repo_root=repo_root)
        )
        id_prompts, id_responses, _labels, record_ids = flatten_partition_records(
            splits, manifest, "train",
        )
        if id_prompts != prompts or id_responses != responses:
            raise ValueError("manifest record IDs do not align with mixture training rows")
        batches = shuffled_train_batch_indices(
            len(prompts), batch_size, n_rounds, np.random.default_rng(seed),
        )
    else:
        manifest = None
        prompts = [prompt for split in splits for prompt in split.prompts]
        responses = [
            response
            for split in splits
            for response in (split.responses or [""] * split.n)
        ]
        record_ids = [f"unsplit:{index}" for index in range(len(prompts))]
        closed_loop = cfg.get("closed_loop") or {}
        batch_size = int(closed_loop.get("batch_size", 64))
        n_rounds = int(closed_loop.get("n_rounds", 2))
        batch_rng = np.random.default_rng(seed)
        batches = [
            batch_rng.integers(0, len(prompts), size=batch_size)
            for _ in range(n_rounds)
        ]
    coordinates = np.asarray(space.project(prompts), dtype=np.float32)
    coordinate_mean = coordinates.mean(axis=0)
    coordinate_scale = coordinates.std(axis=0)
    coordinate_scale = np.where(coordinate_scale > 1e-6, coordinate_scale, 1.0)
    standardized_coordinates = (coordinates - coordinate_mean) / coordinate_scale

    mix = dict(cfg.get("mixture_lora") or {})
    mode = str(mix.get("mode", "dense"))
    if mode not in {"dense", "topk"}:
        raise ValueError("mixture_lora.mode must be dense or topk")
    n_experts = int(mix.get("n_experts", 7))
    expert_rank = int(mix.get("expert_rank", 16))
    top_k = None if mode == "dense" else int(mix.get("top_k", 3))
    balance_coefficient = float(mix.get("load_balance_coefficient", 0.01))
    sft = resolve_sft_block(cfg)
    target_modules = tuple(
        sft.get(
            "lora_target_modules",
            ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        )
    )
    base_model_name = str(sft["base_model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(
        sft.get("bf16", True)
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    dtype = torch.bfloat16 if use_bf16 else (
        torch.float16 if device.type == "cuda" else torch.float32
    )
    try:
        base = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype)
    except TypeError:  # pragma: no cover - old transformers
        base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_mixture_lora_model(
        base,
        trait_dim=space.L,
        n_experts=n_experts,
        expert_rank=expert_rank,
        lora_alpha=float(sft.get("lora_alpha", 32)),
        target_modules=target_modules,
        top_k=top_k,
        lora_dropout=float(sft.get("lora_dropout", 0.0)),
    ).to(device)
    if bool(sft.get("gradient_checkpointing", True)):
        if hasattr(model.base_model, "enable_input_require_grads"):
            model.base_model.enable_input_require_grads()
        model.base_model.gradient_checkpointing_enable()
        model.base_model.config.use_cache = False
    gate_lr = float(mix.get("gate_learning_rate", sft.get("learning_rate", 2e-4)))
    gate_parameters = list(model.gate.parameters())
    gate_ids = {id(parameter) for parameter in gate_parameters}
    lora_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in gate_ids
    ]
    per_device = int(sft.get("per_device_batch_size", 8))
    accumulation = int(sft.get("gradient_accumulation_steps", 2))
    epochs = int(math.ceil(float(sft.get("num_train_epochs", 1.0))))
    max_length = int(sft.get("max_seq_length", 1024))
    output_dir = Path(cfg["output_dir"])
    model_name = "ewora-dense" if mode == "dense" else "trait-gated-topk"
    cfg["agents"] = [{"name": f"component-{i}"} for i in range(n_experts)]
    cfg["closed_loop"] = {
        **dict(cfg.get("closed_loop") or {}),
        "sft_merge_groups": [
            {"train_as": f"component-{i}", "names": [f"component-{i}"]}
            for i in range(n_experts)
        ],
    }
    write_resolved_config(cfg, output_dir / "resolved_config.yaml")
    if manifest is not None:
        (output_dir / "data_split.json").write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8",
        )
    metadata = {
        "model_kind": "ewora_style_dense" if mode == "dense" else "trait_gated_lora_moe",
        "base_model": base_model_name,
        "trait_dim": space.L,
        "n_experts": n_experts,
        "expert_rank": expert_rank,
        "active_rank": n_experts * expert_rank if top_k is None else top_k * expert_rank,
        "lora_alpha": float(sft.get("lora_alpha", 32)),
        "lora_dropout": float(sft.get("lora_dropout", 0.0)),
        "target_modules": list(target_modules),
        "top_k": top_k,
        "load_balance_coefficient": balance_coefficient,
        "trait_mean": coordinate_mean.astype(float).tolist(),
        "trait_scale": coordinate_scale.astype(float).tolist(),
    }
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    history: list[dict[str, Any]] = []
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not use_bf16)
    for round_idx, rows in enumerate(batches):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        losses: list[float] = []
        utilization: list[np.ndarray] = []
        token_exposures = 0
        model_forwards = 0
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": lora_parameters,
                    "lr": float(sft.get("learning_rate", 2e-4)),
                },
                {"params": gate_parameters, "lr": gate_lr},
            ],
            weight_decay=0.0,
        )
        micro_batches = math.ceil(len(rows) / per_device) * epochs
        update_steps = max(1, math.ceil(micro_batches / accumulation))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: max(0.0, 1.0 - step / update_steps),
        )
        optimizer.zero_grad(set_to_none=True)
        step_count = 0
        round_rows = np.asarray(rows, dtype=int)
        for _epoch in range(epochs):
            for start in range(0, len(round_rows), per_device):
                selected = round_rows[start:start + per_device]
                texts = [
                    format_chat_example(prompts[int(i)], responses[int(i)] or None)
                    for i in selected
                ]
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                token_exposures += int(encoded["attention_mask"].sum().item())
                model_forwards += 1
                labels = encoded["input_ids"].clone()
                labels[encoded["attention_mask"] == 0] = -100
                traits = torch.as_tensor(
                    standardized_coordinates[selected], device=device,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=device.type == "cuda",
                ):
                    outputs = model(trait_coords=traits, labels=labels, **encoded)
                    dense_probabilities = model.last_dense_probabilities
                    assert dense_probabilities is not None
                    loss = outputs.loss
                    if top_k is not None:
                        loss = loss + balance_coefficient * load_balance_loss(
                            dense_probabilities
                        )
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
                step_count += 1
                if step_count % accumulation == 0 or start + per_device >= len(round_rows):
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
                utilization.append(dense_probabilities.detach().float().mean(dim=0).cpu().numpy())
        checkpoint = output_dir / "agents" / model_name / f"round-{round_idx:02d}"
        save_mixture_checkpoint(model, tokenizer, checkpoint, metadata)
        with torch.no_grad():
            round_gate, _ = gate_weights(
                model.gate(
                    torch.as_tensor(
                        standardized_coordinates[round_rows], device=device,
                    ).float()
                ),
                top_k=top_k,
            )
        gate_np = round_gate.detach().float().cpu().numpy()
        positions: dict[str, list[float]] = {}
        for expert in range(n_experts):
            mass = gate_np[:, expert]
            if float(mass.sum()) > 0:
                center = (mass[:, None] * coordinates[round_rows]).sum(axis=0) / mass.sum()
            else:
                center = coordinates.mean(axis=0)
            positions[f"component-{expert}"] = center.astype(float).tolist()
        history.append({
            "round": round_idx,
            "positions": positions,
            "routing_mode": "learned_dense" if top_k is None else "learned_topk",
            "soft_top_k": top_k,
            "batch_prompts": [prompts[int(i)] for i in round_rows],
            "batch_responses": [responses[int(i)] for i in round_rows],
            "batch_record_ids": [record_ids[int(i)] for i in round_rows],
            "mean_train_loss": float(np.mean(losses)),
            "gate_utilization": np.mean(utilization, axis=0).tolist(),
            "checkpoint": str(checkpoint),
            "resource_accounting": {
                "trainable_parameters": trainable,
                "token_exposures": token_exposures,
                "peak_memory_bytes": (
                    int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
                ),
                "wall_seconds": float(time.perf_counter() - started),
                "active_lora_rank": metadata["active_rank"],
                "stored_lora_rank": n_experts * expert_rank,
                "model_forwards": model_forwards,
            },
        })
        write_history(output_dir / "history.json", history)
    return 0


__all__ = [
    "build_mixture_lora_model",
    "gate_weights",
    "load_balance_loss",
    "load_mixture_checkpoint",
    "mixture_lora_delta",
    "run_mixture_lora",
    "save_mixture_checkpoint",
]
