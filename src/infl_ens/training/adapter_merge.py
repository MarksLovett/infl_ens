"""Router-free merging of trained LoRA adapter deltas.

All algorithms operate on reconstructed effective updates
:math:`\\Delta W = (\\alpha/r)BA`; factor matrices are never averaged.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from infl_ens.config import load_config
from infl_ens.evaluation.metrics import format_chat_example
from infl_ens.evaluation.routing_eval import (
    final_round,
    load_flat_partition_records,
    score_merge_nll_matrix,
)
from infl_ens.training.setup import (
    load_splits,
    make_trait_space,
    write_history,
    write_resolved_config,
)


def reconstruct_delta(
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    *,
    scaling: float,
) -> np.ndarray:
    """Reconstruct one effective LoRA weight update.

    :param lora_a: A factor, shape ``(R, I)``.
    :type lora_a: numpy.ndarray
    :param lora_b: B factor, shape ``(O, R)``.
    :type lora_b: numpy.ndarray
    :param scaling: PEFT ``lora_alpha / r`` multiplier.
    :type scaling: float
    :returns: Effective delta, shape ``(O, I)``.
    :rtype: numpy.ndarray
    """
    a = np.asarray(lora_a, dtype=np.float32)
    b = np.asarray(lora_b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError("LoRA A/B factors have incompatible shapes")
    return float(scaling) * (b @ a)


def svd_refactor(delta: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Factor a dense delta into rank-limited B and A matrices.

    The returned factors use scaling one, so saving them with
    ``lora_alpha == rank`` reconstructs the truncated SVD exactly.

    :param delta: Dense update matrix.
    :type delta: numpy.ndarray
    :param rank: Serving rank.
    :type rank: int
    :returns: ``(A, B)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    matrix = np.asarray(delta, dtype=np.float32)
    if matrix.ndim != 2 or rank < 1:
        raise ValueError("delta must be 2-D and rank positive")
    if min(matrix.shape) > 256 and rank < min(matrix.shape):
        from scipy.sparse.linalg import svds

        u, singular, vh = svds(
            matrix,
            k=rank,
            which="LM",
            random_state=0,
        )
        order = np.argsort(singular)[::-1]
        u, singular, vh = u[:, order], singular[order], vh[order]
    else:
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    keep = min(rank, len(singular))
    root = np.sqrt(singular[:keep])
    b = u[:, :keep] * root[None, :]
    a = root[:, None] * vh[:keep]
    if keep < rank:
        a = np.pad(a, ((0, rank - keep), (0, 0)))
        b = np.pad(b, ((0, 0), (0, rank - keep)))
    return a.astype(np.float32), b.astype(np.float32)


def trim_by_density(delta: np.ndarray, density: float) -> np.ndarray:
    """Keep the largest-magnitude ``density`` fraction of a delta.

    :param delta: Dense update.
    :type delta: numpy.ndarray
    :param density: Fraction in ``(0, 1]``.
    :type density: float
    :returns: Trimmed update.
    :rtype: numpy.ndarray
    """
    if not 0 < density <= 1:
        raise ValueError("density must lie in (0, 1]")
    flat = np.abs(delta).reshape(-1)
    keep = max(1, int(np.ceil(density * len(flat))))
    threshold = np.partition(flat, len(flat) - keep)[len(flat) - keep]
    return np.where(np.abs(delta) >= threshold, delta, 0.0)


def ties_merge(deltas: Sequence[np.ndarray], *, density: float) -> np.ndarray:
    """Apply trim, elected sign, and disjoint mean merging.

    :param deltas: Task deltas with identical shape.
    :type deltas: Sequence[numpy.ndarray]
    :param density: Per-task retained density.
    :type density: float
    :returns: Merged dense update.
    :rtype: numpy.ndarray
    """
    trimmed = np.stack([trim_by_density(np.asarray(d), density) for d in deltas])
    elected = np.sign(trimmed.sum(axis=0))
    agrees = np.sign(trimmed) == elected[None, ...]
    selected = np.where(agrees, trimmed, 0.0)
    count = np.count_nonzero(selected, axis=0)
    return selected.sum(axis=0) / np.maximum(count, 1)


def dare_ties_merge(
    deltas: Sequence[np.ndarray],
    *,
    density: float,
    drop_rate: float,
    seed: int,
) -> np.ndarray:
    """Apply DARE rescaling before TIES merging.

    :param deltas: Task deltas.
    :type deltas: Sequence[numpy.ndarray]
    :param density: TIES retained density.
    :type density: float
    :param drop_rate: Bernoulli drop probability.
    :type drop_rate: float
    :param seed: RNG seed.
    :type seed: int
    :returns: Merged update.
    :rtype: numpy.ndarray
    """
    if not 0 <= drop_rate < 1:
        raise ValueError("drop_rate must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    retained = [
        np.asarray(delta)
        * (rng.random(np.asarray(delta).shape) >= drop_rate)
        / (1.0 - drop_rate)
        for delta in deltas
    ]
    return ties_merge(retained, density=density)


def knots_align(
    deltas: Sequence[np.ndarray],
    *,
    rank: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Transform task deltas into the shared KnOTS SVD basis.

    The task updates are concatenated across their input dimension, one
    joint SVD is computed, and the corresponding :math:`\\Sigma V^T` blocks
    are returned for merging in the aligned space.

    :param deltas: Task deltas with identical shape.
    :type deltas: Sequence[numpy.ndarray]
    :param rank: Common left/right subspace rank.
    :type rank: int
    :returns: ``(shared_left_basis, aligned_task_components)``.
    :rtype: tuple[numpy.ndarray, list[numpy.ndarray]]
    """
    matrices = [np.asarray(delta, dtype=np.float32) for delta in deltas]
    if not matrices or any(matrix.shape != matrices[0].shape for matrix in matrices):
        raise ValueError("KnOTS requires nonempty, identically shaped task deltas")
    concatenated = np.concatenate(matrices, axis=1)
    max_rank = min(rank, min(concatenated.shape))
    if min(concatenated.shape) > 256 and max_rank < min(concatenated.shape):
        from scipy.sparse.linalg import svds

        left, singular, right = svds(
            concatenated,
            k=max_rank,
            which="LM",
            random_state=0,
        )
        order = np.argsort(singular)[::-1]
        left, singular, right = left[:, order], singular[order], right[order]
    else:
        left, singular, right = np.linalg.svd(concatenated, full_matrices=False)
    keep = min(rank, int(np.sum(singular > 1e-5)))
    if keep < 1:
        return left[:, :1], [np.zeros((1, matrices[0].shape[1])) for _ in matrices]
    basis = left[:, :keep]
    aligned = singular[:keep, None] * right[:keep]
    width = matrices[0].shape[1]
    components = [
        aligned[:, index * width:(index + 1) * width]
        for index in range(len(matrices))
    ]
    return basis, components


def knots_merge(
    deltas: Sequence[np.ndarray],
    *,
    merge_method: str,
    density: float = 0.2,
    rank: int = 112,
) -> np.ndarray:
    """Apply linear or TIES merging in the KnOTS-aligned space.

    :param deltas: Reconstructed task updates.
    :type deltas: Sequence[numpy.ndarray]
    :param merge_method: ``linear`` or ``ties``.
    :type merge_method: str
    :param density: TIES density.
    :type density: float
    :param rank: Maximum joint-SVD rank.
    :type rank: int
    :returns: Reconstructed merged update in the original weight space.
    :rtype: numpy.ndarray
    """
    basis, components = knots_align(deltas, rank=rank)
    if merge_method == "linear":
        merged = np.mean(components, axis=0)
    elif merge_method == "ties":
        merged = ties_merge(components, density=density)
    else:
        raise ValueError("KnOTS merge_method must be linear or ties")
    return np.asarray(basis @ merged, dtype=np.float32)


@dataclass(frozen=True)
class _AdapterFactors:
    factors: dict[str, tuple[np.ndarray, np.ndarray, float]]
    config: dict[str, Any]
    a_keys: dict[str, str]


def _load_adapter_factors(path: Path) -> _AdapterFactors:
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    safe_path = path / "adapter_model.safetensors"
    bin_path = path / "adapter_model.bin"
    if safe_path.is_file():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - environment-level
            raise ImportError("safetensors is required to read this adapter") from exc
        tensors = load_file(str(safe_path), device="cpu")
    elif bin_path.is_file():
        import torch

        tensors = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"no adapter_model.safetensors or adapter_model.bin in {path}")
    rank = int(config["r"])
    scaling = float(config.get("lora_alpha", rank)) / rank
    factors: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    a_keys: dict[str, str] = {}
    for key, tensor in tensors.items():
        if "lora_A" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in tensors:
            raise ValueError(f"missing B factor for {key}")
        module = key.split(".lora_A", 1)[0]
        factors[module] = (
            tensor.float().numpy(),
            tensors[b_key].float().numpy(),
            scaling,
        )
        a_keys[module] = key
    if not factors:
        raise ValueError(f"{path} contains no LoRA factors")
    return _AdapterFactors(factors, config, a_keys)


def _save_delta_adapter(
    deltas: Mapping[str, np.ndarray],
    source_config: Mapping[str, Any],
    a_keys: Mapping[str, str],
    path: Path,
    *,
    rank: int,
) -> Path:
    import torch
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Any] = {}
    for module, delta in deltas.items():
        a, b = svd_refactor(delta, rank)
        a_key = a_keys[module]
        tensors[a_key] = torch.from_numpy(a)
        tensors[a_key.replace("lora_A", "lora_B")] = torch.from_numpy(b)
    config = dict(source_config)
    config["r"] = rank
    config["lora_alpha"] = rank
    config["rank_pattern"] = {}
    config["alpha_pattern"] = {}
    config["use_rslora"] = False
    config["use_dora"] = False
    (path / "adapter_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8",
    )
    save_file(tensors, str(path / "adapter_model.safetensors"))
    return path


def _save_merged_factor_adapter(
    experts: Sequence[_AdapterFactors],
    path: Path,
    *,
    method: str,
    rank: int,
    scale: float,
    density: float,
    drop_rate: float,
    seed: int,
    alignment_rank: int,
) -> Path:
    """Merge one module at a time so source full-rank deltas never accumulate."""
    import torch
    from safetensors.torch import save_file

    modules = set(experts[0].factors)
    if any(set(expert.factors) != modules for expert in experts[1:]):
        raise ValueError("all adapters must expose identical LoRA modules")
    tensors: dict[str, Any] = {}
    for module in sorted(modules):
        deltas = [
            reconstruct_delta(a, b, scaling=factor_scale)
            for a, b, factor_scale in (expert.factors[module] for expert in experts)
        ]
        merged = merge_delta_family(
            [{module: delta} for delta in deltas],
            method=method,
            scale=scale,
            density=density,
            drop_rate=drop_rate,
            seed=seed,
            alignment_rank=alignment_rank,
        )[module]
        a, b = svd_refactor(merged, rank)
        a_key = experts[0].a_keys[module]
        tensors[a_key] = torch.from_numpy(a)
        tensors[a_key.replace("lora_A", "lora_B")] = torch.from_numpy(b)
    path.mkdir(parents=True, exist_ok=True)
    config = dict(experts[0].config)
    config["r"] = rank
    config["lora_alpha"] = rank
    config["rank_pattern"] = {}
    config["alpha_pattern"] = {}
    config["use_rslora"] = False
    config["use_dora"] = False
    (path / "adapter_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8",
    )
    save_file(tensors, str(path / "adapter_model.safetensors"))
    return path


def _serving_parameter_count(expert: _AdapterFactors, rank: int) -> int:
    return int(
        sum(
            rank * (a.shape[1] + b.shape[0])
            for a, b, _scale in expert.factors.values()
        )
    )


def merge_delta_family(
    expert_deltas: Sequence[Mapping[str, np.ndarray]],
    *,
    method: str,
    scale: float = 1.0,
    density: float = 0.2,
    drop_rate: float = 0.5,
    seed: int = 0,
    alignment_rank: int = 112,
) -> dict[str, np.ndarray]:
    """Merge every module of an adapter family with one method.

    :param expert_deltas: Expert module-to-delta mappings.
    :type expert_deltas: Sequence[Mapping[str, numpy.ndarray]]
    :param method: ``linear``, ``task_arithmetic``, ``ties``,
        ``dare_ties``, ``knots_linear``, or ``knots_ties``.
    :type method: str
    :param scale: Task-arithmetic scale.
    :type scale: float
    :param density: TIES density.
    :type density: float
    :param drop_rate: DARE drop rate.
    :type drop_rate: float
    :param seed: DARE seed.
    :type seed: int
    :param alignment_rank: KnOTS joint-subspace rank.
    :type alignment_rank: int
    :returns: Module-to-merged-delta mapping.
    :rtype: dict[str, numpy.ndarray]
    """
    modules = set(expert_deltas[0])
    if any(set(expert) != modules for expert in expert_deltas[1:]):
        raise ValueError("all adapters must expose identical LoRA modules")
    merged: dict[str, np.ndarray] = {}
    for module in sorted(modules):
        deltas = [expert[module] for expert in expert_deltas]
        if method == "linear":
            value = np.mean(deltas, axis=0)
        elif method == "task_arithmetic":
            value = float(scale) * np.sum(deltas, axis=0)
        elif method == "ties":
            value = ties_merge(deltas, density=density)
        elif method == "dare_ties":
            value = dare_ties_merge(
                deltas, density=density, drop_rate=drop_rate, seed=seed,
            )
        elif method in {"knots_linear", "knots_ties"}:
            value = knots_merge(
                deltas,
                merge_method="linear" if method == "knots_linear" else "ties",
                density=density,
                rank=alignment_rank,
            )
        else:
            raise ValueError(f"unknown adapter merge method {method!r}")
        merged[module] = np.asarray(value, dtype=np.float32)
    return merged


def run_adapter_merge(cfg: dict[str, Any]) -> int:
    """Build merge candidates, select hyperparameters on validation NLL, and archive winners.

    :param cfg: Resolved ``adapter_merge`` run configuration.
    :type cfg: dict[str, Any]
    :returns: Process exit code.
    :rtype: int
    """
    from infl_ens.config import resolve_sft_block

    block = dict(cfg.get("adapter_merge") or {})
    source_run = Path(block["source_run_dir"])
    source_round = final_round(source_run / "history.json")
    source_cfg_path = source_run / "resolved_config.yaml"
    if not source_cfg_path.is_file():
        raise FileNotFoundError(source_cfg_path)
    source_cfg = load_config(source_cfg_path, validate=False)
    source_groups = (source_cfg.get("closed_loop") or {}).get("sft_merge_groups") or []
    source_experts = list(block.get("source_experts") or [g["train_as"] for g in source_groups])
    if not source_experts:
        raise ValueError("adapter_merge requires source_experts or source merge groups")
    loaded = [
        _load_adapter_factors(
            source_run / "agents" / name / f"round-{source_round:02d}"
        )
        for name in source_experts
    ]
    output_dir = Path(cfg["output_dir"])
    candidates_root = output_dir / "candidates"
    methods = list(
        block.get("methods")
        or ["linear", "task_arithmetic", "ties", "dare_ties", "knots_linear", "knots_ties"]
    )
    scales = [float(x) for x in block.get("task_arithmetic_scales", (0.25, 0.5, 0.75, 1.0))]
    densities = [float(x) for x in block.get("ties_densities", (0.1, 0.2, 0.5))]
    drops = [float(x) for x in block.get("dare_drop_rates", (0.2, 0.5, 0.8))]
    ranks = [int(x) for x in block.get("serving_ranks", (16, 112))]
    seed = int(block.get("seed", cfg.get("seed", 0)))
    candidate_meta: list[dict[str, Any]] = []
    for method in methods:
        method_scales = scales if method == "task_arithmetic" else [1.0]
        method_densities = densities if method in {"ties", "dare_ties", "knots_ties"} else [1.0]
        method_drops = drops if method == "dare_ties" else [0.0]
        for scale in method_scales:
            for density in method_densities:
                for drop in method_drops:
                    for rank in ranks:
                        name = (
                            f"{method}-r{rank}-s{scale:g}-d{density:g}-p{drop:g}"
                        )
                        path = candidates_root / "agents" / name / f"round-{source_round:02d}"
                        _save_merged_factor_adapter(
                            loaded,
                            path,
                            method=method,
                            rank=rank,
                            scale=scale,
                            density=density,
                            drop_rate=drop,
                            seed=seed,
                            alignment_rank=int(block.get("alignment_rank", 112)),
                        )
                        candidate_meta.append({
                            "name": name,
                            "method": method,
                            "rank": rank,
                            "scale": scale,
                            "density": density,
                            "drop_rate": drop,
                        })

    repo_root = Path(cfg.get("repo_root", Path(__file__).resolve().parents[3]))
    prompts, responses, _labels, _ids = load_flat_partition_records(
        cfg,
        repo_root=repo_root,
        partition="val",
        max_eval_records=(cfg.get("eval") or {}).get("max_eval_records"),
        seed=seed,
    )
    texts = [
        format_chat_example(prompt, response or None)
        for prompt, response in zip(prompts, responses)
    ]
    sft = resolve_sft_block(cfg)
    candidate_names = [row["name"] for row in candidate_meta]
    validation_nll = score_merge_nll_matrix(
        candidate_names,
        texts,
        merge_run_dir=candidates_root,
        round_idx=source_round,
        base_model=str(sft["base_model"]),
        max_seq_length=int(sft.get("max_seq_length", 1024)),
        forward_batch_size=int((cfg.get("eval") or {}).get("forward_batch_size", 8)),
    )
    assert isinstance(validation_nll, np.ndarray)
    mean_validation = validation_nll.mean(axis=0)
    selected: list[dict[str, Any]] = []
    for method in methods:
        indices = [
            i for i, row in enumerate(candidate_meta) if row["method"] == method
        ]
        best = min(
            indices,
            key=lambda index: (float(mean_validation[index]), candidate_names[index]),
        )
        row = dict(candidate_meta[best])
        row["validation_nll"] = float(mean_validation[best])
        row["stored_parameters"] = _serving_parameter_count(
            loaded[0], int(row["rank"])
        )
        row["inference_passes"] = 1
        selected.append(row)
        source = candidates_root / "agents" / row["name"] / f"round-{source_round:02d}"
        target = output_dir / "agents" / method / f"round-{source_round:02d}"
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)

    cfg["agents"] = [{"name": row["method"]} for row in selected]
    cfg["closed_loop"] = {
        **dict(cfg.get("closed_loop") or {}),
        "sft_merge_groups": [
            {"train_as": row["method"], "names": [row["method"]]}
            for row in selected
        ],
    }
    splits = load_splits(cfg)
    space = make_trait_space(cfg, splits)
    mean_position = np.asarray(space.mean, dtype=float).tolist()
    write_resolved_config(cfg, output_dir / "resolved_config.yaml")
    write_history(output_dir / "history.json", [{
        "round": source_round,
        "positions": {row["method"]: mean_position for row in selected},
        "routing_mode": "router_free_merge",
        "selected_merges": selected,
        "source_experts": source_experts,
        "resource_accounting": {
            row["method"]: {
                "trainable_parameters": row["stored_parameters"],
                "active_lora_rank": row["rank"],
                "token_exposures": 0,
                "model_forwards": 0,
                "peak_memory_bytes": 0,
                "wall_seconds": 0.0,
            }
            for row in selected
        },
    }])
    (output_dir / "merge_summary.json").write_text(
        json.dumps({
            "source_run_dir": str(source_run),
            "source_round": source_round,
            "source_experts": source_experts,
            "selected": selected,
            "n_candidates": len(candidate_meta),
        }, indent=2),
        encoding="utf-8",
    )
    return 0


__all__ = [
    "dare_ties_merge",
    "knots_align",
    "knots_merge",
    "merge_delta_family",
    "reconstruct_delta",
    "run_adapter_merge",
    "svd_refactor",
    "ties_merge",
    "trim_by_density",
]
