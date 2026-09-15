"""Flat-pool route-then-score evaluation for merge ensembles.

Scores a concatenated benchmark partition under pooled baseline, learned
routing (argmax or proportional :math:`G`), and oracle merge assignment.
The flat mean NLL across the whole pool is the headline metric; per-benchmark
breakdowns are diagnostic only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from infl_ens.data.splits import flatten_partition_prompts, load_split_manifest
from infl_ens.evaluation.adapters import load_adapter_model, load_base_causal_lm
from infl_ens.config import load_config
from infl_ens.data.benchmarks.loading import subsample_split
from infl_ens.evaluation.metrics import build_chat_formatter
from infl_ens.inflgame.router.allocation import (
    allocation_weights,
    strategic_routing_weights,
)
from infl_ens.training.setup import load_splits, make_trait_space, sigma_from_config

DEFAULT_MERGE_ALIASES: dict[str, str] = {
    "merge-jailbreak": "merge-generalist",
}


@dataclass(frozen=True)
class FlatRoutingScores:
    """Headline flat-pool route-then-score metrics.

    :param pooled_nll: Mean NLL under ``pooled-baseline``.
    :type pooled_nll: float
    :param learned_argmax_nll: Route-then-score with argmax :math:`G`.
    :type learned_argmax_nll: float
    :param learned_expected_nll: Expected NLL under merge-level :math:`G`
        (deterministic proportional match).
    :type learned_expected_nll: float
    :param learned_sampled_nll: Single proportional sample per prompt.
    :type learned_sampled_nll: float
    :param strategic_expected_nll: Expected NLL under merge-level
        :math:`G(1-G)` strategic routing weights.
    :type strategic_expected_nll: float
    :param strategic_argmax_nll: Argmax strategic routing weight per prompt.
    :type strategic_argmax_nll: float
    :param oracle_nll: Per-prompt argmin merge NLL ceiling.
    :type oracle_nll: float
    :param n_prompts: Flat pool size.
    :type n_prompts: int
    :param round_idx: Adapter round index scored.
    :type round_idx: int
    :param universal_nll: Mean NLL of the merged universal adapter with no
        domain adapter attached (residual arms only; ``None`` otherwise).
        Must agree with ``pooled_nll`` up to merge rounding when the
        universal adapter is the pooled generalist.
    :type universal_nll: float | None
    :param fitted_argmax_nll: Route-then-score with the argmax of the
        fitted prompt-level router (``None`` when not fitted).
    :type fitted_argmax_nll: float | None
    :param fitted_expected_nll: Expected NLL under the fitted router's
        probabilities.
    :type fitted_expected_nll: float | None
    :param fitted_val_accuracy: Argmax accuracy of the fitted router on its
        own validation fit.
    :type fitted_val_accuracy: float | None
    """

    pooled_nll: float
    learned_argmax_nll: float
    learned_expected_nll: float
    learned_sampled_nll: float
    strategic_expected_nll: float
    strategic_argmax_nll: float
    oracle_nll: float
    n_prompts: int
    round_idx: int
    universal_nll: float | None = None
    fitted_argmax_nll: float | None = None
    fitted_expected_nll: float | None = None
    fitted_val_accuracy: float | None = None


@dataclass
class FlatRoutingReport:
    """Full route-then-score report for one partition.

    :param flat: Headline scalar metrics.
    :type flat: FlatRoutingScores
    :param merge_names: Resolved on-disk merge adapter names.
    :type merge_names: list[str]
    :param merge_name_map: Config merge name to resolved name.
    :type merge_name_map: dict[str, str]
    :param per_benchmark: Diagnostic per-benchmark breakdown.
    :type per_benchmark: dict[str, dict[str, Any]]
    :param routing_confusion: Argmax learned vs oracle merge counts.
    :type routing_confusion: dict[str, Any]
    :param merge_support_argmax: Hard argmax win counts per merge.
    :type merge_support_argmax: dict[str, Any]
    :param merge_support_expected: Mean merge-level :math:`G` per prompt.
    :type merge_support_expected: dict[str, Any]
    :param clone_support_argmax: Hard argmax win counts per clone.
    :type clone_support_argmax: dict[str, Any]
    :param clone_support_expected: Mean clone-level :math:`G` per prompt.
    :type clone_support_expected: dict[str, Any]
    :param bench_labels: Source benchmark per flat-pool prompt.
    :type bench_labels: list[str]
    """

    flat: FlatRoutingScores
    merge_names: list[str]
    merge_name_map: dict[str, str]
    per_benchmark: dict[str, dict[str, Any]] = field(default_factory=dict)
    routing_confusion: dict[str, Any] = field(default_factory=dict)
    merge_support_argmax: dict[str, Any] = field(default_factory=dict)
    merge_support_expected: dict[str, Any] = field(default_factory=dict)
    clone_support_argmax: dict[str, Any] = field(default_factory=dict)
    clone_support_expected: dict[str, Any] = field(default_factory=dict)
    bench_labels: list[str] = field(default_factory=list)


def parse_merge_groups(
    closed_loop: Mapping[str, Any],
) -> tuple[dict[str, str], list[str]]:
    """Map clone name to config merge adapter name.

    :param closed_loop: ``closed_loop`` block from router YAML.
    :type closed_loop: Mapping[str, Any]
    :returns: ``(clone_to_merge, ordered_config_merge_names)``.
    :rtype: tuple[dict[str, str], list[str]]
    """
    clone_to_merge: dict[str, str] = {}
    merge_names: list[str] = []
    for group in closed_loop.get("sft_merge_groups", []):
        merge = str(group["train_as"])
        merge_names.append(merge)
        for name in group["names"]:
            clone_to_merge[str(name)] = merge
    return clone_to_merge, merge_names


def resolve_merge_adapters(
    merge_run_dir: Path,
    round_idx: int,
    config_merge_names: Sequence[str],
    *,
    aliases: Mapping[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Map config merge names to on-disk adapter directories.

    :param merge_run_dir: Closed-loop run root.
    :type merge_run_dir: pathlib.Path
    :param round_idx: Training round to load.
    :type round_idx: int
    :param config_merge_names: Merge names from YAML ``sft_merge_groups``.
    :type config_merge_names: Sequence[str]
    :param aliases: Optional extra config→disk aliases.
    :type aliases: Mapping[str, str] | None
    :returns: ``(resolved_merge_names, config_to_resolved)``.
    :rtype: tuple[list[str], dict[str, str]]
    :raises FileNotFoundError: If no adapter exists for a config merge name.
    """
    alias_map = dict(DEFAULT_MERGE_ALIASES)
    if aliases:
        alias_map.update(aliases)
    agents_root = merge_run_dir / "agents"
    resolved: list[str] = []
    name_map: dict[str, str] = {}
    for config_name in config_merge_names:
        candidates = [config_name, alias_map.get(config_name, "")]
        seen: set[str] = set()
        pick: str | None = None
        for cand in candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)
            adapter_dir = agents_root / cand / f"round-{round_idx:02d}"
            if adapter_dir.is_dir():
                pick = cand
                break
        if pick is None:
            raise FileNotFoundError(
                f"no merge adapter for {config_name!r} at round {round_idx} "
                f"under {agents_root} (tried {list(seen)})",
            )
        name_map[config_name] = pick
        if pick not in resolved:
            resolved.append(pick)
    return resolved, name_map


def final_round(history_path: Path) -> int:
    """Return the last round index in a closed-loop ``history.json``.

    :param history_path: Path to history file.
    :type history_path: pathlib.Path
    :returns: Final round index.
    :rtype: int
    """
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return int(history[-1]["round"])


def load_final_positions(
    history_path: Path,
    agent_names: Sequence[str],
) -> np.ndarray:
    """Stack final-round router positions.

    :param history_path: Path to ``history.json``.
    :type history_path: pathlib.Path
    :param agent_names: Clone names in config order.
    :type agent_names: Sequence[str]
    :returns: Position matrix, shape ``(N, L)``.
    :rtype: numpy.ndarray
    """
    history = json.loads(history_path.read_text(encoding="utf-8"))
    pos_map = history[-1]["positions"]
    return np.stack(
        [np.asarray(pos_map[name], dtype=float) for name in agent_names],
        axis=0,
    )


def load_flat_partition_pool(
    cfg: Mapping[str, Any],
    *,
    repo_root: Path,
    partition: str,
    max_eval_records: int | None,
    seed: int,
) -> tuple[list[str], list[str | None], list[str]]:
    """Load and cap a flattened partition across benchmarks.

    :param cfg: Router YAML dict.
    :type cfg: Mapping[str, Any]
    :param repo_root: Repository root for relative paths.
    :type repo_root: pathlib.Path
    :param partition: ``train``, ``val``, or ``test``.
    :type partition: str
    :param max_eval_records: Per-benchmark cap, or ``None`` for all rows.
    :type max_eval_records: int | None
    :param seed: Subsample seed when capping.
    :type seed: int
    :returns: ``(prompts, responses, benchmark_labels)``.
    :rtype: tuple[list[str], list[str | None], list[str]]
    """
    full_splits = load_splits(dict(cfg))
    ds = cfg.get("data_split") or {}
    manifest_path = Path(ds["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest = load_split_manifest(manifest_path)
    part_prompts, part_responses, part_bench = flatten_partition_prompts(
        full_splits,
        manifest,
        partition,  # type: ignore[arg-type]
    )
    if max_eval_records is None:
        return part_prompts, part_responses, part_bench

    capped_prompts: list[str] = []
    capped_responses: list[str | None] = []
    capped_bench: list[str] = []
    for split in full_splits:
        idx = list(manifest.partition_for(split.name).select(partition))  # type: ignore[arg-type]
        if not idx:
            continue
        sub = split.take(idx)
        if sub.n > max_eval_records:
            sub = subsample_split(sub, max_eval_records, seed=seed)
        resp = sub.responses or [""] * sub.n
        for i in range(sub.n):
            capped_prompts.append(sub.prompts[i])
            capped_responses.append(resp[i])
            capped_bench.append(split.name)
    return capped_prompts, capped_responses, capped_bench


def aggregate_clone_g_to_merge(
    g_clone: np.ndarray,
    agent_names: Sequence[str],
    clone_to_merge: Mapping[str, str],
    merge_names: Sequence[str],
    merge_name_map: Mapping[str, str],
) -> np.ndarray:
    """Sum clone-level :math:`G` into merge-level weights.

    :param g_clone: Clone allocation matrix, shape ``(N_clone, M)``.
    :type g_clone: numpy.ndarray
    :param agent_names: Clone names aligned with rows of ``g_clone``.
    :type agent_names: Sequence[str]
    :param clone_to_merge: Config clone to config merge name.
    :type clone_to_merge: Mapping[str, str]
    :param merge_names: Resolved merge names (column order for output).
    :type merge_names: Sequence[str]
    :param merge_name_map: Config merge name to resolved merge name.
    :type merge_name_map: Mapping[str, str]
    :returns: Merge weights, shape ``(n_merge, M)``.
    :rtype: numpy.ndarray
    """
    m = g_clone.shape[1]
    out = np.zeros((len(merge_names), m), dtype=float)
    for i, name in enumerate(agent_names):
        resolved = merge_name_map[clone_to_merge[name]]
        j = merge_names.index(resolved)
        out[j] += g_clone[i]
    return out


def per_example_nll(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    device,
) -> np.ndarray:
    """Per-example mean token NLL.

    :param model: Causal LM (possibly with adapter).
    :param tokenizer: Tokenizer for ``model``.
    :param texts: Chat-formatted strings.
    :type texts: Sequence[str]
    :param max_length: Truncation length.
    :type max_length: int
    :param batch_size: Forward batch size.
    :type batch_size: int
    :param device: Torch device.
    :returns: NLL vector, shape ``(len(texts),)``.
    :rtype: numpy.ndarray
    """
    import torch

    out = np.empty(len(texts), dtype=float)
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        with torch.no_grad():
            logits = model(**enc).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(len(chunk), -1)
        mask = shift_labels != -100
        denom = mask.sum(dim=1).clamp(min=1)
        per_ex = (per_token * mask).sum(dim=1) / denom
        out[start : start + len(chunk)] = per_ex.float().cpu().numpy()
    return out


def score_merge_nll_matrix(
    merge_names: Sequence[str],
    texts: Sequence[str],
    *,
    merge_run_dir: Path,
    round_idx: int,
    base_model: str,
    max_seq_length: int,
    forward_batch_size: int,
    universal_adapter_dir: Path | str | None = None,
) -> np.ndarray:
    """Score every merge adapter on every prompt.

    :param universal_adapter_dir: Optional frozen universal LoRA merged into
        the base once before the merge adapters are attached (residual arms).
    :type universal_adapter_dir: pathlib.Path | str | None
    :returns: Matrix shape ``(M, n_merge)``.
    :rtype: numpy.ndarray
    """
    base, tokenizer, device = load_base_causal_lm(
        base_model, universal_adapter_dir=universal_adapter_dir,
    )
    cols: list[np.ndarray] = []
    try:
        for merge in merge_names:
            adapter_dir = merge_run_dir / "agents" / merge / f"round-{round_idx:02d}"
            model = load_adapter_model(base, adapter_dir)
            try:
                cols.append(
                    per_example_nll(
                        model,
                        tokenizer,
                        texts,
                        max_length=max_seq_length,
                        batch_size=forward_batch_size,
                        device=device,
                    ),
                )
            finally:
                import torch
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        import torch
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.stack(cols, axis=1)


def score_pooled_nll(
    texts: Sequence[str],
    *,
    baseline_run_dir: Path,
    round_idx: int,
    base_model: str,
    max_seq_length: int,
    forward_batch_size: int,
) -> np.ndarray:
    """Per-prompt NLL under ``pooled-baseline``.

    :returns: NLL vector, shape ``(len(texts),)``.
    :rtype: numpy.ndarray
    """
    base, tokenizer, device = load_base_causal_lm(base_model)
    adapter_dir = (
        baseline_run_dir / "agents" / "pooled-baseline" / f"round-{round_idx:02d}"
    )
    model = load_adapter_model(base, adapter_dir)
    try:
        return per_example_nll(
            model,
            tokenizer,
            texts,
            max_length=max_seq_length,
            batch_size=forward_batch_size,
            device=device,
        )
    finally:
        import torch
        del model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_universal_nll(
    texts: Sequence[str],
    *,
    universal_adapter_dir: Path | str,
    base_model: str,
    max_seq_length: int,
    forward_batch_size: int,
) -> np.ndarray:
    """Per-prompt NLL of ``base + universal`` with no domain adapter.

    This is the merge-fidelity check of the residual arms: when the
    universal adapter is the pooled generalist, the result must match
    :func:`score_pooled_nll` up to the rounding of the float32 merge.

    :returns: NLL vector, shape ``(len(texts),)``.
    :rtype: numpy.ndarray
    """
    base, tokenizer, device = load_base_causal_lm(
        base_model, universal_adapter_dir=universal_adapter_dir,
    )
    try:
        return per_example_nll(
            base,
            tokenizer,
            texts,
            max_length=max_seq_length,
            batch_size=forward_batch_size,
            device=device,
        )
    finally:
        import torch
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def sample_proportional_merge_idx(
    g_merge: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """One proportional :math:`G` sample per prompt at merge level.

    :param g_merge: Merge weights, shape ``(n_merge, M)``.
    :type g_merge: numpy.ndarray
    :param seed: RNG seed.
    :type seed: int
    :returns: Merge index per prompt, shape ``(M,)``.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    n_merge, m = g_merge.shape
    out = np.empty(m, dtype=int)
    for j in range(m):
        p = g_merge[:, j]
        p = p / max(float(p.sum()), 1e-30)
        out[j] = int(rng.choice(n_merge, p=p))
    return out


def compute_flat_scores(
    *,
    merge_nll: np.ndarray,
    pooled_nll: np.ndarray,
    g_merge: np.ndarray,
    p_merge: np.ndarray,
    argmax_merge_idx: np.ndarray,
    strategic_merge_idx: np.ndarray,
    sampled_merge_idx: np.ndarray,
    bench_labels: Sequence[str],
    round_idx: int,
    universal_nll: np.ndarray | None = None,
    fitted_proba: np.ndarray | None = None,
    fitted_val_accuracy: float | None = None,
) -> tuple[FlatRoutingScores, dict[str, dict[str, Any]]]:
    """Reduce per-prompt NLL matrices to the headline and per-benchmark scores.

    Pure numpy so the routing arithmetic (including the universal-only and
    fitted-router columns) is testable without adapters.

    :param merge_nll: Expert NLL matrix, shape ``(M, n_merge)``.
    :type merge_nll: numpy.ndarray
    :param pooled_nll: Pooled-baseline NLL, shape ``(M,)``.
    :type pooled_nll: numpy.ndarray
    :param g_merge: Merge-level :math:`G`, shape ``(n_merge, M)``.
    :type g_merge: numpy.ndarray
    :param p_merge: Merge-level :math:`G(1-G)` weights, shape ``(n_merge, M)``.
    :type p_merge: numpy.ndarray
    :param argmax_merge_idx: Argmax-:math:`G` merge per prompt.
    :type argmax_merge_idx: numpy.ndarray
    :param strategic_merge_idx: Argmax strategic merge per prompt.
    :type strategic_merge_idx: numpy.ndarray
    :param sampled_merge_idx: One proportional sample per prompt.
    :type sampled_merge_idx: numpy.ndarray
    :param bench_labels: Source benchmark per prompt.
    :type bench_labels: Sequence[str]
    :param round_idx: Round scored.
    :type round_idx: int
    :param universal_nll: Optional universal-only NLL, shape ``(M,)``.
    :type universal_nll: numpy.ndarray | None
    :param fitted_proba: Optional fitted-router probabilities, shape
        ``(M, n_merge)``.
    :type fitted_proba: numpy.ndarray | None
    :param fitted_val_accuracy: Fitted router's validation accuracy.
    :type fitted_val_accuracy: float | None
    :returns: ``(headline scores, per-benchmark breakdown)``.
    :rtype: tuple[FlatRoutingScores, dict[str, dict[str, Any]]]
    """
    m = merge_nll.shape[0]
    rows = np.arange(m)
    expected_nll = (g_merge.T * merge_nll).sum(axis=1)
    strategic_expected_nll = (p_merge.T * merge_nll).sum(axis=1)
    argmax_nll = merge_nll[rows, argmax_merge_idx]
    strategic_argmax_nll = merge_nll[rows, strategic_merge_idx]
    sampled_nll = merge_nll[rows, sampled_merge_idx]
    oracle_merge_idx = np.argmin(merge_nll, axis=1)
    oracle_nll = merge_nll[rows, oracle_merge_idx]

    fitted_argmax_nll: np.ndarray | None = None
    fitted_expected_nll: np.ndarray | None = None
    fitted_idx: np.ndarray | None = None
    if fitted_proba is not None:
        fitted_idx = np.argmax(fitted_proba, axis=1)
        fitted_argmax_nll = merge_nll[rows, fitted_idx]
        fitted_expected_nll = (fitted_proba * merge_nll).sum(axis=1)

    def _mean(v: np.ndarray | None, mask: np.ndarray | None = None) -> float | None:
        if v is None:
            return None
        return float(v.mean() if mask is None else v[mask].mean())

    flat = FlatRoutingScores(
        pooled_nll=float(pooled_nll.mean()),
        learned_argmax_nll=float(argmax_nll.mean()),
        learned_expected_nll=float(expected_nll.mean()),
        learned_sampled_nll=float(sampled_nll.mean()),
        strategic_expected_nll=float(strategic_expected_nll.mean()),
        strategic_argmax_nll=float(strategic_argmax_nll.mean()),
        oracle_nll=float(oracle_nll.mean()),
        n_prompts=m,
        round_idx=round_idx,
        universal_nll=_mean(universal_nll),
        fitted_argmax_nll=_mean(fitted_argmax_nll),
        fitted_expected_nll=_mean(fitted_expected_nll),
        fitted_val_accuracy=fitted_val_accuracy,
    )

    per_bench: dict[str, dict[str, Any]] = {}
    for bench in sorted(set(bench_labels)):
        mask = np.array([b == bench for b in bench_labels])
        row: dict[str, Any] = {
            "n": int(mask.sum()),
            "pooled_nll": float(pooled_nll[mask].mean()),
            "learned_argmax_nll": float(argmax_nll[mask].mean()),
            "learned_expected_nll": float(expected_nll[mask].mean()),
            "learned_sampled_nll": float(sampled_nll[mask].mean()),
            "strategic_expected_nll": float(strategic_expected_nll[mask].mean()),
            "strategic_argmax_nll": float(strategic_argmax_nll[mask].mean()),
            "oracle_nll": float(oracle_nll[mask].mean()),
            "agreement_argmax": float(
                (argmax_merge_idx[mask] == oracle_merge_idx[mask]).mean(),
            ),
        }
        if universal_nll is not None:
            row["universal_nll"] = _mean(universal_nll, mask)
        if fitted_idx is not None:
            row["fitted_argmax_nll"] = _mean(fitted_argmax_nll, mask)
            row["fitted_expected_nll"] = _mean(fitted_expected_nll, mask)
            row["fitted_agreement_argmax"] = float(
                (fitted_idx[mask] == oracle_merge_idx[mask]).mean(),
            )
        per_bench[bench] = row
    return flat, per_bench


def run_flat_routing_eval(
    *,
    router_config: Path,
    history_path: Path,
    merge_run_dir: Path,
    baseline_run_dir: Path,
    repo_root: Path,
    partition: str = "test",
    max_eval_records: int | None = 1000,
    seed: int = 0,
    round_idx: int | None = None,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length: int = 1024,
    forward_batch_size: int = 8,
    score_adapters: bool = True,
    merge_nll: np.ndarray | None = None,
    pooled_nll: np.ndarray | None = None,
    merge_nll_cache: Path | None = None,
    save_merge_nll_cache: Path | None = None,
    universal_adapter_dir: Path | str | None = None,
    merge_aliases: Mapping[str, str] | None = None,
    fitted_router: bool = False,
    val_merge_nll: np.ndarray | None = None,
    val_merge_nll_cache: Path | None = None,
    universal_nll: np.ndarray | None = None,
) -> FlatRoutingReport:
    """Run flat-pool route-then-score evaluation.

    When ``score_adapters`` is false, pass precomputed ``merge_nll`` and
    ``pooled_nll`` (and ``val_merge_nll`` if ``fitted_router``) to skip GPU
    scoring (routing-only analysis).

    :param universal_adapter_dir: Frozen universal LoRA merged under every
        merge adapter (residual ``modula_res`` arms). Also scored alone as
        ``universal_nll``.
    :type universal_adapter_dir: pathlib.Path | str | None
    :param merge_aliases: Extra ``config merge name -> on-disk adapter name``
        aliases (the label arms map ``pair-k`` onto benchmark experts).
    :type merge_aliases: Mapping[str, str] | None
    :param fitted_router: Also fit a prompt-level softmax router on the
        validation pool (argmin-NLL expert) and score it on ``partition``.
    :type fitted_router: bool
    :param val_merge_nll: Precomputed validation NLL matrix for the fitted
        router (``(M_val, n_merge)``); scored on GPU when ``None``.
    :type val_merge_nll: numpy.ndarray | None
    :param val_merge_nll_cache: Optional ``.npy`` path to read/write the
        validation NLL matrix.
    :type val_merge_nll_cache: pathlib.Path | None
    :param universal_nll: Precomputed universal-only NLL vector.
    :type universal_nll: numpy.ndarray | None
    :returns: Full routing report.
    :rtype: FlatRoutingReport
    """
    cfg = load_config(router_config, validate=False)
    cl = cfg.get("closed_loop", {})
    clone_to_merge, config_merge_names = parse_merge_groups(cl)
    agent_names = [a["name"] for a in cfg["agents"]]
    rnd = round_idx if round_idx is not None else final_round(history_path)
    merge_names, merge_name_map = resolve_merge_adapters(
        merge_run_dir, rnd, config_merge_names, aliases=merge_aliases,
    )

    prompts, responses, bench_labels = load_flat_partition_pool(
        cfg,
        repo_root=repo_root,
        partition=partition,
        max_eval_records=max_eval_records,
        seed=seed,
    )
    fmt = build_chat_formatter(base_model)
    texts = [
        fmt(p, r if r else None)
        for p, r in zip(prompts, responses)
    ]

    full_splits = load_splits(cfg)
    space = make_trait_space(cfg, full_splits)
    sigma = sigma_from_config(cfg, len(agent_names), space)
    positions = load_final_positions(history_path, agent_names)
    coords = np.asarray(space.project(prompts), dtype=float)
    cov = float(sigma) ** 2 * np.eye(space.L)
    g_clone = allocation_weights(positions, coords, cov)
    g_merge = aggregate_clone_g_to_merge(
        g_clone, agent_names, clone_to_merge, merge_names, merge_name_map,
    )
    p_clone = strategic_routing_weights(positions, coords, cov)
    p_merge = aggregate_clone_g_to_merge(
        p_clone, agent_names, clone_to_merge, merge_names, merge_name_map,
    )

    clone_win = np.argmax(g_clone, axis=0)
    argmax_merge_idx = np.array(
        [
            merge_names.index(
                merge_name_map[clone_to_merge[agent_names[i]]],
            )
            for i in clone_win
        ],
        dtype=int,
    )
    sampled_merge_idx = sample_proportional_merge_idx(g_merge, seed=seed)
    strategic_merge_idx = np.argmax(p_merge, axis=0)

    if merge_nll is None and merge_nll_cache is not None and merge_nll_cache.is_file():
        merge_nll = np.load(merge_nll_cache)

    if score_adapters:
        merge_nll = score_merge_nll_matrix(
            merge_names,
            texts,
            merge_run_dir=merge_run_dir,
            round_idx=rnd,
            base_model=base_model,
            max_seq_length=max_seq_length,
            forward_batch_size=forward_batch_size,
            universal_adapter_dir=universal_adapter_dir,
        )
        pooled_nll = score_pooled_nll(
            texts,
            baseline_run_dir=baseline_run_dir,
            round_idx=rnd,
            base_model=base_model,
            max_seq_length=max_seq_length,
            forward_batch_size=forward_batch_size,
        )
        if universal_adapter_dir is not None and universal_nll is None:
            universal_nll = score_universal_nll(
                texts,
                universal_adapter_dir=universal_adapter_dir,
                base_model=base_model,
                max_seq_length=max_seq_length,
                forward_batch_size=forward_batch_size,
            )
    if save_merge_nll_cache is not None and merge_nll is not None:
        save_merge_nll_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_merge_nll_cache, merge_nll)
    if merge_nll is None or pooled_nll is None:
        raise ValueError("merge_nll and pooled_nll required when score_adapters=False")

    # Fitted prompt-level router: argmin-NLL expert on the validation pool,
    # predicted from trait coordinates, scored on this partition.
    fitted_proba: np.ndarray | None = None
    fitted_val_accuracy: float | None = None
    if fitted_router:
        from infl_ens.evaluation.fitted_router import fit_argmin_router

        val_prompts, _val_responses, _val_bench = load_flat_partition_pool(
            cfg,
            repo_root=repo_root,
            partition="val",
            max_eval_records=max_eval_records,
            seed=seed,
        )
        if (
            val_merge_nll is None
            and val_merge_nll_cache is not None
            and val_merge_nll_cache.is_file()
        ):
            val_merge_nll = np.load(val_merge_nll_cache)
        if val_merge_nll is None:
            if not score_adapters:
                raise ValueError(
                    "val_merge_nll required for fitted_router when score_adapters=False"
                )
            val_texts = [
                fmt(p, r if r else None)
                for p, r in zip(val_prompts, _val_responses, strict=True)
            ]
            val_merge_nll = score_merge_nll_matrix(
                merge_names,
                val_texts,
                merge_run_dir=merge_run_dir,
                round_idx=rnd,
                base_model=base_model,
                max_seq_length=max_seq_length,
                forward_batch_size=forward_batch_size,
                universal_adapter_dir=universal_adapter_dir,
            )
            if val_merge_nll_cache is not None:
                val_merge_nll_cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(val_merge_nll_cache, val_merge_nll)
        if val_merge_nll.shape != (len(val_prompts), len(merge_names)):
            raise ValueError(
                f"val_merge_nll shape {val_merge_nll.shape} != "
                f"({len(val_prompts)}, {len(merge_names)})"
            )
        val_coords = np.asarray(space.project(val_prompts), dtype=float)
        router = fit_argmin_router(val_coords, val_merge_nll)
        fitted_proba = router.predict_proba(coords)  # (M, n_merge)
        fitted_val_accuracy = float(router.train_accuracy)

    scores, per_bench = compute_flat_scores(
        merge_nll=merge_nll,
        pooled_nll=pooled_nll,
        g_merge=g_merge,
        p_merge=p_merge,
        argmax_merge_idx=argmax_merge_idx,
        strategic_merge_idx=strategic_merge_idx,
        sampled_merge_idx=sampled_merge_idx,
        bench_labels=bench_labels,
        round_idx=rnd,
        universal_nll=universal_nll,
        fitted_proba=fitted_proba,
        fitted_val_accuracy=fitted_val_accuracy,
    )
    flat = scores
    oracle_merge_idx = np.argmin(merge_nll, axis=1)
    bench_names = sorted(set(bench_labels))

    n_merge = len(merge_names)
    confusion = np.zeros((n_merge, n_merge), dtype=int)
    for li, oi in zip(argmax_merge_idx, oracle_merge_idx):
        confusion[int(li), int(oi)] += 1
    confusion_dict = {
        merge_names[i]: {
            merge_names[j]: int(confusion[i, j]) for j in range(n_merge)
        }
        for i in range(n_merge)
    }

    merge_support_argmax: dict[str, Any] = {}
    merge_support_expected: dict[str, Any] = {}
    for j, merge in enumerate(merge_names):
        mask = argmax_merge_idx == j
        merge_support_argmax[merge] = {
            "n_wins": int(mask.sum()),
            "share": float(mask.mean()),
            "by_benchmark": {
                b: int(sum(1 for bb, m in zip(bench_labels, mask) if m and bb == b))
                for b in bench_names
            },
        }
        merge_support_expected[merge] = {
            "mean_g": float(g_merge[j].mean()),
            "by_benchmark": {
                b: float(g_merge[j][np.array([bb == b for bb in bench_labels])].mean())
                if any(bb == b for bb in bench_labels)
                else 0.0
                for b in bench_names
            },
        }

    clone_support_argmax: dict[str, Any] = {}
    clone_support_expected: dict[str, Any] = {}
    for i, name in enumerate(agent_names):
        mask = clone_win == i
        clone_support_argmax[name] = {
            "n_wins": int(mask.sum()),
            "share": float(mask.mean()),
        }
        clone_support_expected[name] = {"mean_g": float(g_clone[i].mean())}

    return FlatRoutingReport(
        flat=flat,
        merge_names=merge_names,
        merge_name_map=merge_name_map,
        per_benchmark=per_bench,
        routing_confusion={
            "merge_names": merge_names,
            "counts": confusion_dict,
        },
        merge_support_argmax=merge_support_argmax,
        merge_support_expected=merge_support_expected,
        clone_support_argmax=clone_support_argmax,
        clone_support_expected=clone_support_expected,
        bench_labels=bench_labels,
    )


def report_to_dict(report: FlatRoutingReport) -> dict[str, Any]:
    """Serialize a :class:`FlatRoutingReport` to JSON-safe dict.

    :param report: Routing report.
    :type report: FlatRoutingReport
    :returns: JSON-serializable mapping.
    :rtype: dict[str, Any]
    """
    f = report.flat
    counts = report.routing_confusion.get("counts", {})
    names = report.merge_names
    total = f.n_prompts
    agree = 0
    for i, merge in enumerate(names):
        agree += int(counts.get(merge, {}).get(merge, 0))
    agreement = agree / max(total, 1)
    return {
        "flat": {
            "pooled_nll": f.pooled_nll,
            "learned_routing_argmax_nll": f.learned_argmax_nll,
            "learned_routing_expected_nll": f.learned_expected_nll,
            "learned_routing_sampled_nll": f.learned_sampled_nll,
            "strategic_routing_expected_nll": f.strategic_expected_nll,
            "strategic_routing_argmax_nll": f.strategic_argmax_nll,
            "learned_routing_nll": f.learned_expected_nll,
            "oracle_routing_nll": f.oracle_nll,
            "routing_agreement_argmax": agreement,
            "universal_nll": f.universal_nll,
            "fitted_routing_argmax_nll": f.fitted_argmax_nll,
            "fitted_routing_expected_nll": f.fitted_expected_nll,
            "fitted_router_val_accuracy": f.fitted_val_accuracy,
            "n_prompts": f.n_prompts,
            "round": f.round_idx,
            "merge_names": report.merge_names,
            "merge_name_map": report.merge_name_map,
        },
        "per_benchmark": report.per_benchmark,
        "routing_confusion": report.routing_confusion,
        "merge_support_argmax": report.merge_support_argmax,
        "merge_support_expected": report.merge_support_expected,
        "clone_support_argmax": report.clone_support_argmax,
        "clone_support_expected": report.clone_support_expected,
    }


def format_headline_markdown(report: FlatRoutingReport) -> str:
    """Render headline flat-pool metrics as markdown.

    Proportional expected routing is the primary learned-routing headline;
    argmax is retained for comparison.

    :param report: Routing report.
    :type report: FlatRoutingReport
    :returns: Markdown section.
    :rtype: str
    """
    f = report.flat
    lines = [
        "## Flat test pool (headline)",
        "",
        "| Metric | Mean NLL | Δ vs pooled | Δ vs oracle |",
        "|---|---:|---:|---:|",
        f"| Pooled baseline | {f.pooled_nll:.4f} | — | {f.pooled_nll - f.oracle_nll:+.4f} |",
        f"| **Learned routing (expected G)** | **{f.learned_expected_nll:.4f}** | "
        f"{f.learned_expected_nll - f.pooled_nll:+.4f} | "
        f"{f.learned_expected_nll - f.oracle_nll:+.4f} |",
        f"| Learned routing (expected G(1−G)) | {f.strategic_expected_nll:.4f} | "
        f"{f.strategic_expected_nll - f.pooled_nll:+.4f} | "
        f"{f.strategic_expected_nll - f.oracle_nll:+.4f} |",
        f"| Learned routing (argmax G) | {f.learned_argmax_nll:.4f} | "
        f"{f.learned_argmax_nll - f.pooled_nll:+.4f} | "
        f"{f.learned_argmax_nll - f.oracle_nll:+.4f} |",
        f"| Learned routing (argmax G(1−G)) | {f.strategic_argmax_nll:.4f} | "
        f"{f.strategic_argmax_nll - f.pooled_nll:+.4f} | "
        f"{f.strategic_argmax_nll - f.oracle_nll:+.4f} |",
        f"| Oracle routing (ceiling) | {f.oracle_nll:.4f} | "
        f"{f.oracle_nll - f.pooled_nll:+.4f} | — |",
    ]
    if f.fitted_expected_nll is not None and f.fitted_argmax_nll is not None:
        lines += [
            f"| Fitted router (expected) | {f.fitted_expected_nll:.4f} | "
            f"{f.fitted_expected_nll - f.pooled_nll:+.4f} | "
            f"{f.fitted_expected_nll - f.oracle_nll:+.4f} |",
            f"| Fitted router (argmax) | {f.fitted_argmax_nll:.4f} | "
            f"{f.fitted_argmax_nll - f.pooled_nll:+.4f} | "
            f"{f.fitted_argmax_nll - f.oracle_nll:+.4f} |",
        ]
    if f.universal_nll is not None:
        lines.append(
            f"| Universal only (merge check) | {f.universal_nll:.4f} | "
            f"{f.universal_nll - f.pooled_nll:+.4f} | "
            f"{f.universal_nll - f.oracle_nll:+.4f} |",
        )
    lines += [
        "",
        f"G(1−G) vs naive-G expected: "
        f"{f.strategic_expected_nll - f.learned_expected_nll:+.4f} "
        f"({'toward oracle' if f.strategic_expected_nll < f.learned_expected_nll else 'away from oracle'})",
        "",
        f"Pool size: **{f.n_prompts}** prompts, round **{f.round_idx}**.",
        "",
        "Per-benchmark breakdown is diagnostic only (see JSON).",
        "",
    ]
    dead_clones = [
        name
        for name, row in report.clone_support_argmax.items()
        if row["n_wins"] == 0 and report.clone_support_expected[name]["mean_g"] < 0.01
    ]
    if dead_clones:
        lines.append(
            f"Dead clones (0 argmax wins, mean G < 0.01): "
            f"{', '.join(dead_clones)}"
        )
        lines.append("")
    return "\n".join(lines)
