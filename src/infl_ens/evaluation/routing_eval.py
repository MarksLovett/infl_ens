"""Flat-pool route-then-score evaluation for merge ensembles.

Scores a concatenated benchmark partition under pooled baseline, learned
routing (argmax or proportional :math:`G`), and oracle merge assignment.
The flat mean NLL across the whole pool is the headline metric; per-benchmark
breakdowns are diagnostic only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from infl_ens.data.splits import (
    flatten_partition_records,
    load_split_manifest,
)
from infl_ens.evaluation.nll_artifact import (
    NLL_ARTIFACT_SCHEMA_VERSION,
    NllMatrixArtifact,
    checkpoint_sha256,
    ordered_records_sha256,
)
from infl_ens.evaluation.routers import (
    hindsight_oracle_router,
    low_support_mask,
    normalize_routing_weights,
    uniform_router,
)
from infl_ens.evaluation.adapters import load_adapter_model, load_base_causal_lm
from infl_ens.config import load_config
from infl_ens.evaluation.metrics import format_chat_example
from infl_ens.inflgame.router.allocation import (
    allocation_weights,
    strategic_routing_weights,
)
from infl_ens.inflgame.dynamics import kernel_allocation_weights
from infl_ens.training.setup import load_splits, make_trait_space, resolve_kernel_setup

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
    ensemble_mixture_nll: float | None = None
    ensemble_uniform_nll: float | None = None
    mean_tokens: float | None = None


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
    :param partition: Evaluated data partition.
    :type partition: str
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
    routers: dict[str, dict[str, Any]] = field(default_factory=dict)
    references: dict[str, dict[str, Any]] = field(default_factory=dict)
    oracle: dict[str, Any] = field(default_factory=dict)
    slices: dict[str, Any] = field(default_factory=dict)
    nll_artifact_digest: str | None = None
    expert_mean_nll: dict[str, float] = field(default_factory=dict)
    partition: str = "test"


@dataclass(frozen=True)
class RoutingWeightScores:
    """Metrics produced by one row-stochastic routing policy.

    :param expected_nll: Expected NLL after sampling one expert.
    :type expected_nll: float
    :param argmax_nll: NLL after selecting the maximum-weight expert.
    :type argmax_nll: float
    :param sampled_nll: NLL from one seeded categorical sample per row.
    :type sampled_nll: float
    :param sequence_mixture_nll: Sequence-level mixture likelihood diagnostic.
    :type sequence_mixture_nll: float
    :param oracle_nll: Hindsight per-record minimum expert NLL.
    :type oracle_nll: float
    :param oracle_regret: ``expected_nll - oracle_nll``.
    :type oracle_regret: float
    :param agreement_argmax: Fraction of argmax choices matching the oracle.
    :type agreement_argmax: float
    :param utilization: Mean routing mass per expert.
    :type utilization: tuple[float, ...]
    :param load_entropy: Entropy of mean expert utilization.
    :type load_entropy: float
    :param effective_experts: Exponential load entropy.
    :type effective_experts: float
    :param per_benchmark: Per-benchmark copies of the scalar metrics.
    :type per_benchmark: dict[str, dict[str, float | int]]
    """

    expected_nll: float
    argmax_nll: float
    sampled_nll: float
    sequence_mixture_nll: float
    oracle_nll: float
    oracle_regret: float
    agreement_argmax: float
    utilization: tuple[float, ...]
    load_entropy: float
    effective_experts: float
    per_benchmark: dict[str, dict[str, float | int]]


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
    prompts, responses, labels, _record_ids = load_flat_partition_records(
        cfg,
        repo_root=repo_root,
        partition=partition,
        max_eval_records=max_eval_records,
        seed=seed,
    )
    return prompts, responses, labels


def load_flat_partition_records(
    cfg: Mapping[str, Any],
    *,
    repo_root: Path,
    partition: str,
    max_eval_records: int | None,
    seed: int,
) -> tuple[list[str], list[str | None], list[str], list[str]]:
    """Load a capped flattened partition with stable record identifiers.

    :param cfg: Router configuration.
    :type cfg: Mapping[str, Any]
    :param repo_root: Repository root used for relative manifest paths.
    :type repo_root: pathlib.Path
    :param partition: ``train``, ``val``, ``test``, or ``train_val``.
    :type partition: str
    :param max_eval_records: Per-benchmark cap, or ``None``.
    :type max_eval_records: int | None
    :param seed: Deterministic per-benchmark subsampling seed.
    :type seed: int
    :returns: ``(prompts, responses, benchmark_labels, record_ids)``.
    :rtype: tuple[list[str], list[str | None], list[str], list[str]]
    """
    full_splits = load_splits(dict(cfg))
    ds = cfg.get("data_split") or {}
    if not ds:
        prompts: list[str] = []
        responses: list[str | None] = []
        labels: list[str] = []
        record_ids: list[str] = []
        for split in full_splits:
            indices = np.arange(split.n, dtype=int)
            if max_eval_records is not None and len(indices) > max_eval_records:
                indices = indices[
                    np.random.default_rng(seed).choice(
                        len(indices), size=max_eval_records, replace=False,
                    )
                ]
            split_responses = split.responses or [""] * split.n
            for index in indices:
                row = int(index)
                prompts.append(split.prompts[row])
                responses.append(split_responses[row])
                labels.append(split.name)
                record_ids.append(f"unsplit:{split.name}:{row}")
        return prompts, responses, labels, record_ids
    manifest_path = Path(ds["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest = load_split_manifest(manifest_path)
    if max_eval_records is None:
        prompts, responses, labels, record_ids = flatten_partition_records(
            full_splits, manifest, partition,  # type: ignore[arg-type]
        )
        return prompts, responses, labels, record_ids

    prompts: list[str] = []
    responses: list[str | None] = []
    labels: list[str] = []
    record_ids: list[str] = []
    for split in full_splits:
        original = np.asarray(
            manifest.partition_for(split.name).select(partition),  # type: ignore[arg-type]
            dtype=int,
        )
        if not len(original):
            continue
        if len(original) > max_eval_records:
            rng = np.random.default_rng(seed)
            original = original[
                rng.choice(len(original), size=max_eval_records, replace=False)
            ]
        split_responses = split.responses or [""] * split.n
        for index in original:
            row = int(index)
            prompts.append(split.prompts[row])
            responses.append(split_responses[row])
            labels.append(split.name)
            record_ids.append(f"{split.name}:{row}")
    return prompts, responses, labels, record_ids


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
    return_tokens: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
    :param return_tokens: Also return the supervised token count per example.
        Needed to convert the mean-token NLL back into a sequence log
        probability for :func:`mixture_nll`.
    :type return_tokens: bool
    :returns: NLL vector shape ``(len(texts),)``, or ``(nll, n_tokens)``.
    :rtype: numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray]
    """
    import torch

    out = np.empty(len(texts), dtype=float)
    ntok = np.empty(len(texts), dtype=float)
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
        ntok[start : start + len(chunk)] = denom.float().cpu().numpy()
    if return_tokens:
        return out, ntok
    return out


def mixture_nll(
    merge_nll: np.ndarray,
    token_counts: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    r"""Sequence-level mixture NLL over the merge adapters.

    ``learned_expected_nll`` is :math:`\sum_p w_p \mathrm{NLL}_p`, the expected
    NLL of *sampling one* expert. It is not an ensemble. The mixture

    .. math::

        \mathrm{NLL}^{\mathrm{mix}}_i \;=\; -\frac{1}{n_i}
        \log \sum_p w_{pi} \, P_p(y_i)
        \;=\; -\frac{1}{n_i} \operatorname*{logsumexp}_p
        \bigl( \log w_{pi} - n_i \, \mathrm{NLL}_{pi} \bigr)

    is what "ensemble" means, and by Jensen's inequality it is never worse:
    ``oracle <= mixture <= expected``. The stored per-example NLL is a *mean
    token* NLL, so :math:`\log P_p(y_i) = -n_i \mathrm{NLL}_{pi}` recovers the
    sequence log probability; the result is divided by :math:`n_i` again to
    stay on the same per-token scale as every other reported figure.

    Note this is a post-hoc reweighting of completed sequences, not token-level
    mixing of next-token distributions at generation time. For long sequences
    the logsumexp is dominated by the best expert, so the mixture approaches
    the oracle and the weights enter only as
    :math:`-\log w_{\mathrm{best}} / n_i`.

    :param merge_nll: Per-example mean-token NLL per merge, shape ``(M, K)``.
    :type merge_nll: numpy.ndarray
    :param token_counts: Supervised tokens per example, shape ``(M,)``.
    :type token_counts: numpy.ndarray
    :param weights: Mixture weights, shape ``(M, K)`` or broadcastable ``(K,)``.
        Rows need not be normalised; zeros are allowed (top-``k`` truncation).
    :type weights: numpy.ndarray
    :returns: Per-example mixture NLL, shape ``(M,)``.
    :rtype: numpy.ndarray
    """
    nll = np.asarray(merge_nll, dtype=float)
    n = np.asarray(token_counts, dtype=float).reshape(-1, 1)
    n = np.clip(n, 1.0, None)
    w = np.broadcast_to(np.asarray(weights, dtype=float), nll.shape)
    tiny = np.finfo(float).tiny
    with np.errstate(divide="ignore"):
        log_w = np.log(np.clip(w, tiny, None))
    terms = log_w - n * nll
    top = terms.max(axis=1, keepdims=True)
    lse = top[:, 0] + np.log(np.exp(terms - top).sum(axis=1))
    return -lse / n[:, 0]


def score_routing_weights(
    weights: np.ndarray,
    expert_nll: np.ndarray,
    token_counts: np.ndarray,
    bench_labels: Sequence[str],
    *,
    seed: int = 0,
) -> RoutingWeightScores:
    """Score one routing matrix without loading any model.

    The public boundary is deliberately ``(M, K)`` and row-stochastic even
    though the influencer-game allocation internals use ``(K, M)``.

    :param weights: Routing matrix, shape ``(M, K)``.
    :type weights: numpy.ndarray
    :param expert_nll: Per-record expert NLL, shape ``(M, K)``.
    :type expert_nll: numpy.ndarray
    :param token_counts: Supervised token count per record, shape ``(M,)``.
    :type token_counts: numpy.ndarray
    :param bench_labels: Benchmark label per record.
    :type bench_labels: Sequence[str]
    :param seed: Categorical-sampling seed.
    :type seed: int
    :returns: Router metrics and per-benchmark diagnostics.
    :rtype: RoutingWeightScores
    :raises ValueError: If arrays are misaligned or invalid.
    """
    routing = normalize_routing_weights(weights)
    nll = np.asarray(expert_nll, dtype=float)
    tokens = np.asarray(token_counts, dtype=float)
    if nll.shape != routing.shape:
        raise ValueError(
            f"expert_nll shape {nll.shape} != routing shape {routing.shape}"
        )
    if tokens.shape != (nll.shape[0],) or len(bench_labels) != nll.shape[0]:
        raise ValueError("token_counts and bench_labels must align with NLL rows")
    if not np.isfinite(nll).all():
        raise ValueError("expert_nll must contain only finite values")

    row = np.arange(nll.shape[0])
    argmax_index = np.argmax(routing, axis=1)
    oracle_index = np.argmin(nll, axis=1)
    rng = np.random.default_rng(seed)
    sampled_index = np.asarray(
        [rng.choice(nll.shape[1], p=routing[i]) for i in range(nll.shape[0])],
        dtype=int,
    )
    expected = (routing * nll).sum(axis=1)
    argmax = nll[row, argmax_index]
    sampled = nll[row, sampled_index]
    oracle = nll[row, oracle_index]
    mixture = mixture_nll(nll, tokens, routing)
    utilization = routing.mean(axis=0)
    positive = utilization[utilization > 0]
    entropy = -float(np.sum(positive * np.log(positive)))

    per_benchmark: dict[str, dict[str, float | int]] = {}
    label_array = np.asarray(bench_labels, dtype=str)
    for label in sorted(set(bench_labels)):
        mask = label_array == label
        per_benchmark[label] = {
            "n": int(mask.sum()),
            "expected_nll": float(expected[mask].mean()),
            "argmax_nll": float(argmax[mask].mean()),
            "sampled_nll": float(sampled[mask].mean()),
            "sequence_mixture_nll": float(mixture[mask].mean()),
            "oracle_nll": float(oracle[mask].mean()),
            "oracle_regret": float(expected[mask].mean() - oracle[mask].mean()),
            "agreement_argmax": float((argmax_index[mask] == oracle_index[mask]).mean()),
        }

    return RoutingWeightScores(
        expected_nll=float(expected.mean()),
        argmax_nll=float(argmax.mean()),
        sampled_nll=float(sampled.mean()),
        sequence_mixture_nll=float(mixture.mean()),
        oracle_nll=float(oracle.mean()),
        oracle_regret=float(expected.mean() - oracle.mean()),
        agreement_argmax=float((argmax_index == oracle_index).mean()),
        utilization=tuple(float(x) for x in utilization),
        load_entropy=entropy,
        effective_experts=float(math.exp(entropy)),
        per_benchmark=per_benchmark,
    )


def routing_scores_to_dict(scores: RoutingWeightScores) -> dict[str, Any]:
    """Serialize :class:`RoutingWeightScores`.

    :param scores: Router scores.
    :type scores: RoutingWeightScores
    :returns: JSON-safe dictionary.
    :rtype: dict[str, Any]
    """
    return {
        "expected_nll": scores.expected_nll,
        "argmax_nll": scores.argmax_nll,
        "sampled_nll": scores.sampled_nll,
        "sequence_mixture_nll": scores.sequence_mixture_nll,
        "oracle_nll": scores.oracle_nll,
        "oracle_regret": scores.oracle_regret,
        "agreement_argmax": scores.agreement_argmax,
        "utilization": list(scores.utilization),
        "load_entropy": scores.load_entropy,
        "effective_experts": scores.effective_experts,
        "per_benchmark": scores.per_benchmark,
    }


def score_merge_nll_matrix(
    merge_names: Sequence[str],
    texts: Sequence[str],
    *,
    merge_run_dir: Path,
    round_idx: int,
    base_model: str,
    max_seq_length: int,
    forward_batch_size: int,
    return_tokens: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Score every merge adapter on every prompt.

    :returns: Matrix shape ``(M, n_merge)``.
    :rtype: numpy.ndarray
    """
    base, tokenizer, device = load_base_causal_lm(base_model)
    cols: list[np.ndarray] = []
    # One tokenizer and one truncation length per run, so the supervised token
    # count is identical for every adapter; keep the first pass's counts.
    ntok: np.ndarray | None = None
    try:
        for merge in merge_names:
            adapter_dir = merge_run_dir / "agents" / merge / f"round-{round_idx:02d}"
            model = load_adapter_model(base, adapter_dir)
            try:
                col, counts = per_example_nll(
                    model,
                    tokenizer,
                    texts,
                    max_length=max_seq_length,
                    batch_size=forward_batch_size,
                    device=device,
                    return_tokens=True,
                )
                cols.append(col)
                if ntok is None:
                    ntok = counts
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
    matrix = np.stack(cols, axis=1)
    if return_tokens:
        return matrix, (ntok if ntok is not None else np.ones(len(texts)))
    return matrix


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


def score_reference_nll_matrix(
    texts: Sequence[str],
    *,
    reference_run_dirs: Mapping[str, Path],
    reference_agent_names: Mapping[str, str] | None,
    round_idx: int,
    base_model: str,
    max_seq_length: int,
    forward_batch_size: int,
) -> np.ndarray:
    """Score multiple pooled/reference adapters in a stable column order.

    :param texts: Chat-formatted evaluation examples.
    :type texts: Sequence[str]
    :param reference_run_dirs: Reference name to run directory mapping.
    :type reference_run_dirs: Mapping[str, pathlib.Path]
    :param reference_agent_names: Optional reference name to adapter name.
    :type reference_agent_names: Mapping[str, str] | None
    :param round_idx: Adapter round.
    :type round_idx: int
    :param base_model: Base causal-LM identifier.
    :type base_model: str
    :param max_seq_length: Tokenization cap.
    :type max_seq_length: int
    :param forward_batch_size: Evaluation batch size.
    :type forward_batch_size: int
    :returns: NLL matrix shape ``(M, B)``.
    :rtype: numpy.ndarray
    """
    base, tokenizer, device = load_base_causal_lm(base_model)
    columns: list[np.ndarray] = []
    try:
        for name, run_dir in reference_run_dirs.items():
            agent_name = (
                reference_agent_names.get(name, "pooled-baseline")
                if reference_agent_names is not None else "pooled-baseline"
            )
            adapter_dir = Path(run_dir) / "agents" / agent_name / f"round-{round_idx:02d}"
            model = load_adapter_model(base, adapter_dir)
            try:
                column = per_example_nll(
                    model,
                    tokenizer,
                    texts,
                    max_length=max_seq_length,
                    batch_size=forward_batch_size,
                    device=device,
                )
                columns.append(np.asarray(column, dtype=float))
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
    return np.stack(columns, axis=1)


def score_mixture_lora_checkpoint(
    texts: Sequence[str],
    coordinates: np.ndarray,
    *,
    checkpoint_dir: Path,
    max_seq_length: int,
    forward_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score a learned-gate LoRA mixture and each forced component.

    :param texts: Chat-formatted evaluation examples.
    :type texts: Sequence[str]
    :param coordinates: Trait coordinates aligned with ``texts``.
    :type coordinates: numpy.ndarray
    :param checkpoint_dir: Learned mixture checkpoint.
    :type checkpoint_dir: pathlib.Path
    :param max_seq_length: Tokenization cap.
    :type max_seq_length: int
    :param forward_batch_size: Evaluation batch size.
    :type forward_batch_size: int
    :returns: ``(learned_nll, component_nll, token_counts, gate_weights)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    import torch
    from torch.nn import functional as functional

    from infl_ens.training.mixture_lora import gate_weights, load_mixture_checkpoint

    model, tokenizer = load_mixture_checkpoint(checkpoint_dir)
    device = next(model.parameters()).device
    metadata = json.loads(
        (checkpoint_dir / "mixture_config.json").read_text(encoding="utf-8")
    )
    trait_mean = np.asarray(
        metadata.get("trait_mean", np.zeros(np.asarray(coordinates).shape[1])),
        dtype=np.float32,
    )
    trait_scale = np.asarray(
        metadata.get("trait_scale", np.ones(np.asarray(coordinates).shape[1])),
        dtype=np.float32,
    )
    standardized_coordinates = (
        np.asarray(coordinates, dtype=np.float32) - trait_mean
    ) / np.where(trait_scale > 1e-6, trait_scale, 1.0)
    n_experts = int(metadata["n_experts"])
    learned = np.empty(len(texts), dtype=float)
    components = np.empty((len(texts), n_experts), dtype=float)
    tokens = np.empty(len(texts), dtype=float)
    gates = np.empty((len(texts), n_experts), dtype=float)

    def batch_nll(logits: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        per_token = functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(shift_labels.shape)
        mask = (shift_labels != -100).to(per_token.dtype)
        counts = mask.sum(dim=1).clamp_min(1.0)
        values = (per_token * mask).sum(dim=1) / counts
        return values.float().cpu().numpy(), counts.float().cpu().numpy()

    try:
        for start in range(0, len(texts), forward_batch_size):
            stop = min(start + forward_batch_size, len(texts))
            encoded = tokenizer(
                list(texts[start:stop]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = encoded["input_ids"].clone()
            labels[encoded["attention_mask"] == 0] = -100
            traits = torch.as_tensor(
                standardized_coordinates[start:stop], device=device,
            )
            with torch.no_grad():
                output = model(trait_coords=traits, **encoded)
                values, counts = batch_nll(output.logits, labels)
                learned[start:stop] = values
                tokens[start:stop] = counts
                dense = model.last_dense_probabilities
                assert dense is not None
                deployed, _ = gate_weights(
                    model.gate(traits.float()), top_k=metadata.get("top_k")
                )
                gates[start:stop] = deployed.float().cpu().numpy()
                for expert in range(n_experts):
                    override = torch.zeros(
                        (stop - start, n_experts), device=device, dtype=deployed.dtype,
                    )
                    override[:, expert] = 1.0
                    component_output = model(
                        trait_coords=traits,
                        gate_override=override,
                        **encoded,
                    )
                    component_values, _counts = batch_nll(
                        component_output.logits, labels,
                    )
                    components[start:stop, expert] = component_values
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return learned, components, tokens, gates


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
    token_counts: np.ndarray | None = None,
    merge_nll_cache: Path | None = None,
    save_merge_nll_cache: Path | None = None,
    save_arrays_dir: Path | None = None,
    nll_artifact_path: Path | None = None,
    force_rescore: bool = False,
    reference_run_dirs: Mapping[str, Path] | None = None,
    reference_agent_names: Mapping[str, str] | None = None,
    extra_router_weights: Mapping[str, np.ndarray] | None = None,
    include_low_support: bool = True,
) -> FlatRoutingReport:
    """Run flat-pool route-then-score evaluation.

    When ``score_adapters`` is false, pass precomputed ``merge_nll`` and
    ``pooled_nll`` to skip GPU scoring (routing-only analysis).

    :returns: Full routing report.
    :rtype: FlatRoutingReport
    """
    cfg = load_config(router_config, validate=False)
    cl = cfg.get("closed_loop", {})
    clone_to_merge, config_merge_names = parse_merge_groups(cl)
    agent_names = [a["name"] for a in cfg["agents"]]
    rnd = round_idx if round_idx is not None else final_round(history_path)
    merge_names, merge_name_map = resolve_merge_adapters(
        merge_run_dir, rnd, config_merge_names,
    )

    prompts, responses, bench_labels, record_ids = load_flat_partition_records(
        cfg,
        repo_root=repo_root,
        partition=partition,
        max_eval_records=max_eval_records,
        seed=seed,
    )
    texts = [
        format_chat_example(p, r if r else None)
        for p, r in zip(prompts, responses)
    ]

    full_splits = load_splits(cfg)
    space = make_trait_space(cfg, full_splits)
    kernel_setup = resolve_kernel_setup(cfg, len(agent_names), space)
    sigma = kernel_setup.sigma
    positions = load_final_positions(history_path, agent_names)
    coords = np.asarray(space.project(prompts), dtype=float)
    cov = float(sigma) ** 2 * np.eye(space.L)
    g_clone = (
        kernel_allocation_weights(positions, coords, kernel_setup.kernel)
        if kernel_setup.kernel is not None
        else allocation_weights(positions, coords, cov)
    )
    g_merge = aggregate_clone_g_to_merge(
        g_clone, agent_names, clone_to_merge, merge_names, merge_name_map,
    )
    if kernel_setup.kernel is not None:
        p_clone = g_clone * (1.0 - g_clone)
        total = p_clone.sum(axis=0, keepdims=True)
        p_clone = np.divide(p_clone, total, out=g_clone.copy(), where=total > 1e-12)
    else:
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

    references = dict(reference_run_dirs or {"pooled_r16": baseline_run_dir})
    reference_agents = dict(reference_agent_names or {})
    reference_names = list(references)
    reference_nll: np.ndarray | None = None
    artifact_digest: str | None = None
    if nll_artifact_path is None and save_arrays_dir is not None:
        nll_artifact_path = save_arrays_dir / f"nll_{partition}_round{rnd:02d}.npz"

    def requested_provenance() -> dict[str, Any]:
        expert_hashes = {
            name: checkpoint_sha256(
                merge_run_dir / "agents" / name / f"round-{rnd:02d}"
            )
            for name in merge_names
        }
        reference_hashes = {
            name: checkpoint_sha256(
                Path(run_dir)
                / "agents"
                / reference_agents.get(name, "pooled-baseline")
                / f"round-{rnd:02d}"
            )
            for name, run_dir in references.items()
        }
        return {
            "schema_version": NLL_ARTIFACT_SCHEMA_VERSION,
            "expert_family": str(merge_run_dir),
            "round": rnd,
            "partition": partition,
            "max_eval_records": max_eval_records,
            "seed": seed,
            "base_model": base_model,
            "max_seq_length": max_seq_length,
            "forward_batch_size": forward_batch_size,
            "ordered_records_sha256": ordered_records_sha256(
                record_ids, prompts, responses,
            ),
            "expert_checkpoints": expert_hashes,
            "reference_checkpoints": reference_hashes,
        }

    provenance: dict[str, Any] | None = None
    if nll_artifact_path is not None and nll_artifact_path.is_file() and not force_rescore:
        provenance = requested_provenance()
        try:
            artifact = NllMatrixArtifact.load(
                nll_artifact_path, expected_provenance=provenance,
            )
        except ValueError:
            artifact = None
        if artifact is not None:
            if list(artifact.expert_names) != merge_names:
                raise ValueError("cached expert column names do not match resolved adapters")
            if list(artifact.reference_names) != reference_names:
                raise ValueError("cached reference column names do not match requested references")
            merge_nll = artifact.expert_nll
            reference_nll = artifact.reference_nll
            pooled_nll = reference_nll[:, 0]
            token_counts = artifact.token_counts
            artifact_digest = artifact.digest

    if merge_nll is None and merge_nll_cache is not None and merge_nll_cache.is_file():
        merge_nll = np.load(merge_nll_cache, allow_pickle=False)
        tok_cache = merge_nll_cache.with_name("token_counts.npy")
        if token_counts is None and tok_cache.is_file():
            token_counts = np.load(tok_cache, allow_pickle=False)

    if merge_nll is None:
        if not score_adapters:
            raise ValueError("expert NLL matrix is missing and score_adapters=False")
        merge_nll, token_counts = score_merge_nll_matrix(
            merge_names,
            texts,
            merge_run_dir=merge_run_dir,
            round_idx=rnd,
            base_model=base_model,
            max_seq_length=max_seq_length,
            forward_batch_size=forward_batch_size,
            return_tokens=True,
        )
    if reference_nll is None:
        if pooled_nll is not None and len(reference_names) == 1:
            reference_nll = np.asarray(pooled_nll, dtype=float).reshape(-1, 1)
        elif score_adapters:
            reference_nll = score_reference_nll_matrix(
                texts,
                reference_run_dirs=references,
                reference_agent_names=reference_agents,
                round_idx=rnd,
                base_model=base_model,
                max_seq_length=max_seq_length,
                forward_batch_size=forward_batch_size,
            )
        else:
            raise ValueError("reference NLL matrix is missing and score_adapters=False")
        pooled_nll = reference_nll[:, 0]
    if token_counts is None:
        token_counts = np.ones(len(texts), dtype=float)
    if save_merge_nll_cache is not None and merge_nll is not None:
        save_merge_nll_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_merge_nll_cache, merge_nll)
    if merge_nll is None or pooled_nll is None or reference_nll is None:
        raise ValueError("expert and reference NLL arrays are required")
    if nll_artifact_path is not None and artifact_digest is None:
        provenance = provenance or requested_provenance()
        artifact = NllMatrixArtifact(
            expert_nll=np.asarray(merge_nll, dtype=float),
            reference_nll=np.asarray(reference_nll, dtype=float),
            token_counts=np.asarray(token_counts, dtype=float),
            expert_names=tuple(merge_names),
            reference_names=tuple(reference_names),
            bench_labels=tuple(bench_labels),
            record_ids=tuple(record_ids),
            provenance=provenance,
        )
        artifact.save(nll_artifact_path)
        artifact_digest = artifact.digest

    router_matrices: dict[str, np.ndarray] = {
        "native": g_merge.T,
        "game_g": g_merge.T,
        "strategic": p_merge.T,
        "uniform": uniform_router(len(texts), len(merge_names)),
        "hindsight_oracle": hindsight_oracle_router(merge_nll),
    }
    router_matrices.update(extra_router_weights or {})
    router_scores = {
        name: score_routing_weights(
            weights,
            merge_nll,
            token_counts,
            bench_labels,
            seed=seed,
        )
        for name, weights in router_matrices.items()
    }
    expected_nll = (normalize_routing_weights(g_merge.T) * merge_nll).sum(axis=1)
    strategic_expected_nll = (
        normalize_routing_weights(p_merge.T) * merge_nll
    ).sum(axis=1)
    mixture_expected = mixture_nll(merge_nll, token_counts, g_merge.T)
    mixture_uniform = mixture_nll(
        merge_nll, token_counts, router_matrices["uniform"],
    )
    if save_arrays_dir is not None:
        save_arrays_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_arrays_dir / "merge_nll.npy", merge_nll)
        np.save(save_arrays_dir / "g_merge.npy", g_merge)
        np.save(save_arrays_dir / "token_counts.npy", token_counts)
        np.save(save_arrays_dir / f"g_merge_{partition}.npy", g_merge)
        np.save(save_arrays_dir / f"coords_{partition}.npy", coords)
    argmax_nll = merge_nll[np.arange(len(texts)), argmax_merge_idx]
    strategic_argmax_nll = merge_nll[np.arange(len(texts)), strategic_merge_idx]
    sampled_nll = merge_nll[np.arange(len(texts)), sampled_merge_idx]
    oracle_merge_idx = np.argmin(merge_nll, axis=1)
    oracle_nll = merge_nll[np.arange(len(texts)), oracle_merge_idx]

    flat = FlatRoutingScores(
        pooled_nll=float(pooled_nll.mean()),
        learned_argmax_nll=float(argmax_nll.mean()),
        learned_expected_nll=float(expected_nll.mean()),
        learned_sampled_nll=float(sampled_nll.mean()),
        strategic_expected_nll=float(strategic_expected_nll.mean()),
        strategic_argmax_nll=float(strategic_argmax_nll.mean()),
        oracle_nll=float(oracle_nll.mean()),
        n_prompts=len(texts),
        round_idx=rnd,
        ensemble_mixture_nll=float(mixture_expected.mean()),
        ensemble_uniform_nll=float(mixture_uniform.mean()),
        mean_tokens=float(np.mean(token_counts)),
    )

    bench_names = sorted(set(bench_labels))
    per_bench: dict[str, dict[str, Any]] = {}
    for bench in bench_names:
        mask = np.array([b == bench for b in bench_labels])
        per_bench[bench] = {
            "n": int(mask.sum()),
            "pooled_nll": float(pooled_nll[mask].mean()),
            "learned_argmax_nll": float(argmax_nll[mask].mean()),
            "learned_expected_nll": float(expected_nll[mask].mean()),
            "learned_sampled_nll": float(sampled_nll[mask].mean()),
            "strategic_expected_nll": float(strategic_expected_nll[mask].mean()),
            "strategic_argmax_nll": float(strategic_argmax_nll[mask].mean()),
            "oracle_nll": float(oracle_nll[mask].mean()),
            "ensemble_mixture_nll": (
                None if mixture_expected is None
                else float(mixture_expected[mask].mean())
            ),
            "ensemble_uniform_nll": (
                None if mixture_uniform is None
                else float(mixture_uniform[mask].mean())
            ),
            "mean_tokens": (
                None if token_counts is None
                else float(np.mean(token_counts[mask]))
            ),
            "agreement_argmax": float(
                (argmax_merge_idx[mask] == oracle_merge_idx[mask]).mean(),
            ),
        }

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

    reference_report: dict[str, dict[str, Any]] = {}
    label_array = np.asarray(bench_labels, dtype=str)
    for j, name in enumerate(reference_names):
        reference_report[name] = {
            "mean_nll": float(reference_nll[:, j].mean()),
            "per_benchmark": {
                bench: float(reference_nll[label_array == bench, j].mean())
                for bench in bench_names
            },
        }
    router_report = {
        name: routing_scores_to_dict(scores)
        for name, scores in router_scores.items()
    }
    slices: dict[str, Any] = {"full": {"routers": router_report}}
    if partition == "test" and include_low_support:
        train_prompts, _train_responses, train_labels, _train_ids = (
            load_flat_partition_records(
                cfg,
                repo_root=repo_root,
                partition="train",
                max_eval_records=None,
                seed=seed,
            )
        )
        train_coords = np.asarray(space.project(train_prompts), dtype=float)
        train_label_array = np.asarray(train_labels, dtype=str)
        train_centroids = {
            label: train_coords[train_label_array == label].mean(axis=0)
            for label in sorted(set(train_labels))
        }
        support_mask = low_support_mask(
            coords,
            bench_labels,
            train_centroids,
            train_coords.mean(axis=0),
            train_coords.std(axis=0),
            fraction=0.2,
        )
        low_router_report = {
            name: routing_scores_to_dict(
                score_routing_weights(
                    weights[support_mask],
                    merge_nll[support_mask],
                    token_counts[support_mask],
                    label_array[support_mask].tolist(),
                    seed=seed,
                )
            )
            for name, weights in router_matrices.items()
        }
        slices["low_support_id"] = {
            "definition": "furthest 20% within each benchmark from its training centroid",
            "n_prompts": int(support_mask.sum()),
            "record_ids": [
                record_ids[i] for i in np.flatnonzero(support_mask)
            ],
            "routers": low_router_report,
            "references": {
                name: {"mean_nll": float(reference_nll[support_mask, j].mean())}
                for j, name in enumerate(reference_names)
            },
        }

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
        routers=router_report,
        references=reference_report,
        oracle={
            "mean_nll": float(oracle_nll.mean()),
            "expert_names": merge_names,
            "selection_counts": {
                merge_names[j]: int(np.sum(oracle_merge_idx == j))
                for j in range(len(merge_names))
            },
        },
        slices=slices,
        nll_artifact_digest=artifact_digest,
        expert_mean_nll={
            name: float(merge_nll[:, i].mean())
            for i, name in enumerate(merge_names)
        },
        partition=partition,
    )


def run_mixture_routing_eval(
    *,
    config_path: Path,
    run_dir: Path,
    reference_run_dirs: Mapping[str, Path],
    reference_agent_names: Mapping[str, str],
    repo_root: Path,
    partition: str,
    max_eval_records: int | None,
    seed: int,
    round_idx: int,
    base_model: str,
    max_seq_length: int,
    forward_batch_size: int,
    force_rescore: bool = False,
    extra_router_weights: Mapping[str, np.ndarray] | None = None,
    include_low_support: bool = True,
) -> dict[str, Any]:
    """Evaluate a learned-gate LoRA mixture and its forced components.

    :param config_path: Resolved mixture run configuration.
    :type config_path: pathlib.Path
    :param run_dir: Mixture run directory.
    :type run_dir: pathlib.Path
    :param reference_run_dirs: Named pooled reference directories.
    :type reference_run_dirs: Mapping[str, pathlib.Path]
    :param reference_agent_names: Adapter name for each reference.
    :type reference_agent_names: Mapping[str, str]
    :param repo_root: Repository root.
    :type repo_root: pathlib.Path
    :param partition: Evaluation partition.
    :type partition: str
    :param max_eval_records: Per-benchmark cap.
    :type max_eval_records: int | None
    :param seed: Evaluation seed.
    :type seed: int
    :param round_idx: Mixture checkpoint round.
    :type round_idx: int
    :param base_model: Base model identifier.
    :type base_model: str
    :param max_seq_length: Tokenization cap.
    :type max_seq_length: int
    :param forward_batch_size: Evaluation batch size.
    :type forward_batch_size: int
    :param force_rescore: Ignore compatible cached arrays.
    :type force_rescore: bool
    :param extra_router_weights: Validation-fitted held-out routers.
    :type extra_router_weights: Mapping[str, numpy.ndarray] | None
    :param include_low_support: Compute the low-support stress slice.
    :type include_low_support: bool
    :returns: Schema-v2 routing report.
    :rtype: dict[str, Any]
    """
    cfg = load_config(config_path, validate=False)
    mix = cfg.get("mixture_lora") or {}
    model_name = "ewora-dense" if str(mix.get("mode", "dense")) == "dense" else "trait-gated-topk"
    checkpoint = run_dir / "agents" / model_name / f"round-{round_idx:02d}"
    prompts, responses, labels, record_ids = load_flat_partition_records(
        cfg,
        repo_root=repo_root,
        partition=partition,
        max_eval_records=max_eval_records,
        seed=seed,
    )
    texts = [
        format_chat_example(prompt, response or None)
        for prompt, response in zip(prompts, responses)
    ]
    splits = load_splits(cfg)
    space = make_trait_space(cfg, splits)
    coordinates = np.asarray(space.project(prompts), dtype=float)
    np.save(run_dir / f"coords_{partition}.npy", coordinates)
    reference_names = list(reference_run_dirs)
    provenance = {
        "schema_version": NLL_ARTIFACT_SCHEMA_VERSION,
        "expert_family": str(run_dir),
        "round": round_idx,
        "partition": partition,
        "max_eval_records": max_eval_records,
        "seed": seed,
        "base_model": base_model,
        "max_seq_length": max_seq_length,
        "forward_batch_size": forward_batch_size,
        "ordered_records_sha256": ordered_records_sha256(record_ids, prompts, responses),
        "expert_checkpoints": {model_name: checkpoint_sha256(checkpoint)},
        "reference_checkpoints": {
            name: checkpoint_sha256(
                Path(path)
                / "agents"
                / reference_agent_names.get(name, "pooled-baseline")
                / f"round-{round_idx:02d}"
            )
            for name, path in reference_run_dirs.items()
        },
    }
    artifact_path = run_dir / f"nll_{partition}_round{round_idx:02d}.npz"
    prediction_path = run_dir / f"mixture_predictions_{partition}_round{round_idx:02d}.npz"
    artifact: NllMatrixArtifact | None = None
    if artifact_path.is_file() and prediction_path.is_file() and not force_rescore:
        try:
            artifact = NllMatrixArtifact.load(
                artifact_path, expected_provenance=provenance,
            )
            with np.load(prediction_path, allow_pickle=False) as predictions:
                learned_nll = np.asarray(predictions["learned_nll"], dtype=float)
                native_weights = np.asarray(predictions["native_weights"], dtype=float)
        except (ValueError, KeyError):
            artifact = None
    if artifact is None:
        learned_nll, component_nll, token_counts, native_weights = (
            score_mixture_lora_checkpoint(
                texts,
                coordinates,
                checkpoint_dir=checkpoint,
                max_seq_length=max_seq_length,
                forward_batch_size=forward_batch_size,
            )
        )
        reference_nll = score_reference_nll_matrix(
            texts,
            reference_run_dirs=reference_run_dirs,
            reference_agent_names=reference_agent_names,
            round_idx=round_idx,
            base_model=base_model,
            max_seq_length=max_seq_length,
            forward_batch_size=forward_batch_size,
        )
        expert_names = tuple(f"component-{i}" for i in range(component_nll.shape[1]))
        artifact = NllMatrixArtifact(
            expert_nll=component_nll,
            reference_nll=reference_nll,
            token_counts=token_counts,
            expert_names=expert_names,
            reference_names=tuple(reference_names),
            bench_labels=tuple(labels),
            record_ids=tuple(record_ids),
            provenance=provenance,
        )
        artifact.save(artifact_path)
        np.savez_compressed(
            prediction_path,
            learned_nll=learned_nll,
            native_weights=native_weights,
        )
    router_matrices = {
        "learned_gate": native_weights,
        "uniform": uniform_router(len(prompts), artifact.expert_nll.shape[1]),
        "hindsight_oracle": hindsight_oracle_router(artifact.expert_nll),
        **dict(extra_router_weights or {}),
    }
    router_report = {
        name: routing_scores_to_dict(
            score_routing_weights(
                weights,
                artifact.expert_nll,
                artifact.token_counts,
                labels,
                seed=seed,
            )
        )
        for name, weights in router_matrices.items()
    }
    references = {
        name: {
            "mean_nll": float(artifact.reference_nll[:, index].mean()),
            "per_benchmark": {
                label: float(
                    artifact.reference_nll[np.asarray(labels) == label, index].mean()
                )
                for label in sorted(set(labels))
            },
        }
        for index, name in enumerate(reference_names)
    }
    oracle = artifact.expert_nll.min(axis=1)
    slices: dict[str, Any] = {"full": {"routers": router_report}}
    if partition == "test" and include_low_support:
        train_prompts, _train_responses, train_labels, _train_ids = load_flat_partition_records(
            cfg,
            repo_root=repo_root,
            partition="train",
            max_eval_records=None,
            seed=seed,
        )
        train_coordinates = np.asarray(space.project(train_prompts), dtype=float)
        train_label_array = np.asarray(train_labels)
        centroids = {
            label: train_coordinates[train_label_array == label].mean(axis=0)
            for label in sorted(set(train_labels))
        }
        mask = low_support_mask(
            coordinates,
            labels,
            centroids,
            train_coordinates.mean(axis=0),
            train_coordinates.std(axis=0),
        )
        slices["low_support_id"] = {
            "definition": "furthest 20% within each benchmark from its training centroid",
            "n_prompts": int(mask.sum()),
            "record_ids": [record_ids[i] for i in np.flatnonzero(mask)],
            "learned_model_nll": float(learned_nll[mask].mean()),
            "routers": {
                name: routing_scores_to_dict(
                    score_routing_weights(
                        weights[mask],
                        artifact.expert_nll[mask],
                        artifact.token_counts[mask],
                        np.asarray(labels)[mask].tolist(),
                        seed=seed,
                    )
                )
                for name, weights in router_matrices.items()
            },
            "references": {
                name: {
                    "mean_nll": float(artifact.reference_nll[mask, index].mean())
                }
                for index, name in enumerate(reference_names)
            },
        }
    primary = reference_names[0]
    native = router_report["learned_gate"]
    return {
        "schema_version": 2,
        "evaluation": {
            "n_prompts": len(prompts),
            "round": round_idx,
            "partition": partition,
            "nll_artifact_digest": artifact.digest,
            "mixture_semantics": "weight_space_prompt_gated_lora",
        },
        "experts": {
            "names": list(artifact.expert_names),
            "mean_nll": {
                name: float(artifact.expert_nll[:, i].mean())
                for i, name in enumerate(artifact.expert_names)
            },
        },
        "references": references,
        "routers": router_report,
        "oracle": {"mean_nll": float(oracle.mean())},
        "slices": slices,
        "learned_model": {
            "name": model_name,
            "mean_nll": float(learned_nll.mean()),
        },
        "flat": {
            "pooled_nll": references[primary]["mean_nll"],
            "learned_routing_expected_nll": native["expected_nll"],
            "learned_routing_nll": native["expected_nll"],
            "learned_routing_argmax_nll": native["argmax_nll"],
            "oracle_routing_nll": float(oracle.mean()),
            "ensemble_mixture_nll": native["sequence_mixture_nll"],
            "learned_model_nll": float(learned_nll.mean()),
            "n_prompts": len(prompts),
            "round": round_idx,
            "merge_names": list(artifact.expert_names),
            "merge_name_map": {
                name: name for name in artifact.expert_names
            },
        },
        "per_benchmark": native["per_benchmark"],
        "routing_confusion": {},
        "merge_support_argmax": {},
        "merge_support_expected": {},
        "clone_support_argmax": {},
        "clone_support_expected": {},
    }


def report_to_dict(report: FlatRoutingReport) -> dict[str, Any]:
    """Serialize a :class:`FlatRoutingReport` to JSON-safe dict.

    :param report: Routing report.
    :type report: FlatRoutingReport
    :returns: JSON-serializable mapping.
    :rtype: dict[str, Any]
    """
    f = report.flat
    strategic_direction = (
        "toward oracle"
        if f.strategic_expected_nll < f.learned_expected_nll
        else "away from oracle"
    )
    counts = report.routing_confusion.get("counts", {})
    names = report.merge_names
    total = f.n_prompts
    agree = 0
    for i, merge in enumerate(names):
        agree += int(counts.get(merge, {}).get(merge, 0))
    agreement = agree / max(total, 1)
    return {
        "schema_version": 2,
        "evaluation": {
            "n_prompts": f.n_prompts,
            "round": f.round_idx,
            "partition": report.partition,
            "nll_artifact_digest": report.nll_artifact_digest,
            "mixture_semantics": "sequence_level_completion_likelihood",
        },
        "experts": {
            "names": report.merge_names,
            "merge_name_map": report.merge_name_map,
            "mean_nll": report.expert_mean_nll,
        },
        "references": report.references,
        "routers": report.routers,
        "oracle": report.oracle,
        "slices": report.slices,
        "flat": {
            "pooled_nll": f.pooled_nll,
            "learned_routing_argmax_nll": f.learned_argmax_nll,
            "learned_routing_expected_nll": f.learned_expected_nll,
            "learned_routing_sampled_nll": f.learned_sampled_nll,
            "strategic_routing_expected_nll": f.strategic_expected_nll,
            "strategic_routing_argmax_nll": f.strategic_argmax_nll,
            "learned_routing_nll": f.learned_expected_nll,
            "oracle_routing_nll": f.oracle_nll,
            "ensemble_mixture_nll": f.ensemble_mixture_nll,
            "ensemble_uniform_nll": f.ensemble_uniform_nll,
            "mean_tokens": f.mean_tokens,
            "routing_agreement_argmax": agreement,
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
        "",
        f"G(1−G) vs naive-G expected: "
        f"{f.strategic_expected_nll - f.learned_expected_nll:+.4f} "
        f"({strategic_direction})",
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
