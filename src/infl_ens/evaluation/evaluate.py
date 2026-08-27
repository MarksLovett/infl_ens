"""Evaluate saved LoRA adapters on BeaverTails and HaluEval.

Two configuration shapes are accepted:

- a standalone evaluation YAML (``task: adapter_eval`` / ``run_eval``),
  parsed by :meth:`EvalJobConfig.from_mapping`;
- a **unified** closed-loop training YAML carrying a top-level ``eval``
  block, parsed by :meth:`EvalJobConfig.from_unified` and driven end to end
  by :func:`run_unified_eval`. The run directory, base model, benchmarks,
  split manifest and seed are read from the training blocks so nothing is
  duplicated between training and evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from infl_ens.data.benchmarks import BenchmarkSplit
from infl_ens.evaluation.adapters import (
    discover_adapters,
    load_adapter_model,
    load_base_causal_lm,
    resolve_adapter_dir,
)
from infl_ens.config import resolve_sft_block
from infl_ens.data.benchmarks.loading import load_benchmark_splits, subsample_split
from infl_ens.evaluation.metrics import mean_token_nll, split_to_texts


@dataclass
class AdapterEvalConfig:
    """Hyperparameters for benchmark evaluation of one adapter.

    :param base_model: HuggingFace id of the shared base model.
    :type base_model: str
    :param max_seq_length: Token cap per example.
    :type max_seq_length: int
    :param forward_batch_size: Inference micro-batch size.
    :type forward_batch_size: int
    :param max_eval_records: Optional cap on rows per benchmark after
        loading (random subsample). ``None`` uses the full split.
    :type max_eval_records: int | None
    :param seed: RNG seed for subsampling.
    :type seed: int
    """

    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_seq_length: int = 1024
    forward_batch_size: int = 8
    max_eval_records: Optional[int] = None
    seed: int = 0


@dataclass(frozen=True)
class BenchmarkEvalResult:
    """Metrics from evaluating one adapter on one benchmark split.

    :param benchmark: Split name (e.g. ``beavertails``).
    :type benchmark: str
    :param axis_name: Trait axis (e.g. ``harm``, ``hallucination``).
    :type axis_name: str
    :param mean_nll: Mean per-token negative log-likelihood.
    :type mean_nll: float
    :param n_examples: Number of formatted examples scored.
    :type n_examples: int
    :param n_tokens: Total non-padding tokens in the loss.
    :type n_tokens: int
    :param adapter_dir: Path to the LoRA checkpoint used.
    :type adapter_dir: str
    :param agent: Optional agent label when evaluating a run checkpoint.
    :type agent: str | None
    :param round: Optional round index for per-round checkpoints.
    :type round: int | None
    """

    benchmark: str
    axis_name: str
    mean_nll: float
    n_examples: int
    n_tokens: int
    adapter_dir: str
    agent: Optional[str] = None
    round: Optional[int] = None


def _seed_all(seed: int) -> None:
    """Best-effort RNG seeding before subsampling or inference.

    :param seed: Seed value.
    :type seed: int
    """
    try:
        from infl_ens.utils.seeding import seed_all
        seed_all(seed)
    except ImportError:  # pragma: no cover
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)


def evaluate_adapter_on_split(
    adapter_dir: str | Path,
    split: BenchmarkSplit,
    cfg: AdapterEvalConfig,
    *,
    agent: Optional[str] = None,
    round_idx: Optional[int] = None,
    base_model_obj=None,
    tokenizer=None,
    device=None,
    formatting_func: Optional[Callable[[str, Optional[str]], str]] = None,
) -> BenchmarkEvalResult:
    """Score one adapter on a single :class:`BenchmarkSplit`.

    The metric is mean per-token NLL on chat-formatted
    ``(prompt, response)`` pairs, matching the SFT objective.

    :param adapter_dir: Directory with LoRA weights.
    :type adapter_dir: str | pathlib.Path
    :param split: Benchmark corpus.
    :type split: BenchmarkSplit
    :param cfg: Evaluation hyperparameters.
    :type cfg: AdapterEvalConfig
    :param agent: Optional agent name for logging.
    :type agent: str | None
    :param round_idx: Optional training round for logging.
    :type round_idx: int | None
    :param base_model_obj: Optional pre-loaded base model (reuse across
        benchmarks or adapters).
    :type base_model_obj: transformers.PreTrainedModel | None
    :param tokenizer: Tokenizer paired with ``base_model_obj``.
    :type tokenizer: transformers.PreTrainedTokenizer | None
    :param device: Torch device paired with ``base_model_obj``.
    :type device: torch.device | None
    :param formatting_func: Optional chat formatter overriding the
        Qwen2.5 default in :mod:`infl_ens.evaluation.metrics`.
    :type formatting_func: Callable[[str, str | None], str] | None
    :returns: Per-benchmark metrics record.
    :rtype: BenchmarkEvalResult
    """
    adapter_path = resolve_adapter_dir(adapter_dir)
    eval_split = split
    if cfg.max_eval_records is not None:
        eval_split = subsample_split(
            split, cfg.max_eval_records, seed=cfg.seed,
        )

    owns_model = base_model_obj is None
    if owns_model:
        base_model_obj, tokenizer, device = load_base_causal_lm(cfg.base_model)

    model = load_adapter_model(base_model_obj, adapter_path)
    texts = split_to_texts(eval_split, formatting_func=formatting_func)
    mean_nll, n_tokens, n_examples = mean_token_nll(
        model,
        tokenizer,
        texts,
        max_length=cfg.max_seq_length,
        batch_size=cfg.forward_batch_size,
        device=device,
    )

    if owns_model:
        import torch
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return BenchmarkEvalResult(
        benchmark=eval_split.name,
        axis_name=eval_split.axis_name,
        mean_nll=mean_nll,
        n_examples=n_examples,
        n_tokens=n_tokens,
        adapter_dir=str(adapter_path),
        agent=agent,
        round=round_idx,
    )


def evaluate_adapter_on_splits(
    adapter_dir: str | Path,
    splits: Sequence[BenchmarkSplit],
    cfg: AdapterEvalConfig,
    *,
    agent: Optional[str] = None,
    round_idx: Optional[int] = None,
    formatting_func: Optional[Callable[[str, Optional[str]], str]] = None,
) -> list[BenchmarkEvalResult]:
    """Evaluate one adapter on every split in ``splits``.

    Loads the base model and LoRA adapter once, then scores each split.

    :param adapter_dir: LoRA checkpoint directory.
    :type adapter_dir: str | pathlib.Path
    :param splits: Benchmark corpora.
    :type splits: Sequence[BenchmarkSplit]
    :param cfg: Evaluation hyperparameters.
    :type cfg: AdapterEvalConfig
    :param agent: Optional agent label.
    :type agent: str | None
    :param round_idx: Optional round index.
    :type round_idx: int | None
    :param formatting_func: Optional chat formatter.
    :type formatting_func: Callable[[str, str | None], str] | None
    :returns: One result per split.
    :rtype: list[BenchmarkEvalResult]
    """
    _seed_all(cfg.seed)
    adapter_path = resolve_adapter_dir(adapter_dir)
    base_model_obj, tokenizer, device = load_base_causal_lm(cfg.base_model)
    model = load_adapter_model(base_model_obj, adapter_path)
    results: list[BenchmarkEvalResult] = []
    try:
        for split in splits:
            eval_split = split
            if cfg.max_eval_records is not None:
                eval_split = subsample_split(
                    split, cfg.max_eval_records, seed=cfg.seed,
                )
            texts = split_to_texts(eval_split, formatting_func=formatting_func)
            mean_nll, n_tokens, n_examples = mean_token_nll(
                model,
                tokenizer,
                texts,
                max_length=cfg.max_seq_length,
                batch_size=cfg.forward_batch_size,
                device=device,
            )
            results.append(
                BenchmarkEvalResult(
                    benchmark=eval_split.name,
                    axis_name=eval_split.axis_name,
                    mean_nll=mean_nll,
                    n_examples=n_examples,
                    n_tokens=n_tokens,
                    adapter_dir=str(adapter_path),
                    agent=agent,
                    round=round_idx,
                )
            )
    finally:
        import torch
        del model, base_model_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def evaluate_run_adapters(
    run_dir: str | Path,
    splits: Sequence[BenchmarkSplit],
    cfg: AdapterEvalConfig,
    *,
    agents: Optional[Sequence[str]] = None,
    rounds: Optional[Sequence[int]] = None,
) -> list[BenchmarkEvalResult]:
    """Evaluate every discovered adapter under ``run_dir/agents/``.

    :param run_dir: Closed-loop or SFT run root.
    :type run_dir: str | pathlib.Path
    :param splits: Benchmark corpora.
    :type splits: Sequence[BenchmarkSplit]
    :param cfg: Evaluation hyperparameters.
    :type cfg: AdapterEvalConfig
    :param agents: Optional subset of agent names.
    :type agents: Sequence[str] | None
    :param rounds: Optional subset of round indices.
    :type rounds: Sequence[int] | None
    :returns: Flat list of per-(adapter, benchmark) results.
    :rtype: list[BenchmarkEvalResult]
    """
    refs = discover_adapters(run_dir, agents=agents, rounds=rounds)
    if not refs:
        raise FileNotFoundError(
            f"no adapters found under {Path(run_dir).resolve() / 'agents'}"
        )

    all_results: list[BenchmarkEvalResult] = []
    for ref in refs:
        all_results.extend(
            evaluate_adapter_on_splits(
                ref.path,
                splits,
                cfg,
                agent=ref.agent,
                round_idx=ref.round,
            )
        )
    return all_results


def write_eval_report(
    results: Sequence[BenchmarkEvalResult],
    output_dir: str | Path,
    *,
    meta: Optional[dict[str, Any]] = None,
) -> Path:
    """Write ``eval_results.json`` under ``output_dir``.

    :param results: Evaluation records.
    :type results: Sequence[BenchmarkEvalResult]
    :param output_dir: Directory to create if needed.
    :type output_dir: str | pathlib.Path
    :param meta: Optional metadata (config snapshot, timestamps).
    :type meta: dict | None
    :returns: Path to the written JSON file.
    :rtype: pathlib.Path
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
        "results": [asdict(r) for r in results],
    }
    path = out / "eval_results.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


@dataclass
class EvalJobConfig:
    """Full evaluation job parsed from YAML or the CLI.

    :param task: ``adapter_eval`` or ``run_eval``.
    :type task: str
    :param seed: Global RNG seed.
    :type seed: int
    :param output_dir: Where to write ``eval_results.json``.
    :type output_dir: str
    :param base_model: HuggingFace base model id.
    :type base_model: str
    :param adapter_dir: LoRA path for ``adapter_eval``.
    :type adapter_dir: str | None
    :param run_dir: Run root for ``run_eval``.
    :type run_dir: str | None
    :param benchmarks: Raw ``benchmarks`` list from YAML.
    :type benchmarks: list[dict[str, Any]]
    :param eval_cfg: Nested ``eval`` block (seq length, batch, cap).
    :type eval_cfg: dict[str, Any]
    :param agents: Optional agent filter for ``run_eval``.
    :type agents: list[str] | None
    :param rounds: Optional round filter for ``run_eval``.
    :type rounds: list[int] | None
    """

    task: str
    seed: int = 0
    output_dir: str = "results/eval"
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_dir: Optional[str] = None
    run_dir: Optional[str] = None
    benchmarks: list[dict[str, Any]] = field(default_factory=list)
    eval_cfg: dict[str, Any] = field(default_factory=dict)
    agents: Optional[list[str]] = None
    rounds: Optional[list[int]] = None
    data_split_manifest: Optional[str] = None
    data_split_partition: Optional[str] = None

    def to_adapter_eval_config(self) -> AdapterEvalConfig:
        """Build :class:`AdapterEvalConfig` from the nested ``eval`` block.

        :returns: Adapter evaluation hyperparameters.
        :rtype: AdapterEvalConfig
        """
        e = self.eval_cfg
        return AdapterEvalConfig(
            base_model=self.base_model,
            max_seq_length=int(e.get("max_seq_length", 1024)),
            forward_batch_size=int(e.get("forward_batch_size", 8)),
            max_eval_records=e.get("max_eval_records"),
            seed=self.seed,
        )

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> "EvalJobConfig":
        """Parse a YAML/CLI configuration mapping.

        :param cfg: Top-level config dict.
        :type cfg: dict
        :returns: Parsed job config.
        :rtype: EvalJobConfig
        """
        return cls(
            task=str(cfg.get("task", "adapter_eval")),
            seed=int(cfg.get("seed", 0)),
            output_dir=str(cfg.get("output_dir", "results/eval")),
            base_model=str(cfg.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")),
            adapter_dir=cfg.get("adapter_dir"),
            run_dir=cfg.get("run_dir"),
            benchmarks=list(cfg.get("benchmarks", [])),
            eval_cfg=dict(cfg.get("eval", {})),
            agents=cfg.get("agents"),
            rounds=cfg.get("rounds"),
            data_split_manifest=cfg.get("data_split", {}).get("manifest")
            if isinstance(cfg.get("data_split"), dict)
            else cfg.get("data_split_manifest"),
            data_split_partition=cfg.get("data_split", {}).get("partition")
            if isinstance(cfg.get("data_split"), dict)
            else cfg.get("data_split_partition"),
        )

    @classmethod
    def from_unified(
        cls,
        cfg: dict[str, Any],
        *,
        partition: str,
        rounds: Optional[Sequence[int]] = None,
    ) -> "EvalJobConfig":
        """Derive a ``run_eval`` job from a unified closed-loop training config.

        The training YAML is the single source of truth: ``run_dir`` is its
        ``output_dir``, the base model and default sequence length come from
        the merged ``sft`` block, the benchmarks from ``benchmarks`` and the
        held-out partitions from ``data_split.manifest``. Only the
        evaluation-specific knobs live in the ``eval`` block
        (``agents``, ``max_eval_records``, ``forward_batch_size``,
        ``max_seq_length``; ``base_model`` may be overridden there).

        :param cfg: Full training config mapping (must contain
            ``closed_loop`` and ``benchmarks``).
        :type cfg: dict
        :param partition: Manifest partition to score (``train``, ``val``,
            ``test`` or ``train_val``); the report lands in
            ``<output_dir>/eval_<partition>/eval_results.json``.
        :type partition: str
        :param rounds: Checkpoint rounds to score; ``None`` scores every
            discovered round.
        :type rounds: Sequence[int] | None
        :returns: Parsed job config.
        :rtype: EvalJobConfig
        :raises ValueError: If the mapping is not a closed-loop training
            config or lacks a split manifest.
        """
        if not isinstance(cfg.get("closed_loop"), dict):
            raise ValueError(
                "from_unified expects a closed-loop training config with a "
                "'closed_loop' block"
            )
        eval_block = dict(cfg.get("eval") or {})
        sft = resolve_sft_block(cfg)
        data_split = cfg.get("data_split")
        manifest = None
        if isinstance(data_split, dict):
            # ``manifest`` is the loaded split; ``write_manifest`` is where
            # the trainer persisted a freshly built one.
            manifest = data_split.get("manifest") or data_split.get(
                "write_manifest",
            )
        if not manifest:
            raise ValueError(
                "unified eval needs data_split.manifest (or write_manifest) "
                "to define the held-out partitions"
            )
        run_dir = str(cfg.get("output_dir", "results/closed_loop"))
        eval_cfg = {
            "max_seq_length": int(
                eval_block.get("max_seq_length")
                or sft.get("max_seq_length", 1024)
            ),
            "forward_batch_size": int(
                eval_block.get("forward_batch_size")
                or sft.get("forward_batch_size", 8)
            ),
            "max_eval_records": eval_block.get("max_eval_records"),
        }
        return cls(
            task="run_eval",
            seed=int(cfg.get("seed", 0)),
            output_dir=str(Path(run_dir) / f"eval_{partition}"),
            base_model=str(
                eval_block.get("base_model")
                or sft.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")
            ),
            run_dir=run_dir,
            benchmarks=list(cfg.get("benchmarks", [])),
            eval_cfg=eval_cfg,
            agents=eval_block.get("agents"),
            rounds=[int(r) for r in rounds] if rounds is not None else None,
            data_split_manifest=str(manifest),
            data_split_partition=str(partition),
        )


def is_unified_config(cfg: dict[str, Any]) -> bool:
    """Whether ``cfg`` is a closed-loop training config with an ``eval`` block.

    :param cfg: Loaded YAML mapping.
    :type cfg: dict
    :returns: ``True`` for the unified training + evaluation shape.
    :rtype: bool
    """
    return isinstance(cfg.get("closed_loop"), dict) and isinstance(
        cfg.get("eval"), dict,
    )


def final_round_from_history(run_dir: str | Path) -> int:
    """Read the last completed round index from ``<run_dir>/history.json``.

    :param run_dir: Closed-loop run root.
    :type run_dir: str | pathlib.Path
    :returns: Round index of the final history entry.
    :rtype: int
    :raises FileNotFoundError: If the history file is missing.
    :raises ValueError: If the history is empty.
    """
    path = Path(run_dir) / "history.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    history = json.loads(path.read_text(encoding="utf-8"))
    if not history:
        raise ValueError(f"{path} holds no rounds")
    return int(history[-1]["round"])


def run_unified_eval(
    cfg: dict[str, Any],
    *,
    final_round: Optional[int] = None,
) -> list[Path]:
    """Run every evaluation described by a unified training config's ``eval`` block.

    ``eval.partitions`` (default ``[train, test]``) selects the manifest
    partitions; ``eval.rounds`` (default ``final``) either the literal
    ``final`` or an explicit list of round indices. When
    ``eval.baseline_run_dir`` is set, that run (typically the pooled
    replay) is scored on the same partitions and rounds into
    ``<baseline_run_dir>/eval_<partition>/``, optionally filtered by
    ``eval.baseline_agents``.

    The closed-loop trainer calls this right after training with
    ``final_round`` known; the standalone CLI resolves it from
    ``history.json``.

    :param cfg: Unified training config.
    :type cfg: dict
    :param final_round: Last trained round, if the caller knows it.
    :type final_round: int | None
    :returns: Paths of the written ``eval_results.json`` reports.
    :rtype: list[pathlib.Path]
    """
    eval_block = dict(cfg.get("eval") or {})
    partitions = list(eval_block.get("partitions") or ["train", "test"])
    rounds_spec = eval_block.get("rounds", "final")
    run_dir = str(cfg.get("output_dir", "results/closed_loop"))
    if rounds_spec in (None, "final"):
        if final_round is None:
            final_round = final_round_from_history(run_dir)
        rounds: list[int] = [int(final_round)]
    else:
        rounds = [int(r) for r in rounds_spec]

    baseline_run_dir = eval_block.get("baseline_run_dir")
    reports: list[Path] = []
    for partition in partitions:
        job = EvalJobConfig.from_unified(cfg, partition=partition, rounds=rounds)
        run_eval_job(job)
        reports.append(Path(job.output_dir) / "eval_results.json")
        if baseline_run_dir:
            baseline_job = replace(
                job,
                run_dir=str(baseline_run_dir),
                output_dir=str(Path(baseline_run_dir) / f"eval_{partition}"),
                agents=eval_block.get("baseline_agents"),
            )
            run_eval_job(baseline_job)
            reports.append(Path(baseline_job.output_dir) / "eval_results.json")
    return reports


def run_eval_job(job: EvalJobConfig) -> list[BenchmarkEvalResult]:
    """Execute an evaluation job described by :class:`EvalJobConfig`.

    :param job: Parsed job configuration.
    :type job: EvalJobConfig
    :returns: All benchmark metrics produced by the job.
    :rtype: list[BenchmarkEvalResult]
    :raises ValueError: If required paths are missing for the task.
    """
    if not job.benchmarks:
        raise ValueError("benchmarks list must be non-empty")

    if job.data_split_manifest and job.data_split_partition:
        from infl_ens.data.benchmarks.loading import load_benchmark_splits_with_partition

        splits = load_benchmark_splits_with_partition(
            job.benchmarks,
            manifest_path=job.data_split_manifest,
            partition=job.data_split_partition,
        )
    else:
        splits = load_benchmark_splits(job.benchmarks)
    eval_cfg = job.to_adapter_eval_config()

    if job.task == "adapter_eval":
        if not job.adapter_dir:
            raise ValueError("adapter_eval requires adapter_dir")
        results = evaluate_adapter_on_splits(
            job.adapter_dir, splits, eval_cfg,
        )
    elif job.task == "run_eval":
        if not job.run_dir:
            raise ValueError("run_eval requires run_dir")
        results = evaluate_run_adapters(
            job.run_dir,
            splits,
            eval_cfg,
            agents=job.agents,
            rounds=job.rounds,
        )
    else:
        raise ValueError(
            f"unknown task {job.task!r}; expected 'adapter_eval' or 'run_eval'"
        )

    write_eval_report(
        results,
        job.output_dir,
        meta={
            "task": job.task,
            "base_model": job.base_model,
            "adapter_dir": job.adapter_dir,
            "run_dir": job.run_dir,
            "benchmarks": job.benchmarks,
            "eval": job.eval_cfg,
            "data_split_manifest": job.data_split_manifest,
            "data_split_partition": job.data_split_partition,
        },
    )
    return results
