"""Evaluate a base (non-adapter) causal LM on benchmark splits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from infl_ens.evaluation.adapters import load_base_causal_lm
from infl_ens.evaluation.benchmarks import load_benchmark_splits, subsample_split
from infl_ens.evaluation.metrics import mean_token_nll, split_to_texts


@dataclass(frozen=True)
class BaseEvalResult:
    """One benchmark score for a base model.

    :param benchmark: Benchmark id.
    :type benchmark: str
    :param axis_name: Axis name (e.g. ``harm``).
    :type axis_name: str
    :param mean_nll: Mean per-token NLL.
    :type mean_nll: float
    :param n_examples: Number of examples scored.
    :type n_examples: int
    :param n_tokens: Total non-pad tokens scored.
    :type n_tokens: int
    """

    benchmark: str
    axis_name: str
    mean_nll: float
    n_examples: int
    n_tokens: int


def evaluate_base_model(
    benchmarks: Sequence[dict[str, Any]],
    *,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length: int = 1024,
    forward_batch_size: int = 8,
    max_eval_records: Optional[int] = None,
    seed: int = 0,
) -> tuple[list[BaseEvalResult], dict[str, Any]]:
    """Score a base model on benchmark corpora.

    :param benchmarks: Benchmark config entries for
        :func:`infl_ens.evaluation.benchmarks.load_benchmark_splits`.
    :type benchmarks: Sequence[dict]
    :param base_model: HuggingFace model id.
    :type base_model: str
    :param max_seq_length: Tokenizer truncation length.
    :type max_seq_length: int
    :param forward_batch_size: Inference batch size.
    :type forward_batch_size: int
    :param max_eval_records: Optional per-benchmark subsample cap.
    :type max_eval_records: int | None
    :param seed: Subsampling seed.
    :type seed: int
    :returns: ``(results, meta)`` where ``meta`` holds run configuration.
    :rtype: tuple[list[BaseEvalResult], dict]
    """
    splits = load_benchmark_splits(list(benchmarks))
    model, tokenizer, device = load_base_causal_lm(base_model)
    results: list[BaseEvalResult] = []
    for split in splits:
        eval_split = split
        if max_eval_records is not None:
            eval_split = subsample_split(split, max_eval_records, seed=seed)
        texts = split_to_texts(eval_split)
        mean_nll, n_tokens, n_examples = mean_token_nll(
            model,
            tokenizer,
            texts,
            max_length=max_seq_length,
            batch_size=forward_batch_size,
            device=device,
        )
        results.append(
            BaseEvalResult(
                benchmark=eval_split.name,
                axis_name=eval_split.axis_name,
                mean_nll=mean_nll,
                n_examples=n_examples,
                n_tokens=n_tokens,
            )
        )
    meta = {
        "base_model": base_model,
        "max_seq_length": max_seq_length,
        "forward_batch_size": forward_batch_size,
        "max_eval_records": max_eval_records,
        "seed": seed,
        "benchmarks": list(benchmarks),
    }
    return results, meta


def write_base_eval_report(
    output_dir: Path,
    results: Sequence[BaseEvalResult],
    meta: dict[str, Any],
) -> Path:
    """Write ``base_eval.json`` under ``output_dir``.

    :param output_dir: Destination directory.
    :type output_dir: pathlib.Path
    :param results: Benchmark scores.
    :type results: Sequence[BaseEvalResult]
    :param meta: Run metadata.
    :type meta: dict
    :returns: Path to the written JSON file.
    :rtype: pathlib.Path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "base_eval.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
