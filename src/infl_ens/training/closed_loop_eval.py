"""Periodic validation during closed-loop training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from infl_ens.data.benchmarks import BenchmarkSplit
from infl_ens.evaluation.evaluate import (
    AdapterEvalConfig,
    evaluate_run_adapters,
    write_eval_report,
)


def run_closed_loop_val_eval(
    run_dir: str | Path,
    val_splits: Sequence[BenchmarkSplit],
    *,
    round_idx: int,
    sft_cfg: dict[str, Any],
    seed: int = 0,
    agents: Optional[Sequence[str]] = None,
    max_eval_records: Optional[int] = None,
    output_subdir: str = "eval_val",
) -> Path:
    """Evaluate merge adapters on the validation partition mid-training.

    :param run_dir: Closed-loop run root containing ``agents/``.
    :type run_dir: str | pathlib.Path
    :param val_splits: Validation benchmark splits.
    :type val_splits: Sequence[BenchmarkSplit]
    :param round_idx: Current training round (used to select checkpoints).
    :type round_idx: int
    :param sft_cfg: Nested ``closed_loop.sft`` dict (for ``base_model``).
    :type sft_cfg: dict
    :param seed: RNG seed forwarded to eval subsampling.
    :type seed: int
    :param agents: Optional merge-agent filter.
    :type agents: Sequence[str] | None
    :param max_eval_records: Optional cap per benchmark; ``None`` uses full
        validation split.
    :type max_eval_records: int | None
    :param output_subdir: Directory name under ``run_dir``.
    :type output_subdir: str
    :returns: Path to ``eval_results.json``.
    :rtype: pathlib.Path
    """
    run_path = Path(run_dir)
    out_dir = run_path / output_subdir / f"round-{round_idx:02d}"
    eval_cfg = AdapterEvalConfig(
        base_model=str(sft_cfg.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")),
        max_seq_length=int(sft_cfg.get("max_seq_length", 1024)),
        forward_batch_size=int(sft_cfg.get("forward_batch_size", 8)),
        max_eval_records=max_eval_records,
        seed=seed,
    )
    results = evaluate_run_adapters(
        run_path,
        val_splits,
        eval_cfg,
        agents=agents,
        rounds=[round_idx],
    )
    report_path = write_eval_report(
        results,
        out_dir,
        meta={
            "partition": "val",
            "round": round_idx,
            "max_eval_records": max_eval_records,
        },
    )
    print(f"val eval round {round_idx}: wrote {report_path}")
    return report_path


def append_val_eval_summary(
    history_path: Path,
    round_idx: int,
    report_path: Path,
) -> None:
    """Attach a val-eval report path to the matching history round entry.

    :param history_path: Path to ``history.json``.
    :type history_path: pathlib.Path
    :param round_idx: Round index to annotate.
    :type round_idx: int
    :param report_path: Written validation eval report.
    :type report_path: pathlib.Path
    """
    if not history_path.is_file():
        return
    history = json.loads(history_path.read_text(encoding="utf-8"))
    for entry in history:
        if int(entry.get("round", -1)) == round_idx:
            entry["val_eval_report"] = str(report_path)
            break
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
