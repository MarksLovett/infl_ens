"""Task registry for ``python -m infl_ens.training``.

- ``closed_loop``: :func:`infl_ens.training.closed_loop.run_closed_loop`.
- ``baseline_replay``: :func:`run_baseline_replay`, one pooled cumulative
  LoRA trained on the union of the per-round routed batches logged in an
  existing ``history.json`` (see :mod:`infl_ens.training.baseline_replay`).
- ``modula_res``: :func:`run_modula_res`, one cumulative LoRA expert per
  domain (benchmark label or game pair) replayed on a closed-loop run's
  round schedule, optionally as a residual on a frozen universal adapter
  (see :mod:`infl_ens.training.modula_res`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from infl_ens.config import resolve_sft_block
from infl_ens.training.closed_loop import run_closed_loop
from infl_ens.training.setup import load_splits, make_trait_space, write_resolved_config


def run_baseline_replay(cfg: dict[str, Any]) -> int:
    """Replay pooled baseline SFT from a closed-loop ``history.json``.

    :param cfg: Configuration with ``history_path``, ``output_dir``,
        ``benchmarks``, ``trait_space``, ``sft``, and ``baseline_replay``.
    :type cfg: dict
    :returns: Exit code.
    :rtype: int
    """
    from infl_ens.training.baseline_replay import (
        make_pooled_baseline_agent,
        replay_pooled_baseline_sft,
    )
    from infl_ens.training.sft_training import SFTTrainingConfig

    hist_path = Path(cfg["history_path"])
    if not hist_path.is_file():
        raise FileNotFoundError(hist_path)

    splits = load_splits(cfg)
    space = make_trait_space(cfg, splits)
    br = cfg.get("baseline_replay", {})
    agent_name = str(br.get("agent_name", "pooled-baseline"))
    agent = make_pooled_baseline_agent(space, name=agent_name)

    sft_cfg_dict = resolve_sft_block(cfg)
    sft_cfg = SFTTrainingConfig(**sft_cfg_dict)
    rounds = br.get("rounds")

    summaries = replay_pooled_baseline_sft(
        hist_path,
        agent,
        sft_cfg,
        project=space.project,
        output_dir=Path(sft_cfg.output_dir),
        rounds=rounds,
        save_per_round=bool(br.get("save_per_round", True)),
    )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "replay_summary.json"
    history_path = out_dir / "history.json"
    baseline_history = [
        {
            "round": s["round"],
            "positions": {agent_name: s["position"]},
            "batch_centroid": s["batch_centroid"],
            "cumulative_centroid": s["cumulative_centroid"],
            "n_train": s["n_train"],
            "cumulative_n_train": s["cumulative_n_train"],
            "output_dir": s["output_dir"],
        }
        for s in summaries
    ]
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "history_path": str(hist_path.resolve()),
                "agent": agent_name,
                "n_rounds": len(summaries),
                "rounds": summaries,
                "centroid_history_path": str(history_path),
            },
            fh,
            indent=2,
        )
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(baseline_history, fh, indent=2)
    print(
        f"baseline replay done: {len(summaries)} rounds, wrote "
        f"{summary_path} and {history_path}"
    )
    return 0


def run_modula_res(cfg: dict[str, Any]) -> int:
    """Train per-domain (residual) LoRA experts on a closed-loop schedule.

    Reads the source run named by ``history_path``, builds per-round
    per-domain batches according to ``modula_res.domain_source``, trains one
    cumulative expert per domain (with the universal adapter merged
    underneath when ``modula_res.universal_run_dir`` is set), then writes
    ``history.json`` (positions copied from the source so the game router
    is unchanged), ``modula_res_summary.json`` and a ``resolved_config.yaml``
    carrying the source ``agents`` / merge groups plus the resolved
    ``merge_aliases`` and ``universal_adapter_dir`` for the eval stages.

    :param cfg: Configuration with ``history_path``, ``output_dir``,
        ``benchmarks``, ``data_split``, ``trait_space``, ``sft`` and
        ``modula_res``.
    :type cfg: dict
    :returns: Exit code.
    :rtype: int
    :raises ValueError: On an unknown ``domain_source``.
    """
    from infl_ens.data.splits import flatten_partition_prompts, load_split_manifest
    from infl_ens.training.baseline_replay import load_closed_loop_history
    from infl_ens.training.modula_res import (
        DOMAIN_SOURCES,
        history_domain_batches,
        label_domain_batches,
        pair_to_benchmark_aliases,
        resolve_universal_adapter_dir,
        source_router_blocks,
        train_modula_res,
    )
    from infl_ens.training.sft_training import SFTTrainingConfig

    block = dict(cfg.get("modula_res") or {})
    domain_source = str(block.get("domain_source", "labels"))
    if domain_source not in DOMAIN_SOURCES:
        raise ValueError(
            f"modula_res.domain_source must be one of {sorted(DOMAIN_SOURCES)}, "
            f"got {domain_source!r}"
        )
    repo_root = Path(cfg["repo_root"]) if cfg.get("repo_root") else Path.cwd()
    hist_path = Path(cfg["history_path"])
    if not hist_path.is_absolute():
        hist_path = repo_root / hist_path
    if not hist_path.is_file():
        raise FileNotFoundError(hist_path)
    history = load_closed_loop_history(hist_path)
    source_run_dir = hist_path.parent
    blocks = source_router_blocks(source_run_dir, history)
    universal_dir = resolve_universal_adapter_dir(cfg, repo_root=repo_root)

    splits = load_splits(cfg)
    space = make_trait_space(cfg, splits)
    benchmark_names = [str(s.name) for s in splits]
    final_positions = {
        str(k): [float(x) for x in v] for k, v in history[-1]["positions"].items()
    }

    def _pair_position(pair: str) -> list[float]:
        members = next(
            (g["names"] for g in blocks.merge_groups if g["train_as"] == pair), [],
        )
        for m in members:
            if m in final_positions:
                return final_positions[m]
        return [float(x) for x in space.mean]

    merge_aliases: dict[str, str] = {}
    if domain_source == "labels":
        ds = cfg.get("data_split") or {}
        manifest_rel = ds.get("manifest") or ds.get("write_manifest")
        if not manifest_rel:
            raise ValueError("modula_res with domain_source=labels needs data_split.manifest")
        manifest_path = Path(str(manifest_rel))
        if not manifest_path.is_absolute():
            manifest_path = repo_root / manifest_path
        manifest = load_split_manifest(manifest_path)
        train_prompts, train_responses, train_labels = flatten_partition_prompts(
            splits, manifest, "train",
        )
        domain_names = list(benchmark_names)
        merge_aliases = pair_to_benchmark_aliases(blocks, benchmark_names)
        print(
            "modula_res label partition: pair -> benchmark aliases "
            + ", ".join(f"{p}->{b}" for p, b in merge_aliases.items()),
            flush=True,
        )
        bench_to_pair = {b: p for p, b in merge_aliases.items()}
        initial_positions = {
            b: _pair_position(bench_to_pair[b]) for b in domain_names
        }

        def batches_for_round(record: dict[str, Any]) -> dict[str, Any]:
            return label_domain_batches(
                record,
                train_prompts=train_prompts,
                train_responses=train_responses,
                train_labels=train_labels,
                domain_names=domain_names,
            )
    else:
        domain_names = list(blocks.pair_names)
        initial_positions = {p: _pair_position(p) for p in domain_names}

        def batches_for_round(record: dict[str, Any]) -> dict[str, Any]:
            return history_domain_batches(record, domain_names=domain_names)

    sft_cfg = SFTTrainingConfig(**resolve_sft_block(cfg))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolved config first, so a crash mid-training still leaves the eval
    # stages something to point at.
    resolved = dict(cfg)
    resolved["agents"] = blocks.agents
    for key, value in blocks.router_keys.items():
        resolved.setdefault(key, value)
    resolved["closed_loop"] = {
        **(cfg.get("closed_loop") or {}),
        "sft_merge_groups": blocks.merge_groups,
    }
    resolved["modula_res"] = {
        **block,
        "domain_source": domain_source,
        "merge_aliases": merge_aliases,
        "universal_adapter_dir": str(universal_dir) if universal_dir is not None else None,
    }
    write_resolved_config(resolved, out_dir / "resolved_config.yaml")

    summaries = train_modula_res(
        history,
        domain_names=domain_names,
        batches_for_round=batches_for_round,
        sft_cfg=sft_cfg,
        project=space.project,
        output_dir=Path(sft_cfg.output_dir),
        universal_adapter_dir=universal_dir,
        initial_positions=initial_positions,
        rounds=block.get("rounds"),
        save_per_round=bool(block.get("save_per_round", True)),
    )

    by_round = {int(r["round"]): r for r in history}
    new_history: list[dict[str, Any]] = []
    for s in summaries:
        src = by_round[int(s["round"])]
        new_history.append({
            "round": s["round"],
            "positions": src["positions"],
            "domain_source": domain_source,
            "experts": domain_names,
            "n_train": {k: v["n_train"] for k, v in s["experts"].items()},
            "n_train_total": s["n_train_total"],
            "output_dirs": {k: v["output_dir"] for k, v in s["experts"].items()},
            "agent_prompts": _expert_prompts(batches_for_round(src)),
            **(
                {"pair_members": {g["train_as"]: list(g["names"]) for g in blocks.merge_groups}}
                if s is summaries[0] else {}
            ),
        })
    history_path = out_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(new_history, fh, indent=2)
    with (out_dir / "modula_res_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "history_path": str(hist_path.resolve()),
                "source_run_dir": str(source_run_dir.resolve()),
                "domain_source": domain_source,
                "experts": domain_names,
                "universal_adapter_dir": str(universal_dir) if universal_dir else None,
                "merge_aliases": merge_aliases,
                "n_rounds": len(summaries),
                "rounds": summaries,
            },
            fh,
            indent=2,
        )
    print(
        f"modula_res ({domain_source}, universal={'yes' if universal_dir else 'no'}) done: "
        f"{len(summaries)} rounds, {len(domain_names)} experts -> {out_dir}"
    )

    eval_block = cfg.get("eval") or {}
    if eval_block.get("after_training") and summaries:
        from infl_ens.evaluation.evaluate import run_unified_eval

        for report in run_unified_eval(resolved, final_round=int(summaries[-1]["round"])):
            print(f"eval: wrote {report}")
    return 0


def _expert_prompts(batches: dict[str, Any]) -> dict[str, list[str]]:
    """Per-expert prompt lists for the history record."""
    return {name: list(b.prompts) for name, b in batches.items()}


#: ``task`` value -> runner. Every runner takes the resolved config and
#: returns a process exit code.
TASKS: dict[str, Callable[[dict[str, Any]], int]] = {
    "closed_loop": run_closed_loop,
    "baseline_replay": run_baseline_replay,
    "modula_res": run_modula_res,
}

__all__ = ["TASKS", "run_baseline_replay", "run_closed_loop", "run_modula_res"]
