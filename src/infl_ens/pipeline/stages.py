"""Pipeline stages, each a function of a :class:`PipelineContext`.

Stages call the package APIs directly (no subprocesses except the pytest
smoke gate) and are individually re-runnable: a stage whose outputs already
exist is skipped unless ``force`` is set, so a failure late in the pipeline
costs no GPU time on the next launch.

Stage order and what each one reads/writes (paths relative to the working
directory, i.e. the repository root):

``download``
    Fetch every benchmark named by the first arm that is missing on disk.
``manifest``
    Build ``data_split.manifest`` of the first specialist arm from its
    config (skipped when the file exists).
``train``
    Run every arm's task (``closed_loop`` or ``baseline_replay``) in order.
``perround``
    Score each specialist arm's per-round adapters on
    ``eval.perround_partition`` at ``eval.perround_rounds`` and write the
    pair NLL tables under ``<run>/tables/``.
``routing``
    Route-then-score each specialist arm on ``eval.routing_partition``
    against the generalist replay, writing
    ``<run>/routing_ensemble_diagnostics.json``.
``figures``
    Render the experiment's figures into ``figures_dir``.
``prune``
    Delete intermediate ``round-NN`` adapters, keeping the final round.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from infl_ens.config import apply_overrides, load_config
from infl_ens.experiment import ALL_STAGES, ArmSpec, ExperimentConfig

log = logging.getLogger("infl_ens.pipeline")


@dataclass
class PipelineContext:
    """Everything a stage needs.

    :param exp: The experiment.
    :type exp: ExperimentConfig
    :param force: Re-run stages whose outputs already exist.
    :type force: bool
    :param only_arms: Restrict the per-arm stages to these arm names
        (empty = all).
    :type only_arms: tuple[str, ...]
    :param repo_root: Root used for the split manifest paths.
    :type repo_root: pathlib.Path
    :param status: Per-stage timing/outcome, written to ``stage_status.json``.
    :type status: dict[str, dict[str, Any]]
    """

    exp: ExperimentConfig
    force: bool = False
    only_arms: tuple[str, ...] = ()
    repo_root: Path = field(default_factory=Path.cwd)
    status: dict[str, dict[str, Any]] = field(default_factory=dict)

    def arms(self, *, specialists_only: bool = False) -> list[ArmSpec]:
        """Arms selected for this run, in experiment order.

        :param specialists_only: Drop the generalist.
        :type specialists_only: bool
        :returns: Selected arms.
        :rtype: list[ArmSpec]
        """
        arms = list(self.exp.specialists if specialists_only else self.exp.arms)
        if self.only_arms:
            arms = [a for a in arms if a.name in self.only_arms]
        return arms

    @property
    def status_path(self) -> Path:
        """Location of ``stage_status.json``."""
        return Path(self.exp.results_dir) / "stage_status.json"

    def write_status(self) -> None:
        """Persist :attr:`status`."""
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(self.status, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def resolved_run_config(arm: ArmSpec) -> dict[str, Any]:
    """The flattened config a finished run wrote (``resolved_config.yaml``).

    Falls back to resolving the arm's YAML when the run has not started.

    :param arm: The arm.
    :type arm: ArmSpec
    :returns: Flat config.
    :rtype: dict
    """
    path = arm.run_dir / "resolved_config.yaml"
    if path.is_file():
        return load_config(path, validate=False)
    return arm.load()


def final_round(arm: ArmSpec) -> int:
    """Last trained round of an arm, read from its history.

    :param arm: The arm.
    :type arm: ArmSpec
    :returns: Round index.
    :rtype: int
    """
    from infl_ens.evaluation.evaluate import final_round_from_history

    return final_round_from_history(arm.run_dir)


def expected_rounds(arm: ArmSpec, cfg: dict[str, Any]) -> Optional[int]:
    """How many rounds a finished run of ``arm`` should hold, if knowable.

    :param arm: The arm.
    :type arm: ArmSpec
    :param cfg: The arm's config.
    :type cfg: dict
    :returns: Round count, or ``None`` when it depends on an unbuilt manifest.
    :rtype: int | None
    """
    split_meta = arm.run_dir / "data_split.json"
    if split_meta.is_file():
        try:
            meta = json.loads(split_meta.read_text(encoding="utf-8")).get("meta") or {}
            if meta.get("n_rounds"):
                return int(meta["n_rounds"])
        except (OSError, ValueError):
            pass
    if cfg.get("task") == "baseline_replay":
        return None
    if not cfg.get("data_split"):
        return int((cfg.get("closed_loop") or {}).get("n_rounds", 5))
    return None


def run_is_complete(arm: ArmSpec, cfg: dict[str, Any]) -> bool:
    """Whether an arm's training outputs are already present.

    :param arm: The arm.
    :type arm: ArmSpec
    :param cfg: The arm's config.
    :type cfg: dict
    :returns: ``True`` when ``train`` can skip this arm.
    :rtype: bool
    """
    history = arm.run_dir / "history.json"
    if not history.is_file():
        return False
    if cfg.get("task") == "baseline_replay":
        return (arm.run_dir / "replay_summary.json").is_file()
    try:
        records = json.loads(history.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not records:
        return False
    want = expected_rounds(arm, cfg)
    return want is None or len(records) >= want


def smoke_config(arm: ArmSpec, exp: ExperimentConfig) -> dict[str, Any]:
    """The arm's config with the smoke overrides and redirected outputs.

    :param arm: The arm.
    :type arm: ArmSpec
    :param exp: The experiment (``smoke`` settings).
    :type exp: ExperimentConfig
    :returns: Flat config writing under ``smoke.output_root/<arm>/seed0``.
    :rtype: dict
    """
    cfg = arm.load(exp.smoke.overrides)
    out = Path(exp.smoke.output_root) / arm.name / "seed0"
    overrides: dict[str, Any] = {"output_dir": str(out)}
    if cfg.get("sft") is not None:
        overrides["sft.output_dir"] = str(out / "agents")
    if (cfg.get("closed_loop") or {}).get("sft") is not None:
        overrides["closed_loop.sft.output_dir"] = str(out / "agents")
    apply_overrides(cfg, overrides)
    return cfg


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_download(ctx: PipelineContext) -> None:
    """Fetch every benchmark of the first arm that is missing on disk."""
    from infl_ens.data.download import download_for_entry, entry_is_present

    cfg = ctx.exp.arms[0].load()
    for entry in cfg.get("benchmarks", []):
        if entry_is_present(entry) and not ctx.force:
            log.info("download: %s present at %s", entry["kind"], entry["path"])
            continue
        log.info("download: fetching %s -> %s", entry["kind"], entry["path"])
        download_for_entry(entry)


def stage_manifest(ctx: PipelineContext) -> None:
    """Build the shared split manifest from the first specialist arm."""
    arms = ctx.arms(specialists_only=True) or list(ctx.exp.arms)
    arm = arms[0]
    cfg = arm.load()
    ds = cfg.get("data_split") or {}
    manifest_path = ds.get("manifest")
    if not manifest_path:
        log.info("manifest: %s has no data_split.manifest; nothing to build", arm.name)
        return
    target = ctx.repo_root / str(manifest_path)
    if target.is_file() and not ctx.force:
        log.info("manifest: already present: %s", target)
        return
    from infl_ens.data.splits import build_manifest_from_config, save_split_manifest
    from infl_ens.training.setup import load_splits

    splits = load_splits(cfg)
    manifest = build_manifest_from_config(cfg, splits, config_label=str(arm.config_path))
    path = save_split_manifest(manifest, target)
    log.info("manifest: wrote %s (%s)", path, json.dumps(manifest.meta.get("per_benchmark", {})))
    log.info(
        "manifest: train=%d val=%d test=%d batch_size=%s n_rounds=%s",
        manifest.n_train, manifest.n_val, manifest.n_test,
        manifest.meta.get("batch_size"), manifest.meta.get("n_rounds"),
    )


def stage_train(ctx: PipelineContext) -> None:
    """Run every selected arm's task in experiment order."""
    from infl_ens.training.tasks import TASKS

    for arm in ctx.arms():
        cfg = arm.load()
        if not ctx.force and run_is_complete(arm, cfg):
            log.info("train: %s already complete at %s", arm.name, arm.run_dir)
            continue
        task = str(cfg.get("task"))
        log.info("train: %s (%s) -> %s", arm.name, task, arm.run_dir)
        rc = int(TASKS[task](cfg))
        if rc != 0:
            raise RuntimeError(f"train: arm {arm.name!r} exited with code {rc}")


def stage_perround(ctx: PipelineContext) -> None:
    """Score per-round adapters on the validation partition and tabulate."""
    from infl_ens.evaluation.evaluate import run_unified_eval
    from infl_ens.figures.per_round_tables import build_per_round_tables, eval_rows_cover

    settings = ctx.exp.eval
    partition = settings.perround_partition
    for arm in ctx.arms(specialists_only=True):
        last = final_round(arm)
        rounds = settings.resolve_rounds(last)
        eval_dir = arm.run_dir / f"eval_{partition}"
        if eval_rows_cover(eval_dir, rounds) and not ctx.force:
            log.info("perround: %s already covers rounds %s", arm.name, rounds)
        else:
            cfg = resolved_run_config(arm)
            apply_overrides(cfg, {
                "eval.partitions": [partition],
                "eval.rounds": rounds,
                "eval.max_eval_records": settings.max_eval_records,
                "eval.baseline_run_dir": None,
            })
            log.info("perround: scoring %s rounds %s on %s", arm.name, rounds, partition)
            for report in run_unified_eval(cfg, final_round=last):
                log.info("perround: wrote %s", report)
        written = build_per_round_tables(
            eval_dir,
            arm.run_dir / "tables" / "pair_nll_by_round",
            label=f"{arm.title} ({partition})",
            rounds=rounds,
        )
        log.info("perround: tables %s", ", ".join(str(p) for p in written.values()))


def stage_routing(ctx: PipelineContext) -> None:
    """Route-then-score each specialist arm against the generalist replay."""
    from infl_ens.config import resolve_sft_block
    from infl_ens.evaluation.routing_eval import (
        format_headline_markdown,
        report_to_dict,
        run_flat_routing_eval,
    )

    gen = ctx.exp.generalist
    if gen is None:
        raise ValueError("routing stage needs a generalist arm (role: generalist)")
    settings = ctx.exp.eval
    for arm in ctx.arms(specialists_only=True):
        out_json = arm.run_dir / "routing_ensemble_diagnostics.json"
        if out_json.is_file() and not ctx.force:
            log.info("routing: %s already has %s", arm.name, out_json)
            continue
        resolved = arm.run_dir / "resolved_config.yaml"
        if not resolved.is_file():
            raise FileNotFoundError(f"routing: {arm.name} has no {resolved}; train it first")
        cfg = resolved_run_config(arm)
        sft = resolve_sft_block(cfg)
        eval_block = cfg.get("eval") or {}
        last = final_round(arm)
        log.info("routing: %s round %d on %s", arm.name, last, settings.routing_partition)
        report = run_flat_routing_eval(
            router_config=resolved,
            history_path=arm.run_dir / "history.json",
            merge_run_dir=arm.run_dir,
            baseline_run_dir=gen.run_dir,
            repo_root=ctx.repo_root,
            partition=settings.routing_partition,
            max_eval_records=settings.max_eval_records,
            seed=int(cfg.get("seed", 0)),
            round_idx=last,
            base_model=str(sft.get("base_model")),
            max_seq_length=int(sft.get("max_seq_length", 1024)),
            forward_batch_size=int(eval_block.get("forward_batch_size", 8)),
        )
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
        log.info("routing: wrote %s\n%s", out_json, format_headline_markdown(report))


def stage_figures(ctx: PipelineContext) -> None:
    """Render the experiment's figures."""
    from infl_ens.figures.render import render_all

    written = render_all(ctx.exp)
    for name, paths in written.items():
        log.info("figures: %s -> %d file(s)", name, len(paths))


def stage_prune(ctx: PipelineContext) -> None:
    """Delete intermediate ``round-NN`` adapters of every selected arm."""
    from infl_ens.utils.checkpoints import prune_intermediate_adapters

    for arm in ctx.arms():
        agents = arm.run_dir / "agents"
        if not agents.is_dir():
            log.info("prune: %s has no agents/ directory", arm.name)
            continue
        stats = prune_intermediate_adapters(agents)
        log.info("prune: %s %s", arm.name, stats)


#: Stage name -> function, in execution order.
STAGES: dict[str, Callable[[PipelineContext], None]] = {
    "download": stage_download,
    "manifest": stage_manifest,
    "train": stage_train,
    "perround": stage_perround,
    "routing": stage_routing,
    "figures": stage_figures,
    "prune": stage_prune,
}
assert tuple(STAGES) == ALL_STAGES


def run_pipeline(ctx: PipelineContext, stages: Sequence[str]) -> None:
    """Run ``stages`` in canonical order, recording status after each.

    :param ctx: Pipeline context.
    :type ctx: PipelineContext
    :param stages: Stage names (any order; executed in :data:`STAGES` order).
    :type stages: Sequence[str]
    :raises KeyError: For an unknown stage name.
    :raises Exception: Re-raises the first stage failure after recording it.
    """
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise KeyError(f"unknown stage(s) {unknown}; known: {list(STAGES)}")
    ordered = [s for s in STAGES if s in set(stages)]
    log.info("pipeline %s: stages %s", ctx.exp.name, ordered)
    for name in ordered:
        started = time.time()
        ctx.status[name] = {"started": started, "ok": None}
        ctx.write_status()
        log.info("=== stage %s ===", name)
        try:
            STAGES[name](ctx)
        except Exception as exc:
            ctx.status[name].update({"finished": time.time(), "ok": False, "error": repr(exc)})
            ctx.write_status()
            log.error("stage %s failed: %r", name, exc)
            raise
        ctx.status[name].update({
            "finished": time.time(), "ok": True, "seconds": round(time.time() - started, 1),
        })
        ctx.write_status()
    log.info("pipeline %s: done", ctx.exp.name)


def run_smoke(ctx: PipelineContext) -> None:
    """Cheap end-to-end gate: pytest subset, then tiny runs of the smoke arms.

    :param ctx: Pipeline context.
    :type ctx: PipelineContext
    :raises RuntimeError: If the tests or a smoke run fail.
    """
    from infl_ens.training.tasks import TASKS

    smoke = ctx.exp.smoke
    if smoke.tests:
        cmd = [sys.executable, "-m", "pytest", "-q", *smoke.tests]
        log.info("smoke: %s", " ".join(cmd))
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"smoke: pytest exited with {proc.returncode}")
    for name in smoke.arms:
        arm = ctx.exp.arm(name)
        cfg = smoke_config(arm, ctx.exp)
        log.info("smoke: %s -> %s", arm.name, cfg["output_dir"])
        rc = int(TASKS[str(cfg.get("task"))](cfg))
        if rc != 0:
            raise RuntimeError(f"smoke: arm {arm.name!r} exited with code {rc}")
    log.info("smoke: OK")


__all__ = [
    "STAGES",
    "PipelineContext",
    "expected_rounds",
    "final_round",
    "resolved_run_config",
    "run_is_complete",
    "run_pipeline",
    "run_smoke",
    "smoke_config",
    "stage_download",
    "stage_figures",
    "stage_manifest",
    "stage_perround",
    "stage_prune",
    "stage_routing",
    "stage_train",
]
