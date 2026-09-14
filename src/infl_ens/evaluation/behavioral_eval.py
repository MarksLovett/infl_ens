"""End-to-end generation, grading and caching for behavioral safety suites."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
from dataclasses import asdict, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from infl_ens.data.behavioral import (
    BehavioralCase,
    BehavioralSuite,
    audit_contamination,
    load_behavioral_suites,
)
from infl_ens.evaluation.behavioral_artifact import (
    GenerationArtifact,
    canonical_digest,
    generation_artifact_is_current,
    write_generation_artifact,
    write_score_artifact,
)
from infl_ens.evaluation.behavioral_graders import (
    aggregate_behavioral_scores,
    grade_generation_records,
    resolved_grader_identity,
)
from infl_ens.evaluation.generation import (
    generate_probability_mixture,
    generate_standard_model,
    generate_trait_mixture_model,
)
from infl_ens.evaluation.model_runners import (
    BehavioralTarget,
    load_multi_adapter_model,
    resolve_behavioral_targets,
)
from infl_ens.experiment import BehavioralGenerationSettings, ExperimentConfig

log = logging.getLogger("infl_ens.behavioral")


def behavioral_suite_digest(suite: BehavioralSuite) -> str:
    """Hash ordered cases, private scoring payloads and release metadata.

    :param suite: Loaded behavioral suite.
    :type suite: BehavioralSuite
    :returns: SHA-256 digest.
    :rtype: str
    """
    payload = {
        "name": suite.name,
        "version": suite.version,
        "source": suite.source,
        "metadata": dict(suite.metadata),
        "cases": [
            {
                "case_id": case.case_id,
                "messages": [message.to_dict() for message in case.messages],
                "category": case.category,
                "metadata": dict(case.metadata),
                "scoring_payload": dict(case.scoring_payload),
            }
            for case in suite.cases
        ],
    }
    return canonical_digest(payload)


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    hasher = hashlib.sha256()
    hasher.update(str(values.shape).encode("ascii"))
    hasher.update(values.tobytes())
    return hasher.hexdigest()


def _checkpoint_hashes(target: BehavioralTarget) -> dict[str, str]:
    from infl_ens.evaluation.nll_artifact import checkpoint_sha256

    if not target.checkpoint_paths:
        return {}
    if len(target.checkpoint_paths) == 1:
        name = target.expert_names[0] if len(target.expert_names) == 1 else "mixture_lora"
        return {name: checkpoint_sha256(target.checkpoint_paths[0])}
    if len(target.checkpoint_paths) != len(target.expert_names):
        raise ValueError("checkpoint paths and expert names must align")
    return {
        name: checkpoint_sha256(path)
        for name, path in zip(target.expert_names, target.checkpoint_paths)
    }


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("torch", "transformers", "peft"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return versions


def _release_model_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - lightweight environment
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _training_prompts(cfg: Mapping[str, Any], repo_root: Path) -> list[str]:
    from infl_ens.evaluation.routing_eval import load_flat_partition_records

    prompts, _responses, _labels, _ids = load_flat_partition_records(
        cfg,
        repo_root=repo_root,
        partition="train_val",
        max_eval_records=None,
        seed=int(cfg.get("seed", 0)),
    )
    return prompts


def _load_space(cfg: Mapping[str, Any]) -> Any:
    from infl_ens.training.setup import load_splits, make_trait_space

    mutable = dict(cfg)
    return make_trait_space(mutable, load_splits(mutable))


def _native_routing_weights(
    target: BehavioralTarget,
    coordinates: np.ndarray,
    space: Any,
    *,
    hard_argmax: bool,
) -> np.ndarray:
    """Compute final-position merge-level routing weights for a target."""
    from infl_ens.evaluation.routing_eval import (
        aggregate_clone_g_to_merge,
        load_final_positions,
        parse_merge_groups,
        resolve_merge_adapters,
    )
    from infl_ens.inflgame.router.allocation import allocation_weights
    from infl_ens.training.setup import sigma_from_config

    if target.config is None or target.run_dir is None or target.round_idx is None:
        raise ValueError("routed target is missing config/run metadata")
    cfg = dict(target.config)
    agents = [str(entry["name"]) for entry in cfg.get("agents", [])]
    clone_to_merge, config_merge_names = parse_merge_groups(cfg.get("closed_loop") or {})
    if not clone_to_merge:
        clone_to_merge = {name: name for name in agents}
        config_merge_names = list(agents)
    resolved, name_map = resolve_merge_adapters(
        target.run_dir, target.round_idx, config_merge_names,
    )
    if tuple(resolved) != target.expert_names:
        raise ValueError("behavioral target expert order disagrees with routing merge order")
    positions = load_final_positions(target.run_dir / "history.json", agents)
    sigma = sigma_from_config(cfg, len(agents), space)
    covariance = float(sigma) ** 2 * np.eye(space.L)
    clone_weights = allocation_weights(positions, coordinates, covariance)
    weights = aggregate_clone_g_to_merge(
        clone_weights, agents, clone_to_merge, resolved, name_map,
    ).T
    row_sum = weights.sum(axis=1, keepdims=True)
    weights = weights / np.maximum(row_sum, 1e-30)
    if hard_argmax:
        hard = np.zeros_like(weights)
        hard[np.arange(len(weights)), np.argmax(weights, axis=1)] = 1.0
        return hard
    top_k = (cfg.get("closed_loop") or {}).get("soft_top_k")
    if top_k is not None and 0 < int(top_k) < weights.shape[1]:
        keep = np.argpartition(weights, -int(top_k), axis=1)[:, -int(top_k):]
        mask = np.zeros_like(weights, dtype=bool)
        mask[np.arange(len(weights))[:, None], keep] = True
        weights = np.where(mask, weights, 0.0)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)
    return weights


def _grader_spec(
    suite_entry: Mapping[str, Any],
    global_graders: Mapping[str, Any],
    suite_name: str,
) -> dict[str, Any]:
    value = suite_entry.get("grader", global_graders.get(suite_name))
    if value is None:
        return {}
    if isinstance(value, str):
        return {"kind": value}
    if not isinstance(value, Mapping):
        raise ValueError(f"grader for {suite_name!r} must be a string or mapping")
    return dict(value)


def _suite_generation_settings(
    experiment: ExperimentConfig,
    suite_name: str,
) -> BehavioralGenerationSettings:
    settings = experiment.behavioral_eval
    assert settings is not None
    entry = next(value for value in settings.suites if str(value["kind"]) == suite_name)
    override = dict(entry.get("generation") or {})
    return replace(settings.generation, **override)


def _target_provenance(
    target: BehavioralTarget,
    suite: BehavioralSuite,
    protocol: str,
    experiment: ExperimentConfig,
    weights: np.ndarray | None,
    coordinates: np.ndarray,
    checkpoint_hashes: Mapping[str, str],
    generation_settings: BehavioralGenerationSettings,
) -> dict[str, Any]:
    settings = experiment.behavioral_eval
    assert settings is not None
    from infl_ens.data.trait_space_cache import trait_space_fingerprint

    cfg = dict(target.config or {})
    return {
        "schema_version": settings.schema_version,
        "experiment": experiment.name,
        "target": target.name,
        "target_kind": target.kind,
        "arm": target.arm_name,
        "round": target.round_idx,
        "base_model": target.base_model,
        "checkpoints": dict(checkpoint_hashes),
        "runtime_versions": _runtime_versions(),
        "generation_implementation_version": 1,
        "protocol": protocol,
        "generation": asdict(generation_settings),
        "suite": suite.name,
        "suite_version": suite.version,
        "suite_digest": behavioral_suite_digest(suite),
        "ordered_case_ids": [case.case_id for case in suite.cases],
        "routing_weights_sha256": _array_sha256(weights) if weights is not None else None,
        "trait_coordinates_sha256": _array_sha256(coordinates),
        "trait_space_fingerprint": trait_space_fingerprint(cfg) if cfg else None,
    }


def _generate_target_batch(
    target: BehavioralTarget,
    suite_items: Sequence[tuple[BehavioralSuite, np.ndarray]],
    protocol: str,
    space: Any,
    experiment: ExperimentConfig,
    *,
    force: bool,
    checkpoint_hashes: Mapping[str, str],
) -> list[tuple[BehavioralSuite, GenerationArtifact, Path]]:
    """Generate every configured suite while loading a target only once."""
    from infl_ens.evaluation.adapters import load_adapter_model, load_base_causal_lm
    from infl_ens.training.mixture_lora import load_mixture_checkpoint

    requests: list[
        tuple[
            BehavioralSuite,
            np.ndarray,
            Path,
            np.ndarray | None,
            dict[str, Any],
            BehavioralGenerationSettings,
        ]
    ] = []
    cached: dict[str, tuple[GenerationArtifact, Path]] = {}
    for suite, coordinates in suite_items:
        output = (
            Path(experiment.results_dir)
            / "behavioral"
            / target.artifact_name
            / protocol
            / suite.name
        )
        weights = (
            _native_routing_weights(
                target, coordinates, space, hard_argmax=protocol == "hard_argmax",
            )
            if target.kind == "routed" else None
        )
        generation_settings = _suite_generation_settings(experiment, suite.name)
        provenance = _target_provenance(
            target,
            suite,
            protocol,
            experiment,
            weights,
            coordinates,
            checkpoint_hashes,
            generation_settings,
        )
        if not force and generation_artifact_is_current(output, provenance):
            log.info("behavioral: cache hit %s/%s/%s", target.name, protocol, suite.name)
            cached[suite.name] = (GenerationArtifact.load(output), output)
        else:
            requests.append((
                suite,
                coordinates,
                output,
                weights,
                provenance,
                generation_settings,
            ))

    generated: dict[str, tuple[GenerationArtifact, Path]] = {}
    if not requests:
        return [
            (suite, *cached[suite.name])
            for suite, _coordinates in suite_items
        ]
    for suite, _coordinates, _output, _weights, _provenance, _settings in requests:
        log.info("behavioral: generating %s/%s/%s", target.name, protocol, suite.name)
    if target.kind == "base":
        model, tokenizer, _device = load_base_causal_lm(target.base_model)
        try:
            for suite, coordinates, output, _weights, provenance, suite_settings in requests:
                records = generate_standard_model(
                    model, tokenizer, suite.cases,
                    target=target.name, protocol=protocol, settings=suite_settings,
                    trait_coordinates=coordinates,
                )
                generated[suite.name] = (
                    write_generation_artifact(output, records, provenance), output,
                )
        finally:
            del model, tokenizer
            _release_model_memory()
    elif target.kind == "adapter":
        base, tokenizer, _device = load_base_causal_lm(target.base_model)
        model = load_adapter_model(base, target.checkpoint_paths[0])
        try:
            for suite, coordinates, output, _weights, provenance, suite_settings in requests:
                one = np.ones((len(suite.cases), 1), dtype=float)
                records = generate_standard_model(
                    model, tokenizer, suite.cases,
                    target=target.name, protocol=protocol, settings=suite_settings,
                    trait_coordinates=coordinates, router_weights=one,
                    expert_names=target.expert_names,
                )
                generated[suite.name] = (
                    write_generation_artifact(output, records, provenance), output,
                )
        finally:
            del model, base, tokenizer
            _release_model_memory()
    elif target.kind == "routed":
        model, tokenizer, _device, mapping = load_multi_adapter_model(target)
        try:
            for suite, coordinates, output, weights, provenance, suite_settings in requests:
                assert weights is not None
                records = generate_probability_mixture(
                    model, tokenizer, suite.cases, weights, target.expert_names,
                    lambda name: model.set_adapter(mapping[name]),
                    coordinates=coordinates, target=target.name, protocol=protocol,
                    settings=suite_settings,
                )
                generated[suite.name] = (
                    write_generation_artifact(output, records, provenance), output,
                )
        finally:
            del model, tokenizer
            _release_model_memory()
    elif target.kind == "mixture_lora":
        model, tokenizer = load_mixture_checkpoint(target.checkpoint_paths[0])
        metadata = json.loads(
            (target.checkpoint_paths[0] / "mixture_config.json").read_text(encoding="utf-8"),
        )
        mean = np.asarray(metadata.get("trait_mean", np.zeros(space.L)), dtype=float)
        scale = np.asarray(metadata.get("trait_scale", np.ones(space.L)), dtype=float)
        try:
            for suite, coordinates, output, _weights, provenance, suite_settings in requests:
                standardized = (coordinates - mean) / np.where(scale > 1e-6, scale, 1.0)
                records = generate_trait_mixture_model(
                    model, tokenizer, suite.cases, standardized,
                    recorded_coordinates=coordinates,
                    target=target.name,
                    protocol=protocol,
                    settings=suite_settings,
                    hard_argmax=protocol == "hard_argmax",
                    expert_names=target.expert_names,
                    top_k=(
                        int(metadata["top_k"])
                        if metadata.get("top_k") is not None else None
                    ),
                )
                generated[suite.name] = (
                    write_generation_artifact(output, records, provenance), output,
                )
        finally:
            del model, tokenizer
            _release_model_memory()
    else:  # pragma: no cover - target construction validates kinds
        raise ValueError(f"unknown behavioral target kind {target.kind!r}")
    return [
        (suite, *(cached.get(suite.name) or generated[suite.name]))
        for suite, _coordinates in suite_items
    ]


def _score_artifact(
    artifact: GenerationArtifact,
    output: Path,
    suite: BehavioralSuite,
    grader_spec: Mapping[str, Any],
    *,
    force: bool,
) -> dict[str, Any]:
    grader_identity = resolved_grader_identity(suite.name, grader_spec)
    score_path = output / "scores" / "scores.json"
    expected_digest = canonical_digest({
        "generation_digest": artifact.digest,
        "grader": grader_identity,
    })
    if score_path.is_file() and not force:
        try:
            cached = json.loads(score_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if cached.get("score_digest") == expected_digest:
            return cached
    scores = grade_generation_records(suite, artifact.records, grader_spec)
    payload = write_score_artifact(
        score_path,
        [score.to_dict() for score in scores],
        generation_digest=artifact.digest,
        grader=grader_identity,
    )
    payload["aggregate"] = aggregate_behavioral_scores(scores)
    score_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _contamination_report(
    suites: Sequence[BehavioralSuite],
    training_prompts: Sequence[str],
    settings: Mapping[str, Any],
) -> tuple[list[BehavioralSuite], dict[str, Any]]:
    clean: list[BehavioralSuite] = []
    report: dict[str, Any] = {"policy": dict(settings), "suites": {}}
    for suite in suites:
        rows = audit_contamination(
            suite.cases,
            training_prompts,
            exact_policy=str(settings.get("exact_match", "exclude")),
            fuzzy_policy=str(settings.get("fuzzy_match", "flag")),
            ngram_size=int(settings.get("ngram_size", 5)),
            fuzzy_threshold=float(settings.get("fuzzy_threshold", 0.85)),
        )
        excluded = {row.case_id for row in rows if row.excluded}
        retained = tuple(case for case in suite.cases if case.case_id not in excluded)
        if not retained:
            raise ValueError(f"contamination policy removed every {suite.name} case")
        clean.append(BehavioralSuite(
            name=suite.name,
            version=suite.version,
            source=suite.source,
            cases=retained,
            metadata=dict(suite.metadata),
        ))
        report["suites"][suite.name] = {
            "n_input": len(suite.cases),
            "n_retained": len(retained),
            "n_exact": sum(row.exact_match for row in rows),
            "n_fuzzy": sum(row.fuzzy_match for row in rows),
            "rows": [asdict(row) for row in rows],
        }
    return clean, report


def run_behavioral_eval(
    experiment: ExperimentConfig,
    *,
    repo_root: str | Path,
    selected_arms: Sequence[str] | None = None,
    force: bool = False,
) -> Path:
    """Run configured behavioral suites over final experiment targets.

    :param experiment: Parsed experiment.
    :type experiment: ExperimentConfig
    :param repo_root: Repository root for split-manifest paths.
    :type repo_root: str | pathlib.Path
    :param selected_arms: Optional per-arm restriction from the pipeline CLI.
    :type selected_arms: Sequence[str] | None
    :param force: Regenerate and regrade even when cache provenance matches.
    :type force: bool
    :returns: Experiment-level behavioral summary path.
    :rtype: pathlib.Path
    """
    settings = experiment.behavioral_eval
    if settings is None:
        raise ValueError("experiment has no behavioral_eval block")
    suites = load_behavioral_suites(settings.suites)
    targets = resolve_behavioral_targets(experiment, selected_arms=selected_arms)
    first_cfg = next(
        (target.config for target in targets if target.config is not None),
        None,
    )
    if first_cfg is None:
        first_cfg = experiment.arms[0].load()
    training_prompts = _training_prompts(first_cfg, Path(repo_root))
    suites, contamination = _contamination_report(
        suites, training_prompts, settings.contamination,
    )
    output_root = Path(experiment.results_dir) / "behavioral"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "contamination_report.json").write_text(
        json.dumps(contamination, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    space = _load_space(first_cfg)
    from infl_ens.data.trait_space_cache import trait_space_fingerprint

    reference_fingerprint = trait_space_fingerprint(first_cfg)
    incompatible = [
        target.name
        for target in targets
        if target.config is not None
        and trait_space_fingerprint(dict(target.config)) != reference_fingerprint
    ]
    if incompatible:
        raise ValueError(
            "behavioral routing requires one frozen trait space; mismatched targets: "
            f"{incompatible}",
        )
    suite_entries = {str(entry["kind"]): entry for entry in settings.suites}
    suite_items = [
        (
            suite,
            np.asarray(
                space.project([case.routing_text for case in suite.cases]), dtype=float,
            ),
        )
        for suite in suites
    ]
    summary_rows: list[dict[str, Any]] = []
    for target in targets:
        checkpoint_hashes = _checkpoint_hashes(target)
        protocols = (
            ("native",)
            if target.kind in {"base", "adapter"}
            else settings.protocols
        )
        for protocol in protocols:
            artifacts = _generate_target_batch(
                target,
                suite_items,
                protocol,
                space,
                experiment,
                force=force,
                checkpoint_hashes=checkpoint_hashes,
            )
            for suite, artifact, output in artifacts:
                grader = _grader_spec(
                    suite_entries[suite.name], settings.graders, suite.name,
                )
                scored = _score_artifact(
                    artifact, output, suite, grader, force=force,
                )
                records = artifact.records
                summary_rows.append({
                    "target": target.name,
                    "target_kind": target.kind,
                    "protocol": protocol,
                    "suite": suite.name,
                    "generation_digest": artifact.digest,
                    "artifact_dir": str(output),
                    "generation": {
                        "input_tokens": sum(record.input_tokens for record in records),
                        "output_tokens": sum(record.output_tokens for record in records),
                        "model_forwards": sum(record.model_forwards for record in records),
                        "latency_seconds": sum(record.latency_seconds for record in records),
                    },
                    "aggregate": scored["aggregate"],
                })
    summary = {
        "schema_version": settings.schema_version,
        "experiment": experiment.name,
        "protocol": {
            "generation": asdict(settings.generation),
            "contamination": dict(settings.contamination),
            "audit": dict(settings.audit),
        },
        "rows": summary_rows,
    }
    summary_path = output_root / "behavioral_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


__all__ = ["behavioral_suite_digest", "run_behavioral_eval"]
