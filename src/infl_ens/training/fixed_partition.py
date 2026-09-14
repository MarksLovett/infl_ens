"""Cumulative LoRA training on fixed benchmark, k-means, or random shards."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from infl_ens.data.splits import flatten_partition_records
from infl_ens.inflgame.router.agents import RouterAgent
from infl_ens.training.baseline_replay import load_closed_loop_history
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


@dataclass(frozen=True)
class PartitionPlan:
    """A fixed expert assignment over the flattened training partition.

    :param mode: ``benchmark``, ``kmeans``, or ``random``.
    :type mode: str
    :param expert_names: Expert names in assignment-column order.
    :type expert_names: tuple[str, ...]
    :param assignment: Expert index per training record.
    :type assignment: numpy.ndarray
    :param centroids: Fixed train-only expert centroids, shape ``(K, L)``.
    :type centroids: numpy.ndarray
    """

    mode: str
    expert_names: tuple[str, ...]
    assignment: np.ndarray
    centroids: np.ndarray

    def validate(self, n_records: int) -> None:
        """Validate assignment coverage and nonempty experts.

        :param n_records: Expected record count.
        :type n_records: int
        :raises ValueError: If the plan is malformed.
        """
        assignment = np.asarray(self.assignment, dtype=int)
        if assignment.shape != (n_records,):
            raise ValueError(
                f"assignment must have shape ({n_records},), got {assignment.shape}"
            )
        if self.centroids.shape[0] != len(self.expert_names):
            raise ValueError("centroid rows do not match expert names")
        if assignment.size and (
            assignment.min() < 0 or assignment.max() >= len(self.expert_names)
        ):
            raise ValueError("assignment contains an invalid expert index")
        counts = np.bincount(assignment, minlength=len(self.expert_names))
        if np.any(counts == 0):
            raise ValueError(f"every expert needs training records, got {counts.tolist()}")


def _kmeans_once(
    coordinates: np.ndarray,
    n_clusters: int,
    *,
    rng: np.random.Generator,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one deterministic-seeded k-means++ solve."""
    x = np.asarray(coordinates, dtype=float)
    centers = np.empty((n_clusters, x.shape[1]), dtype=float)
    centers[0] = x[int(rng.integers(len(x)))]
    closest = ((x - centers[0]) ** 2).sum(axis=1)
    for cluster in range(1, n_clusters):
        total = float(closest.sum())
        index = int(rng.integers(len(x))) if total <= 0 else int(
            rng.choice(len(x), p=closest / total)
        )
        centers[cluster] = x[index]
        closest = np.minimum(closest, ((x - centers[cluster]) ** 2).sum(axis=1))
    assignment = np.zeros(len(x), dtype=int)
    for _ in range(max_iter):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assignment = np.argmin(distances, axis=1)
        updated = centers.copy()
        for cluster in range(n_clusters):
            rows = x[assignment == cluster]
            if len(rows):
                updated[cluster] = rows.mean(axis=0)
            else:
                nearest_distance = distances[np.arange(len(x)), assignment]
                updated[cluster] = x[int(np.argmax(nearest_distance))]
        shift = float(np.linalg.norm(updated - centers, axis=1).max())
        centers = updated
        if shift <= tol:
            break
    distances = ((x - centers[assignment]) ** 2).sum(axis=1)
    return assignment, centers, float(distances.sum())


def hard_kmeans_partition(
    coordinates: np.ndarray,
    n_clusters: int,
    *,
    seed: int = 0,
    n_init: int = 20,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit train-only hard k-means with k-means++ restarts.

    :param coordinates: Training coordinates, shape ``(M, L)``.
    :type coordinates: numpy.ndarray
    :param n_clusters: Number of experts.
    :type n_clusters: int
    :param seed: Base restart seed.
    :type seed: int
    :param n_init: Number of k-means++ restarts.
    :type n_init: int
    :param max_iter: Maximum Lloyd iterations per restart.
    :type max_iter: int
    :param tol: Maximum centroid-shift tolerance.
    :type tol: float
    :returns: ``(assignment, centroids)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    x = np.asarray(coordinates, dtype=float)
    if x.ndim != 2 or len(x) < n_clusters or n_clusters < 1:
        raise ValueError("coordinates must be 2-D with at least n_clusters rows")
    candidates = [
        _kmeans_once(
            x,
            n_clusters,
            rng=np.random.default_rng(seed + restart),
            max_iter=max_iter,
            tol=tol,
        )
        for restart in range(n_init)
    ]
    assignment, centers, _inertia = min(candidates, key=lambda item: item[2])
    order = np.lexsort(centers[:, ::-1].T)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(n_clusters)
    return inverse[assignment], centers[order]


def build_partition_plan(
    mode: str,
    coordinates: np.ndarray,
    bench_labels: Sequence[str],
    *,
    n_experts: int = 7,
    seed: int = 0,
    n_init: int = 20,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> PartitionPlan:
    """Build one fixed train-only expert partition.

    :param mode: ``benchmark``, ``kmeans``, or ``random``.
    :type mode: str
    :param coordinates: Training trait coordinates.
    :type coordinates: numpy.ndarray
    :param bench_labels: Benchmark label per record.
    :type bench_labels: Sequence[str]
    :param n_experts: Requested expert count.
    :type n_experts: int
    :param seed: Partition seed.
    :type seed: int
    :param n_init: K-means restarts.
    :type n_init: int
    :param max_iter: K-means iterations.
    :type max_iter: int
    :param tol: K-means tolerance.
    :type tol: float
    :returns: Validated partition plan.
    :rtype: PartitionPlan
    """
    x = np.asarray(coordinates, dtype=float)
    if x.ndim != 2 or len(bench_labels) != len(x):
        raise ValueError("coordinates and bench_labels must be aligned")
    if mode == "benchmark":
        ordered_labels = tuple(dict.fromkeys(str(label) for label in bench_labels))
        if len(ordered_labels) != n_experts:
            raise ValueError(
                f"benchmark mode found {len(ordered_labels)} labels, expected {n_experts}"
            )
        lookup = {label: i for i, label in enumerate(ordered_labels)}
        assignment = np.asarray([lookup[str(label)] for label in bench_labels], dtype=int)
        names = tuple(f"domain-{label}" for label in ordered_labels)
    elif mode == "kmeans":
        assignment, centers = hard_kmeans_partition(
            x,
            n_experts,
            seed=seed,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
        )
        names = tuple(f"cluster-{i}" for i in range(n_experts))
        plan = PartitionPlan(mode, names, assignment, centers)
        plan.validate(len(x))
        return plan
    elif mode == "random":
        order = np.random.default_rng(seed).permutation(len(x))
        assignment = np.empty(len(x), dtype=int)
        for expert, rows in enumerate(np.array_split(order, n_experts)):
            assignment[rows] = expert
        names = tuple(f"random-{i}" for i in range(n_experts))
    else:
        raise ValueError("partition mode must be benchmark, kmeans, or random")
    centroids = np.stack(
        [x[assignment == expert].mean(axis=0) for expert in range(n_experts)],
        axis=0,
    )
    plan = PartitionPlan(mode, names, assignment, centroids)
    plan.validate(len(x))
    return plan


def audit_reference_batches(
    history: Sequence[dict[str, Any]],
    batches: Sequence[np.ndarray],
    prompts: Sequence[str],
    responses: Sequence[str],
    record_ids: Sequence[str],
) -> None:
    """Assert that reconstructed batches exactly match reference history.

    :param history: Reference round records.
    :type history: Sequence[dict[str, Any]]
    :param batches: Reconstructed global row indices per round.
    :type batches: Sequence[numpy.ndarray]
    :param prompts: Flattened train prompts.
    :type prompts: Sequence[str]
    :param responses: Flattened train responses.
    :type responses: Sequence[str]
    :param record_ids: Flattened stable record IDs.
    :type record_ids: Sequence[str]
    :raises ValueError: If any round differs.
    """
    by_round = {int(record["round"]): record for record in history}
    for round_idx, rows in enumerate(batches):
        if round_idx not in by_round:
            raise ValueError(f"reference history has no round {round_idx}")
        record = by_round[round_idx]
        expected_ids = [record_ids[int(i)] for i in rows]
        logged_ids = record.get("batch_record_ids")
        if logged_ids is not None:
            if list(logged_ids) != expected_ids:
                raise ValueError(f"round {round_idx} record IDs differ from reference")
            continue
        expected_prompts = [prompts[int(i)] for i in rows]
        expected_responses = [responses[int(i)] for i in rows]
        if list(record.get("batch_prompts") or []) != expected_prompts:
            raise ValueError(f"round {round_idx} prompts differ from reference")
        if list(record.get("batch_responses") or []) != expected_responses:
            raise ValueError(f"round {round_idx} responses differ from reference")


def run_partition_replay(cfg: dict[str, Any]) -> int:
    """Train cumulative LoRA experts on one fixed partition of each round.

    :param cfg: Resolved ``partition_replay`` run configuration.
    :type cfg: dict[str, Any]
    :returns: Process exit code.
    :rtype: int
    """
    from infl_ens.config import resolve_sft_block
    from infl_ens.training.sft_training import SFTTrainingConfig, sft_train_agent

    source_history_path = Path(cfg["history_path"])
    source_history = load_closed_loop_history(source_history_path)
    splits = load_splits(cfg)
    space = make_trait_space(cfg, splits)
    repo_root = Path(cfg.get("repo_root", Path(__file__).resolve().parents[3]))
    if cfg.get("data_split"):
        (
            manifest,
            train_prompts,
            train_responses,
            _pool_p,
            _pool_r,
            batch_size,
            n_rounds,
        ) = resolve_closed_loop_data_split(cfg, splits, repo_root=repo_root)
        id_prompts, id_responses, bench_labels, record_ids = flatten_partition_records(
            splits, manifest, "train",
        )
        if id_prompts != train_prompts or id_responses != train_responses:
            raise ValueError("manifest record IDs do not align with replay training rows")
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        batches = shuffled_train_batch_indices(
            len(train_prompts), batch_size, n_rounds, rng,
        )
    else:
        manifest = None
        train_prompts = [prompt for split in splits for prompt in split.prompts]
        train_responses = [
            response
            for split in splits
            for response in (split.responses or [""] * split.n)
        ]
        bench_labels = [split.name for split in splits for _ in split.prompts]
        record_ids = [f"unsplit:{index}" for index in range(len(train_prompts))]
        id_to_row = {record_id: index for index, record_id in enumerate(record_ids)}
        batches = [
            np.asarray([id_to_row[str(value)] for value in record["batch_record_ids"]])
            for record in source_history
        ]
        n_rounds = len(batches)
        batch_size = max((len(rows) for rows in batches), default=0)
    audit_reference_batches(
        source_history, batches, train_prompts, train_responses, record_ids,
    )

    block = dict(cfg.get("partition_replay") or {})
    mode = str(block.get("mode", "benchmark"))
    coordinates = np.asarray(space.project(train_prompts), dtype=float)
    plan = build_partition_plan(
        mode,
        coordinates,
        bench_labels,
        n_experts=int(block.get("n_experts", 7)),
        seed=int(block.get("seed", cfg.get("seed", 0))),
        n_init=int(block.get("n_init", 20)),
        max_iter=int(block.get("max_iter", 300)),
        tol=float(block.get("tol", 1e-6)),
    )
    agents = [
        RouterAgent(name=name, position=plan.centroids[i].copy())
        for i, name in enumerate(plan.expert_names)
    ]
    cfg["agents"] = [{"name": name} for name in plan.expert_names]
    closed_loop = dict(cfg.get("closed_loop") or {})
    closed_loop["sft_merge_groups"] = [
        {"train_as": name, "names": [name]} for name in plan.expert_names
    ]
    cfg["closed_loop"] = closed_loop

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_dict = resolve_sft_block(cfg)
    sft_dict["output_dir"] = str(output_dir / "agents")
    sft_dict["cumulative_lora"] = True
    cfg["sft"] = dict(sft_dict)
    sft_cfg = SFTTrainingConfig(**sft_dict)
    write_resolved_config(cfg, output_dir / "resolved_config.yaml")
    if manifest is not None:
        (output_dir / "data_split.json").write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8",
        )

    history: list[dict[str, Any]] = []
    cumulative_counts = np.zeros(len(agents), dtype=int)
    for round_idx, batch_rows in enumerate(batches):
        batch_assignment = plan.assignment[batch_rows]
        agent_prompts: dict[str, list[str]] = {}
        agent_responses: dict[str, list[str]] = {}
        agent_record_ids: dict[str, list[str]] = {}
        agent_batch_indices: dict[str, list[int]] = {}
        agent_sft_logs: dict[str, list[dict[str, Any]]] = {}
        loaded_prior: dict[str, str | None] = {}
        resources: dict[str, dict[str, Any]] = {}
        for expert_index, agent in enumerate(agents):
            local = np.flatnonzero(batch_assignment == expert_index)
            global_rows = batch_rows[local]
            prompts = [train_prompts[int(i)] for i in global_rows]
            responses = [train_responses[int(i)] for i in global_rows]
            ids = [record_ids[int(i)] for i in global_rows]
            agent_prompts[agent.name] = prompts
            agent_responses[agent.name] = responses
            agent_record_ids[agent.name] = ids
            agent_batch_indices[agent.name] = [int(i) for i in local]
            cumulative_counts[expert_index] += len(prompts)
            if not prompts:
                agent_sft_logs[agent.name] = []
                prior_dir = next(
                    (
                        output_dir
                        / "agents"
                        / agent.name
                        / f"round-{prior_round:02d}"
                        for prior_round in reversed(range(round_idx))
                        if (
                            output_dir
                            / "agents"
                            / agent.name
                            / f"round-{prior_round:02d}"
                        ).is_dir()
                    ),
                    None,
                )
                loaded_prior[agent.name] = (
                    str(prior_dir) if prior_dir is not None else None
                )
                if prior_dir is not None:
                    current_dir = (
                        output_dir
                        / "agents"
                        / agent.name
                        / f"round-{round_idx:02d}"
                    )
                    shutil.copytree(prior_dir, current_dir, dirs_exist_ok=True)
                    previous_resource = next(
                        (
                            record.get("resource_accounting", {}).get(agent.name, {})
                            for record in reversed(history)
                            if record.get("resource_accounting", {}).get(agent.name)
                        ),
                        {},
                    )
                    resources[agent.name] = {
                        "n_train": 0,
                        "trainable_parameters": int(
                            previous_resource.get("trainable_parameters", 0)
                        ),
                        "token_exposures": 0,
                        "peak_memory_bytes": 0,
                        "wall_seconds": 0.0,
                        "active_lora_rank": int(
                            previous_resource.get("active_lora_rank", sft_cfg.lora_r)
                        ),
                        "model_forwards": 0,
                        "carried_forward": True,
                    }
                continue
            result = sft_train_agent(
                agent,
                prompts=prompts,
                responses=responses if any(responses) else None,
                cfg=sft_cfg,
                eval_prompts=prompts,
                project=space.project,
                out_dir_override=str(
                    output_dir / "agents" / agent.name / f"round-{round_idx:02d}"
                ),
                skip_position_update=True,
            )
            agent_sft_logs[agent.name] = list(result.get("log_history", []))
            loaded_prior[agent.name] = result.get("loaded_prior_lora")
            resources[agent.name] = {
                key: result[key]
                for key in (
                    "n_train",
                    "trainable_parameters",
                    "token_exposures",
                    "peak_memory_bytes",
                    "wall_seconds",
                    "active_lora_rank",
                    "model_forwards",
                )
                if key in result
            }
        history.append({
            "round": round_idx,
            "positions": {
                agent.name: plan.centroids[i].tolist()
                for i, agent in enumerate(agents)
            },
            "routing_mode": "fixed_partition",
            "partition_mode": mode,
            "batch_prompts": [train_prompts[int(i)] for i in batch_rows],
            "batch_responses": [train_responses[int(i)] for i in batch_rows],
            "batch_record_ids": [record_ids[int(i)] for i in batch_rows],
            "agent_prompts": agent_prompts,
            "agent_responses": agent_responses,
            "agent_record_ids": agent_record_ids,
            "agent_batch_indices": agent_batch_indices,
            "agent_sft_logs": agent_sft_logs,
            "agent_loaded_prior": loaded_prior,
            "cumulative_n_train": {
                agent.name: int(cumulative_counts[i])
                for i, agent in enumerate(agents)
            },
            "resource_accounting": resources,
            "sft_merge_mode": "fixed",
            "pair_members": {agent.name: [agent.name] for agent in agents},
        })
        write_history(output_dir / "history.json", history)

    summary = {
        "mode": mode,
        "source_history": str(source_history_path),
        "n_rounds": len(history),
        "expert_names": list(plan.expert_names),
        "counts": np.bincount(
            plan.assignment, minlength=len(plan.expert_names),
        ).tolist(),
        "centroids": plan.centroids.tolist(),
        "data_match": "exact",
    }
    (output_dir / "partition_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    return 0


__all__ = [
    "PartitionPlan",
    "audit_reference_batches",
    "build_partition_plan",
    "hard_kmeans_partition",
    "run_partition_replay",
]
