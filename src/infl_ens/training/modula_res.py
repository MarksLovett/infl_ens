"""MoDULA-Res-style residual experts replayed on a closed-loop schedule.

The ``modula_res`` task trains one LoRA expert per *domain* on the same
per-round batches a finished closed-loop run drew, optionally as a
**residual** on top of a frozen *universal* adapter (the pooled generalist
merged into the base weights). It fills three cells of the
``{game partition, label partition} x {from scratch, residual on U}`` grid;
the fourth (game partition, from scratch) is the closed loop itself.

Two switches define an arm:

``domain_source``
    ``labels``: one expert per benchmark, trained on that benchmark's rows
    of every round batch (``agent_batch_indices`` of the source history,
    falling back to ``batch_prompts``). This is the MoDULA-Res baseline
    when a universal adapter is given and "one LoRA per benchmark" when it
    is not.
    ``history``: one expert per source merge group (``pair-k``), trained on
    exactly the prompts and loss weights the game routed to that pair. With
    a universal adapter this is the "frozen-U + residual pairs" ablation.

``universal_run_dir``
    A ``baseline_replay`` run whose ``pooled-baseline`` adapter is merged
    into the base before every expert is trained (``null`` = no merge).

Positions never move: the source run's per-round positions are copied into
the new ``history.json`` so route-then-score uses the game router
unchanged, and the source ``agents`` / merge groups are copied into the
resolved config. For ``labels`` the source pairs are aliased onto the
benchmark experts through ``theory_init.pair_dominant_axis``.

The module keeps to the training rules: the reward stays in
:mod:`infl_ens.inflgame`, the LoRA trainer is
:func:`infl_ens.training.sft_training.sft_train_agent`, and nothing here
touches evaluation artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from infl_ens.inflgame.router.agents import RouterAgent

PathLike = str | Path

#: Accepted ``modula_res.domain_source`` values.
DOMAIN_SOURCES: frozenset[str] = frozenset({"labels", "history"})


@dataclass(frozen=True)
class DomainBatch:
    """One expert's training batch for one round.

    :param prompts: Training prompts.
    :type prompts: list[str]
    :param responses: Aligned target responses (``None`` for prompt-only).
    :type responses: list[str | None]
    :param sample_weights: Optional per-example loss weights.
    :type sample_weights: list[float] | None
    """

    prompts: list[str]
    responses: list[str | None]
    sample_weights: list[float] | None = None

    @property
    def n(self) -> int:
        """Number of examples."""
        return len(self.prompts)


@dataclass(frozen=True)
class SourceRouterBlocks:
    """Router description copied from the source closed-loop run.

    :param agents: Literal ``agents`` list (``[{name: ...}, ...]``).
    :type agents: list[dict]
    :param merge_groups: Resolved ``sft_merge_groups`` (``train_as``/``names``).
    :type merge_groups: list[dict]
    :param router_keys: Top-level routing keys (``policy``, ``sigma_mode``,
        ``sigma``, ``sigma_fraction``) present in the source config.
    :type router_keys: dict
    :param pair_dominant_axis: ``pair-k -> axis index`` from the theory
        initialisation, or an empty mapping when unavailable.
    :type pair_dominant_axis: dict[str, int]
    :param pair_positions: ``pair-k -> trait coordinates`` of the pair at
        initialisation, or an empty mapping when unavailable. Used to
        assign pairs to benchmarks one-to-one when the dominant axes
        collide.
    :type pair_positions: dict[str, list[float]]
    """

    agents: list[dict[str, Any]]
    merge_groups: list[dict[str, Any]]
    router_keys: dict[str, Any] = field(default_factory=dict)
    pair_dominant_axis: dict[str, int] = field(default_factory=dict)
    pair_positions: dict[str, list[float]] = field(default_factory=dict)

    @property
    def pair_names(self) -> list[str]:
        """Merge-group names in config order."""
        return [str(g["train_as"]) for g in self.merge_groups]


_ROUTER_TOP_LEVEL_KEYS: tuple[str, ...] = ("policy", "sigma_mode", "sigma", "sigma_fraction")


def resolve_universal_adapter_dir(
    cfg: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> Path | None:
    """Locate the frozen universal adapter named by ``cfg["modula_res"]``.

    :param cfg: Resolved run config.
    :type cfg: Mapping
    :param repo_root: Root used for relative ``universal_run_dir`` paths.
        Defaults to the current working directory.
    :type repo_root: pathlib.Path | None
    :returns: ``<universal_run_dir>/agents/<universal_agent>/round-NN``, or
        ``None`` when ``modula_res.universal_run_dir`` is unset / ``null``.
    :rtype: pathlib.Path | None
    :raises FileNotFoundError: If the run directory or adapter is missing.
    :raises ValueError: If ``universal_round`` is neither ``final`` nor an int.
    """
    block = cfg.get("modula_res") or {}
    run_dir_raw = block.get("universal_run_dir")
    if not run_dir_raw:
        return None
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    run_dir = Path(str(run_dir_raw))
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    if not run_dir.is_dir():
        raise FileNotFoundError(f"universal_run_dir not found: {run_dir}")
    agent = str(block.get("universal_agent", "pooled-baseline"))
    round_spec = block.get("universal_round", "final")
    if round_spec in (None, "final"):
        history = load_closed_loop_history_lenient(run_dir / "history.json")
        round_idx = int(history[-1]["round"])
    elif isinstance(round_spec, int) and not isinstance(round_spec, bool):
        round_idx = int(round_spec)
    else:
        raise ValueError(
            f"modula_res.universal_round must be 'final' or an int, got {round_spec!r}"
        )
    adapter_dir = run_dir / "agents" / agent / f"round-{round_idx:02d}"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"universal adapter not found: {adapter_dir}")
    return adapter_dir


def load_closed_loop_history_lenient(path: PathLike) -> list[dict[str, Any]]:
    """Load a ``history.json`` that need not carry ``agent_prompts``.

    The pooled generalist's history has positions and centroids only, so
    the strict :func:`infl_ens.training.baseline_replay.load_closed_loop_history`
    check does not apply to it.

    :param path: Path to ``history.json``.
    :type path: str | pathlib.Path
    :returns: Per-round records.
    :rtype: list[dict]
    :raises ValueError: If the file is empty.
    """
    p = Path(path)
    records = json.loads(p.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"{p} is empty")
    return records


def round_batch_indices(record: Mapping[str, Any]) -> list[int] | None:
    """Recover the round's row indices into the train partition.

    Soft-pair runs log ``agent_batch_indices`` (per merge group, indices
    into the flattened train partition). Their union, sorted, is the round
    batch. Returns ``None`` when the record carries no indices, in which
    case callers fall back to ``batch_prompts`` matched by text.

    :param record: One history record.
    :type record: Mapping
    :returns: Sorted unique row indices, or ``None``.
    :rtype: list[int] | None
    """
    per_agent = record.get("agent_batch_indices")
    if not isinstance(per_agent, Mapping) or not per_agent:
        return None
    rows: set[int] = set()
    for idx in per_agent.values():
        rows.update(int(i) for i in idx)
    return sorted(rows)


def label_domain_batches(
    record: Mapping[str, Any],
    *,
    train_prompts: Sequence[str],
    train_responses: Sequence[str | None],
    train_labels: Sequence[str],
    domain_names: Sequence[str],
) -> dict[str, DomainBatch]:
    """Split one round batch by benchmark label.

    :param record: One source history record.
    :type record: Mapping
    :param train_prompts: Flattened train partition prompts (the order the
        closed loop indexed with ``agent_batch_indices``).
    :type train_prompts: Sequence[str]
    :param train_responses: Aligned responses.
    :type train_responses: Sequence[str | None]
    :param train_labels: Aligned benchmark names.
    :type train_labels: Sequence[str]
    :param domain_names: Benchmark names, one expert each (config order).
    :type domain_names: Sequence[str]
    :returns: ``benchmark -> DomainBatch`` (every domain present, possibly
        empty).
    :rtype: dict[str, DomainBatch]
    :raises ValueError: If the record has neither indices nor prompts, or a
        logged prompt is not in the train partition.
    """
    rows = round_batch_indices(record)
    if rows is None:
        batch_prompts = record.get("batch_prompts")
        if not batch_prompts:
            # Hard-routing runs log only per-agent prompt lists.
            agent_prompts = record.get("agent_prompts") or {}
            seen: set[str] = set()
            batch_prompts = []
            for name in sorted(agent_prompts):
                for p in agent_prompts[name]:
                    if p not in seen:
                        seen.add(p)
                        batch_prompts.append(p)
        if not batch_prompts:
            raise ValueError(
                f"round {record.get('round')} logs neither agent_batch_indices "
                "nor batch_prompts / agent_prompts"
            )
        index_of: dict[str, int] = {}
        for i, p in enumerate(train_prompts):
            index_of.setdefault(p, i)
        try:
            rows = sorted({index_of[str(p)] for p in batch_prompts})
        except KeyError as exc:
            raise ValueError(
                f"round {record.get('round')}: logged prompt not in the train "
                f"partition: {str(exc)[:80]!r}"
            ) from exc

    prompts: dict[str, list[str]] = {d: [] for d in domain_names}
    responses: dict[str, list[str | None]] = {d: [] for d in domain_names}
    for i in rows:
        label = str(train_labels[i])
        if label not in prompts:
            raise ValueError(f"benchmark {label!r} is not one of {list(domain_names)}")
        prompts[label].append(str(train_prompts[i]))
        r = train_responses[i]
        responses[label].append(r if r else None)
    return {
        d: DomainBatch(prompts=prompts[d], responses=responses[d], sample_weights=None)
        for d in domain_names
    }


def history_domain_batches(
    record: Mapping[str, Any],
    *,
    domain_names: Sequence[str],
) -> dict[str, DomainBatch]:
    """Replay the per-merge-group batches the game logged for one round.

    Loss weights are carried over only when the source round trained with
    ``soft_loss == 'weighted'``; otherwise the experts train at unit weight
    exactly as the pairs did.

    :param record: One source history record.
    :type record: Mapping
    :param domain_names: Merge-group names (``pair-k``) to replay.
    :type domain_names: Sequence[str]
    :returns: ``pair -> DomainBatch``.
    :rtype: dict[str, DomainBatch]
    """
    agent_prompts: Mapping[str, Sequence[str]] = record.get("agent_prompts") or {}
    agent_responses: Mapping[str, Sequence[str]] = record.get("agent_responses") or {}
    agent_weights: Mapping[str, Sequence[float]] = record.get("agent_sample_weights") or {}
    weighted = (
        str(record.get("routing_mode", "hard")) == "soft"
        and str(record.get("soft_loss") or "") == "weighted"
    )
    out: dict[str, DomainBatch] = {}
    for name in domain_names:
        p_list = [str(p) for p in agent_prompts.get(name, [])]
        r_raw = list(agent_responses.get(name, []))
        r_list: list[str | None] = (
            [r if r else None for r in r_raw]
            if len(r_raw) == len(p_list) else [None] * len(p_list)
        )
        w_raw = list(agent_weights.get(name, []))
        w_list = (
            [float(w) for w in w_raw]
            if weighted and len(w_raw) == len(p_list) and p_list else None
        )
        out[name] = DomainBatch(prompts=p_list, responses=r_list, sample_weights=w_list)
    return out


def rekey_by_merge_group(
    per_pair: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Re-key a theory-init per-pair mapping by merge-group name.

    The paired theory initialisation logs per-pair quantities
    (``pair_dominant_axis``, ``pair_positions``) under a membership key,
    ``pair_<clone-a>_<clone-b>`` (see
    :func:`infl_ens.training.agent_init.init_agents_theory_gradient_paired`),
    while the SFT merge groups the rest of the pipeline uses are named
    ``pair-k`` (``merge_group_prefix``). This maps each group's ``train_as``
    name onto the value logged for the group's members, in any member order.
    Keys that already equal a ``train_as`` name are kept as they are.

    :param per_pair: Raw theory-init mapping keyed by membership.
    :type per_pair: Mapping[str, Any]
    :param groups: Merge groups, each with ``train_as`` and ``names``.
    :type groups: Sequence[Mapping]
    :returns: ``train_as -> value`` for every group that could be matched;
        empty when nothing matches.
    :rtype: dict[str, Any]
    """
    from itertools import permutations

    raw = {str(k): v for k, v in per_pair.items()}
    out: dict[str, Any] = {}
    for group in groups:
        train_as = str(group["train_as"])
        names = [str(n) for n in group["names"]]
        if train_as in raw:
            out[train_as] = raw[train_as]
            continue
        for order in permutations(names):
            key = "pair_" + "_".join(order)
            if key in raw:
                out[train_as] = raw[key]
                break
    return out


def dominant_axis_by_merge_group(
    pair_dominant_axis: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Re-key ``theory_init.pair_dominant_axis`` by merge-group name.

    Thin wrapper over :func:`rekey_by_merge_group` that casts to ``int``.

    :param pair_dominant_axis: Raw ``theory_init.pair_dominant_axis`` mapping.
    :type pair_dominant_axis: Mapping[str, Any]
    :param groups: Merge groups, each with ``train_as`` and ``names``.
    :type groups: Sequence[Mapping]
    :returns: ``train_as -> axis index`` for every matched group.
    :rtype: dict[str, int]
    """
    return {k: int(v) for k, v in rekey_by_merge_group(pair_dominant_axis, groups).items()}


def assign_pairs_to_axes(
    positions: Mapping[str, Sequence[float]],
    pairs: Sequence[str],
    n_axes: int,
) -> dict[str, int]:
    """Assign pairs to distinct axes maximising the summed coordinate.

    A pair's argmax coordinate is not a bijection in general: at the game
    equilibrium two pairs may share a dominant axis while another axis has
    none. The label partition needs exactly one expert per benchmark, so
    this solves the linear assignment
    :math:`\\max_{\\pi} \\sum_k x_{k,\\pi(k)}` over permutations
    :math:`\\pi` (Hungarian method via :func:`scipy.optimize.linear_sum_assignment`
    when SciPy is available, otherwise exhaustive search for up to nine
    pairs and a greedy fallback beyond).

    :param positions: ``pair -> coordinates`` (length ``n_axes``).
    :type positions: Mapping[str, Sequence[float]]
    :param pairs: Pair names to assign, in output order.
    :type pairs: Sequence[str]
    :param n_axes: Number of axes; must equal ``len(pairs)``.
    :type n_axes: int
    :returns: ``pair -> axis index``, a bijection onto ``range(n_axes)``.
    :rtype: dict[str, int]
    :raises ValueError: If a position is missing, the wrong length, or
        ``len(pairs) != n_axes``.
    """
    if len(pairs) != n_axes:
        raise ValueError(f"{len(pairs)} pairs but {n_axes} axes")
    rows = []
    for pair in pairs:
        if pair not in positions:
            raise ValueError(f"pair_positions lacks {pair!r}")
        vec = np.asarray(positions[pair], dtype=float).ravel()
        if vec.shape[0] != n_axes:
            raise ValueError(f"{pair}: position has {vec.shape[0]} coordinates, expected {n_axes}")
        rows.append(vec)
    score = np.stack(rows, axis=0)

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover - scipy optional
        linear_sum_assignment = None
    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(score, maximize=True)
        return {pairs[int(r)]: int(c) for r, c in zip(row_ind, col_ind, strict=True)}

    from itertools import permutations

    if n_axes <= 9:
        best, best_val = None, -np.inf
        for perm in permutations(range(n_axes)):
            val = float(sum(score[k, perm[k]] for k in range(n_axes)))
            if val > best_val:
                best, best_val = perm, val
        assert best is not None
        return {pairs[k]: int(best[k]) for k in range(n_axes)}
    # Greedy: repeatedly take the largest remaining (pair, axis) cell.
    out: dict[str, int] = {}
    free_rows, free_cols = set(range(n_axes)), set(range(n_axes))
    while free_rows:
        r, c = max(
            ((r, c) for r in free_rows for c in free_cols), key=lambda rc: score[rc[0], rc[1]],
        )
        out[pairs[r]] = c
        free_rows.discard(r)
        free_cols.discard(c)
    return out


def source_router_blocks(
    source_run_dir: PathLike,
    history: Sequence[Mapping[str, Any]],
) -> SourceRouterBlocks:
    """Copy the router description of the source closed-loop run.

    Reads ``agents``, the routing keys and the literal ``sft_merge_groups``
    from the source ``resolved_config.yaml``; falls back to the history's
    ``theory_init.sft_merge_groups_resolved`` and ``pair_members`` when
    the config is missing or still holds the ``from_init`` sentinel.

    :param source_run_dir: Directory holding the source run.
    :type source_run_dir: str | pathlib.Path
    :param history: The source history records.
    :type history: Sequence[Mapping]
    :returns: The copied router blocks.
    :rtype: SourceRouterBlocks
    :raises ValueError: If no agents or merge groups can be recovered.
    """
    from infl_ens.config import load_config

    run_dir = Path(source_run_dir)
    cfg: dict[str, Any] = {}
    resolved = run_dir / "resolved_config.yaml"
    if resolved.is_file():
        cfg = load_config(resolved, validate=False)

    agents_raw = cfg.get("agents")
    agents: list[dict[str, Any]] = []
    if isinstance(agents_raw, list):
        agents = [{"name": str(a["name"])} for a in agents_raw]
    if not agents:
        positions = history[-1].get("positions") or {}
        agents = [{"name": str(n)} for n in positions]
    if not agents:
        raise ValueError(f"cannot recover agents from {run_dir}")

    groups_raw = (cfg.get("closed_loop") or {}).get("sft_merge_groups")
    groups: list[dict[str, Any]] = []
    if isinstance(groups_raw, list) and groups_raw and isinstance(groups_raw[0], Mapping):
        groups = [
            {"train_as": str(g["train_as"]), "names": [str(n) for n in g["names"]]}
            for g in groups_raw
        ]
    theory_init = history[0].get("theory_init") or {}
    if not groups:
        resolved_groups = theory_init.get("sft_merge_groups_resolved")
        if isinstance(resolved_groups, list) and resolved_groups:
            groups = [
                {"train_as": str(g["train_as"]), "names": [str(n) for n in g["names"]]}
                for g in resolved_groups
            ]
    if not groups:
        members = history[0].get("pair_members")
        if isinstance(members, Mapping) and members:
            groups = [
                {"train_as": str(k), "names": [str(n) for n in v]}
                for k, v in sorted(members.items())
            ]
    if not groups:
        raise ValueError(f"cannot recover sft_merge_groups from {run_dir}")

    router_keys = {k: cfg[k] for k in _ROUTER_TOP_LEVEL_KEYS if k in cfg}
    pair_dominant_axis = dominant_axis_by_merge_group(
        theory_init.get("pair_dominant_axis") or {}, groups,
    )
    # Pair coordinates: the theory init's own record first, else the
    # round-0 position of each group's first member (partners are
    # co-located after the paired init).
    pair_positions = {
        str(k): [float(x) for x in v]
        for k, v in rekey_by_merge_group(theory_init.get("pair_positions") or {}, groups).items()
    }
    if not pair_positions:
        positions0 = history[0].get("positions") or {}
        for g in groups:
            first = str(g["names"][0]) if g["names"] else None
            if first in positions0:
                pair_positions[str(g["train_as"])] = [float(x) for x in positions0[first]]
    return SourceRouterBlocks(
        agents=agents,
        merge_groups=groups,
        router_keys=router_keys,
        pair_dominant_axis=pair_dominant_axis,
        pair_positions=pair_positions,
    )


def pair_to_benchmark_aliases(
    blocks: SourceRouterBlocks,
    benchmark_names: Sequence[str],
) -> dict[str, str]:
    """Map every source pair onto one benchmark expert, one-to-one.

    Axis ``k`` is the ``k``-th benchmark of the config (the trait space
    learns one axis per benchmark in loader order). The mapping is chosen
    in this order of preference:

    1. the theory initialisation's ``pair_dominant_axis`` (each pair's
       argmax coordinate) when it happens to be a bijection;
    2. otherwise the optimal one-to-one assignment of pairs to axes from
       the pairs' initial coordinates (:func:`assign_pairs_to_axes`), which
       agrees with the argmax wherever that is unambiguous;
    3. otherwise, when neither is recorded, the pairs are matched to the
       benchmarks in order.

    :param blocks: Source router blocks.
    :type blocks: SourceRouterBlocks
    :param benchmark_names: Benchmark names in config order.
    :type benchmark_names: Sequence[str]
    :returns: ``pair-k -> benchmark`` covering every pair.
    :rtype: dict[str, str]
    :raises ValueError: If the pair and benchmark counts differ, a
        recorded axis is out of range, a recorded mapping is incomplete,
        or no bijection can be formed.
    """
    pairs = blocks.pair_names
    benches = [str(b) for b in benchmark_names]
    n = len(benches)
    if len(pairs) != n:
        raise ValueError(
            f"{len(pairs)} merge groups but {n} benchmarks; the "
            "label partition needs one pair per benchmark"
        )

    axis_of: dict[str, int]
    if blocks.pair_dominant_axis:
        axis_of = {}
        for pair in pairs:
            if pair not in blocks.pair_dominant_axis:
                raise ValueError(f"pair_dominant_axis lacks {pair!r}")
            axis = blocks.pair_dominant_axis[pair]
            if not 0 <= axis < n:
                raise ValueError(f"{pair}: axis {axis} outside 0..{n - 1}")
            axis_of[pair] = axis
        if len(set(axis_of.values())) != n:
            if not blocks.pair_positions:
                raise ValueError(
                    "pair dominant axes collide and no pair_positions are "
                    f"available to break the tie: {axis_of}"
                )
            axis_of = assign_pairs_to_axes(blocks.pair_positions, pairs, n)
    elif blocks.pair_positions:
        axis_of = assign_pairs_to_axes(blocks.pair_positions, pairs, n)
    else:
        axis_of = {pair: k for k, pair in enumerate(pairs)}

    aliases = {pair: benches[axis_of[pair]] for pair in pairs}
    if len(set(aliases.values())) != n:  # pragma: no cover - assignment guarantees this
        raise ValueError(f"pair -> benchmark aliases are not one-to-one: {aliases}")
    return aliases


def train_modula_res(
    history: Sequence[Mapping[str, Any]],
    *,
    domain_names: Sequence[str],
    batches_for_round: Callable[[Mapping[str, Any]], dict[str, DomainBatch]],
    sft_cfg: Any,
    project: Any,
    output_dir: PathLike,
    universal_adapter_dir: Path | None,
    initial_positions: Mapping[str, Sequence[float]],
    rounds: Sequence[int] | None = None,
    save_per_round: bool = True,
    sft_train: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Train one cumulative LoRA expert per domain over the source rounds.

    :param history: Source closed-loop history.
    :type history: Sequence[Mapping]
    :param domain_names: Expert names (benchmarks or ``pair-k``).
    :type domain_names: Sequence[str]
    :param batches_for_round: Maps a history record to per-domain batches.
    :type batches_for_round: Callable[[Mapping], dict[str, DomainBatch]]
    :param sft_cfg: :class:`~infl_ens.training.sft_training.SFTTrainingConfig`.
    :type sft_cfg: SFTTrainingConfig
    :param project: Trait-space projector (forwarded; positions are frozen).
    :type project: Callable
    :param output_dir: Adapter root (``<output_dir>/<expert>/round-NN``).
    :type output_dir: str | pathlib.Path
    :param universal_adapter_dir: Frozen universal adapter merged under every
        expert, or ``None`` for from-scratch experts.
    :type universal_adapter_dir: pathlib.Path | None
    :param initial_positions: Placeholder trait-space position per expert
        (never updated; kept on the agent for bookkeeping).
    :type initial_positions: Mapping[str, Sequence[float]]
    :param rounds: Optional subset of round indices; ``None`` = every round.
    :type rounds: Sequence[int] | None
    :param save_per_round: Write ``round-NN`` subdirectories.
    :type save_per_round: bool
    :param sft_train: Trainer callable (defaults to
        :func:`infl_ens.training.sft_training.sft_train_agent`; injectable
        for tests).
    :type sft_train: Callable | None
    :returns: One summary per round with per-expert training records.
    :rtype: list[dict]
    """
    if sft_train is None:
        from infl_ens.training.sft_training import sft_train_agent as sft_train

    by_round = {int(r["round"]): r for r in history}
    target_rounds = (
        sorted(by_round) if rounds is None else sorted(int(r) for r in rounds)
    )
    agents = {
        name: RouterAgent(
            name=name,
            position=np.asarray(initial_positions[name], dtype=float),
        )
        for name in domain_names
    }
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    universal = str(universal_adapter_dir) if universal_adapter_dir is not None else None

    summaries: list[dict[str, Any]] = []
    for r in target_rounds:
        rec = by_round.get(r)
        if rec is None:
            continue
        batches = batches_for_round(rec)
        per_expert: dict[str, dict[str, Any]] = {}
        for name in domain_names:
            batch = batches.get(name)
            if batch is None or batch.n == 0:
                per_expert[name] = {"n_train": 0, "output_dir": None, "skipped": True}
                continue
            agent = agents[name]
            out_override = (
                str(out_root / name / f"round-{r:02d}") if save_per_round else None
            )
            result = sft_train(
                agent,
                prompts=batch.prompts,
                responses=batch.responses if any(batch.responses) else None,
                cfg=sft_cfg,
                eval_prompts=batch.prompts,
                project=project,
                blend=1.0,
                out_dir_override=out_override,
                sample_weights=batch.sample_weights,
                skip_position_update=True,
                frozen_base_adapter_dir=universal,
            )
            per_expert[name] = {
                "n_train": int(result["n_train"]),
                "train_loss": result.get("train_loss"),
                "output_dir": result["output_dir"],
                "loaded_prior_lora": result.get("loaded_prior_lora"),
                "frozen_base_adapter": result.get("frozen_base_adapter"),
                "skipped": False,
            }
        summaries.append({
            "round": r,
            "experts": per_expert,
            "n_train_total": int(sum(e["n_train"] for e in per_expert.values())),
        })
    return summaries


__all__ = [
    "DOMAIN_SOURCES",
    "DomainBatch",
    "SourceRouterBlocks",
    "assign_pairs_to_axes",
    "dominant_axis_by_merge_group",
    "history_domain_batches",
    "label_domain_batches",
    "load_closed_loop_history_lenient",
    "pair_to_benchmark_aliases",
    "rekey_by_merge_group",
    "resolve_universal_adapter_dir",
    "round_batch_indices",
    "source_router_blocks",
    "train_modula_res",
]
