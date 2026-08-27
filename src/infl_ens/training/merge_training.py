"""Pair-merged SFT: four router agents, two physical LoRA trainers.

The router keeps four strategic positions for game-theoretic routing and
centroid updates. When ``closed_loop.sft_merge_groups`` is set, routed
prompts from each pair of nearby clones are concatenated and used to
train a single cumulative LoRA per pair.

With ``closed_loop.sft_merge_mode: proximity``, merge groups are resolved
each round from live positions: harm-axis pairs when layout is ``2,2``,
else nearest-neighbour pairs under ``merge_distance_threshold``; if no
pair qualifies, the round logs a non-``2,2`` layout and routers train
individually.

Under ``closed_loop.routing_mode: soft`` the merge group, not the clone, is
the routing unit: :func:`soft_pair_assignments` sums each group's member
allocations into a group share, keeps the top-``soft_top_k`` groups per
query, and renormalises. Each group then trains its single LoRA on those
queries with the group share as a per-example loss weight and takes ONE
position step toward :func:`soft_pair_position_target`, which is written
back to every member so partners stay exactly co-located.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from infl_ens.inflgame.router.agents import RouterAgent
from infl_ens.utils.agent_init import harm_pair_indices


def merge_train_name(members: Sequence[str]) -> str:
    """Stable adapter subdirectory name for a merge group.

    :param members: Router agent names in the group.
    :type members: Sequence[str]
    :returns: Name such as ``merge-clone-0-clone-1``.
    :rtype: str
    """
    return "merge-" + "-".join(sorted(members))


def classify_layout_extended(
    pos: np.ndarray,
    *,
    spread_thresh: float = 0.45,
) -> str:
    """Classify layout as ``2,2``, ``collapsed``, or ``other``.

    :param pos: ``(N, L)`` agent positions.
    :type pos: numpy.ndarray
    :param spread_thresh: Collapse threshold passed to :func:`classify_layout`.
    :type spread_thresh: float
    :returns: Layout label.
    :rtype: str
    """
    from infl_ens.training.pool_dynamics import classify_layout, pairwise_spread

    if pairwise_spread(pos) < spread_thresh:
        return "collapsed"
    if classify_layout(pos, spread_thresh=spread_thresh) == "2,2":
        return "2,2"
    return "other"


def proximity_pairs(
    router_names: Sequence[str],
    positions: np.ndarray,
    distance_threshold: float,
) -> tuple[list[list[str]], list[str]]:
    """Greedy pair agents whose L2 distance is below ``distance_threshold``.

    :param router_names: Agent names aligned with ``positions`` rows.
    :type router_names: Sequence[str]
    :param positions: ``(N, L)`` positions.
    :type positions: numpy.ndarray
    :param distance_threshold: Maximum distance to treat as co-located.
    :type distance_threshold: float
    :returns: ``(pairs, unpaired)`` where each pair is two names.
    :rtype: tuple[list[list[str]], list[str]]
    """
    n = len(router_names)
    dists: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            dists.append((d, i, j))
    dists.sort(key=lambda x: x[0])
    used: set[int] = set()
    pairs: list[list[str]] = []
    pair_dists: dict[str, float] = {}
    for d, i, j in dists:
        if d > distance_threshold:
            break
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        pair = [router_names[i], router_names[j]]
        pairs.append(pair)
        key = merge_train_name(pair)
        pair_dists[key] = d
    unpaired = [router_names[k] for k in range(n) if k not in used]
    return pairs, unpaired


def resolve_dynamic_merge_groups(
    router_agents: Sequence[RouterAgent],
    router_names: Sequence[str],
    *,
    distance_threshold: float = 0.08,
    spread_thresh: float = 0.45,
) -> tuple[list[tuple[str, list[str]]], list[str], dict[str, Any]]:
    """Resolve merge groups from current router positions.

    When layout is ``2,2``, pairs are the low- and high-harm groups from
    :func:`harm_pair_indices`. Otherwise, pairs agents by proximity; agents
    that match no partner are returned in ``unpaired``.

    :param router_agents: Live router agents.
    :type router_agents: Sequence[RouterAgent]
    :param router_names: Agent name order.
    :type router_names: Sequence[str]
    :param distance_threshold: Proximity pairing cutoff.
    :type distance_threshold: float
    :param spread_thresh: Layout classifier spread threshold.
    :type spread_thresh: float
    :returns: ``(merge_groups, unpaired_names, metadata)``.
    :rtype: tuple
    """
    by_name = {a.name: a for a in router_agents}
    positions = np.stack([by_name[n].position for n in router_names], axis=0)
    layout = classify_layout_extended(positions, spread_thresh=spread_thresh)
    meta: dict[str, Any] = {
        "layout": layout,
        "distance_threshold": distance_threshold,
        "positions": {n: by_name[n].position.tolist() for n in router_names},
    }

    groups: list[tuple[str, list[str]]] = []
    unpaired: list[str] = []

    if layout == "2,2":
        low_idx, high_idx = harm_pair_indices(positions)
        low_names = [router_names[int(i)] for i in low_idx]
        high_names = [router_names[int(i)] for i in high_idx]
        groups = [
            (merge_train_name(low_names), low_names),
            (merge_train_name(high_names), high_names),
        ]
        meta["pairing_method"] = "harm_22"
        meta["low_harm_pair"] = low_names
        meta["high_harm_pair"] = high_names
        d_low = float(np.linalg.norm(positions[low_idx[0]] - positions[low_idx[1]]))
        d_high = float(np.linalg.norm(positions[high_idx[0]] - positions[high_idx[1]]))
        meta["within_pair_distance"] = {
            merge_train_name(low_names): d_low,
            merge_train_name(high_names): d_high,
        }
    else:
        pairs, unpaired = proximity_pairs(
            router_names, positions, distance_threshold,
        )
        if pairs:
            groups = [(merge_train_name(m), m) for m in pairs]
            meta["pairing_method"] = "proximity"
            meta["proximity_pairs"] = pairs
        else:
            meta["pairing_method"] = "none"
        meta["unpaired"] = unpaired

    meta["merge_groups"] = [{"train_as": t, "names": m} for t, m in groups]
    return groups, unpaired, meta


def parse_sft_merge_groups(
    cl: dict[str, Any],
    router_names: Sequence[str],
) -> Optional[list[tuple[str, list[str]]]]:
    """Parse ``closed_loop.sft_merge_groups`` into training groups.

    Each entry is either a list of router agent names or a mapping with
    ``names`` and optional ``train_as`` (default ``merge-{i}``).

    :param cl: ``closed_loop`` config block.
    :type cl: dict
    :param router_names: Router agent names from the config.
    :type router_names: Sequence[str]
    :returns: List of ``(train_name, member_names)`` or ``None`` if disabled.
    :rtype: list[tuple[str, list[str]]] | None
    :raises ValueError: If groups are malformed or do not partition agents.
    """
    raw = cl.get("sft_merge_groups")
    if raw is None:
        return None
    if isinstance(raw, str):
        raise ValueError(
            f"closed_loop.sft_merge_groups={raw!r} is a sentinel that the "
            "closed-loop driver must resolve into concrete groups before "
            "parsing (see merge_groups_from_theory_pairs)",
        )

    router_set = set(router_names)
    groups: list[tuple[str, list[str]]] = []
    covered: set[str] = set()

    for i, entry in enumerate(raw):
        if isinstance(entry, dict):
            members = list(entry["names"])
            train_as = str(entry.get("train_as", f"merge-{i}"))
        else:
            members = list(entry)
            train_as = f"merge-{i}"
        missing = set(members) - router_set
        if missing:
            raise ValueError(f"sft_merge_groups references unknown agents {missing}")
        if covered & set(members):
            raise ValueError("sft_merge_groups must partition router agents")
        covered.update(members)
        groups.append((train_as, members))

    if covered != router_set:
        raise ValueError(
            f"sft_merge_groups must cover all router agents; missing "
            f"{router_set - covered}",
        )
    return groups


def make_merge_train_agents(
    groups: list[tuple[str, list[str]]],
    router_agents: Sequence[RouterAgent],
) -> dict[str, RouterAgent]:
    """Create one :class:`RouterAgent` per merge group for LoRA training.

    Initial position is the mean of member router positions (metadata only;
    merge trainers do not participate in routing).

    :param groups: ``(train_name, member_names)`` pairs.
    :type groups: list[tuple[str, list[str]]]
    :param router_agents: Live router agents.
    :type router_agents: Sequence[RouterAgent]
    :returns: Mapping ``train_name -> RouterAgent`` for SFT.
    :rtype: dict[str, RouterAgent]
    """
    by_name = {a.name: a for a in router_agents}
    out: dict[str, RouterAgent] = {}
    for train_name, members in groups:
        positions = np.stack([by_name[m].position for m in members], axis=0)
        out[train_name] = RouterAgent(
            name=train_name,
            position=positions.mean(axis=0),
        )
    return out


def merge_routed_batch(
    agent_prompts: dict[str, list[str]],
    agent_responses: dict[str, list[str]],
    members: Sequence[str],
    *,
    agent_sample_weights: Optional[dict[str, list[float]]] = None,
    loss_reweight: Optional[str] = None,
) -> tuple[list[str], list[str | None], Optional[list[float]]]:
    """Concatenate routed examples from ``members`` for merged SFT.

    :param agent_prompts: Per-router prompts for this round.
    :type agent_prompts: dict[str, list[str]]
    :param agent_responses: Per-router responses.
    :type agent_responses: dict[str, list[str]]
    :param members: Router agent names in this merge group.
    :type members: Sequence[str]
    :param agent_sample_weights: Optional per-router (1-G) weights.
    :type agent_sample_weights: dict[str, list[float]] | None
    :param loss_reweight: ``closed_loop.loss_reweight`` mode.
    :type loss_reweight: str | None
    :returns: ``(prompts, responses, sample_weights)`` for SFT.
    :rtype: tuple[list[str], list[str | None], list[float] | None]
    """
    prompts: list[str] = []
    responses: list[str | None] = []
    weights: list[float] = []
    use_weights = loss_reweight == "one_minus_G"

    for name in members:
        p_list = list(agent_prompts.get(name, []))
        r_list = list(agent_responses.get(name, []))
        w_list = (
            list(agent_sample_weights.get(name, []))
            if agent_sample_weights is not None else []
        )
        if r_list and len(r_list) == len(p_list):
            for i, p in enumerate(p_list):
                prompts.append(p)
                responses.append(r_list[i] if r_list[i] else None)
                if use_weights and w_list:
                    weights.append(w_list[i])
        else:
            prompts.extend(p_list)
            responses.extend([None] * len(p_list))
            if use_weights and w_list:
                weights.extend(w_list)

    sample_weights_arg: Optional[list[float]] = None
    if use_weights and weights:
        if len(weights) != len(prompts):
            raise ValueError(
                f"merged weight length {len(weights)} != prompts {len(prompts)}",
            )
        sample_weights_arg = weights
    return prompts, responses, sample_weights_arg


def closed_loop_weight_args(
    loss_reweight: Optional[str],
    centroid_mode: str,
    weights_i: Optional[list[float]],
    *,
    position_update: str = "theory_matched",
) -> tuple[Optional[list[float]], Optional[list[float]], bool]:
    """Map the hard-routing rule knobs to SFT sample weights, centroid weights, and skip flag.

    ``loss_reweight`` governs the loss side only: ``'one_minus_G'`` trains
    with ``weights_i`` as per-example loss weights, anything else trains at
    unit weight. ``position_update`` governs the centroid side:
    ``'theory_matched'`` uses ``weights_i`` (the per-query ``(1 - G)`` that
    completes the :math:`G_i(1 - G_i)` gradient coefficient under canonical
    routing) and ``'naive'`` uses the unweighted centroid. The deprecated
    ``loss_reweight='position_only'`` is accepted as an alias for unit loss
    plus ``position_update='theory_matched'``.

    Under ``centroid_mode='expected_pool'`` the per-batch position update is
    skipped altogether (the driver applies the expected-pool centroid, which
    is gradient-matched by construction).

    :param loss_reweight: ``closed_loop.loss_reweight`` value.
    :type loss_reweight: str | None
    :param centroid_mode: ``batch`` or ``expected_pool``.
    :type centroid_mode: str
    :param weights_i: Per-query ``(1 - G)`` weights for this agent's batch;
        ``None`` when routing already carries the ``(1 - G)`` factor
        (strategic routing) or the centroid is unweighted.
    :type weights_i: list[float] | None
    :param position_update: ``closed_loop.position_update`` value.
    :type position_update: str
    :returns: ``(sample_weights, eval_weights, skip_position_update)``.
    :rtype: tuple
    """
    if loss_reweight == "position_only":
        position_update = "theory_matched"
    skip = centroid_mode == "expected_pool"
    sample_weights = weights_i if loss_reweight == "one_minus_G" else None
    if skip or position_update != "theory_matched":
        return sample_weights, None, skip
    return sample_weights, weights_i, skip


def snap_configured_merge_pairs(
    router_agents: Sequence[RouterAgent],
    merge_groups: Sequence[tuple[str, Sequence[str]]],
    *,
    threshold: float = 0.01,
) -> dict[str, Any]:
    """Snap co-located merge partners to their shared centroid.

    Router agents keep separate identities for routing and position updates;
    only their stored positions are aligned when within ``threshold``.

    :param router_agents: Live router agents.
    :type router_agents: Sequence[RouterAgent]
    :param merge_groups: Configured ``(train_as, members)`` pairs.
    :type merge_groups: Sequence[tuple[str, Sequence[str]]]
    :param threshold: Maximum L2 distance to treat as co-located.
    :type threshold: float
    :returns: Snap metadata keyed by merge group.
    :rtype: dict
    """
    by_name = {a.name: a for a in router_agents}
    snapped: dict[str, Any] = {}
    for train_as, members in merge_groups:
        members = list(members)
        if len(members) != 2:
            continue
        p0 = by_name[members[0]].position
        p1 = by_name[members[1]].position
        dist = float(np.linalg.norm(p0 - p1))
        if dist <= threshold:
            centroid = 0.5 * (p0 + p1)
            by_name[members[0]].position = centroid.copy()
            by_name[members[1]].position = centroid.copy()
            snapped[train_as] = {
                "distance_before": dist,
                "snapped": True,
                "members": members,
            }
        else:
            snapped[train_as] = {
                "distance_before": dist,
                "snapped": False,
                "members": members,
            }
    return snapped


def collapsed_sft_merge_groups(
    router_agents: Sequence[RouterAgent],
    merge_groups: Sequence[tuple[str, Sequence[str]]],
    *,
    threshold: float = 0.01,
) -> tuple[list[tuple[str, list[str]]], dict[str, Any]]:
    """Keep only merge groups whose partners are co-located within ``threshold``.

    :param router_agents: Live router agents.
    :type router_agents: Sequence[RouterAgent]
    :param merge_groups: Candidate ``(train_as, members)`` pairs.
    :type merge_groups: Sequence[tuple[str, Sequence[str]]]
    :param threshold: Maximum L2 distance for SFT merge eligibility.
    :type threshold: float
    :returns: Active merge groups and diagnostic metadata.
    :rtype: tuple[list[tuple[str, list[str]]], dict]
    """
    by_name = {a.name: a for a in router_agents}
    active: list[tuple[str, list[str]]] = []
    collapsed: dict[str, float] = {}
    skipped: dict[str, float] = {}
    for train_as, members in merge_groups:
        members = list(members)
        if len(members) != 2:
            raise ValueError(
                f"collapsed_sft_merge_groups expects pairs; "
                f"{train_as!r} has {len(members)} members",
            )
        dist = float(
            np.linalg.norm(by_name[members[0]].position - by_name[members[1]].position),
        )
        if dist <= threshold:
            active.append((train_as, members))
            collapsed[train_as] = dist
        else:
            skipped[train_as] = dist
    return active, {
        "collapsed_sft_merge": collapsed,
        "skipped_sft_merge": skipped,
    }


def merge_mode_from_config(cl: dict[str, Any]) -> str:
    """Return ``fixed``, ``proximity``, or ``none`` for SFT merging.

    :param cl: ``closed_loop`` config block.
    :type cl: dict
    :returns: Merge mode label.
    :rtype: str
    """
    if cl.get("sft_merge_groups") is not None:
        return "fixed"
    mode = str(cl.get("sft_merge_mode", "none"))
    if mode in ("proximity", "dynamic", "by_proximity"):
        return "proximity"
    return "none"


def get_merge_train_agent(
    registry: dict[str, RouterAgent],
    train_name: str,
    members: Sequence[str],
    router_agents: Sequence[RouterAgent],
) -> RouterAgent:
    """Fetch or create a cumulative merge trainer for ``train_name``.

    :param registry: Persistent ``train_name -> RouterAgent`` map.
    :type registry: dict[str, RouterAgent]
    :param train_name: Adapter subdirectory name.
    :type train_name: str
    :param members: Router names in this merge group.
    :type members: Sequence[str]
    :param router_agents: Live routers (for centroid of member positions).
    :type router_agents: Sequence[RouterAgent]
    :returns: Merge trainer agent.
    :rtype: RouterAgent
    """
    by_name = {a.name: a for a in router_agents}
    positions = np.stack([by_name[m].position for m in members], axis=0)
    centroid = positions.mean(axis=0)
    if train_name in registry:
        registry[train_name].position = centroid
        return registry[train_name]
    agent = RouterAgent(name=train_name, position=centroid)
    registry[train_name] = agent
    return agent


def merge_groups_from_theory_pairs(
    paired_harm_order: Sequence[Sequence[str]],
    *,
    name_prefix: str = "pair",
) -> list[dict[str, Any]]:
    """Turn a paired theory init into ``sft_merge_groups`` entries.

    Consumes ``theory_init_meta['paired_harm_order']`` produced by
    :func:`infl_ens.utils.agent_init.init_agents_theory_gradient_paired`,
    whose entries are the two agent names co-located at each theory
    endpoint, ordered by the harm coordinate. The result is the ordinary
    list-of-mappings form accepted by :func:`parse_sft_merge_groups` and by
    :func:`infl_ens.evaluation.routing_eval.parse_merge_groups`, so a run
    that derives its groups from the theory solve stays readable by every
    downstream evaluation tool.

    :param paired_harm_order: Pairs of agent names in harm order.
    :type paired_harm_order: Sequence[Sequence[str]]
    :param name_prefix: Prefix for the generated adapter names.
    :type name_prefix: str
    :returns: ``[{"train_as": "<prefix>-<k>", "names": [a, b]}, ...]``.
    :rtype: list[dict]
    :raises ValueError: If the pairing is empty or any entry does not hold
        exactly two distinct names.
    """
    pairs = [list(entry) for entry in paired_harm_order]
    if not pairs:
        raise ValueError("paired_harm_order is empty")
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for k, members in enumerate(pairs):
        if len(members) != 2:
            raise ValueError(
                f"paired_harm_order entry {k} must hold exactly two names, "
                f"got {members}",
            )
        if seen & set(members):
            raise ValueError(
                f"paired_harm_order repeats agents in entry {k}: {members}",
            )
        seen.update(members)
        out.append({
            "train_as": f"{name_prefix}-{k}",
            "names": [str(m) for m in members],
        })
    return out


def group_index_for_merge_groups(
    merge_groups: Sequence[tuple[str, Sequence[str]]],
    router_names: Sequence[str],
) -> np.ndarray:
    """Map each router agent row to its merge-group index.

    :param merge_groups: ``(train_as, members)`` pairs in group order.
    :type merge_groups: Sequence[tuple[str, Sequence[str]]]
    :param router_names: Router agent names in population (row) order.
    :type router_names: Sequence[str]
    :returns: Group id per agent row, shape ``(N,)``.
    :rtype: numpy.ndarray
    :raises ValueError: If the groups do not cover every router agent.
    """
    position = {name: i for i, name in enumerate(router_names)}
    out = np.full(len(router_names), -1, dtype=int)
    for g, (_train_as, members) in enumerate(merge_groups):
        for name in members:
            if name not in position:
                raise ValueError(f"merge group references unknown agent {name!r}")
            out[position[name]] = g
    missing = [
        router_names[i] for i in range(len(router_names)) if out[i] < 0
    ]
    if missing:
        raise ValueError(
            f"merge groups do not cover router agents {missing}",
        )
    return out


def soft_pair_assignments(
    G_clone: np.ndarray,
    group_index: np.ndarray,
    n_groups: int,
    soft_top_k: int,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Top-``k`` soft routing weights at the merge-group level (training only).

    Sums clone allocations into group allocations with
    :func:`infl_ens.inflgame.router.allocation.group_allocation_weights`,
    then sparsifies and renormalises each query column with
    :func:`infl_ens.inflgame.router.allocation.top_k_allocation_weights`.
    With co-located pairs of equal size the group share is exactly the
    allocation of one agent at the shared position, so the *training*
    assignment over pairs is the ``n_groups``-player game the theory init
    solved.

    This decides only which queries a group's adapter trains on. The
    position step is never taken at the group level: every clone follows
    its own theory-matched update, and co-located partners stay together
    because identical positions receive identical steps — not because a
    shared step is written to both.

    :param G_clone: Clone allocation matrix, shape ``(N, M)``.
    :type G_clone: numpy.ndarray
    :param group_index: Group id per clone row, shape ``(N,)``.
    :type group_index: numpy.ndarray
    :param n_groups: Number of merge groups :math:`P`.
    :type n_groups: int
    :param soft_top_k: Groups retained per query.
    :type soft_top_k: int
    :returns: ``(W, idx, weights)`` where ``W`` is the ``(P, M)`` weight
        matrix, ``idx[p]`` the batch positions this group trains on and
        ``weights[p]`` the aligned per-example weights.
    :rtype: tuple[numpy.ndarray, list[numpy.ndarray], list[numpy.ndarray]]
    """
    from infl_ens.inflgame.router.allocation import (
        group_allocation_weights,
        top_k_allocation_weights,
    )

    G_group = group_allocation_weights(G_clone, group_index, n_groups)
    W = top_k_allocation_weights(G_group, soft_top_k)
    idx = [np.flatnonzero(W[p] > 0.0) for p in range(n_groups)]
    weights = [W[p, idx[p]] for p in range(n_groups)]
    return W, idx, weights


def soft_pair_position_target(
    coords: np.ndarray,
    idx: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted centroid of already-projected batch coordinates.

    Equivalent to
    :func:`infl_ens.data.trait_space.position_from_corpus` with
    ``scores=weights``, but takes coordinates that the caller already
    projected. That matters in the closed loop: the trait-space projector
    re-encodes with the sentence encoder on every call, so reusing the
    round's batch coordinates avoids one full encode per merge group per
    round.

    :param coords: Projected batch coordinates, shape ``(M, L)``.
    :type coords: numpy.ndarray
    :param idx: Rows of ``coords`` this group trains on, shape ``(m,)``.
    :type idx: numpy.ndarray
    :param weights: Per-row weights aligned with ``idx``, shape ``(m,)``.
    :type weights: numpy.ndarray
    :returns: Target position, shape ``(L,)``.
    :rtype: numpy.ndarray
    :raises ValueError: If ``idx`` is empty, ``weights`` is misaligned, or
        any weight is negative.
    """
    rows = np.asarray(idx, dtype=int)
    if rows.size == 0:
        raise ValueError("soft_pair_position_target requires a non-empty idx")
    w = np.asarray(weights, dtype=float)
    if w.shape != rows.shape:
        raise ValueError(
            f"weights shape {w.shape} does not match idx shape {rows.shape}"
        )
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    sub = np.asarray(coords, dtype=float)[rows]
    total = float(w.sum())
    if total <= 0.0:
        return sub.mean(axis=0)
    return (w[:, None] * sub).sum(axis=0) / total
