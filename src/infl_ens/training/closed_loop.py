"""The closed loop: route -> SFT -> position update, round after round.

:func:`run_closed_loop` is the ``closed_loop`` task behind
``python -m infl_ens.training``.  Each round it routes a batch of prompts
through the influencer-game router, fine-tunes the agents (or their merge
groups) on what they were routed, and moves every agent toward the
gradient-matched centroid of its routed mass.  ``history.json`` is
rewritten after every round.

Configuration knobs (``closed_loop`` block; see
:data:`infl_ens.config.CLOSED_LOOP_KEYS` for the full list):

- ``init_mode``: ``mean_noise`` (default; every clone at
  :math:`\\mathbb{E}_B[b]` plus ``init_noise``), ``theory_gradient`` (grid-Nash
  gradient ascent from a random separated start) or
  ``theory_gradient_paired`` (the same, then partners co-located; supplies
  the ``from_init`` merge groups).  ``theory_gradient`` holds the solver
  hyperparameters (``learning_rate``, ``n_steps``, ``tol``,
  ``min_pairwise``, ``pairing``).
- ``routing_weight``: ``G`` (canonical, Lovett & Fu 2024) or
  ``G_times_1mG`` (strategic).
- ``routing_mode``: ``hard`` (sample one agent per query) or ``soft``
  (every query trains its top-``soft_top_k`` agents or merge groups;
  ``soft_loss`` = ``weighted`` share-weighted loss or ``unit``).
- ``position_update``: ``theory_matched`` (default; expected drift
  parallel to :math:`\\nabla_{x_i} u_i`) or ``naive`` (ablation).
- ``loss_reweight``: ``null`` or ``one_minus_G`` (hard routing only;
  ``position_only`` is a deprecated alias for the default).
- ``centroid_mode``: ``batch`` or ``expected_pool``; ``blend``: EMA weight
  of the position step.
- ``sft_merge_groups``: list of ``{train_as, names}`` groups sharing one
  LoRA, or ``from_init``; ``snap_collapsed_pairs`` /
  ``collapse_merge_threshold`` snap co-located partners together.
- ``save_per_round``, ``val_eval``, ``sft`` (overlaid on the top-level
  ``sft`` block), ``n_rounds`` / ``batch_size`` (only without
  ``data_split``).

See :func:`validate_routing_and_loss_modes` for the allowed combinations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from infl_ens.config import resolve_sft_block
from infl_ens.data.benchmarks import BenchmarkSplit
from infl_ens.data.trait_space import TraitSpace
from infl_ens.inflgame.router import InfluencerRouter, RouterAgent
from infl_ens.training.agent_init import resolve_agent_entries
from infl_ens.training.setup import (
    coords_for_prompts,
    init_agents,
    load_splits,
    make_trait_space,
    sigma_from_config,
    write_history,
    write_resolved_config,
)

# Module-level aliases: tests monkeypatch these two to stub out the data
# layer, so the loop looks them up here at call time.
_load_splits = load_splits
_make_trait_space = make_trait_space

#: Valid values for ``closed_loop.loss_reweight``.
#:
#: - ``None``: no per-query weighting; both the SFT loss and the
#:   centroid update use uniform weights.
#: - ``"one_minus_G"``: per-query weight :math:`w_m = 1 - G_i(b_m)`
#:   applied to BOTH the loss and the centroid update. Gradient-matched
#:   position drift; reduced LoRA ESS.
#: - ``"position_only"``: same weight applied to the centroid update
#:   ONLY; the SFT loss runs at unit weight. Decouples LoRA capability
#:   from trait-space drift, recovering full ESS while keeping the
#:   gradient-matched position update.
_VALID_LOSS_REWEIGHT_MODES: tuple[Optional[str], ...] = (
    None, "one_minus_G", "position_only",
)

#: Valid values for ``closed_loop.position_update``. ``theory_matched``
#: (default) always applies the centroid mass that makes the expected
#: trait-space drift proportional to the strategic gradient coefficient
#: :math:`G_i(1-G_i)`; ``naive`` keeps the historical uninstrumented centroid
#: (unweighted under hard routing, renormalised share under soft routing).
_VALID_POSITION_UPDATE_MODES: tuple[str, ...] = ("theory_matched", "naive")

#: Valid values for ``closed_loop.soft_loss`` (soft routing only).
#: ``weighted`` trains each assigned agent with its renormalised share as the
#: per-example loss weight; ``unit`` trains the top-k agents at unit weight.
_VALID_SOFT_LOSS_MODES: tuple[str, ...] = ("weighted", "unit")

#: Valid values for ``closed_loop.soft_select`` (soft routing only).
#: ``topk`` keeps the ``soft_top_k`` largest shares per query (argmax gate);
#: ``sample`` draws ``soft_top_k`` distinct units without replacement from the
#: shares (:func:`infl_ens.inflgame.router.allocation.sampled_top_k_mask`).
_VALID_SOFT_SELECT_MODES: tuple[str, ...] = ("topk", "sample")


def validate_routing_and_loss_modes(
    routing_weight: str,
    loss_reweight: Optional[str],
    *,
    routing_mode: str = "hard",
    soft_top_k: int = 1,
    n_agents: Optional[int] = None,
    has_merge_groups: bool = False,
    n_groups: Optional[int] = None,
    position_update: str = "theory_matched",
    soft_loss: str = "weighted",
    soft_select: str = "topk",
    centroid_mode: str = "batch",
) -> None:
    """Validate combinations of the closed-loop rule knobs.

    Also validates the soft (dense) routing knobs. ``routing_mode='soft'``
    assigns each query to its top-``soft_top_k`` agents by allocation share
    and trains them either share-weighted or at unit weight
    (``soft_loss``). It requires the canonical routing weight ``'G'`` and
    carries its own loss weighting, so it is mutually exclusive with
    ``loss_reweight``; the centroid side is chosen by ``position_update``
    in both routing modes.

    With fixed ``sft_merge_groups`` the merge group replaces the clone as
    the routing unit, so ``soft_top_k`` counts groups rather than agents.

    :param routing_mode: ``'hard'`` (sample one agent per query) or
        ``'soft'`` (dense, top-k weighted).
    :type routing_mode: str
    :param soft_top_k: Number of agents (or merge groups) each query trains
        under soft mode.
    :type soft_top_k: int
    :param n_agents: Agent count, used to bound ``soft_top_k``.
    :type n_agents: int | None
    :param has_merge_groups: Whether ``sft_merge_groups`` is configured.
    :type has_merge_groups: bool
    :param n_groups: Merge-group count, used to bound ``soft_top_k`` when
        merge groups are configured.
    :type n_groups: int | None
    :param position_update: Value of ``closed_loop.position_update``.
    :type position_update: str
    :param soft_loss: Value of ``closed_loop.soft_loss``.
    :type soft_loss: str
    :param centroid_mode: Value of ``closed_loop.centroid_mode``.
    :type centroid_mode: str

    The centroid mass applied by ``position_update='theory_matched'`` is

    +--------------+------------------+----------------------------------+
    | routing_mode | routing_weight   | centroid mass                    |
    +==============+==================+==================================+
    | ``hard``     | ``'G'``          | ``(1 - G_i)`` on routed prompts  |
    +--------------+------------------+----------------------------------+
    | ``hard``     | ``'G_times_1mG'``| uniform (routing carries (1-G))  |
    +--------------+------------------+----------------------------------+
    | ``soft``     | ``'G'``          | ``G_i (1 - G_i)`` over the whole |
    |              |                  | batch (not renormalised)         |
    +--------------+------------------+----------------------------------+

    while ``position_update='naive'`` gives the uniform centroid under hard
    routing and the renormalised top-k share under soft routing.

    The loss-side matrix under hard routing is

    +------------------+----------------------+--------------------------+
    | routing_weight   | loss_reweight        | semantics                |
    +==================+======================+==========================+
    | ``'G'``          | ``None``             | unit-weight loss         |
    +------------------+----------------------+--------------------------+
    | ``'G_times_1mG'``| ``None``             | strategic routing        |
    +------------------+----------------------+--------------------------+
    | ``'G'``          | ``'one_minus_G'``    | (1-G)-weighted loss      |
    |                  |                      | (reduced ESS)            |
    +------------------+----------------------+--------------------------+
    | ``'G'``          | ``'position_only'``  | deprecated alias for     |
    |                  |                      | ``None`` +               |
    |                  |                      | ``theory_matched``       |
    +------------------+----------------------+--------------------------+
    | ``'G_times_1mG'``| ``'one_minus_G'`` or | **rejected**: strategic  |
    |                  | ``'position_only'``  | routing already carries  |
    |                  |                      | the (1-G) factor; adding |
    |                  |                      | it on the loss weight    |
    |                  |                      | double-counts and breaks |
    |                  |                      | gradient alignment.      |
    +------------------+----------------------+--------------------------+

    and under soft routing ``loss_reweight`` must be ``None`` while
    ``soft_loss`` picks ``'weighted'`` (renormalised share) or ``'unit'``.

    :param routing_weight: Value of ``closed_loop.routing_weight``.
    :type routing_weight: str
    :param loss_reweight: Value of ``closed_loop.loss_reweight``; may be
        ``None``.
    :type loss_reweight: str | None
    :raises ValueError: For unknown values or the disallowed combinations.
    """
    if routing_weight not in ("G", "G_times_1mG"):
        raise ValueError(
            f"routing_weight must be 'G' or 'G_times_1mG', got {routing_weight!r}"
        )
    if loss_reweight not in _VALID_LOSS_REWEIGHT_MODES:
        raise ValueError(
            f"loss_reweight must be one of {_VALID_LOSS_REWEIGHT_MODES}, "
            f"got {loss_reweight!r}"
        )
    if position_update not in _VALID_POSITION_UPDATE_MODES:
        raise ValueError(
            "closed_loop.position_update must be one of "
            f"{_VALID_POSITION_UPDATE_MODES}, got {position_update!r}"
        )
    if soft_loss not in _VALID_SOFT_LOSS_MODES:
        raise ValueError(
            f"closed_loop.soft_loss must be one of {_VALID_SOFT_LOSS_MODES}, "
            f"got {soft_loss!r}"
        )
    if soft_select not in _VALID_SOFT_SELECT_MODES:
        raise ValueError(
            "closed_loop.soft_select must be one of "
            f"{_VALID_SOFT_SELECT_MODES}, got {soft_select!r}"
        )
    if loss_reweight == "position_only" and position_update == "naive":
        raise ValueError(
            "loss_reweight='position_only' is a deprecated alias for the "
            "gradient-matched centroid (position_update='theory_matched'); "
            "it contradicts position_update='naive'. Drop loss_reweight or "
            "set position_update='theory_matched'."
        )
    if centroid_mode == "expected_pool" and position_update == "naive":
        raise ValueError(
            "centroid_mode='expected_pool' is the gradient-matched "
            "expected-pool centroid; it has no naive variant. Use "
            "position_update='theory_matched' or centroid_mode='batch'."
        )
    if (
        loss_reweight in ("one_minus_G", "position_only")
        and routing_weight == "G_times_1mG"
    ):
        raise ValueError(
            f"loss_reweight={loss_reweight!r} applies a (1-G) factor on top "
            "of strategic routing (which already carries a (1-G) factor in "
            "p_i ∝ G_i(1-G_i)). The combination double-counts the factor "
            "and breaks gradient alignment. Pick one of: "
            "(a) routing_weight='G_times_1mG', loss_reweight=null  "
            "(strategic routing, approximate gradient match); "
            "(b) routing_weight='G', loss_reweight='one_minus_G'  "
            "(full reweight, exact gradient match, reduced ESS); "
            "(c) routing_weight='G', loss_reweight='position_only'  "
            "(decoupled: exact gradient match in position, full ESS in loss)."
        )
    if routing_mode not in ("hard", "soft"):
        raise ValueError(
            f"closed_loop.routing_mode must be 'hard' or 'soft', "
            f"got {routing_mode!r}"
        )
    if routing_mode != "soft" and soft_loss != "weighted":
        raise ValueError(
            f"closed_loop.soft_loss={soft_loss!r} applies only to "
            "routing_mode='soft'; hard routing always trains the sampled "
            "winner at unit weight (use loss_reweight for hard-mode loss "
            "weighting)."
        )
    if routing_mode != "soft" and soft_select != "topk":
        raise ValueError(
            f"closed_loop.soft_select={soft_select!r} applies only to "
            "routing_mode='soft'; hard routing already samples its single "
            "winner. For sampled multi-winner routing use "
            "routing_mode='soft' with soft_select='sample'."
        )
    if routing_mode == "soft":
        if routing_weight != "G":
            raise ValueError(
                "routing_mode='soft' requires routing_weight='G'; the soft "
                "per-query weights already are the renormalised G_i shares."
            )
        if loss_reweight is not None:
            raise ValueError(
                "routing_mode='soft' carries its own loss weighting "
                f"(closed_loop.soft_loss); loss_reweight must be null, got "
                f"{loss_reweight!r}. The centroid side is chosen by "
                "closed_loop.position_update ('theory_matched' = dense "
                "G_i(1-G_i) mass over the batch, 'naive' = renormalised share)."
            )
        if soft_top_k < 1:
            raise ValueError(
                f"closed_loop.soft_top_k must be >= 1, got {soft_top_k}"
            )
        if has_merge_groups and n_groups is not None:
            if soft_top_k > n_groups:
                raise ValueError(
                    f"closed_loop.soft_top_k={soft_top_k} exceeds the number "
                    f"of merge groups ({n_groups}); under soft routing with "
                    "sft_merge_groups the group is the routing unit."
                )
        elif n_agents is not None and soft_top_k > n_agents:
            raise ValueError(
                f"closed_loop.soft_top_k={soft_top_k} exceeds the number of "
                f"agents ({n_agents})."
            )


# Backwards-compatible private name.
_validate_routing_and_loss_modes = validate_routing_and_loss_modes


def init_agents_closed_loop(
    cfg: dict[str, Any],
    space: TraitSpace,
    splits: list[BenchmarkSplit],
    cl: dict[str, Any],
    *,
    sigma: float,
    rng: Optional[np.random.Generator],
) -> tuple[list[RouterAgent], Optional[dict[str, Any]]]:
    """Initialize agents for the closed loop.

    :param cfg: Full training config.
    :type cfg: dict
    :param space: Trait space.
    :type space: TraitSpace
    :param splits: Benchmark splits.
    :type splits: list[BenchmarkSplit]
    :param cl: ``closed_loop`` config block.
    :type cl: dict
    :param sigma: Absolute competitive reach for the theory solve.
    :type sigma: float
    :param rng: RNG for mean-noise init.
    :type rng: numpy.random.Generator | None
    :returns: Router agents and optional theory-init metadata.
    :rtype: tuple[list[RouterAgent], dict | None]
    """
    init_noise = float(cl.get("init_noise", 0.0))
    init_mode = str(cl.get("init_mode", "mean_noise"))
    if init_mode == "mean_noise":
        return init_agents(cfg, space, splits, init_noise=init_noise, rng=rng), None

    if init_mode == "theory_gradient":
        from infl_ens.training.agent_init import init_agents_theory_gradient

        seed = int(cfg.get("seed", 0))
        agents, meta = init_agents_theory_gradient(
            cfg,
            space,
            sigma=sigma,
            seed=seed,
            init_noise=init_noise,
            theory_cfg=cl.get("theory_gradient"),
        )
        log_meta = {
            "init_mode": "theory_gradient",
            "theory_layout": meta["layout"],
            "theory_converged": meta["converged"],
            "theory_n_steps": meta["n_steps"],
            "theory_final_spread": meta["final_spread"],
            "theory_initial": meta["initial"].tolist(),
            "theory_end": meta["theory_end"].tolist(),
        }
        return agents, log_meta

    if init_mode == "theory_gradient_paired":
        from infl_ens.training.agent_init import init_agents_theory_gradient_paired

        seed = int(cfg.get("seed", 0))
        agents, meta = init_agents_theory_gradient_paired(
            cfg,
            space,
            sigma=sigma,
            seed=seed,
            init_noise=init_noise,
            theory_cfg=cl.get("theory_gradient"),
        )
        log_meta = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in meta.items()
            if k not in ("initial",)
        }
        if "initial" in meta:
            log_meta["theory_initial"] = meta["initial"].tolist()
        log_meta["theory_end"] = np.asarray(meta["theory_end"]).tolist()
        return agents, log_meta

    raise ValueError(
        "closed_loop.init_mode must be mean_noise, theory_gradient, or "
        f"theory_gradient_paired, got {init_mode!r}",
    )


def run_closed_loop(cfg: dict[str, Any]) -> int:
    """Alternating route → SFT → position update across ``rounds``.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :returns: Exit code.
    :rtype: int
    """
    from infl_ens.training.merge_training import (
        closed_loop_weight_args,
        get_merge_train_agent,
        group_index_for_merge_groups,
        merge_groups_from_theory_pairs,
        merge_mode_from_config,
        merge_routed_batch,
        parse_sft_merge_groups,
        snap_configured_merge_pairs,
        soft_pair_assignments,
        soft_pair_position_target,
    )
    from infl_ens.training.sft_training import SFTTrainingConfig, sft_train_agent

    splits = _load_splits(cfg)
    space = _make_trait_space(cfg, splits)
    cfg["agents"] = resolve_agent_entries(cfg.get("agents"), space.L)
    cl = cfg.get("closed_loop", {})
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    n_agents = len(cfg.get("agents", []))
    sigma = sigma_from_config(cfg, n_agents, space)
    agents, theory_init_meta = init_agents_closed_loop(
        cfg, space, splits, cl, sigma=sigma, rng=rng,
    )
    if theory_init_meta is not None:
        layout = theory_init_meta.get(
            "theory_layout",
            theory_init_meta.get("theory_layout_paired_refine"),
        )
        print(
            f"theory init ({theory_init_meta.get('init_mode', '?')}): "
            f"layout={layout} "
            f"converged={theory_init_meta.get('theory_converged')} "
            f"steps={theory_init_meta.get('theory_n_steps')}",
        )
        if "paired_harm_order" in theory_init_meta:
            # After co-location partners coincide by construction; the
            # honest measure of the pairing is how far apart they were
            # BEFORE each co-location step.
            first = theory_init_meta.get("within_pair_distance_first_pass") or {}
            second = (
                theory_init_meta.get("within_pair_distance_second_pass") or {}
            )
            print(
                f"  pairing={theory_init_meta.get('pairing_method', '?')} "
                f"pairs={theory_init_meta['paired_harm_order']}",
            )
            print(
                "  within-pair L2 before co-location: "
                f"max_first_pass="
                f"{max(first.values()) if first else float('nan'):.4f} "
                f"max_second_pass="
                f"{max(second.values()) if second else float('nan'):.4f}",
            )

    # ``sft_merge_groups: from_init`` derives one merge group per co-located
    # theory pair, so a run can set the population size and the adapter
    # partition from the trait space alone (pairs = axes).
    if cl.get("sft_merge_groups") == "from_init":
        if not theory_init_meta or "paired_harm_order" not in theory_init_meta:
            raise ValueError(
                "closed_loop.sft_merge_groups='from_init' requires a paired "
                "theory initialization (closed_loop.init_mode="
                "'theory_gradient_paired'), which supplies paired_harm_order",
            )
        cl["sft_merge_groups"] = merge_groups_from_theory_pairs(
            theory_init_meta["paired_harm_order"],
            name_prefix=str(cl.get("merge_group_prefix", "pair")),
        )
        theory_init_meta["sft_merge_groups_resolved"] = cl["sft_merge_groups"]
        print(
            "merge groups from theory init: "
            + ", ".join(
                f"{g['train_as']}<-{g['names']}" for g in cl["sft_merge_groups"]
            ),
        )

    collapse_merge_threshold = float(cl.get("collapse_merge_threshold", 0.01))
    router_names = [a.name for a in agents]
    sft_merge_mode = merge_mode_from_config(cl)
    static_merge_groups = (
        parse_sft_merge_groups(cl, router_names)
        if sft_merge_mode == "fixed" else None
    )
    if static_merge_groups and cl.get("snap_collapsed_pairs"):
        snap_meta = snap_configured_merge_pairs(
            agents,
            static_merge_groups,
            threshold=collapse_merge_threshold,
        )
        if theory_init_meta is None:
            theory_init_meta = {}
        theory_init_meta["merge_pair_snap"] = snap_meta
        snapped = [k for k, v in snap_meta.items() if v.get("snapped")]
        if snapped:
            print(
                "snapped co-located merge pairs to shared centroids: "
                + ", ".join(snapped),
            )

    router = InfluencerRouter(
        space, agents, sigma=sigma,
        policy=cfg.get("policy", "proportional"),
    )

    routing_weight = str(cl.get("routing_weight", "G"))
    # Loss-side weighting under hard routing. See
    # ``_VALID_LOSS_REWEIGHT_MODES`` and ``_validate_routing_and_loss_modes``
    # for the full semantics. 'one_minus_G' applies (1-G_i) to the SFT
    # loss (reduced ESS). The deprecated 'position_only' value is an alias
    # for a unit-weight loss plus the gradient-matched centroid, which is
    # now the default `position_update`, so it is folded away here.
    loss_reweight = cl.get("loss_reweight", None)
    # Centroid mass of the position step. 'theory_matched' (default) makes
    # the expected trait-space drift proportional to the strategic
    # gradient coefficient G_i(1-G_i) in every routing mode; 'naive' keeps
    # the historical uninstrumented centroid as an ablation arm.
    position_update = str(cl.get("position_update", "theory_matched"))
    centroid_mode = str(cl.get("centroid_mode", "batch"))
    if loss_reweight == "position_only":
        if str(cl.get("position_update", "theory_matched")) == "naive":
            raise ValueError(
                "loss_reweight='position_only' contradicts "
                "position_update='naive'; drop one of them."
            )
        print(
            "closed_loop.loss_reweight='position_only' is deprecated: it is "
            "the default (loss_reweight: null, position_update: "
            "theory_matched).",
        )
        loss_reweight = None
        position_update = "theory_matched"
    # Soft (dense) routing: assign every query to its top-`soft_top_k`
    # agents by allocation share, bounding the ~Nx SFT cost.
    # routing_mode='hard' (default) preserves the original sample-one-agent
    # behaviour. `soft_loss` picks share-weighted vs unit-weight training
    # of the assigned agents.
    routing_mode = str(cl.get("routing_mode", "hard"))
    soft_top_k = int(cl.get("soft_top_k", 1))
    soft_loss = str(cl.get("soft_loss", "weighted"))
    # How the top-k set is chosen: 'topk' takes the k largest shares,
    # 'sample' draws k distinct units per query without replacement from
    # those shares (hard routing's draw, generalised past one winner).
    soft_select = str(cl.get("soft_select", "topk"))
    validate_routing_and_loss_modes(
        routing_weight,
        loss_reweight,
        routing_mode=routing_mode,
        soft_top_k=soft_top_k,
        n_agents=len(agents),
        has_merge_groups=static_merge_groups is not None,
        n_groups=(
            len(static_merge_groups) if static_merge_groups is not None else None
        ),
        position_update=position_update,
        soft_loss=soft_loss,
        soft_select=soft_select,
        centroid_mode=centroid_mode,
    )
    # Whether hard canonical routing needs the per-query (1-G) weights this
    # round: for the loss (one_minus_G) and/or the matched centroid.
    hard_needs_one_minus_g = routing_mode != "soft" and (
        loss_reweight == "one_minus_G"
        or (position_update == "theory_matched" and routing_weight == "G")
    )
    # Soft routing over merge groups: the group is the routing, training and
    # position-update unit. Requires equal-size groups whose members are
    # co-located, because only then does the summed member allocation equal
    # the allocation of one agent at the shared position.
    soft_pairs = routing_mode == "soft" and static_merge_groups is not None
    if soft_pairs:
        assert static_merge_groups is not None
        sizes = {len(members) for _t, members in static_merge_groups}
        if sizes != {2}:
            raise ValueError(
                "routing_mode='soft' with sft_merge_groups requires every "
                f"group to hold exactly two agents, got sizes {sorted(sizes)}"
            )
        by_name_check = {a.name: a for a in agents}
        drifted = {
            train_as: float(
                np.linalg.norm(
                    by_name_check[members[0]].position
                    - by_name_check[members[1]].position
                )
            )
            for train_as, members in static_merge_groups
        }
        worst = max(drifted, key=drifted.get)
        if drifted[worst] > collapse_merge_threshold:
            raise ValueError(
                "routing_mode='soft' with sft_merge_groups requires "
                f"co-located partners; {worst} members are "
                f"{drifted[worst]:.4f} apart (> collapse_merge_threshold="
                f"{collapse_merge_threshold}). Use "
                "init_mode='theory_gradient_paired' with init_noise: 0.0, or "
                "snap_collapsed_pairs: true."
            )

    save_per_round = bool(cl.get("save_per_round", False))
    position_step = cl.get("position_step")
    blend_base = float(cl.get("blend", 0.5))
    blend_schedule = cl.get("blend_schedule")
    blend_start = cl.get("blend_start")
    if centroid_mode not in ("batch", "expected_pool"):
        raise ValueError(
            f"closed_loop.centroid_mode must be 'batch' or 'expected_pool', "
            f"got {centroid_mode!r}"
        )
    if centroid_mode == "expected_pool" and routing_weight != "G":
        raise ValueError(
            "centroid_mode='expected_pool' requires routing_weight='G'"
        )
    if (
        routing_mode == "soft"
        and centroid_mode == "expected_pool"
        and position_update != "theory_matched"
    ):
        raise ValueError(
            "routing_mode='soft' with centroid_mode='expected_pool' is the "
            "gradient-matched pool centroid; it requires "
            "position_update='theory_matched'."
        )

    data_split_cfg = cfg.get("data_split")
    split_manifest = None
    val_splits: list[BenchmarkSplit] = []
    train_batch_indices: list[np.ndarray] | None = None
    if data_split_cfg:
        from infl_ens.training.data_split import (
            partitioned_splits_for_eval,
            resolve_closed_loop_data_split,
            shuffled_train_batch_indices,
        )

        repo_root = (
            Path(cfg["repo_root"])
            if "repo_root" in cfg
            else Path(__file__).resolve().parents[3]
        )
        (
            split_manifest,
            train_prompts,
            train_responses,
            pool_prompts,
            _pool_responses,
            batch_size,
            n_rounds,
        ) = resolve_closed_loop_data_split(cfg, splits, repo_root=repo_root)
        train_batch_indices = shuffled_train_batch_indices(
            len(train_prompts), batch_size, n_rounds, rng,
        )
        val_splits = partitioned_splits_for_eval(
            splits, split_manifest, "val",
        )
        all_prompts = train_prompts
        all_responses = train_responses
        print(
            f"data split: train={len(train_prompts)} val={split_manifest.n_val} "
            f"test={split_manifest.n_test} pool={len(pool_prompts)} "
            f"batch_size={batch_size} n_rounds={n_rounds}",
        )
    else:
        n_rounds = int(cl.get("n_rounds", 5))
        batch_size = int(cl.get("batch_size", 256))
        all_prompts = [p for s in splits for p in s.prompts]
        all_responses = [
            r for s in splits for r in (s.responses or [""] * s.n)
        ]
        pool_prompts = all_prompts

    if static_merge_groups is not None:
        from infl_ens.training.pool_dynamics import agent_pairwise_geometry

        if theory_init_meta is None:
            theory_init_meta = {}
        positions = np.stack([a.position for a in agents], axis=0)
        names = [a.name for a in agents]
        geom = agent_pairwise_geometry(
            positions, names, merge_groups=static_merge_groups,
        )
        theory_init_meta["agent_geometry"] = {
            "geometry_phase": "post_init",
            **geom,
        }
        print(
            "post-init geometry: "
            f"within_merge={geom.get('within_merge_l2', {})}",
        )

    val_eval_cfg = cl.get("val_eval") or {}
    val_eval_every = int(val_eval_cfg.get("every_n_rounds", 0))
    val_eval_agents = val_eval_cfg.get("agents")
    val_eval_max_records = val_eval_cfg.get("max_eval_records")

    # Base-model + LoRA settings: top-level ``sft`` (model fragment) overlaid
    # by ``closed_loop.sft``; the merged block is what resolved_config.yaml
    # records so every later stage sees the same base model.
    sft_cfg_dict = resolve_sft_block(cfg)
    cl["sft"] = dict(sft_cfg_dict)
    sft_cfg = SFTTrainingConfig(**sft_cfg_dict)
    sft_base_output_dir = Path(sft_cfg.output_dir)
    merge_train_registry: dict[str, RouterAgent] = {}
    use_router_only_sft = sft_merge_mode == "fixed"
    print(
        f"position update: {position_update} "
        f"(routing_mode={routing_mode}, routing_weight={routing_weight}, "
        f"loss_reweight={loss_reweight}"
        + (
            f", soft_loss={soft_loss}, soft_select={soft_select}"
            if routing_mode == "soft" else ""
        )
        + ")",
    )
    if soft_pairs:
        assert static_merge_groups is not None
        unit = (
            "one adapter per pair, independent per-clone position steps"
            if position_update == "theory_matched"
            else "one adapter and one shared position step per pair"
        )
        print(
            f"soft routing over pairs: top_k={soft_top_k} of "
            f"{len(static_merge_groups)} groups, soft_loss={soft_loss}; "
            f"{unit}: "
            + ", ".join(f"{t}<-{m}" for t, m in static_merge_groups),
        )
    elif static_merge_groups:
        print(
            "pair-merge SFT only (routing + position updates stay per-clone): "
            + ", ".join(f"{t}<-{m}" for t, m in static_merge_groups),
        )

    from infl_ens.training.position_step import (
        apply_position_update,
        blend_for_round,
        expected_pool_centroid,
    )

    history_path = (
        Path(cfg.get("output_dir", "results/closed_loop")) / "history.json"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path = write_resolved_config(
        cfg, history_path.parent / "resolved_config.yaml",
    )
    if resolved_path is not None:
        print(f"resolved config: {resolved_path}")
    if split_manifest is not None:
        split_meta_path = history_path.parent / "data_split.json"
        with split_meta_path.open("w", encoding="utf-8") as fh:
            json.dump(split_manifest.to_dict(), fh, indent=2)
        print(f"data split manifest: {split_meta_path}")

    # The projector re-encodes on every call, and the pool is the same every
    # round; project it once and serve batch coordinates from the cache.
    pool_coords = space.project(pool_prompts)
    coord_by_text: dict[str, np.ndarray] = {
        text: pool_coords[i] for i, text in enumerate(pool_prompts)
    }

    by_name: dict[str, RouterAgent] = {a.name: a for a in agents}
    group_index: Optional[np.ndarray] = None
    if soft_pairs:
        assert static_merge_groups is not None
        group_index = group_index_for_merge_groups(
            static_merge_groups, router_names,
        )

    history: list[dict[str, Any]] = []
    for r in range(n_rounds):
        blend_r = blend_for_round(
            r, n_rounds, blend_base, blend_schedule, blend_start=blend_start,
        )
        if train_batch_indices is not None:
            idx = train_batch_indices[r]
            batch_prompts = [all_prompts[int(i)] for i in idx]
            batch_responses = [all_responses[int(i)] for i in idx]
        else:
            idx = rng.integers(0, len(all_prompts), size=batch_size)
            batch_prompts = [all_prompts[i] for i in idx]
            batch_responses = [all_responses[i] for i in idx]
        choices = router.route_batch(
            batch_prompts, rng=rng, routing_weight=routing_weight,
        )

        # Pre-compute G_i(b) over the batch once when ANY (1-G) factor
        # is in play. Both 'one_minus_G' and 'position_only' need it;
        # the difference between them is only in which of sample_weights
        # or eval_weights ultimately receives the values.
        G_batch: Optional[np.ndarray] = None
        name_to_idx: dict[str, int] = {}
        if hard_needs_one_minus_g:
            from infl_ens.inflgame.router.allocation import allocation_weights
            batch_coords = coords_for_prompts(
                batch_prompts, coord_by_text, space.project,
            )                                                       # (M, L)
            G_batch = allocation_weights(
                router.positions, batch_coords, router.cov,
            )                                                       # (N, M)
            name_to_idx = {a.name: i for i, a in enumerate(agents)}

        # Soft (dense) routing: every agent trains on each of its top-k
        # queries, share-weighted or at unit weight (`soft_loss`). This
        # bypasses the hard `choices` partition below; `choices` is still
        # sampled for the `observed_share` diagnostic in the history log.
        # `soft_pos_mass` carries every clone's theory-matched centroid
        # mass G_i(1-G_i) over the WHOLE batch: the position step is
        # per clone and independent of the top-k training gate (and of
        # merge groups).
        soft_weights: Optional[np.ndarray] = None
        soft_pos_mass: Optional[np.ndarray] = None
        pair_weight_matrix: Optional[np.ndarray] = None
        pair_idx: Optional[list[np.ndarray]] = None
        pair_weights: Optional[list[np.ndarray]] = None
        batch_coords_soft: Optional[np.ndarray] = None
        if routing_mode == "soft":
            from infl_ens.inflgame.router.allocation import (
                allocation_weights,
                matched_centroid_mass,
                top_k_allocation_weights,
            )
            batch_coords_soft = coords_for_prompts(
                batch_prompts, coord_by_text, space.project,
            )                                                       # (M, L)
            G_soft = allocation_weights(
                router.positions, batch_coords_soft, router.cov,
            )                                                       # (N, M)
            soft_pos_mass = matched_centroid_mass(G_soft)
            if soft_pairs:
                assert static_merge_groups is not None
                assert group_index is not None
                # Members of a group are co-located, so summing their
                # shares gives the allocation of one agent at the shared
                # position: the top-k TRAINING gate is taken over groups,
                # not clones. Positions still move per clone.
                pair_weight_matrix, pair_idx, pair_weights = (
                    soft_pair_assignments(
                        G_soft,
                        group_index,
                        len(static_merge_groups),
                        soft_top_k,
                        select=soft_select,
                        rng=rng,
                    )
                )                                                   # (P, M)
            elif soft_select == "sample":
                from infl_ens.inflgame.router.allocation import (
                    sampled_top_k_mask,
                )

                keep = sampled_top_k_mask(G_soft, soft_top_k, rng)
                masked = np.where(keep, G_soft, 0.0)
                col = masked.sum(axis=0, keepdims=True)
                safe = col > 0.0
                soft_weights = np.where(
                    safe, masked / np.where(safe, col, 1.0), 0.0,
                )
            else:
                soft_weights = top_k_allocation_weights(G_soft, soft_top_k)

        agent_prompts: dict[str, list[str]] = {a.name: [] for a in agents}
        agent_responses: dict[str, list[str]] = {a.name: [] for a in agents}
        agent_sft_logs: dict[str, list[dict[str, Any]]] = {
            a.name: [] for a in agents
        }
        agent_loaded_prior: dict[str, Optional[str]] = {
            a.name: None for a in agents
        }
        agent_blend_effective: dict[str, list[float]] = {
            a.name: [] for a in agents
        }
        # Per-agent per-query weights computed this round. Under hard
        # canonical routing these are the (1-G) weights (used for the loss
        # when ``loss_reweight == 'one_minus_G'`` and for the centroid when
        # ``position_update == 'theory_matched'``); under soft routing they
        # are the renormalised top-k shares, logged even when the loss ran
        # at unit weight (``soft_loss == 'unit'``) so per-query share totals
        # stay auditable. Empty list when nothing was computed.
        agent_sample_weights: dict[str, list[float]] = {
            a.name: [] for a in agents
        }
        # The centroid weights actually applied by the position step, when
        # they differ from ``agent_sample_weights`` (soft routing with the
        # theory-matched G(1-G) mass over the whole batch, aligned with
        # ``batch_prompts``). Keyed by training unit.
        agent_position_weights: dict[str, list[float]] = {}
        merge_sft_logs: dict[str, list[dict[str, Any]]] = {}
        merge_prompt_counts: dict[str, int] = {}
        merge_loaded_prior: dict[str, Optional[str]] = {}
        # Soft-pair bookkeeping: which batch rows each pair trained on, the
        # shared position it moved to, and its mean share of the batch.
        agent_batch_indices: dict[str, list[int]] = {}
        pair_positions: dict[str, list[float]] = {}
        pair_blend_effective: dict[str, float] = {}
        pair_share_batch: dict[str, float] = {}
        active_merge_groups: Optional[list[tuple[str, list[str]]]] = None
        for i_agent, agent in enumerate(agents):
            if soft_pairs:
                # The pair, not the clone, routes and trains; the per-pair
                # block below owns SFT and the single shared position step.
                break
            if routing_mode == "soft":
                assert soft_weights is not None
                mine_idx_soft = np.flatnonzero(soft_weights[i_agent] > 0.0)
                mine_p = [batch_prompts[int(m)] for m in mine_idx_soft]
                mine_r = [batch_responses[int(m)] for m in mine_idx_soft]
            else:
                mine_p = [
                    q for q, c in zip(batch_prompts, choices)
                    if c.name == agent.name
                ]
                mine_r = [
                    t for t, c in zip(batch_responses, choices)
                    if c.name == agent.name
                ]
            agent_prompts[agent.name] = list(mine_p)
            agent_responses[agent.name] = list(mine_r)
            if not mine_p:
                continue

            weights_i: Optional[list[float]] = None
            if routing_mode == "soft":
                assert soft_weights is not None
                # Renormalised shares are always logged; the loss uses them
                # only under soft_loss='weighted', and the centroid target
                # (via eval_weights → scores) uses them only under
                # position_update='naive'. The theory-matched position step
                # uses the whole batch and is applied in the dense block
                # after this loop, so SFT skips its own position update.
                share_i = soft_weights[i_agent, mine_idx_soft].tolist()
                weights_i = share_i
                agent_sample_weights[agent.name] = list(share_i)
                sample_weights_arg: Optional[list[float]] = (
                    None if soft_loss == "unit" else list(share_i)
                )
                if position_update == "theory_matched":
                    eval_weights_arg: Optional[list[float]] = None
                    skip_pos = True
                else:
                    eval_weights_arg = list(share_i)
                    skip_pos = False
            else:
                if hard_needs_one_minus_g:
                    mine_idx = [
                        m for m, c in enumerate(choices) if c.name == agent.name
                    ]
                    assert G_batch is not None
                    i_idx = name_to_idx[agent.name]
                    weights_i = (1.0 - G_batch[i_idx, mine_idx]).tolist()
                    agent_sample_weights[agent.name] = list(weights_i)

                sample_weights_arg, eval_weights_arg, skip_pos = (
                    closed_loop_weight_args(
                        loss_reweight, centroid_mode, weights_i,
                        position_update=position_update,
                    )
                )

            if use_router_only_sft:
                if not skip_pos:
                    blend_eff = agent.update_position_from_corpus(
                        list(mine_p),
                        space.project,
                        scores=(
                            list(eval_weights_arg)
                            if eval_weights_arg is not None else None
                        ),
                        blend=blend_r,
                        position_step=position_step,
                    )
                    agent_blend_effective[agent.name].append(float(blend_eff))
                continue

            out_override = (
                str(sft_base_output_dir / agent.name / f"round-{r:02d}")
                if save_per_round else None
            )
            sft_result = sft_train_agent(
                agent,
                prompts=mine_p,
                responses=mine_r if any(mine_r) else None,
                cfg=sft_cfg,
                eval_prompts=mine_p,
                project=space.project,
                blend=blend_r,
                position_step=position_step,
                out_dir_override=out_override,
                sample_weights=sample_weights_arg,
                eval_weights=eval_weights_arg,
                skip_position_update=skip_pos,
            )
            agent_sft_logs[agent.name] = sft_result.get("log_history", [])
            agent_loaded_prior[agent.name] = sft_result.get("loaded_prior_lora")
            if not skip_pos and "position_blend_effective" in sft_result:
                agent_blend_effective[agent.name].append(
                    float(sft_result["position_blend_effective"])
                )

        from infl_ens.inflgame.router.allocation import (
            allocation_weights,
            empirical_utility,
            strategic_routing_weights,
        )

        if centroid_mode == "expected_pool" and position_update == "theory_matched":
            G_pool = allocation_weights(
                router.positions, pool_coords, router.cov,
            )
            for i, agent in enumerate(agents):
                target = expected_pool_centroid(i, pool_coords, G_pool)
                agent.position, beta_eff = apply_position_update(
                    agent.position,
                    target,
                    blend=blend_r,
                    position_step=position_step,
                )
                agent_blend_effective[agent.name].append(float(beta_eff))

        if (
            routing_mode == "soft"
            and position_update == "theory_matched"
            and centroid_mode == "batch"
        ):
            # Dense theory-matched position step: every clone moves toward
            # the G_i(1-G_i)-weighted centroid of the WHOLE batch, whether
            # or not it trained on a given query, and independently of any
            # merge group. Nothing is sampled under soft routing, so this
            # mass is the full strategic-gradient coefficient and the
            # expected drift is exactly parallel to grad u_i (no top-k
            # truncation, no per-query renormaliser). Co-located partners
            # have identical G_i rows and therefore take identical steps:
            # pairs persist because the theory says so, not by fiat.
            assert soft_pos_mass is not None and batch_coords_soft is not None
            all_rows = np.arange(len(batch_prompts))
            for i_agent, agent in enumerate(agents):
                mass_i = soft_pos_mass[i_agent]
                agent_position_weights[agent.name] = mass_i.tolist()
                target = soft_pair_position_target(
                    batch_coords_soft, all_rows, mass_i,
                )
                agent.position, beta_eff = apply_position_update(
                    agent.position,
                    target,
                    blend=blend_r,
                    position_step=position_step,
                )
                agent_blend_effective[agent.name].append(float(beta_eff))

        if soft_pairs:
            assert static_merge_groups is not None
            assert pair_idx is not None and pair_weights is not None
            assert pair_weight_matrix is not None
            assert batch_coords_soft is not None
            active_merge_groups = static_merge_groups
            for i_pair, (train_name, members) in enumerate(
                static_merge_groups,
            ):
                idx_p = pair_idx[i_pair]
                w_p = pair_weights[i_pair]
                mine_p = [batch_prompts[int(m)] for m in idx_p]
                mine_r = [batch_responses[int(m)] for m in idx_p]
                agent_prompts[train_name] = list(mine_p)
                agent_responses[train_name] = list(mine_r)
                agent_sample_weights[train_name] = [float(x) for x in w_p]
                agent_batch_indices[train_name] = [int(m) for m in idx_p]
                merge_prompt_counts[train_name] = len(mine_p)
                pair_share_batch[train_name] = float(
                    pair_weight_matrix[i_pair].mean()
                )
                merge_agent = get_merge_train_agent(
                    merge_train_registry, train_name, members, agents,
                )
                if mine_p:
                    out_override = (
                        str(sft_base_output_dir / train_name / f"round-{r:02d}")
                        if save_per_round else None
                    )
                    sft_result = sft_train_agent(
                        merge_agent,
                        prompts=mine_p,
                        responses=mine_r if any(mine_r) else None,
                        cfg=sft_cfg,
                        eval_prompts=mine_p,
                        project=space.project,
                        blend=blend_r,
                        position_step=position_step,
                        out_dir_override=out_override,
                        sample_weights=(
                            None if soft_loss == "unit"
                            else [float(x) for x in w_p]
                        ),
                        eval_weights=None,
                        skip_position_update=True,
                    )
                    merge_sft_logs[train_name] = sft_result.get("log_history", [])
                    merge_loaded_prior[train_name] = sft_result.get(
                        "loaded_prior_lora",
                    )
                    agent_sft_logs[train_name] = merge_sft_logs[train_name]
                    agent_loaded_prior[train_name] = merge_loaded_prior[train_name]
                if position_update == "theory_matched":
                    # Each member already took its own theory-matched step
                    # (dense batch block or expected-pool block above). The
                    # trainer's routing position simply tracks its members;
                    # nothing forces them together — co-location is an
                    # outcome the theory predicts, audited via the logged
                    # within-pair distance in `agent_geometry`.
                    member_pos = np.stack(
                        [by_name[m].position for m in members], axis=0,
                    )
                    merge_agent.position = member_pos.mean(axis=0)
                    pair_positions[train_name] = merge_agent.position.tolist()
                    betas = [
                        agent_blend_effective[m][-1]
                        for m in members
                        if agent_blend_effective[m]
                    ]
                    pair_blend_effective[train_name] = (
                        float(np.mean(betas)) if betas else float("nan")
                    )
                else:
                    # Historical naive arm: ONE renormalised-share step per
                    # pair on its top-k queries, written to both members.
                    if not mine_p:
                        continue
                    target = soft_pair_position_target(
                        batch_coords_soft, idx_p, w_p,
                    )
                    new_pos, beta_eff = apply_position_update(
                        merge_agent.position,
                        target,
                        blend=blend_r,
                        position_step=position_step,
                    )
                    for member in members:
                        by_name[member].position = new_pos.copy()
                        agent_blend_effective[member].append(float(beta_eff))
                    merge_agent.position = new_pos.copy()
                    pair_positions[train_name] = new_pos.tolist()
                    pair_blend_effective[train_name] = float(beta_eff)
        elif use_router_only_sft:
            active_merge_groups = static_merge_groups

            for train_name, members in active_merge_groups or []:
                merge_agent = get_merge_train_agent(
                    merge_train_registry,
                    train_name,
                    members,
                    agents,
                )
                mp, mr, merged_weights = merge_routed_batch(
                    agent_prompts,
                    agent_responses,
                    members,
                    agent_sample_weights=agent_sample_weights,
                    loss_reweight=loss_reweight,
                )
                merge_prompt_counts[train_name] = len(mp)
                if not mp:
                    continue
                m_sample_weights = (
                    merged_weights if loss_reweight == "one_minus_G" else None
                )
                out_override = (
                    str(sft_base_output_dir / train_name / f"round-{r:02d}")
                    if save_per_round else None
                )
                sft_result = sft_train_agent(
                    merge_agent,
                    prompts=mp,
                    responses=mr if any(mr) else None,
                    cfg=sft_cfg,
                    eval_prompts=mp,
                    project=space.project,
                    blend=blend_r,
                    position_step=position_step,
                    out_dir_override=out_override,
                    sample_weights=m_sample_weights,
                    eval_weights=None,
                    skip_position_update=True,
                )
                merge_sft_logs[train_name] = sft_result.get("log_history", [])
                merge_loaded_prior[train_name] = sft_result.get(
                    "loaded_prior_lora",
                )


        observed = np.array(
            [sum(1 for c in choices if c.name == a.name) / max(len(choices), 1)
             for a in agents]
        )
        strategic_share_pool = strategic_routing_weights(
            router.positions, pool_coords, router.cov,
        ).mean(axis=1)
        u_pool_round = empirical_utility(
            router.positions, pool_coords, router.cov,
        ).tolist()
        from infl_ens.training.pool_dynamics import agent_pairwise_geometry

        pos_stack = np.stack([a.position for a in agents], axis=0)
        round_geometry = agent_pairwise_geometry(
            pos_stack,
            router_names,
            merge_groups=static_merge_groups,
        )
        round_geometry["geometry_phase"] = f"round_{r}"
        history.append({
            "round": r,
            "positions": {a.name: a.position.tolist() for a in agents},
            "agent_geometry": round_geometry,
            "u_grid": router.expected_utilities().tolist(),
            "u_pool": u_pool_round,
            "strategic_share_pool": strategic_share_pool.tolist(),
            "observed_share": observed.tolist(),
            "routing_weight": routing_weight,
            "routing_mode": routing_mode,
            "soft_top_k": soft_top_k if routing_mode == "soft" else None,
            "soft_loss": soft_loss if routing_mode == "soft" else None,
            "soft_select": soft_select if routing_mode == "soft" else None,
            "loss_reweight": loss_reweight,
            "position_update": position_update,
            "agent_prompts": agent_prompts,
            "agent_responses": agent_responses,
            "agent_sft_logs": agent_sft_logs,
            # Per-agent per-query weights computed this round: (1-G) under
            # hard canonical routing, renormalised shares under soft
            # routing. Populated whenever they were computed, even if the
            # SFT loss itself ran at unit weight.
            "agent_sample_weights": agent_sample_weights,
            # Centroid weights actually applied when they differ from the
            # above: under soft theory-matched routing the dense G(1-G)
            # mass over the whole batch, aligned with `batch_prompts`.
            "agent_position_weights": agent_position_weights,
            **(
                {
                    "batch_prompts": list(batch_prompts),
                    "batch_responses": list(batch_responses),
                }
                if routing_mode == "soft" and not soft_pairs
                else {}
            ),
            "agent_loaded_prior": agent_loaded_prior,
            "agent_blend_effective": agent_blend_effective,
            "position_step": position_step,
            "blend_base": blend_base,
            "blend_round": blend_r,
            "centroid_mode": centroid_mode,
            "blend_schedule": blend_schedule,
            "save_per_round": save_per_round,
            **(
                {
                    "sft_merge_mode": sft_merge_mode,
                    "merge_sft_logs": merge_sft_logs,
                    "merge_prompt_counts": merge_prompt_counts,
                    "merge_loaded_prior": merge_loaded_prior,
                }
                if use_router_only_sft
                else {}
            ),
            **(
                {
                    "soft_routing_units": "pairs",
                    "pair_members": {
                        t: list(m) for t, m in static_merge_groups or []
                    },
                    "pair_positions": pair_positions,
                    "pair_blend_effective": pair_blend_effective,
                    "pair_share_batch": pair_share_batch,
                    "pair_u_pool": {
                        t: sum(
                            u_pool_round[router_names.index(name)]
                            for name in m
                        )
                        for t, m in static_merge_groups or []
                    },
                    "agent_batch_indices": agent_batch_indices,
                    "batch_prompts": list(batch_prompts),
                    "batch_responses": list(batch_responses),
                }
                if soft_pairs
                else {}
            ),
            **(
                {"theory_init": theory_init_meta}
                if r == 0 and theory_init_meta is not None
                else {}
            ),
            **(
                {
                    "data_split": {
                        "manifest_seed": split_manifest.seed,
                        "n_train": split_manifest.n_train,
                        "n_val": split_manifest.n_val,
                        "n_test": split_manifest.n_test,
                        "batch_size": batch_size,
                    },
                }
                if r == 0 and split_manifest is not None
                else {}
            ),
        })

        write_history(history_path, history)

        if (
            split_manifest is not None
            and val_eval_every > 0
            and val_splits
            and (r + 1) % val_eval_every == 0
        ):
            from infl_ens.training.closed_loop_eval import run_closed_loop_val_eval

            run_dir = Path(cfg.get("output_dir", "results/closed_loop"))
            run_closed_loop_val_eval(
                run_dir,
                val_splits,
                round_idx=r,
                sft_cfg=sft_cfg_dict,
                seed=int(cfg.get("seed", 0)),
                agents=val_eval_agents,
                max_eval_records=val_eval_max_records,
            )

    write_history(history_path, history)
    print(f"closed-loop done: {n_rounds} rounds, wrote {history_path}")

    eval_block = cfg.get("eval")
    if eval_block and bool(eval_block.get("after_training", True)):
        from infl_ens.evaluation.evaluate import run_unified_eval

        if split_manifest is None:
            print(
                "eval: skipped — the eval block needs data_split.manifest "
                "to define held-out partitions.",
            )
        else:
            reports = run_unified_eval(cfg, final_round=n_rounds - 1)
            for path in reports:
                print(f"eval: wrote {path}")
    return 0


__all__ = [
    "init_agents_closed_loop",
    "run_closed_loop",
    "validate_routing_and_loss_modes",
]
