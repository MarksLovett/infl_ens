"""Single CLI entry point for all training in ``infl_ens``.

Per AGENTS.md §4 rule 1, every fine-tune / training script is reached
through this one entry point, with behaviour controlled by a Hydra-style
config. The CLI accepts either:

- a YAML config path via ``--config configs/benchmark/router/safety_truth.yaml``
  with optional ``KEY=VAL`` overrides appended after ``--``,
- or ``--task <name>`` plus task-specific flags for ad-hoc use.

Supported tasks (selected by the ``task`` field of the config):

- ``router_training``: gradient-ascent on agent positions in a fixed
  trait space. Delegates to
  :func:`infl_ens.training.router_training.train_router_positions`.
- ``sft_training``: LoRA SFT of a single :class:`RouterAgent` on a
  benchmark split. Delegates to
  :func:`infl_ens.training.sft_training.sft_train_agent`.
- ``closed_loop``: alternating SFT + position re-estimation across rounds,
  using the influencer-game router to assign queries to agents.

Closed-loop honours three orthogonal "rule" knobs:

- ``closed_loop.routing_weight`` selects how queries are routed:
  ``'G'`` (canonical, Lovett & Fu 2024) or ``'G_times_1mG'`` (strategic).
- ``closed_loop.loss_reweight`` selects whether per-query weights are
  applied, and where:

  - ``null``: no weighting; loss is unit-weight CE, centroid is the
    unweighted mean of routed embeddings (naive canonical when paired
    with ``routing_weight='G'``).
  - ``'one_minus_G'``: per-query weight :math:`w_m = 1 - G_i(\\mathbf{x},
    b_m)` applied to BOTH the SFT loss and the centroid update.
    Gradient-aligned position drift; full ESS cost (most weight goes to
    contested-trait queries, near-stronghold queries get weight ≈ 0).
  - ``'position_only'``: per-query weight :math:`w_m = 1 - G_i(\\mathbf{x},
    b_m)` applied to ONLY the centroid update; the SFT loss runs at unit
    weight. Decouples LoRA capability training from trait-space drift —
    you get the gradient-aligned position update of ``one_minus_G`` plus
    the full ESS of the naive / strategic methods, because every routed
    query is trained at unit weight and contributes equally to the LoRA
    gradient.

- ``closed_loop.save_per_round`` selects per-round adapter archiving.
- ``closed_loop.init_noise``: std of Gaussian perturbation added to each
  clone's starting position around :math:`\\mathbb{E}_B[b]`. With
  ``init_noise=0`` (default) every agent starts at the identical resource
  mean, which makes symmetry-breaking path-dependent on the routing RNG
  alone. A small value (e.g. ``1e-4``) mirrors
  :mod:`scripts.closed_loop_demo` and stabilises bifurcation across
  sigma sweeps.
- ``closed_loop.init_mode``: ``mean_noise`` (default), ``pairs_near_theory``
  (jitter around a stored (2,2) reference), or ``theory_gradient`` (grid
  Nash gradient ascent from a random separated start, then place SFT agents
  at the theoretical endpoints; see :mod:`infl_ens.utils.agent_init`).
- ``closed_loop.theory_gradient``: hyperparameters when
  ``init_mode='theory_gradient'`` (``learning_rate``, ``n_steps``, ``tol``,
  ``min_pairwise``).
- ``closed_loop.theory_ref``: optional path to a ``history.json`` whose
  final positions define the reference (defaults: per-:math:`\\sigma`
  template under ``results/``).
- ``closed_loop.position_step``: adaptive EMA blend for position updates
  (see :mod:`infl_ens.utils.position_step`). Modes: ``static`` (fixed
  ``blend``), ``cap_linf``, ``cap_l2``, ``trust_box``.

Run with::

    python -m infl_ens.training --config configs/benchmark/router/safety_truth.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from infl_ens.data.benchmarks import (
    BenchmarkSplit,
    build_safety_trait_space,
    load_beavertails,
    load_halueval,
)
from infl_ens.data.encoders import SentenceTransformerEncoder
from infl_ens.data.trait_space import TraitSpace
from infl_ens.inflgame.router import InfluencerRouter, RouterAgent
from infl_ens.training.router_training import (
    RouterTrainingConfig,
    train_router_positions,
)
from infl_ens.utils.resource import gaussian_stability_threshold


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file using PyYAML if available, else a tiny fallback.

    :param path: Path to the YAML file.
    :type path: pathlib.Path
    :returns: Parsed mapping.
    :rtype: dict
    """
    try:
        import yaml
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:  # pragma: no cover
        out: dict[str, Any] = {}
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
        return out


def _apply_overrides(cfg: dict[str, Any], overrides: Sequence[str]) -> None:
    """Apply ``key.subkey=value`` overrides in place.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :param overrides: Sequence of dotted overrides.
    :type overrides: Sequence[str]
    """
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        path = key.split(".")
        node = cfg
        for p in path[:-1]:
            node = node.setdefault(p, {})
        try:
            node[path[-1]] = json.loads(val)
        except json.JSONDecodeError:
            node[path[-1]] = val


def _load_splits(cfg: dict[str, Any]) -> list[BenchmarkSplit]:
    """Load all benchmark splits referenced by ``cfg['benchmarks']``.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :returns: List of loaded :class:`BenchmarkSplit`.
    :rtype: list[BenchmarkSplit]
    """
    splits: list[BenchmarkSplit] = []
    for entry in cfg.get("benchmarks", []):
        kind = entry["kind"]
        path = entry["path"]
        max_records = entry.get("max_records")
        if kind == "beavertails":
            splits.append(load_beavertails(path, max_records=max_records))
        elif kind == "halueval":
            splits.append(load_halueval(
                path,
                tasks=entry.get("tasks"),
                max_records=max_records,
            ))
        else:  # pragma: no cover
            raise ValueError(f"unknown benchmark kind {kind!r}")
    return splits


def _make_trait_space(cfg: dict[str, Any], splits: list[BenchmarkSplit]) -> TraitSpace:
    """Build a :class:`TraitSpace` from benchmark splits and config.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :param splits: Already-loaded benchmark splits.
    :type splits: list[BenchmarkSplit]
    :returns: The constructed trait space.
    :rtype: TraitSpace
    """
    ts_cfg = cfg.get("trait_space", {})
    encoder_name = ts_cfg.get(
        "encoder", "sentence-transformers/all-MiniLM-L6-v2",
    )
    encoder = SentenceTransformerEncoder(model_name=encoder_name)
    return build_safety_trait_space(
        splits, encoder,
        n_grid=int(ts_cfg.get("n_grid", 32)),
        kde_bandwidth=ts_cfg.get("kde_bandwidth"),
        threshold=float(ts_cfg.get("threshold", 0.5)),
    )


def _init_agents(
    cfg: dict[str, Any],
    space: TraitSpace,
    splits: list[BenchmarkSplit],
    *,
    init_noise: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> list[RouterAgent]:
    """Initialise router agents from the config's ``agents`` list.

    When no per-agent ``calibration`` split is given, each agent's
    position defaults to the resource-weighted mean :math:`\\mathbb{E}_B[b]`
    plus optional i.i.d. Gaussian noise
    ``init_noise * \\mathcal{N}(0, I_L)`` to break exact clone symmetry.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :param space: Trait space defining the projector.
    :type space: TraitSpace
    :param splits: Already-loaded benchmark splits.
    :type splits: list[BenchmarkSplit]
    :param init_noise: Standard deviation of the initial position
        perturbation around ``space.mean``. Ignored for calibrated agents.
    :type init_noise: float
    :param rng: RNG used for the perturbation. Defaults to
        ``numpy.random.default_rng(cfg['seed'])`` when ``init_noise > 0``.
    :type rng: numpy.random.Generator | None
    :returns: List of :class:`RouterAgent` ready for routing/training.
    :rtype: list[RouterAgent]
    :raises ValueError: If ``init_noise < 0``.
    """
    if init_noise < 0.0:
        raise ValueError(f"init_noise must be >= 0, got {init_noise}")
    by_name = {s.name: s for s in splits}
    agents: list[RouterAgent] = []
    x0 = space.mean.copy()
    if init_noise > 0.0 and rng is None:
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
    for entry in cfg.get("agents", []):
        name = entry["name"]
        cal_name = entry.get("calibration")
        if cal_name and cal_name in by_name:
            agents.append(RouterAgent.from_calibration(
                name=name,
                calibration_queries=by_name[cal_name].prompts[:200],
                project=space.project,
            ))
        else:
            if init_noise > 0.0:
                assert rng is not None
                pos = x0 + init_noise * rng.standard_normal(space.L)
            else:
                pos = x0.copy()
            agents.append(RouterAgent(name=name, position=pos))
    return agents


def _init_agents_closed_loop(
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
    :param sigma: Absolute competitive reach (for reference resolution).
    :type sigma: float
    :param rng: RNG for mean-noise init.
    :type rng: numpy.random.Generator | None
    :returns: Router agents and optional theory-init metadata.
    :rtype: tuple[list[RouterAgent], dict | None]
    """
    init_noise = float(cl.get("init_noise", 0.0))
    init_mode = str(cl.get("init_mode", "mean_noise"))
    if init_mode == "mean_noise":
        return _init_agents(cfg, space, splits, init_noise=init_noise, rng=rng), None

    if init_mode == "theory_gradient":
        from infl_ens.utils.agent_init import init_agents_theory_gradient

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

    if init_mode == "pairs_near_theory":
        from infl_ens.utils.agent_init import (
            init_agents_pairs_near_reference,
            load_reference_from_history,
            resolve_theory_22_reference,
        )

        repo_root = Path(__file__).resolve().parents[3]
        ref_path = cl.get("theory_ref")
        if ref_path:
            ref_p = Path(ref_path)
            if not ref_p.is_absolute():
                ref_p = repo_root / ref_p
            reference = load_reference_from_history(ref_p)
        else:
            names = sorted(a["name"] for a in cfg.get("agents", []))
            sigma_frac = float(cfg.get("sigma_fraction", 0.5))
            reference = resolve_theory_22_reference(
                space,
                names,
                sigma=sigma,
                repo_root=repo_root,
                sigma_fraction=sigma_frac,
            )
        seed = int(cfg.get("seed", 0))
        return init_agents_pairs_near_reference(
            cfg, space, reference, seed=seed, init_noise=init_noise,
        ), None

    raise ValueError(
        f"closed_loop.init_mode must be mean_noise, pairs_near_theory, or "
        f"theory_gradient, got {init_mode!r}",
    )


def _sigma_from_cfg(
    cfg: dict[str, Any], n_agents: int, space: TraitSpace,
) -> float:
    """Resolve the competitive reach :math:`\\sigma` from the config.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :param n_agents: Number of agents.
    :type n_agents: int
    :param space: Trait space (for the stability threshold).
    :type space: TraitSpace
    :returns: Scalar :math:`\\sigma`.
    :rtype: float
    """
    mode = cfg.get("sigma_mode", "absolute")
    if mode == "absolute":
        return float(cfg["sigma"])
    if mode == "stability_fraction":
        frac = float(cfg.get("sigma_fraction", 0.8))
        s0 = gaussian_stability_threshold(n_agents, space.grid, space.weights)
        return frac * max(s0, 1e-3)
    raise ValueError(f"unknown sigma_mode {mode!r}")


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


def _validate_routing_and_loss_modes(
    routing_weight: str, loss_reweight: Optional[str],
) -> None:
    """Validate combinations of ``routing_weight`` and ``loss_reweight``.

    The matrix of valid combinations is

    +------------------+----------------------+--------------------------+
    | routing_weight   | loss_reweight        | semantics                |
    +==================+======================+==========================+
    | ``'G'``          | ``None``             | naive canonical          |
    +------------------+----------------------+--------------------------+
    | ``'G_times_1mG'``| ``None``             | strategic routing        |
    +------------------+----------------------+--------------------------+
    | ``'G'``          | ``'one_minus_G'``    | full reweight (loss +    |
    |                  |                      | centroid both weighted)  |
    +------------------+----------------------+--------------------------+
    | ``'G'``          | ``'position_only'``  | centroid weighted only;  |
    |                  |                      | SFT loss is unit-weight  |
    +------------------+----------------------+--------------------------+
    | ``'G_times_1mG'``| ``'one_minus_G'`` or | **rejected**: strategic  |
    |                  | ``'position_only'``  | routing already carries  |
    |                  |                      | the (1-G) factor; adding |
    |                  |                      | it on the centroid       |
    |                  |                      | weight double-counts and |
    |                  |                      | breaks gradient          |
    |                  |                      | alignment.               |
    +------------------+----------------------+--------------------------+

    :param routing_weight: Value of ``closed_loop.routing_weight``.
    :type routing_weight: str
    :param loss_reweight: Value of ``closed_loop.loss_reweight``; may be
        ``None``.
    :type loss_reweight: str | None
    :raises ValueError: For unknown values or the disallowed
        strategic-routing-plus-reweight combinations.
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


def _task_router_training(cfg: dict[str, Any]) -> int:
    """Run gradient-ascent training on agent positions.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :returns: Exit code.
    :rtype: int
    """
    splits = _load_splits(cfg)
    space = _make_trait_space(cfg, splits)
    agents = _init_agents(cfg, space, splits)
    sigma = _sigma_from_cfg(cfg, len(agents), space)

    train_cfg = cfg.get("training", {})
    rt_cfg = RouterTrainingConfig(
        sigma=sigma,
        learning_rate=float(train_cfg.get("learning_rate", 5e-3)),
        n_steps=int(train_cfg.get("n_steps", 5000)),
        tol=float(train_cfg.get("tol", 1e-8)),
        clip_to_box=bool(train_cfg.get("clip_to_box", True)),
    )
    info = train_router_positions(space, agents, rt_cfg, seed=int(cfg.get("seed", 0)))

    out = Path(cfg.get("output_dir", "results/router")) / "positions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "sigma": sigma,
                "converged": info["converged"],
                "n_steps": info["n_steps"],
                "positions": {a.name: a.position.tolist() for a in agents},
            },
            fh,
            indent=2,
        )
    print(f"router training: converged={info['converged']} in {info['n_steps']} steps")
    print(f"wrote {out}")
    return 0


def _task_sft_training(cfg: dict[str, Any]) -> int:
    """Run one LoRA SFT round for a single agent.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :returns: Exit code.
    :rtype: int
    """
    from infl_ens.training.sft_training import SFTTrainingConfig, sft_train_agent

    splits = _load_splits(cfg)
    space = _make_trait_space(cfg, splits)
    agents = _init_agents(cfg, space, splits)
    target_name = cfg["sft"]["agent"]
    agent = next(a for a in agents if a.name == target_name)
    split = next(s for s in splits if s.name == cfg["sft"]["benchmark"])

    sft_cfg_dict = dict(cfg.get("sft", {}))
    sft_cfg_dict.pop("agent", None)
    sft_cfg_dict.pop("benchmark", None)
    sft_cfg = SFTTrainingConfig(**sft_cfg_dict)

    result = sft_train_agent(
        agent,
        prompts=split.prompts,
        responses=split.responses or None,
        cfg=sft_cfg,
        eval_prompts=split.prompts[: min(256, split.n)],
        project=space.project,
    )
    print(f"sft training done: {json.dumps(result)}")
    return 0


def _task_closed_loop(cfg: dict[str, Any]) -> int:
    """Alternating route → SFT → position update across ``rounds``.

    :param cfg: Configuration dictionary.
    :type cfg: dict
    :returns: Exit code.
    :rtype: int
    """
    from infl_ens.training.sft_training import SFTTrainingConfig, sft_train_agent

    splits = _load_splits(cfg)
    space = _make_trait_space(cfg, splits)
    cl = cfg.get("closed_loop", {})
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    n_agents = len(cfg.get("agents", []))
    sigma = _sigma_from_cfg(cfg, n_agents, space)
    agents, theory_init_meta = _init_agents_closed_loop(
        cfg, space, splits, cl, sigma=sigma, rng=rng,
    )
    if theory_init_meta is not None:
        print(
            f"theory_gradient init: layout={theory_init_meta['theory_layout']} "
            f"converged={theory_init_meta['theory_converged']} "
            f"steps={theory_init_meta['theory_n_steps']}",
        )
    router = InfluencerRouter(
        space, agents, sigma=sigma,
        policy=cfg.get("policy", "proportional"),
    )

    n_rounds = int(cl.get("n_rounds", 5))
    batch_size = int(cl.get("batch_size", 256))
    routing_weight = str(cl.get("routing_weight", "G"))
    # Per-query weighting mode. See ``_VALID_LOSS_REWEIGHT_MODES`` and
    # ``_validate_routing_and_loss_modes`` for the full semantics.
    #
    # 'one_minus_G' applies (1-G_i) to BOTH the SFT loss and the
    # centroid update — gradient-matched position drift with reduced
    # ESS. 'position_only' applies (1-G_i) to the centroid update ONLY
    # while the SFT loss runs at unit weight — same gradient-matched
    # drift in trait space, full ESS in the LoRA, broader effective
    # training distribution (canonical, including strongholds at unit
    # weight).
    loss_reweight = cl.get("loss_reweight", None)
    _validate_routing_and_loss_modes(routing_weight, loss_reweight)

    save_per_round = bool(cl.get("save_per_round", False))
    position_step = cl.get("position_step")
    blend_base = float(cl.get("blend", 0.5))
    blend_schedule = cl.get("blend_schedule")
    blend_start = cl.get("blend_start")
    centroid_mode = str(cl.get("centroid_mode", "batch"))
    if centroid_mode not in ("batch", "expected_pool"):
        raise ValueError(
            f"closed_loop.centroid_mode must be 'batch' or 'expected_pool', "
            f"got {centroid_mode!r}"
        )
    if centroid_mode == "expected_pool" and routing_weight != "G":
        raise ValueError(
            "centroid_mode='expected_pool' requires routing_weight='G'"
        )
    all_prompts = [p for s in splits for p in s.prompts]
    all_responses = [r for s in splits for r in (s.responses or [""] * s.n)]
    sft_cfg_dict = dict(cl.get("sft", {}))
    sft_cfg = SFTTrainingConfig(**sft_cfg_dict)
    sft_base_output_dir = Path(sft_cfg.output_dir)

    from infl_ens.utils.position_step import (
        apply_position_update,
        blend_for_round,
        expected_pool_centroid,
    )

    history: list[dict[str, Any]] = []
    for r in range(n_rounds):
        blend_r = blend_for_round(
            r, n_rounds, blend_base, blend_schedule, blend_start=blend_start,
        )
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
        if loss_reweight in ("one_minus_G", "position_only"):
            from infl_ens.inflgame.router.allocation import allocation_weights
            batch_coords = space.project(batch_prompts)             # (M, L)
            G_batch = allocation_weights(
                router.positions, batch_coords, router.cov,
            )                                                       # (N, M)
            name_to_idx = {a.name: i for i, a in enumerate(agents)}

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
        # Per-agent per-query (1-G) weights computed this round. Used
        # for the loss when ``loss_reweight == 'one_minus_G'`` and for
        # the centroid when ``loss_reweight in {'one_minus_G',
        # 'position_only'}``. Empty list when ``loss_reweight is None``.
        agent_sample_weights: dict[str, list[float]] = {
            a.name: [] for a in agents
        }
        for agent in agents:
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

            # Build per-query (1-G) weights once; the loss_reweight
            # mode decides whether to apply them to sample_weights
            # (loss) or only to eval_weights (centroid).
            weights_i: Optional[list[float]] = None
            if loss_reweight in ("one_minus_G", "position_only"):
                mine_idx = [
                    m for m, c in enumerate(choices) if c.name == agent.name
                ]
                assert G_batch is not None  # established above
                i_idx = name_to_idx[agent.name]
                weights_i = (1.0 - G_batch[i_idx, mine_idx]).tolist()
                agent_sample_weights[agent.name] = list(weights_i)

            # Decide which weights to apply to the loss vs the centroid.
            if loss_reweight == "one_minus_G":
                sample_weights_arg = weights_i
                eval_weights_arg = weights_i
            elif loss_reweight == "position_only":
                # SFT loss is unit-weight (full ESS, broader training
                # distribution including strongholds); centroid is
                # (1-G)-weighted so the position update still matches
                # the strategic gradient direction in expectation.
                sample_weights_arg = None
                eval_weights_arg = (
                    None if centroid_mode == "expected_pool" else weights_i
                )
            else:  # loss_reweight is None
                sample_weights_arg = None
                eval_weights_arg = None

            out_override = (
                str(sft_base_output_dir / agent.name / f"round-{r:02d}")
                if save_per_round else None
            )
            skip_pos = (
                centroid_mode == "expected_pool"
                and loss_reweight in ("one_minus_G", "position_only")
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
        pool_coords = space.project(all_prompts)

        if (
            centroid_mode == "expected_pool"
            and loss_reweight in ("one_minus_G", "position_only")
        ):
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
        observed = np.array(
            [sum(1 for c in choices if c.name == a.name) / max(len(choices), 1)
             for a in agents]
        )
        strategic_share_pool = strategic_routing_weights(
            router.positions, pool_coords, router.cov,
        ).mean(axis=1)
        history.append({
            "round": r,
            "positions": {a.name: a.position.tolist() for a in agents},
            "u_grid": router.expected_utilities().tolist(),
            "u_pool": empirical_utility(
                router.positions, pool_coords, router.cov,
            ).tolist(),
            "strategic_share_pool": strategic_share_pool.tolist(),
            "observed_share": observed.tolist(),
            "routing_weight": routing_weight,
            "loss_reweight": loss_reweight,
            "agent_prompts": agent_prompts,
            "agent_responses": agent_responses,
            "agent_sft_logs": agent_sft_logs,
            # Per-agent per-query (1-G) weights computed this round.
            # See the agent_sample_weights variable above: these values
            # are populated whenever loss_reweight is set, even if the
            # SFT loss itself ran at unit weight (position_only mode).
            "agent_sample_weights": agent_sample_weights,
            "agent_loaded_prior": agent_loaded_prior,
            "agent_blend_effective": agent_blend_effective,
            "position_step": position_step,
            "blend_base": blend_base,
            "blend_round": blend_r,
            "centroid_mode": centroid_mode,
            "blend_schedule": blend_schedule,
            "save_per_round": save_per_round,
            **(
                {"theory_init": theory_init_meta}
                if r == 0 and theory_init_meta is not None
                else {}
            ),
        })

    out = Path(cfg.get("output_dir", "results/closed_loop")) / "history.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"closed-loop done: {n_rounds} rounds, wrote {out}")
    return 0


_TASKS = {
    "router_training": _task_router_training,
    "sft_training": _task_sft_training,
    "closed_loop": _task_closed_loop,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m infl_ens.training``.

    :param argv: Argument vector (defaults to ``sys.argv[1:]``).
    :type argv: Sequence[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        prog="python -m infl_ens.training",
        description="Single CLI for router-position training, SFT, and the closed loop.",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a YAML config under configs/.")
    parser.add_argument("overrides", nargs="*",
                        help="Optional KEY=VALUE config overrides.")
    args = parser.parse_args(argv)

    cfg = _load_yaml(Path(args.config))
    _apply_overrides(cfg, args.overrides)
    task = cfg.get("task")
    if task not in _TASKS:
        print(f"error: unknown task {task!r}; expected one of {sorted(_TASKS)}",
              file=sys.stderr)
        return 2
    return int(_TASKS[task](cfg))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
