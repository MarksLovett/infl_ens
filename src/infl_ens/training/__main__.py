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
- ``baseline_replay``: train one pooled cumulative LoRA on the union of
  per-round routed batches logged in an existing ``history.json`` (see
  :mod:`infl_ens.training.baseline_replay`).

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
  (jitter around a stored (2,2) reference), ``theory_gradient`` (grid Nash
  gradient ascent from a random separated start, then place SFT agents at
  the theoretical endpoints; see :mod:`infl_ens.utils.agent_init`),
  ``merge_near_theory`` (one Nash anchor per ``sft_merge_groups`` pair,
  partners offset along the merge axis; optional ``theory_pre`` warm-up),
  or ``fixed_positions`` (load exact agent-name → position coordinates from
  JSON).
- ``closed_loop.theory_pre``: optional matched-pool position dynamics
  (adaptive ``position_step``) before round 0; enabled by default for
  ``merge_near_theory`` unless ``theory_pre.enabled: false``.
- ``closed_loop.merge_near``: ``pair_separation`` (default ``0.04``) for
  ``merge_near_theory`` partner spacing.
- ``closed_loop.theory_gradient``: hyperparameters when
  ``init_mode='theory_gradient'`` (``learning_rate``, ``n_steps``, ``tol``,
  ``min_pairwise``).
- ``closed_loop.theory_ref``: optional path to a ``history.json`` whose
  final positions define the reference (defaults: per-:math:`\\sigma`
  template under ``results/``).
- ``closed_loop.position_step``: adaptive EMA blend for position updates
  (see :mod:`infl_ens.utils.position_step`). Modes: ``static`` (fixed
  ``blend``), ``cap_linf``, ``cap_l2``, ``trust_box``.
- ``closed_loop.sft_merge_groups``: optional list of router-name groups
  that share one physical LoRA trainer (see
  :mod:`infl_ens.training.merge_training`). Router agents still route and
  receive centroid updates; merged trainers concatenate each group's routed
  batch each round. At deployment, copy each merge checkpoint to every
  clone in that group.

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
    load_ai4privacy,
    load_beavertails,
    load_do_not_answer,
    load_halueval,
    load_jbb_behaviors,
    load_orbench,
    load_prompt_injection,
    load_toxicchat,
)
from infl_ens.data.trait_space import TraitSpace
from infl_ens.data.trait_space_cache import build_or_load_safety_trait_space
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
        elif kind == "jbb_behaviors":
            splits.append(load_jbb_behaviors(
                path,
                include_benign=bool(entry.get("include_benign", True)),
                max_records=max_records,
            ))
        elif kind == "toxicchat":
            splits.append(load_toxicchat(
                path,
                score_mode=entry.get("score_mode", "jailbreaking"),
                human_annotated_only=bool(entry.get("human_annotated_only", False)),
                max_records=max_records,
            ))
        elif kind == "ai4privacy":
            splits.append(load_ai4privacy(
                path,
                score_mode=entry.get("score_mode", "density"),
                english_only=bool(entry.get("english_only", True)),
                max_records=max_records,
            ))
        elif kind == "orbench":
            splits.append(load_orbench(
                path,
                configs=tuple(entry["configs"]) if entry.get("configs") else None,
                max_records=max_records,
            ))
        elif kind == "prompt_injection":
            splits.append(load_prompt_injection(
                path,
                max_records=max_records,
            ))
        elif kind == "do_not_answer":
            splits.append(load_do_not_answer(
                path,
                benign_path=entry.get("benign_path"),
                include_benign=bool(entry.get("include_benign", True)),
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
    return build_or_load_safety_trait_space(cfg, splits)


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

    if init_mode == "theory_gradient_paired":
        from infl_ens.utils.agent_init import init_agents_theory_gradient_paired

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

    if init_mode == "merge_near_theory":
        from infl_ens.training.merge_training import parse_sft_merge_groups
        from infl_ens.utils.agent_init import init_agents_merge_near_theory

        router_names = [a["name"] for a in cfg.get("agents", [])]
        merge_groups = parse_sft_merge_groups(cl, router_names)
        if not merge_groups:
            raise ValueError(
                "closed_loop.init_mode='merge_near_theory' requires "
                "closed_loop.sft_merge_groups with disjoint pairs",
            )
        seed = int(cfg.get("seed", 0))
        agents, meta = init_agents_merge_near_theory(
            cfg,
            space,
            merge_groups,
            sigma=sigma,
            seed=seed,
            init_noise=init_noise,
            near_cfg=cl.get("merge_near"),
            theory_cfg=cl.get("theory_gradient"),
        )
        log_meta = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in meta.items()
        }
        log_meta["theory_end"] = {
            a.name: a.position.tolist() for a in agents
        }
        return agents, log_meta

    if init_mode == "fixed_positions":
        repo_root = Path(__file__).resolve().parents[3]
        pos_path = cl.get("fixed_positions")
        if not pos_path:
            raise ValueError(
                "closed_loop.init_mode='fixed_positions' requires "
                "closed_loop.fixed_positions"
            )
        fixed_path = Path(pos_path)
        if not fixed_path.is_absolute():
            fixed_path = repo_root / fixed_path
        payload = json.loads(fixed_path.read_text(encoding="utf-8"))
        positions = payload.get("positions", payload.get("final_positions", payload))
        if not isinstance(positions, dict):
            raise ValueError(
                f"{fixed_path} must contain a positions mapping or "
                "final_positions mapping"
            )
        seed = int(cfg.get("seed", 0))
        rng_local = np.random.default_rng(seed)
        agents: list[RouterAgent] = []
        missing: list[str] = []
        for entry in cfg.get("agents", []):
            name = entry["name"]
            if name not in positions:
                missing.append(name)
                continue
            pos = np.asarray(positions[name], dtype=float)
            from infl_ens.data.trait_linear_transform import load_transform_from_cfg

            transform = load_transform_from_cfg(cfg, repo_root=repo_root)
            if transform is not None:
                pos = transform.apply(pos)
            if pos.shape != (space.L,):
                raise ValueError(
                    f"{fixed_path}: position for {name!r} must have shape "
                    f"({space.L},), got {pos.shape}"
                )
            if init_noise > 0.0:
                pos = pos + init_noise * rng_local.standard_normal(space.L)
            agents.append(RouterAgent(name=name, position=pos))
        if missing:
            raise ValueError(
                f"{fixed_path} missing fixed positions for: {', '.join(missing)}"
            )
        log_meta = {
            "init_mode": "fixed_positions",
            "fixed_positions": str(fixed_path),
            "init_noise": init_noise,
            "theory_end": {a.name: a.position.tolist() for a in agents},
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
        f"closed_loop.init_mode must be mean_noise, pairs_near_theory, "
        f"theory_gradient, theory_gradient_paired, merge_near_theory, or "
        f"fixed_positions, got {init_mode!r}",
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


def _init_agents_for_router_training(
    cfg: dict[str, Any],
    space: TraitSpace,
    splits: list[BenchmarkSplit],
    train_cfg: dict[str, Any],
    *,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[list[RouterAgent], Optional[dict[str, Any]]]:
    """Initialize agents for :func:`_task_router_training`.

    Supports the same theory initializers used by the closed loop so a
    theory-only Nash solve can start from co-located harm pairs.

    :param cfg: Full training config.
    :type cfg: dict
    :param space: Trait space.
    :type space: TraitSpace
    :param splits: Benchmark splits (for calibrated agents).
    :type splits: list[BenchmarkSplit]
    :param train_cfg: ``training`` config block.
    :type train_cfg: dict
    :param sigma: Absolute competitive reach.
    :type sigma: float
    :param rng: RNG for mean-noise init.
    :type rng: numpy.random.Generator
    :returns: Router agents and optional theory-init metadata.
    :rtype: tuple[list[RouterAgent], dict | None]
    :raises ValueError: If ``init_mode`` is unknown.
    """
    init_mode = str(train_cfg.get("init_mode", "mean_noise"))
    init_noise = float(train_cfg.get("init_noise", 0.0))
    theory_cfg = train_cfg.get("theory_gradient", {})

    if init_mode == "mean_noise":
        return _init_agents(
            cfg, space, splits, init_noise=init_noise, rng=rng,
        ), None

    if init_mode == "theory_gradient":
        from infl_ens.utils.agent_init import init_agents_theory_gradient

        agents, meta = init_agents_theory_gradient(
            cfg,
            space,
            sigma=sigma,
            seed=int(cfg.get("seed", 0)),
            init_noise=init_noise,
            theory_cfg=theory_cfg,
        )
        log_meta = {
            "init_mode": "theory_gradient",
            "theory_layout": meta["layout"],
            "theory_converged": meta["converged"],
            "theory_n_steps": meta["n_steps"],
            "theory_end": np.asarray(meta["theory_end"]).tolist(),
        }
        return agents, log_meta

    if init_mode == "theory_gradient_paired":
        from infl_ens.utils.agent_init import init_agents_theory_gradient_paired

        overrides = train_cfg.get("agent_start_overrides")
        agents, meta = init_agents_theory_gradient_paired(
            cfg,
            space,
            sigma=sigma,
            seed=int(cfg.get("seed", 0)),
            init_noise=init_noise,
            theory_cfg=theory_cfg,
            skip_initial_theory=bool(overrides),
        )
        log_meta = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in meta.items()
            if k != "initial"
        }
        if "initial" in meta:
            log_meta["theory_initial"] = meta["initial"].tolist()
        log_meta["theory_end"] = np.asarray(meta["theory_end"]).tolist()
        overrides = train_cfg.get("agent_start_overrides")
        if overrides:
            from infl_ens.utils.agent_init import apply_agent_start_overrides

            resolved = apply_agent_start_overrides(
                agents,
                overrides,
                space,
                seed=int(cfg.get("seed", 0)),
            )
            log_meta["agent_start_overrides"] = overrides
            log_meta["resolved_start_overrides"] = resolved
            log_meta["positions_after_overrides"] = {
                a.name: a.position.tolist() for a in agents
            }
        return agents, log_meta

    raise ValueError(
        f"training.init_mode must be mean_noise, theory_gradient, or "
        f"theory_gradient_paired, got {init_mode!r}"
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
    train_cfg = cfg.get("training", {})
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    sigma = _sigma_from_cfg(cfg, len(cfg.get("agents", [])), space)
    agents, theory_init_meta = _init_agents_for_router_training(
        cfg, space, splits, train_cfg, sigma=sigma, rng=rng,
    )
    if theory_init_meta is not None:
        print(
            f"theory init ({theory_init_meta.get('init_mode', '?')}): "
            f"layout={theory_init_meta.get('theory_layout_paired_refine', theory_init_meta.get('theory_layout'))} "
            f"pairs={theory_init_meta.get('paired_harm_order', [])}",
        )

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
    payload: dict[str, Any] = {
        "sigma": sigma,
        "converged": info["converged"],
        "n_steps": info["n_steps"],
        "positions": {a.name: a.position.tolist() for a in agents},
    }
    if theory_init_meta is not None:
        payload["theory_init"] = theory_init_meta
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
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
    from infl_ens.training.merge_training import (
        closed_loop_weight_args,
        collapsed_sft_merge_groups,
        get_merge_train_agent,
        merge_mode_from_config,
        merge_routed_batch,
        parse_sft_merge_groups,
        resolve_dynamic_merge_groups,
        snap_configured_merge_pairs,
    )
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
    if static_merge_groups and cl.get("sft_merge_only_if_collapsed"):
        static_merge_groups, collapse_meta = collapsed_sft_merge_groups(
            agents,
            static_merge_groups,
            threshold=collapse_merge_threshold,
        )
        if theory_init_meta is None:
            theory_init_meta = {}
        theory_init_meta["collapsed_sft_merge_filter"] = collapse_meta
        print(
            "SFT merge (collapsed pairs only): "
            + ", ".join(f"{t}<-{m}" for t, m in static_merge_groups),
        )
        skipped = collapse_meta.get("skipped_sft_merge", {})
        if skipped:
            print(
                "SFT merge skipped (partners not co-located): "
                + ", ".join(f"{k} (d={v:.4f})" for k, v in skipped.items()),
            )

    router = InfluencerRouter(
        space, agents, sigma=sigma,
        policy=cfg.get("policy", "proportional"),
    )

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
            pool_responses,
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
        pool_responses = all_responses

    theory_pre_cfg = dict(cl.get("theory_pre") or {})
    if (
        str(cl.get("init_mode", "mean_noise")) == "merge_near_theory"
        and "enabled" not in (cl.get("theory_pre") or {})
    ):
        theory_pre_cfg["enabled"] = True
    if theory_pre_cfg.get("enabled", False):
        from infl_ens.training.pool_dynamics import (
            agent_pairwise_geometry,
            run_theory_pre_warmup,
        )

        pre_cfg = dict(theory_pre_cfg)
        if static_merge_groups is not None:
            pre_cfg["merge_groups"] = static_merge_groups
        pre_meta = run_theory_pre_warmup(
            space,
            agents,
            pool_prompts,
            sigma=sigma,
            cfg=pre_cfg,
        )
        if theory_init_meta is None:
            theory_init_meta = {}
        pre_meta["geometry_phase"] = "post_theory_pre"
        theory_init_meta["theory_pre"] = pre_meta
        theory_init_meta["theory_end"] = {
            a.name: a.position.tolist() for a in agents
        }
        geom = pre_meta.get("agent_geometry", {})
        print(
            f"theory pre-warmup: {pre_meta['n_rounds']} matched-pool rounds "
            f"layout={pre_meta['theory_pre_layout']} "
            f"spread={pre_meta['theory_pre_final_spread']:.4f} "
            f"within_merge={geom.get('within_merge_l2', {})}",
        )
    elif static_merge_groups is not None:
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
            "post-init geometry (theory_pre disabled): "
            f"within_merge={geom.get('within_merge_l2', {})}",
        )

    val_eval_cfg = cl.get("val_eval") or {}
    val_eval_every = int(val_eval_cfg.get("every_n_rounds", 0))
    val_eval_agents = val_eval_cfg.get("agents")
    val_eval_max_records = val_eval_cfg.get("max_eval_records")

    sft_cfg_dict = dict(cl.get("sft", {}))
    sft_cfg = SFTTrainingConfig(**sft_cfg_dict)
    sft_base_output_dir = Path(sft_cfg.output_dir)
    merge_distance_threshold = float(cl.get("merge_distance_threshold", 0.08))
    merge_layout_spread_thresh = float(cl.get("merge_layout_spread_thresh", 0.45))
    merge_train_registry: dict[str, RouterAgent] = {}
    use_router_only_sft = sft_merge_mode in ("fixed", "proximity")
    also_train_individual = bool(cl.get("sft_also_train_individual", False))
    if static_merge_groups:
        print(
            "pair-merge SFT only (routing + position updates stay per-clone): "
            + ", ".join(f"{t}<-{m}" for t, m in static_merge_groups),
        )
    elif sft_merge_mode == "proximity":
        extra = " + per-clone SFT" if also_train_individual else ""
        print(
            f"pair-merge SFT (proximity): threshold={merge_distance_threshold}{extra}",
        )

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
        merge_sft_logs: dict[str, list[dict[str, Any]]] = {}
        merge_prompt_counts: dict[str, int] = {}
        merge_loaded_prior: dict[str, Optional[str]] = {}
        round_merge_meta: Optional[dict[str, Any]] = None
        active_merge_groups: Optional[list[tuple[str, list[str]]]] = None
        unpaired_agents: list[str] = []
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

            weights_i: Optional[list[float]] = None
            if loss_reweight in ("one_minus_G", "position_only"):
                mine_idx = [
                    m for m, c in enumerate(choices) if c.name == agent.name
                ]
                assert G_batch is not None
                i_idx = name_to_idx[agent.name]
                weights_i = (1.0 - G_batch[i_idx, mine_idx]).tolist()
                agent_sample_weights[agent.name] = list(weights_i)

            sample_weights_arg, eval_weights_arg, skip_pos = (
                closed_loop_weight_args(loss_reweight, centroid_mode, weights_i)
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
        pool_coords = space.project(pool_prompts)

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

        if use_router_only_sft:
            if sft_merge_mode == "fixed":
                active_merge_groups = static_merge_groups
                unpaired_agents = []
            else:
                active_merge_groups, unpaired_agents, round_merge_meta = (
                    resolve_dynamic_merge_groups(
                        agents,
                        router_names,
                        distance_threshold=merge_distance_threshold,
                        spread_thresh=merge_layout_spread_thresh,
                    )
                )
                layout = round_merge_meta.get("layout", "?")
                method = round_merge_meta.get("pairing_method", "?")
                if layout != "2,2" and not active_merge_groups:
                    print(
                        f"round {r}: layout={layout} — no merge pairs "
                        f"(config not 2,2); per-agent SFT",
                    )
                elif active_merge_groups:
                    print(
                        f"round {r}: layout={layout} pairing={method} "
                        + ", ".join(
                            f"{t}<-{m}" for t, m in active_merge_groups
                        ),
                    )

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
                if loss_reweight == "one_minus_G":
                    m_sample_weights = merged_weights
                elif loss_reweight == "position_only":
                    m_sample_weights = None
                else:
                    m_sample_weights = None
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

            if also_train_individual:
                for agent in agents:
                    mine_p = agent_prompts.get(agent.name, [])
                    mine_r = agent_responses.get(agent.name, [])
                    if not mine_p:
                        continue
                    weights_i = agent_sample_weights.get(agent.name) or None
                    sw, ew, _skip_pos = closed_loop_weight_args(
                        loss_reweight, centroid_mode, weights_i,
                    )
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
                        sample_weights=sw,
                        eval_weights=ew,
                        skip_position_update=True,
                    )
                    agent_sft_logs[agent.name] = sft_result.get(
                        "log_history", [],
                    )
                    agent_loaded_prior[agent.name] = sft_result.get(
                        "loaded_prior_lora",
                    )

            for name in unpaired_agents:
                agent = next(a for a in agents if a.name == name)
                mine_p = agent_prompts.get(name, [])
                mine_r = agent_responses.get(name, [])
                if not mine_p:
                    continue
                weights_i = agent_sample_weights.get(name) or None
                sw, ew, skip_pos = closed_loop_weight_args(
                    loss_reweight, centroid_mode, weights_i,
                )
                out_override = (
                    str(sft_base_output_dir / name / f"round-{r:02d}")
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
                    sample_weights=sw,
                    eval_weights=ew,
                    skip_position_update=True,
                )
                agent_sft_logs[name] = sft_result.get("log_history", [])
                agent_loaded_prior[name] = sft_result.get("loaded_prior_lora")

            if (
                sft_merge_mode == "proximity"
                and not active_merge_groups
                and unpaired_agents == list(router_names)
            ):
                for agent in agents:
                    mine_p = agent_prompts.get(agent.name, [])
                    mine_r = agent_responses.get(agent.name, [])
                    if not mine_p or agent_sft_logs.get(agent.name):
                        continue
                    weights_i = agent_sample_weights.get(agent.name) or None
                    sw, ew, skip_pos = closed_loop_weight_args(
                        loss_reweight, centroid_mode, weights_i,
                    )
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
                        sample_weights=sw,
                        eval_weights=ew,
                        skip_position_update=True,
                    )
                    agent_sft_logs[agent.name] = sft_result.get("log_history", [])
                    agent_loaded_prior[agent.name] = sft_result.get(
                        "loaded_prior_lora",
                    )

        observed = np.array(
            [sum(1 for c in choices if c.name == a.name) / max(len(choices), 1)
             for a in agents]
        )
        strategic_share_pool = strategic_routing_weights(
            router.positions, pool_coords, router.cov,
        ).mean(axis=1)
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
                {
                    "sft_merge_mode": sft_merge_mode,
                    "sft_also_train_individual": also_train_individual,
                    "merge_round": round_merge_meta,
                    "merge_sft_logs": merge_sft_logs,
                    "merge_prompt_counts": merge_prompt_counts,
                    "merge_loaded_prior": merge_loaded_prior,
                    "merge_unpaired": unpaired_agents,
                }
                if use_router_only_sft
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

    out = Path(cfg.get("output_dir", "results/closed_loop")) / "history.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    if split_manifest is not None:
        split_meta_path = out.parent / "data_split.json"
        with split_meta_path.open("w", encoding="utf-8") as fh:
            json.dump(split_manifest.to_dict(), fh, indent=2)
        print(f"data split manifest: {split_meta_path}")
    print(f"closed-loop done: {n_rounds} rounds, wrote {out}")
    return 0


def _task_baseline_replay(cfg: dict[str, Any]) -> int:
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

    splits = _load_splits(cfg)
    space = _make_trait_space(cfg, splits)
    br = cfg.get("baseline_replay", {})
    agent_name = str(br.get("agent_name", "pooled-baseline"))
    agent = make_pooled_baseline_agent(space, name=agent_name)

    sft_cfg_dict = dict(cfg.get("sft", {}))
    sft_cfg = SFTTrainingConfig(**sft_cfg_dict)
    rounds = br.get("rounds")

    summaries = replay_pooled_baseline_sft(
        hist_path,
        agent,
        sft_cfg,
        project=space.project,
        output_dir=Path(cfg["output_dir"]) / "agents",
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


_TASKS = {
    "router_training": _task_router_training,
    "sft_training": _task_sft_training,
    "closed_loop": _task_closed_loop,
    "baseline_replay": _task_baseline_replay,
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
