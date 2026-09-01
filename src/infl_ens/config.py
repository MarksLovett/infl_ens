"""Layered YAML configuration for every ``infl_ens`` entry point.

One loader serves the training, evaluation, figure and pipeline CLIs so a
config means the same thing everywhere.  A YAML file may compose other
files through an ``includes:`` list (paths relative to the including file);
fragments are deep-merged in order and the including file's own keys win.
Dotted ``KEY=VALUE`` overrides are applied on top, then the flat result is
validated against the key tables below and returned as a plain ``dict``.

The validator is deliberately *read-only*: it never injects defaults or
rewrites values.  The resolved ``benchmarks`` and ``trait_space`` blocks are
hashed into the trait-space cache fingerprint
(:func:`infl_ens.data.trait_space_cache.trait_space_fingerprint`), so the
loader must hand them over exactly as the YAML spells them.

Example::

    # configs/arms/soft_topk3_pairs.yaml
    includes: [_closed_loop_base.yaml]
    output_dir: results/seven_axis_soft_topk3_pairs/seed0
    closed_loop: {routing_mode: soft, soft_top_k: 3, soft_loss: weighted}

    $ python -m infl_ens.training --config configs/arms/soft_topk3_pairs.yaml \\
          -- closed_loop.n_rounds=2 data_split=null
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

INCLUDES_KEY = "includes"

#: Tasks accepted by ``python -m infl_ens.training``.
KNOWN_TASKS: frozenset[str] = frozenset({"closed_loop", "baseline_replay"})

#: Keys allowed at the top level of a run config.
TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "task",
        "seed",
        "output_dir",
        "policy",
        "sigma_mode",
        "sigma",
        "sigma_fraction",
        "history_path",
        "repo_root",
        "benchmarks",
        "data_split",
        "trait_space",
        "encoder",
        "agents",
        "closed_loop",
        "sft",
        "eval",
        "baseline_replay",
    },
)

#: Allowed keys per benchmark ``kind`` (mirrors the loader signatures in
#: :mod:`infl_ens.data.benchmarks`).
BENCHMARK_ENTRY_KEYS: dict[str, frozenset[str]] = {
    "beavertails": frozenset({"kind", "path", "categories", "max_records"}),
    "halueval": frozenset({"kind", "path", "tasks", "max_records"}),
    "jbb_behaviors": frozenset({"kind", "path", "include_benign", "max_records"}),
    "ai4privacy": frozenset(
        {"kind", "path", "score_mode", "english_only", "max_records"},
    ),
    "orbench": frozenset({"kind", "path", "configs", "max_records"}),
    "prompt_injection": frozenset({"kind", "path", "max_records"}),
    "do_not_answer": frozenset(
        {"kind", "path", "benign_path", "include_benign", "max_records"},
    ),
}

DATA_SPLIT_KEYS: frozenset[str] = frozenset(
    {
        "seed",
        "train_frac",
        "val_frac",
        "test_frac",
        "manifest",
        "write_manifest",
        "cover_train_exactly",
        "preferred_batch_sizes",
        "target_n_rounds",
        "min_rounds",
        "max_rounds",
    },
)

#: Every key here except ``encoder_batch_size``, ``cache``, ``cache_dir``
#: and ``cache_path`` is part of the cache fingerprint.
TRAIT_SPACE_KEYS: frozenset[str] = frozenset(
    {
        "encoder",
        "encoder_batch_size",
        "cache",
        "cache_dir",
        "cache_path",
        "n_grid",
        "kde_bandwidth",
        "threshold",
        "coordinate_residualize",
        "mode_alignment_weight",
        "mode_alignment_weights",
        "coordinate_stretch_gamma",
        "coordinate_stretch_gammas",
        "quantile_knots",
    },
)

#: Keyword arguments of :class:`infl_ens.data.encoders.HuggingFaceEncoder`.
ENCODER_KEYS: frozenset[str] = frozenset(
    {
        "model_name",
        "device",
        "batch_size",
        "max_length",
        "pooling",
        "normalize",
        "torch_dtype",
        "device_map",
        "padding_side",
        "attn_implementation",
        "trust_remote_code",
    },
)

AGENTS_MAPPING_KEYS: frozenset[str] = frozenset({"pairs_from_axes", "name_prefix"})
AGENT_ENTRY_KEYS: frozenset[str] = frozenset({"name", "calibration"})

CLOSED_LOOP_KEYS: frozenset[str] = frozenset(
    {
        "init_mode",
        "init_noise",
        "theory_gradient",
        "snap_collapsed_pairs",
        "collapse_merge_threshold",
        "routing_mode",
        "soft_top_k",
        "soft_loss",
        "soft_select",
        "routing_weight",
        "loss_reweight",
        "position_update",
        "centroid_mode",
        "blend",
        "blend_schedule",
        "blend_start",
        "position_step",
        "sft_merge_groups",
        "merge_group_prefix",
        "save_per_round",
        "val_eval",
        "sft",
        "n_rounds",
        "batch_size",
    },
)

THEORY_GRADIENT_KEYS: frozenset[str] = frozenset(
    {"learning_rate", "n_steps", "tol", "min_pairwise", "pairing"},
)

VAL_EVAL_KEYS: frozenset[str] = frozenset(
    {"every_n_rounds", "agents", "max_eval_records"},
)

#: Fields of :class:`infl_ens.training.sft_training.SFTTrainingConfig`.
SFT_KEYS: frozenset[str] = frozenset(
    {
        "base_model",
        "output_dir",
        "max_seq_length",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "learning_rate",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "bf16",
        "gradient_checkpointing",
        "logging_steps",
        "cumulative_lora",
        "seed",
    },
)

EVAL_KEYS: frozenset[str] = frozenset(
    {
        "partitions",
        "rounds",
        "agents",
        "max_eval_records",
        "forward_batch_size",
        "max_seq_length",
        "after_training",
        "baseline_run_dir",
        "baseline_agents",
        "base_model",
    },
)

BASELINE_REPLAY_KEYS: frozenset[str] = frozenset(
    {"agent_name", "save_per_round", "rounds"},
)


class ConfigError(ValueError):
    """Raised for malformed configuration files or overrides."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Parse one YAML file into a mapping.

    :param path: File to read.
    :type path: str | pathlib.Path
    :returns: The parsed mapping (an empty file yields ``{}``).
    :rtype: dict
    :raises ConfigError: If the file is missing or its top level is not a
        mapping.
    """
    import yaml

    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base`` without mutating either.

    Nested mappings merge recursively; lists, scalars and ``None`` replace
    the base value outright (so ``data_split: null`` disables a block).

    :param base: Lower-priority mapping.
    :type base: Mapping
    :param override: Higher-priority mapping.
    :type override: Mapping
    :returns: A new, deep-copied mapping.
    :rtype: dict
    """
    out: dict[str, Any] = {k: copy.deepcopy(v) for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_includes(
    cfg: Mapping[str, Any],
    base_dir: str | Path,
    *,
    _stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Expand the ``includes:`` list of a loaded mapping.

    Included files are resolved relative to ``base_dir``, expanded
    recursively, merged in list order, and finally overlaid by the
    including mapping's own keys.  The ``includes`` key itself is dropped
    from the result.

    :param cfg: Mapping possibly carrying an ``includes`` list.
    :type cfg: Mapping
    :param base_dir: Directory the include paths are relative to.
    :type base_dir: str | pathlib.Path
    :returns: Flattened mapping with no ``includes`` key.
    :rtype: dict
    :raises ConfigError: On a non-list ``includes`` value or an include
        cycle.
    """
    base_dir = Path(base_dir)
    own = {k: v for k, v in cfg.items() if k != INCLUDES_KEY}
    includes = cfg.get(INCLUDES_KEY) or []
    if not isinstance(includes, list):
        raise ConfigError(
            f"{base_dir}: '{INCLUDES_KEY}' must be a list of paths, "
            f"got {type(includes).__name__}",
        )
    merged: dict[str, Any] = {}
    for rel in includes:
        inc_path = (base_dir / str(rel)).resolve()
        if inc_path in _stack:
            chain = " -> ".join(str(p) for p in (*_stack, inc_path))
            raise ConfigError(f"include cycle: {chain}")
        fragment = resolve_includes(
            load_yaml(inc_path), inc_path.parent, _stack=(*_stack, inc_path),
        )
        merged = deep_merge(merged, fragment)
    return deep_merge(merged, own)


def _parse_override_value(raw: str) -> Any:
    """Parse an override value as JSON, falling back to the raw string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(
    cfg: dict[str, Any],
    overrides: Sequence[str] | Mapping[str, Any],
) -> None:
    """Apply dotted-path overrides to ``cfg`` in place.

    Accepts either ``["closed_loop.n_rounds=2", "data_split=null"]`` (values
    parsed as JSON when possible) or a mapping ``{"closed_loop.n_rounds": 2}``
    (values used verbatim).  Missing intermediate mappings are created;
    a scalar in the way is replaced by a mapping.

    :param cfg: Configuration to mutate.
    :type cfg: dict
    :param overrides: Overrides in either form.
    :type overrides: Sequence[str] | Mapping[str, Any]
    :raises ConfigError: If a string override lacks ``=`` or a key is empty.
    """
    if isinstance(overrides, Mapping):
        items = list(overrides.items())
    else:
        items = []
        for ov in overrides:
            if "=" not in ov:
                raise ConfigError(f"override {ov!r} must look like KEY=VALUE")
            key, raw = ov.split("=", 1)
            items.append((key, _parse_override_value(raw)))
    for key, value in items:
        parts = [p for p in str(key).split(".") if p]
        if not parts:
            raise ConfigError(f"override has an empty key: {key!r}")
        node: dict[str, Any] = cfg
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = copy.deepcopy(value)


def _check_keys(
    block: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    label: str,
    source: str,
) -> None:
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise ConfigError(
            f"{label}: unknown key {unknown[0]!r} ({source}); "
            f"allowed: {sorted(allowed)}",
        )


def _check_mapping(value: Any, *, label: str, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(
            f"{label}: expected a mapping ({source}), got {type(value).__name__}",
        )
    return value


def _validate_benchmarks(entries: Any, *, source: str) -> None:
    if not isinstance(entries, list):
        raise ConfigError(
            f"benchmarks: expected a list ({source}), got {type(entries).__name__}",
        )
    for i, entry in enumerate(entries):
        label = f"benchmarks[{i}]"
        entry = _check_mapping(entry, label=label, source=source)
        kind = entry.get("kind")
        if kind not in BENCHMARK_ENTRY_KEYS:
            raise ConfigError(
                f"{label}: unknown benchmark kind {kind!r} ({source}); "
                f"known: {sorted(BENCHMARK_ENTRY_KEYS)}",
            )
        if "path" not in entry:
            raise ConfigError(f"{label}: missing required key 'path' ({source})")
        _check_keys(entry, BENCHMARK_ENTRY_KEYS[kind], label=label, source=source)


def _validate_agents(value: Any, *, source: str) -> None:
    if isinstance(value, list):
        for i, entry in enumerate(value):
            label = f"agents[{i}]"
            entry = _check_mapping(entry, label=label, source=source)
            if "name" not in entry:
                raise ConfigError(f"{label}: missing required key 'name' ({source})")
            _check_keys(entry, AGENT_ENTRY_KEYS, label=label, source=source)
        return
    if isinstance(value, Mapping):
        _check_keys(value, AGENTS_MAPPING_KEYS, label="agents", source=source)
        if not value.get("pairs_from_axes", False):
            raise ConfigError(
                f"agents: a mapping must set pairs_from_axes: true ({source})",
            )
        return
    raise ConfigError(
        f"agents: expected a list or mapping ({source}), got {type(value).__name__}",
    )


def _validate_closed_loop(block: Mapping[str, Any], *, source: str) -> None:
    _check_keys(block, CLOSED_LOOP_KEYS, label="closed_loop", source=source)
    tg = block.get("theory_gradient")
    if tg is not None:
        tg = _check_mapping(tg, label="closed_loop.theory_gradient", source=source)
        _check_keys(
            tg, THEORY_GRADIENT_KEYS, label="closed_loop.theory_gradient", source=source,
        )
    ve = block.get("val_eval")
    if ve is not None:
        ve = _check_mapping(ve, label="closed_loop.val_eval", source=source)
        _check_keys(ve, VAL_EVAL_KEYS, label="closed_loop.val_eval", source=source)
    sft = block.get("sft")
    if sft is not None:
        sft = _check_mapping(sft, label="closed_loop.sft", source=source)
        _check_keys(sft, SFT_KEYS, label="closed_loop.sft", source=source)


def validate_config(cfg: Mapping[str, Any], *, source: str = "<config>") -> None:
    """Check a resolved run config against the key tables.

    Only structure is checked (unknown keys, wrong container types,
    missing ``kind``/``path``/``name``).  Values are left to the consumers,
    and nothing is mutated.

    :param cfg: Flattened configuration.
    :type cfg: Mapping
    :param source: Label used in error messages (normally the file path).
    :type source: str
    :raises ConfigError: On the first problem found.
    """
    _check_mapping(cfg, label="config", source=source)
    _check_keys(cfg, TOP_LEVEL_KEYS, label="config", source=source)

    task = cfg.get("task")
    if task is not None and task not in KNOWN_TASKS:
        raise ConfigError(
            f"task: unknown task {task!r} ({source}); known: {sorted(KNOWN_TASKS)}",
        )

    simple_blocks = (
        ("data_split", DATA_SPLIT_KEYS),
        ("trait_space", TRAIT_SPACE_KEYS),
        ("encoder", ENCODER_KEYS),
        ("sft", SFT_KEYS),
        ("eval", EVAL_KEYS),
        ("baseline_replay", BASELINE_REPLAY_KEYS),
    )
    for name, allowed in simple_blocks:
        block = cfg.get(name)
        if block is None:
            continue
        block = _check_mapping(block, label=name, source=source)
        _check_keys(block, allowed, label=name, source=source)

    ts = cfg.get("trait_space")
    if ts is not None:
        enc = ts.get("encoder")
        if enc is not None and not isinstance(enc, (str, Mapping)):
            raise ConfigError(
                f"trait_space.encoder: expected a model-name string or mapping "
                f"({source}), got {type(enc).__name__}",
            )
        if isinstance(enc, Mapping):
            _check_keys(enc, ENCODER_KEYS, label="trait_space.encoder", source=source)

    if cfg.get("benchmarks") is not None:
        _validate_benchmarks(cfg["benchmarks"], source=source)
    if cfg.get("agents") is not None:
        _validate_agents(cfg["agents"], source=source)
    cl = cfg.get("closed_loop")
    if cl is not None:
        cl = _check_mapping(cl, label="closed_loop", source=source)
        _validate_closed_loop(cl, source=source)


def load_config(
    path: str | Path,
    overrides: Sequence[str] | Mapping[str, Any] = (),
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Load, compose, override and validate a run config.

    :param path: YAML file to load.
    :type path: str | pathlib.Path
    :param overrides: Dotted overrides applied after include resolution
        (see :func:`apply_overrides`).
    :type overrides: Sequence[str] | Mapping[str, Any]
    :param validate: Run :func:`validate_config` on the result.  Pass
        ``False`` for files with a different schema (experiment files
        validate themselves).
    :type validate: bool
    :returns: Flat configuration mapping with no ``includes`` key.
    :rtype: dict
    :raises ConfigError: On any loading, composition or validation error.
    """
    p = Path(path)
    cfg = resolve_includes(load_yaml(p), p.parent, _stack=(p.resolve(),))
    apply_overrides(cfg, overrides)
    if validate:
        validate_config(cfg, source=str(p))
    return cfg


def resolve_sft_block(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Merge the SFT (base model + LoRA) settings of a run config.

    The model fragment (``configs/models/*.yaml``) provides a top-level
    ``sft`` block; a closed-loop config may overlay ``closed_loop.sft``.
    When neither sets ``output_dir`` it defaults to ``<output_dir>/agents``.

    :param cfg: Resolved run config.
    :type cfg: Mapping
    :returns: Keyword arguments for
        :class:`infl_ens.training.sft_training.SFTTrainingConfig`.
    :rtype: dict
    """
    top = cfg.get("sft") or {}
    nested = (cfg.get("closed_loop") or {}).get("sft") or {}
    sft = deep_merge(top, nested)
    if "output_dir" not in sft:
        run_dir = str(cfg.get("output_dir", "results/closed_loop"))
        sft["output_dir"] = str(Path(run_dir) / "agents")
    return sft


__all__ = [
    "BASELINE_REPLAY_KEYS",
    "BENCHMARK_ENTRY_KEYS",
    "CLOSED_LOOP_KEYS",
    "ConfigError",
    "DATA_SPLIT_KEYS",
    "ENCODER_KEYS",
    "EVAL_KEYS",
    "KNOWN_TASKS",
    "SFT_KEYS",
    "THEORY_GRADIENT_KEYS",
    "TOP_LEVEL_KEYS",
    "TRAIT_SPACE_KEYS",
    "VAL_EVAL_KEYS",
    "apply_overrides",
    "deep_merge",
    "load_config",
    "load_yaml",
    "resolve_includes",
    "resolve_sft_block",
    "validate_config",
]
