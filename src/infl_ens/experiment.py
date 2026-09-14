"""Experiment files: the arms, stages and analysis settings of one study.

An experiment YAML (``configs/experiments/*.yaml``) names the arm configs
that make up a comparison and the settings shared by the analysis stages
(evaluation window, figure options, smoke-test overrides).  It is the
single input of ``python -m infl_ens.pipeline`` and
``python -m infl_ens.figures``.

Arm ``config`` paths are relative to the experiment file; every other path
(``results_dir``, ``figures_dir``, the arms' ``output_dir``) is relative to
the working directory, which is the repository root for every CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from infl_ens.config import ConfigError, load_config, load_yaml

#: Pipeline stages in execution order. ``download`` and ``prune`` are opt-in.
ALL_STAGES: tuple[str, ...] = (
    "download",
    "manifest",
    "train",
    "perround",
    "routing",
    "behavioral",
    "figures",
    "prune",
)
DEFAULT_STAGES: tuple[str, ...] = ("manifest", "train", "perround", "routing", "figures")
ARM_ROLES: frozenset[str] = frozenset({"specialist", "generalist"})

EXPERIMENT_KEYS: frozenset[str] = frozenset(
    {
        "name", "results_dir", "figures_dir", "arms", "stages", "eval",
        "behavioral_eval", "figures", "smoke",
    },
)
ARM_KEYS: frozenset[str] = frozenset({"name", "label", "title", "role", "config"})
EVAL_SETTING_KEYS: frozenset[str] = frozenset(
    {"perround_rounds", "perround_partition", "routing_partition", "max_eval_records"},
)
FIGURE_SETTING_KEYS: frozenset[str] = frozenset(
    {"axis_labels", "formats", "compile_tex", "include"},
)
SMOKE_KEYS: frozenset[str] = frozenset({"tests", "arms", "output_root", "overrides"})
BEHAVIORAL_SETTING_KEYS: frozenset[str] = frozenset({
    "schema_version", "checkpoints", "protocols", "targets", "generation",
    "suites", "contamination", "graders", "audit",
})
BEHAVIORAL_TARGET_KEYS: frozenset[str] = frozenset({"include_base_model", "arms"})
BEHAVIORAL_GENERATION_KEYS: frozenset[str] = frozenset({
    "do_sample", "temperature", "batch_size", "max_new_tokens",
    "max_input_tokens", "fail_on_truncation", "seed",
})
BEHAVIORAL_CONTAMINATION_KEYS: frozenset[str] = frozenset({
    "exact_match", "fuzzy_match", "ngram_size", "fuzzy_threshold",
})
BEHAVIORAL_SUITE_KINDS: frozenset[str] = frozenset({
    "harmbench", "strongreject", "xstest", "truthfulqa", "confaide", "bipia", "ifeval",
})
BEHAVIORAL_SUITE_GENERATION_KEYS: frozenset[str] = frozenset({
    "batch_size", "max_new_tokens", "max_input_tokens",
})


def _check_keys(block: Mapping[str, Any], allowed: frozenset[str], label: str, source: str) -> None:
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise ConfigError(
            f"{label}: unknown key {unknown[0]!r} ({source}); allowed: {sorted(allowed)}",
        )


@dataclass(frozen=True)
class ArmSpec:
    """One arm of an experiment.

    :param name: Short identifier used for stage selection and file stems.
    :type name: str
    :param label: Short legend label (cross-arm figures).
    :type label: str
    :param title: Longer figure title.
    :type title: str
    :param role: ``specialist`` (a routed closed loop) or ``generalist`` (the
        pooled replay comparator).
    :type role: str
    :param config_path: Absolute path of the arm's run config.
    :type config_path: pathlib.Path
    :param output_dir: The arm's ``output_dir`` as written in its config.
    :type output_dir: pathlib.Path
    """

    name: str
    label: str
    title: str
    role: str
    config_path: Path
    output_dir: Path

    def load(self, overrides: Sequence[str] | Mapping[str, Any] = ()) -> dict[str, Any]:
        """Resolve the arm's run config (includes + optional overrides).

        :param overrides: Dotted overrides, see :func:`infl_ens.config.apply_overrides`.
        :type overrides: Sequence[str] | Mapping[str, Any]
        :returns: Flat run config.
        :rtype: dict
        """
        return load_config(self.config_path, overrides)

    @property
    def run_dir(self) -> Path:
        """Alias of :attr:`output_dir`: where the run writes ``history.json``."""
        return self.output_dir

    @property
    def is_specialist(self) -> bool:
        """Whether this arm is a routed closed loop."""
        return self.role == "specialist"


@dataclass(frozen=True)
class EvalSettings:
    """Held-out evaluation window shared by the analysis stages.

    :param perround_rounds: Rounds scored on ``perround_partition``; the
        string ``final`` stands for the last trained round.
    :type perround_rounds: tuple[int | str, ...]
    :param perround_partition: Manifest partition for the per-round table.
    :type perround_partition: str
    :param routing_partition: Manifest partition for route-then-score.
    :type routing_partition: str
    :param max_eval_records: Per-benchmark cap for both stages.
    :type max_eval_records: int | None
    """

    perround_rounds: tuple[int | str, ...] = (4, "final")
    perround_partition: str = "val"
    routing_partition: str = "test"
    max_eval_records: int | None = 1000

    def resolve_rounds(self, final_round: int) -> list[int]:
        """Replace ``final`` by the last trained round and sort.

        :param final_round: Last round index present in the history.
        :type final_round: int
        :returns: Concrete, sorted, de-duplicated round indices.
        :rtype: list[int]
        """
        out: set[int] = set()
        for r in self.perround_rounds:
            out.add(final_round if r == "final" else int(r))
        return sorted(out)


@dataclass(frozen=True)
class FigureSettings:
    """Figure options shared by every arm.

    :param axis_labels: Trait-axis names in benchmark order.
    :type axis_labels: tuple[str, ...]
    :param formats: Raster/vector formats written per matplotlib figure.
    :type formats: tuple[str, ...]
    :param compile_tex: ``auto`` (latexmk when available), ``always`` or
        ``never``.
    :type compile_tex: str
    :param include: Figure names to render (see
        :data:`infl_ens.figures.render.FIGURES`); empty means all
        CPU-only figures.
    :type include: tuple[str, ...]
    """

    axis_labels: tuple[str, ...] = ()
    formats: tuple[str, ...] = ("pdf", "png")
    compile_tex: str = "auto"
    include: tuple[str, ...] = ()


@dataclass(frozen=True)
class SmokeSettings:
    """Cheap end-to-end gate: a pytest subset plus tiny closed loops.

    :param tests: Test files to run first.
    :type tests: tuple[str, ...]
    :param arms: Arm names to run with the smoke overrides.
    :type arms: tuple[str, ...]
    :param output_root: Where the smoke runs write (``<root>/<arm>/seed0``).
    :type output_root: pathlib.Path
    :param overrides: Dotted overrides applied to each smoke arm.
    :type overrides: dict[str, Any]
    """

    tests: tuple[str, ...] = ()
    arms: tuple[str, ...] = ()
    output_root: Path = Path("results/_smoke")
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BehavioralGenerationSettings:
    """Deterministic generation settings for behavioral evaluation.

    :param do_sample: Whether to sample tokens. Publication runs default to
        greedy decoding.
    :type do_sample: bool
    :param temperature: Sampling temperature. Ignored by greedy decoding.
    :type temperature: float
    :param batch_size: Target-model generation batch size.
    :type batch_size: int
    :param max_new_tokens: Default output-token limit; a suite can override it.
    :type max_new_tokens: int
    :param max_input_tokens: Optional input-token ceiling.
    :type max_input_tokens: int | None
    :param fail_on_truncation: Raise rather than silently truncate long cases.
    :type fail_on_truncation: bool
    :param seed: Generation seed.
    :type seed: int
    """

    do_sample: bool = False
    temperature: float = 0.0
    batch_size: int = 8
    max_new_tokens: int = 256
    max_input_tokens: int | None = None
    fail_on_truncation: bool = True
    seed: int = 0


@dataclass(frozen=True)
class BehavioralSettings:
    """External generation-based safety-evaluation settings.

    Behavioral suites are intentionally separate from ``benchmarks`` so they
    cannot enter SFT, router fitting, or the trait-space fingerprint.

    :param schema_version: Behavioral artifact schema version.
    :type schema_version: int
    :param checkpoints: Checkpoint selector; currently ``"final"``.
    :type checkpoints: str
    :param protocols: Generation protocols, ``native`` and/or ``hard_argmax``.
    :type protocols: tuple[str, ...]
    :param include_base_model: Include the unmodified base instruct model.
    :type include_base_model: bool
    :param target_arms: Arm names or the sentinel ``("all",)``.
    :type target_arms: tuple[str, ...]
    :param generation: Shared decoding settings.
    :type generation: BehavioralGenerationSettings
    :param suites: Raw suite-loader mappings in declared order.
    :type suites: tuple[dict[str, Any], ...]
    :param contamination: Exact/fuzzy overlap policy.
    :type contamination: dict[str, Any]
    :param graders: Optional grader overrides.
    :type graders: dict[str, Any]
    :param audit: Optional human-audit metadata.
    :type audit: dict[str, Any]
    """

    schema_version: int = 1
    checkpoints: str = "final"
    protocols: tuple[str, ...] = ("native", "hard_argmax")
    include_base_model: bool = True
    target_arms: tuple[str, ...] = ("all",)
    generation: BehavioralGenerationSettings = field(
        default_factory=BehavioralGenerationSettings,
    )
    suites: tuple[dict[str, Any], ...] = ()
    contamination: dict[str, Any] = field(default_factory=dict)
    graders: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """A parsed experiment file.

    :param name: Experiment name (also the default results/figures subdir).
    :type name: str
    :param path: The experiment YAML this was loaded from.
    :type path: pathlib.Path
    :param results_dir: Pipeline log and stage status location.
    :type results_dir: pathlib.Path
    :param figures_dir: Where rendered figures and tables go.
    :type figures_dir: pathlib.Path
    :param arms: Arms in execution order.
    :type arms: tuple[ArmSpec, ...]
    :param stages: Default stage list.
    :type stages: tuple[str, ...]
    :param eval: Evaluation window.
    :type eval: EvalSettings
    :param behavioral_eval: Optional generation-based safety evaluation.
    :type behavioral_eval: BehavioralSettings | None
    :param figures: Figure options.
    :type figures: FigureSettings
    :param smoke: Smoke-gate settings.
    :type smoke: SmokeSettings
    """

    name: str
    path: Path
    results_dir: Path
    figures_dir: Path
    arms: tuple[ArmSpec, ...]
    stages: tuple[str, ...]
    eval: EvalSettings
    figures: FigureSettings
    smoke: SmokeSettings
    behavioral_eval: BehavioralSettings | None = None

    @property
    def specialists(self) -> tuple[ArmSpec, ...]:
        """Arms with ``role: specialist``, in order."""
        return tuple(a for a in self.arms if a.is_specialist)

    @property
    def generalists(self) -> tuple[ArmSpec, ...]:
        """All ``role: generalist`` arms, in experiment order.

        The first remains the primary legacy reference exposed by
        :attr:`generalist`.

        :returns: Generalist arms.
        :rtype: tuple[ArmSpec, ...]
        """
        return tuple(a for a in self.arms if a.role == "generalist")

    @property
    def generalist(self) -> ArmSpec | None:
        """The primary (first) ``role: generalist`` arm, if any."""
        return self.generalists[0] if self.generalists else None

    def arm(self, name: str) -> ArmSpec:
        """Look an arm up by name.

        :param name: Arm name.
        :type name: str
        :returns: The arm.
        :rtype: ArmSpec
        :raises KeyError: If no arm has that name.
        """
        for a in self.arms:
            if a.name == name:
                return a
        raise KeyError(f"unknown arm {name!r}; known: {[a.name for a in self.arms]}")


def _parse_arm(raw: Any, index: int, base_dir: Path, source: str) -> ArmSpec:
    label = f"arms[{index}]"
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label}: expected a mapping ({source})")
    _check_keys(raw, ARM_KEYS, label, source)
    for key in ("name", "config"):
        if key not in raw:
            raise ConfigError(f"{label}: missing required key {key!r} ({source})")
    role = str(raw.get("role", "specialist"))
    if role not in ARM_ROLES:
        raise ConfigError(f"{label}: role must be one of {sorted(ARM_ROLES)} ({source})")
    config_path = (base_dir / str(raw["config"])).resolve()
    if not config_path.is_file():
        raise ConfigError(f"{label}: config not found: {config_path} ({source})")
    cfg = load_config(config_path)
    if "output_dir" not in cfg:
        raise ConfigError(f"{label}: {config_path} sets no output_dir ({source})")
    name = str(raw["name"])
    return ArmSpec(
        name=name,
        label=str(raw.get("label", name)),
        title=str(raw.get("title", raw.get("label", name))),
        role=role,
        config_path=config_path,
        output_dir=Path(str(cfg["output_dir"])),
    )


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Parse and validate an experiment file.

    Every arm config is resolved once here, so a broken include or an
    unknown key anywhere in the experiment fails before any GPU work.

    :param path: Experiment YAML.
    :type path: str | pathlib.Path
    :returns: Parsed experiment.
    :rtype: ExperimentConfig
    :raises ConfigError: On any structural problem.
    """
    p = Path(path)
    raw = load_yaml(p)
    source = str(p)
    _check_keys(raw, EXPERIMENT_KEYS, "experiment", source)
    if "name" not in raw or "arms" not in raw:
        raise ConfigError(f"experiment: 'name' and 'arms' are required ({source})")
    name = str(raw["name"])

    arms_raw = raw["arms"]
    if not isinstance(arms_raw, list) or not arms_raw:
        raise ConfigError(f"arms: expected a non-empty list ({source})")
    arms = tuple(_parse_arm(a, i, p.parent, source) for i, a in enumerate(arms_raw))
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ConfigError(f"arms: duplicate arm names {names} ({source})")
    stages = tuple(str(s) for s in (raw.get("stages") or DEFAULT_STAGES))
    unknown_stages = [s for s in stages if s not in ALL_STAGES]
    if unknown_stages:
        raise ConfigError(
            f"stages: unknown stage {unknown_stages[0]!r} ({source}); known: {list(ALL_STAGES)}",
        )

    ev = raw.get("eval") or {}
    _check_keys(ev, EVAL_SETTING_KEYS, "eval", source)
    rounds_raw = ev.get("perround_rounds", [4, "final"])
    if not isinstance(rounds_raw, list) or not rounds_raw:
        raise ConfigError(f"eval.perround_rounds: expected a non-empty list ({source})")
    rounds: list[int | str] = []
    for r in rounds_raw:
        if r == "final":
            rounds.append("final")
        elif isinstance(r, int) and not isinstance(r, bool):
            rounds.append(int(r))
        else:
            raise ConfigError(
                f"eval.perround_rounds: entries must be ints or 'final', got {r!r} ({source})",
            )
    eval_settings = EvalSettings(
        perround_rounds=tuple(rounds),
        perround_partition=str(ev.get("perround_partition", "val")),
        routing_partition=str(ev.get("routing_partition", "test")),
        max_eval_records=(
            int(ev["max_eval_records"]) if ev.get("max_eval_records") is not None else None
        ),
    )

    behavioral_settings: BehavioralSettings | None = None
    behavioral_raw = raw.get("behavioral_eval")
    if behavioral_raw is not None:
        if not isinstance(behavioral_raw, Mapping):
            raise ConfigError(f"behavioral_eval: expected a mapping ({source})")
        _check_keys(
            behavioral_raw, BEHAVIORAL_SETTING_KEYS, "behavioral_eval", source,
        )
        schema_version = int(behavioral_raw.get("schema_version", 1))
        if schema_version != 1:
            raise ConfigError(
                f"behavioral_eval.schema_version must be 1, got {schema_version} ({source})",
            )
        checkpoints = str(behavioral_raw.get("checkpoints", "final"))
        if checkpoints != "final":
            raise ConfigError(
                f"behavioral_eval.checkpoints must be 'final', got {checkpoints!r} ({source})",
            )
        protocols = tuple(
            str(value) for value in behavioral_raw.get(
                "protocols", ["native", "hard_argmax"],
            )
        )
        allowed_protocols = {"native", "hard_argmax"}
        unknown_protocols = sorted(set(protocols) - allowed_protocols)
        if not protocols or unknown_protocols:
            raise ConfigError(
                "behavioral_eval.protocols must be a non-empty subset of "
                f"{sorted(allowed_protocols)}, got {list(protocols)} ({source})",
            )
        targets = behavioral_raw.get("targets") or {}
        if not isinstance(targets, Mapping):
            raise ConfigError(f"behavioral_eval.targets: expected a mapping ({source})")
        _check_keys(targets, BEHAVIORAL_TARGET_KEYS, "behavioral_eval.targets", source)
        target_arms_raw = targets.get("arms", "all")
        if target_arms_raw == "all":
            target_arms = ("all",)
        elif isinstance(target_arms_raw, list) and target_arms_raw:
            target_arms = tuple(str(value) for value in target_arms_raw)
            unknown_targets = sorted(set(target_arms) - set(names))
            if unknown_targets:
                raise ConfigError(
                    "behavioral_eval.targets.arms: unknown arm "
                    f"{unknown_targets[0]!r} ({source})",
                )
        else:
            raise ConfigError(
                "behavioral_eval.targets.arms must be 'all' or a non-empty list "
                f"({source})",
            )
        generation = behavioral_raw.get("generation") or {}
        if not isinstance(generation, Mapping):
            raise ConfigError(
                f"behavioral_eval.generation: expected a mapping ({source})",
            )
        _check_keys(
            generation, BEHAVIORAL_GENERATION_KEYS,
            "behavioral_eval.generation", source,
        )
        generation_settings = BehavioralGenerationSettings(
            do_sample=bool(generation.get("do_sample", False)),
            temperature=float(generation.get("temperature", 0.0)),
            batch_size=int(generation.get("batch_size", 8)),
            max_new_tokens=int(generation.get("max_new_tokens", 256)),
            max_input_tokens=(
                int(generation["max_input_tokens"])
                if generation.get("max_input_tokens") is not None else None
            ),
            fail_on_truncation=bool(generation.get("fail_on_truncation", True)),
            seed=int(generation.get("seed", 0)),
        )
        if generation_settings.batch_size <= 0 or generation_settings.max_new_tokens <= 0:
            raise ConfigError(
                "behavioral_eval.generation batch_size and max_new_tokens must be > 0 "
                f"({source})",
            )
        if (
            generation_settings.max_input_tokens is not None
            and generation_settings.max_input_tokens <= 0
        ):
            raise ConfigError(
                "behavioral_eval.generation.max_input_tokens must be > 0 "
                f"({source})",
            )
        if generation_settings.do_sample and generation_settings.temperature <= 0.0:
            raise ConfigError(
                "behavioral_eval.generation.temperature must be > 0 when sampling "
                f"({source})",
            )
        suites_raw = behavioral_raw.get("suites") or []
        if not isinstance(suites_raw, list) or not suites_raw:
            raise ConfigError(
                f"behavioral_eval.suites: expected a non-empty list ({source})",
            )
        suites: list[dict[str, Any]] = []
        suite_names: list[str] = []
        for index, suite in enumerate(suites_raw):
            if not isinstance(suite, Mapping) or "kind" not in suite or "path" not in suite:
                raise ConfigError(
                    "behavioral_eval.suites"
                    f"[{index}]: expected a mapping with kind and path ({source})",
                )
            kind = str(suite["kind"])
            if kind not in BEHAVIORAL_SUITE_KINDS:
                raise ConfigError(
                    "behavioral_eval.suites"
                    f"[{index}].kind: unknown kind {kind!r}; known: "
                    f"{sorted(BEHAVIORAL_SUITE_KINDS)} ({source})",
                )
            grader = suite.get("grader")
            if grader is not None and not isinstance(grader, (str, Mapping)):
                raise ConfigError(
                    f"behavioral_eval.suites[{index}].grader must be a string or mapping "
                    f"({source})",
                )
            suite_generation = suite.get("generation") or {}
            if not isinstance(suite_generation, Mapping):
                raise ConfigError(
                    f"behavioral_eval.suites[{index}].generation must be a mapping "
                    f"({source})",
                )
            _check_keys(
                suite_generation,
                BEHAVIORAL_SUITE_GENERATION_KEYS,
                f"behavioral_eval.suites[{index}].generation",
                source,
            )
            for key in ("batch_size", "max_new_tokens", "max_input_tokens"):
                if suite_generation.get(key) is not None and int(suite_generation[key]) <= 0:
                    raise ConfigError(
                        f"behavioral_eval.suites[{index}].generation.{key} must be > 0 "
                        f"({source})",
                    )
            suite_names.append(kind)
            suites.append(dict(suite))
        if len(suite_names) != len(set(suite_names)):
            raise ConfigError(
                f"behavioral_eval.suites: duplicate suite kinds {suite_names} ({source})",
            )
        contamination = behavioral_raw.get("contamination") or {}
        if not isinstance(contamination, Mapping):
            raise ConfigError(
                f"behavioral_eval.contamination: expected a mapping ({source})",
            )
        _check_keys(
            contamination, BEHAVIORAL_CONTAMINATION_KEYS,
            "behavioral_eval.contamination", source,
        )
        exact_policy = str(contamination.get("exact_match", "exclude"))
        fuzzy_policy = str(contamination.get("fuzzy_match", "flag"))
        if exact_policy not in {"exclude", "flag"}:
            raise ConfigError(
                "behavioral_eval.contamination.exact_match must be exclude or flag "
                f"({source})",
            )
        if fuzzy_policy not in {"exclude", "flag", "off"}:
            raise ConfigError(
                "behavioral_eval.contamination.fuzzy_match must be exclude, flag or off "
                f"({source})",
            )
        contamination_settings = {
            "exact_match": exact_policy,
            "fuzzy_match": fuzzy_policy,
            "ngram_size": int(contamination.get("ngram_size", 5)),
            "fuzzy_threshold": float(contamination.get("fuzzy_threshold", 0.85)),
        }
        if contamination_settings["ngram_size"] <= 0:
            raise ConfigError(
                f"behavioral_eval.contamination.ngram_size must be > 0 ({source})",
            )
        if not 0.0 <= contamination_settings["fuzzy_threshold"] <= 1.0:
            raise ConfigError(
                "behavioral_eval.contamination.fuzzy_threshold must lie in [0,1] "
                f"({source})",
            )
        graders = behavioral_raw.get("graders") or {}
        audit = behavioral_raw.get("audit") or {}
        if not isinstance(graders, Mapping) or not isinstance(audit, Mapping):
            raise ConfigError(
                f"behavioral_eval.graders and audit must be mappings ({source})",
            )
        behavioral_settings = BehavioralSettings(
            schema_version=schema_version,
            checkpoints=checkpoints,
            protocols=protocols,
            include_base_model=bool(targets.get("include_base_model", True)),
            target_arms=target_arms,
            generation=generation_settings,
            suites=tuple(suites),
            contamination=contamination_settings,
            graders=dict(graders),
            audit=dict(audit),
        )

    fg = raw.get("figures") or {}
    _check_keys(fg, FIGURE_SETTING_KEYS, "figures", source)
    compile_tex = str(fg.get("compile_tex", "auto"))
    if compile_tex not in ("auto", "always", "never"):
        raise ConfigError(
            f"figures.compile_tex must be auto, always or never, got {compile_tex!r} ({source})",
        )
    figure_settings = FigureSettings(
        axis_labels=tuple(str(x) for x in (fg.get("axis_labels") or ())),
        formats=tuple(str(x) for x in (fg.get("formats") or ("pdf", "png"))),
        compile_tex=compile_tex,
        include=tuple(str(x) for x in (fg.get("include") or ())),
    )

    sm = raw.get("smoke") or {}
    _check_keys(sm, SMOKE_KEYS, "smoke", source)
    smoke_arms = tuple(str(x) for x in (sm.get("arms") or ()))
    for arm_name in smoke_arms:
        if arm_name not in names:
            raise ConfigError(f"smoke.arms: unknown arm {arm_name!r} ({source})")
    overrides = sm.get("overrides") or {}
    if not isinstance(overrides, Mapping):
        raise ConfigError(f"smoke.overrides: expected a mapping ({source})")
    smoke_settings = SmokeSettings(
        tests=tuple(str(x) for x in (sm.get("tests") or ())),
        arms=smoke_arms,
        output_root=Path(str(sm.get("output_root", "results/_smoke"))),
        overrides=dict(overrides),
    )

    return ExperimentConfig(
        name=name,
        path=p,
        results_dir=Path(str(raw.get("results_dir", f"results/{name}"))),
        figures_dir=Path(str(raw.get("figures_dir", f"figures/{name}"))),
        arms=arms,
        stages=stages,
        eval=eval_settings,
        behavioral_eval=behavioral_settings,
        figures=figure_settings,
        smoke=smoke_settings,
    )


__all__ = [
    "ALL_STAGES",
    "DEFAULT_STAGES",
    "ArmSpec",
    "BehavioralGenerationSettings",
    "BehavioralSettings",
    "EvalSettings",
    "ExperimentConfig",
    "FigureSettings",
    "SmokeSettings",
    "load_experiment",
]
