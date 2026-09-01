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
    "figures",
    "prune",
)
DEFAULT_STAGES: tuple[str, ...] = ("manifest", "train", "perround", "routing", "figures")
ARM_ROLES: frozenset[str] = frozenset({"specialist", "generalist"})

EXPERIMENT_KEYS: frozenset[str] = frozenset(
    {"name", "results_dir", "figures_dir", "arms", "stages", "eval", "figures", "smoke"},
)
ARM_KEYS: frozenset[str] = frozenset(
    {"name", "label", "title", "role", "config", "family", "scale"},
)
EVAL_SETTING_KEYS: frozenset[str] = frozenset(
    {"perround_rounds", "perround_partition", "routing_partition", "max_eval_records"},
)
FIGURE_SETTING_KEYS: frozenset[str] = frozenset(
    {"axis_labels", "formats", "compile_tex", "include"},
)
SMOKE_KEYS: frozenset[str] = frozenset({"tests", "arms", "output_root", "overrides"})


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
    :param family: Optional model family label (e.g. ``Qwen2.5``) used to
        group arms into a family x scale grid and to pair each specialist
        with its same-family/scale generalist.
    :type family: str | None
    :param scale: Optional nominal model scale bucket (e.g. ``1b``).
    :type scale: str | None
    """

    name: str
    label: str
    title: str
    role: str
    config_path: Path
    output_dir: Path
    family: str | None = None
    scale: str | None = None

    @property
    def cell(self) -> tuple[str | None, str | None]:
        """The ``(family, scale)`` grid coordinate of this arm."""
        return (self.family, self.scale)

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

    @property
    def specialists(self) -> tuple[ArmSpec, ...]:
        """Arms with ``role: specialist``, in order."""
        return tuple(a for a in self.arms if a.is_specialist)

    @property
    def generalists(self) -> tuple[ArmSpec, ...]:
        """Every ``role: generalist`` arm, in order."""
        return tuple(a for a in self.arms if a.role == "generalist")

    @property
    def generalist(self) -> ArmSpec | None:
        """The single ``role: generalist`` arm, if exactly one is defined."""
        gens = self.generalists
        return gens[0] if len(gens) == 1 else (gens[0] if gens else None)

    def generalist_for(self, specialist: ArmSpec) -> ArmSpec | None:
        """The generalist paired with ``specialist`` for route-then-score.

        When arms carry ``family``/``scale`` metadata (the scale-family
        sweep) the generalist sharing the specialist's ``(family, scale)``
        cell is returned. Otherwise, and to preserve single-generalist
        experiments, the sole generalist is returned.

        :param specialist: A specialist arm.
        :type specialist: ArmSpec
        :returns: The matching generalist, or ``None`` if none is defined.
        :rtype: ArmSpec | None
        """
        gens = self.generalists
        if not gens:
            return None
        if specialist.family is not None or specialist.scale is not None:
            for gen in gens:
                if gen.cell == specialist.cell:
                    return gen
            return None
        return gens[0] if len(gens) == 1 else None

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
    family = raw.get("family")
    scale = raw.get("scale")
    return ArmSpec(
        name=name,
        label=str(raw.get("label", name)),
        title=str(raw.get("title", raw.get("label", name))),
        role=role,
        config_path=config_path,
        output_dir=Path(str(cfg["output_dir"])),
        family=None if family is None else str(family),
        scale=None if scale is None else str(scale),
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
    generalists = [a for a in arms if a.role == "generalist"]
    if len(generalists) > 1:
        # Multiple generalists are only meaningful when each is pinned to a
        # distinct (family, scale) cell so it pairs 1:1 with a specialist.
        cells: list[tuple[str | None, str | None]] = []
        for gen in generalists:
            if gen.family is None or gen.scale is None:
                raise ConfigError(
                    f"arms: generalist {gen.name!r} must set family and scale "
                    f"when more than one generalist is defined ({source})",
                )
            cells.append(gen.cell)
        if len(set(cells)) != len(cells):
            raise ConfigError(
                f"arms: generalist (family, scale) cells must be unique, "
                f"got {cells} ({source})",
            )

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
        figures=figure_settings,
        smoke=smoke_settings,
    )


__all__ = [
    "ALL_STAGES",
    "DEFAULT_STAGES",
    "ArmSpec",
    "EvalSettings",
    "ExperimentConfig",
    "FigureSettings",
    "SmokeSettings",
    "load_experiment",
]
