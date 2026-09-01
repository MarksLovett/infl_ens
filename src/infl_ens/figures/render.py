"""The only impure layer of :mod:`infl_ens.figures`: read run artifacts, plot, save.

Every figure of an experiment is a :class:`FigureSpec` in :data:`FIGURES`.
A spec reads what the pipeline stages wrote under each arm's ``output_dir``
(``history.json``, ``resolved_config.yaml``, ``routing_ensemble_diagnostics.json``,
``eval_<partition>/``), calls the pure plot functions of the sibling modules,
and writes into the experiment's ``figures_dir``.

A figure whose inputs are missing (a stage that has not run yet) is
skipped with a log line rather than aborting the whole render, so
``python -m infl_ens.figures`` can be re-run as results accumulate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from infl_ens.experiment import ArmSpec, ExperimentConfig
from infl_ens.figures.save import save_figure
from infl_ens.figures.style import apply_paper_style

log = logging.getLogger("infl_ens.figures")

RenderFn = Callable[[ExperimentConfig, Path], list[Path]]


@dataclass(frozen=True)
class FigureSpec:
    """One renderable figure or table of an experiment.

    :param name: Registry key (used in ``figures.include`` and ``--only``).
    :type name: str
    :param description: One-line description for ``--list``.
    :type description: str
    :param render: Callable writing the outputs for an experiment.
    :type render: Callable[[ExperimentConfig, pathlib.Path], list[pathlib.Path]]
    :param requires_gpu: Whether the figure needs the encoder (skipped
        unless asked for explicitly).
    :type requires_gpu: bool
    """

    name: str
    description: str
    render: RenderFn
    requires_gpu: bool = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _history(arm: ArmSpec) -> list[dict[str, Any]]:
    from infl_ens.figures.cross_arm_report import load_history

    return load_history(arm.run_dir)


def _resolved_config(arm: ArmSpec) -> dict[str, Any]:
    """The run's flattened config, falling back to resolving the arm YAML."""
    from infl_ens.config import load_config

    path = arm.run_dir / "resolved_config.yaml"
    if path.is_file():
        return load_config(path, validate=False)
    return arm.load()


def _merge_groups(arm: ArmSpec, records: Sequence[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    from infl_ens.figures.pair_positions import (
        merge_groups_from_config,
        merge_groups_from_history,
    )

    groups = merge_groups_from_config(_resolved_config(arm)) or merge_groups_from_history(records)
    if not groups:
        raise ValueError(
            f"{arm.name}: no merge groups in resolved_config.yaml or history pair_members; "
            "was this run configured with sft_merge_groups?"
        )
    return groups


def _routing_report(arm: ArmSpec) -> dict[str, Any]:
    return _load_json(arm.run_dir / "routing_ensemble_diagnostics.json")


def _final_round(arm: ArmSpec) -> int:
    from infl_ens.evaluation.evaluate import final_round_from_history

    return final_round_from_history(arm.run_dir)


def _save(fig: Any, stem: Path, exp: ExperimentConfig) -> list[Path]:
    import matplotlib.pyplot as plt

    written = save_figure(fig, stem, formats=exp.figures.formats)
    plt.close(fig)
    return written


def _write_tex(exp: ExperimentConfig, path: Path, tex: str) -> list[Path]:
    from infl_ens.figures.pgf_tex import compile_tex, latexmk_available

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")
    written = [path]
    mode = exp.figures.compile_tex
    if mode == "always" or (mode == "auto" and latexmk_available()):
        if compile_tex(path):
            written.append(path.with_suffix(".pdf"))
        else:
            log.warning("could not compile %s (latexmk missing or failed)", path)
    return written


def _axis_labels(exp: ExperimentConfig) -> tuple[str, ...]:
    return tuple(exp.figures.axis_labels)


# ---------------------------------------------------------------------------
# renderers (CPU)
# ---------------------------------------------------------------------------


def render_oracle_routing(exp: ExperimentConfig, out: Path) -> list[Path]:
    """One oracle / pooled / learned pgfplots figure per specialist arm."""
    from infl_ens.figures.pgf_tex import oracle_routing_tex

    written: list[Path] = []
    for arm in exp.specialists:
        report = _routing_report(arm)
        tex = oracle_routing_tex(report, experiment_label=arm.title)
        written += _write_tex(exp, out / f"{arm.name}_vs_oracle.tex", tex)
    return written


def render_arm_comparison(exp: ExperimentConfig, out: Path) -> list[Path]:
    """Cross-arm overlay of every specialist arm with a routing report."""
    from infl_ens.figures.pgf_tex import arm_comparison_tex

    arms = [(arm.label, _routing_report(arm)) for arm in exp.specialists]
    return _write_tex(exp, out / "arm_comparison.tex", arm_comparison_tex(arms))


def render_pair_positions(exp: ExperimentConfig, out: Path) -> list[Path]:
    """Final pair positions over every axis pair, per specialist arm."""
    from infl_ens.figures.pair_positions import plot_final_positions

    apply_paper_style()
    written: list[Path] = []
    for arm in exp.specialists:
        records = _history(arm)
        groups = _merge_groups(arm, records)
        fig = plot_final_positions(
            records[-1], groups, axis_labels=_axis_labels(exp), title=arm.title,
        )
        written += _save(fig, out / f"{arm.name}_final_positions", exp)
    return written


def render_within_pair(exp: ExperimentConfig, out: Path) -> list[Path]:
    """Within-pair separation over rounds, per specialist arm."""
    from infl_ens.figures.pair_positions import plot_within_pair

    apply_paper_style()
    written: list[Path] = []
    for arm in exp.specialists:
        records = _history(arm)
        groups = _merge_groups(arm, records)
        fig = plot_within_pair(records, groups, title=arm.title)
        written += _save(fig, out / f"{arm.name}_within_pair", exp)
    return written


def render_closed_loop_history(exp: ExperimentConfig, out: Path) -> list[Path]:
    """Trajectories + utility tracking, per specialist arm."""
    from infl_ens.figures.closed_loop import plot_history

    apply_paper_style()
    written: list[Path] = []
    labels = _axis_labels(exp) or ("axis 0", "axis 1")
    for arm in exp.specialists:
        records = _history(arm)
        fig = plot_history(records, axis_labels=labels, title=arm.title)
        written += _save(fig, out / f"{arm.name}_history", exp)
    return written


def render_cross_arm_report(exp: ExperimentConfig, out: Path) -> list[Path]:
    """Data-matching check, routing headline, pair stability, NLL movement."""
    from infl_ens.figures.cross_arm_report import write_cross_arm_report

    arms = [(arm.label, arm.run_dir) for arm in exp.specialists]
    gen = exp.generalist
    return write_cross_arm_report(arms, out, generalist_run_dir=gen.run_dir if gen else None)


def render_per_round_tables(exp: ExperimentConfig, out: Path) -> list[Path]:
    """Held-out NLL by pair at the evaluation rounds, per specialist arm.

    Written under each run's ``tables/`` (where the cross-arm report reads
    them) and mirrored into the figures directory.
    """
    from infl_ens.figures.per_round_tables import build_per_round_tables

    partition = exp.eval.perround_partition
    written: list[Path] = []
    for arm in exp.specialists:
        rounds = exp.eval.resolve_rounds(_final_round(arm))
        label = f"{arm.title} ({partition})"
        stem = arm.run_dir / "tables" / "pair_nll_by_round"
        paths = build_per_round_tables(
            arm.run_dir / f"eval_{partition}", stem, label=label, rounds=rounds,
        )
        written += list(paths.values())
        mirror = out / f"{arm.name}_pair_nll_by_round"
        mirror.parent.mkdir(parents=True, exist_ok=True)
        for ext, path in paths.items():
            target = mirror.with_suffix(f".{ext}")
            target.write_bytes(path.read_bytes())
            written.append(target)
    return written


def render_family_scale_nll(exp: ExperimentConfig, out: Path) -> list[Path]:
    """Family x scale held-out NLL grid + table for the scale-family sweep.

    Reads each specialist cell's ``routing_ensemble_diagnostics.json`` (the
    route-then-score report), taking the learned-routing NLL as the headline
    metric plus pooled/oracle where present, groups the cells by
    ``(family, scale)`` and writes a heatmap and a csv/md/tex/json table.
    Cells whose routing report is missing are skipped.
    """
    from infl_ens.figures.scale_family import (
        CellNLL,
        plot_family_scale_nll,
        write_family_scale_table,
    )

    cells: list[CellNLL] = []
    families: list[str] = []
    scales: list[str] = []
    for arm in exp.specialists:
        if arm.family is None or arm.scale is None:
            continue
        if arm.family not in families:
            families.append(arm.family)
        if arm.scale not in scales:
            scales.append(arm.scale)
        try:
            flat = _routing_report(arm).get("flat", {})
        except FileNotFoundError:
            log.warning("family_scale_nll: %s has no routing report; skipping cell", arm.name)
            continue
        learned = flat.get("learned_routing_nll", flat.get("learned_routing_expected_nll"))
        if learned is None:
            log.warning("family_scale_nll: %s report has no learned NLL; skipping", arm.name)
            continue
        cells.append(
            CellNLL(
                family=arm.family,
                scale=arm.scale,
                learned_nll=float(learned),
                pooled_nll=(float(flat["pooled_nll"]) if flat.get("pooled_nll") is not None else None),
                oracle_nll=(
                    float(flat["oracle_routing_nll"])
                    if flat.get("oracle_routing_nll") is not None else None
                ),
            )
        )
    if not cells:
        raise ValueError("no specialist cell with a routing report and family/scale metadata")

    apply_paper_style()
    fig = plot_family_scale_nll(
        cells, families=families, scales=scales,
        title=f"{exp.name}: family x scale held-out NLL",
    )
    written = _save(fig, out / "family_scale_nll", exp)
    tables = write_family_scale_table(
        cells, out / "family_scale_nll", families=families, scales=scales, label=exp.name,
    )
    written += list(tables.values())
    return written


# ---------------------------------------------------------------------------
# renderers (encoder required)
# ---------------------------------------------------------------------------


def _reference_arm(exp: ExperimentConfig) -> ArmSpec:
    if not exp.specialists:
        raise ValueError("experiment has no specialist arm to take the trait space from")
    return exp.specialists[0]


def _load_space(cfg: dict[str, Any]) -> tuple[Any, list[Any]]:
    from infl_ens.data.trait_space_cache import build_or_load_safety_trait_space
    from infl_ens.training.setup import load_splits

    splits = load_splits(cfg)
    space = build_or_load_safety_trait_space(cfg, splits)
    return space, splits


def render_trait_representation(exp: ExperimentConfig, out: Path) -> list[Path]:
    """Clipped-vs-quantile trait representation figures (needs the encoder)."""
    from infl_ens.data.benchmarks.safety_trait_space import (
        _pre_normalizer_coordinates,
        build_safety_trait_space_bundle,
    )
    from infl_ens.data.encoders import make_encoder
    from infl_ens.data.trait_space_cache import (
        _trait_space_build_kwargs,
        artifacts_from_bundle,
        load_cache_artifacts,
    )
    from infl_ens.figures.trait_representation import (
        legacy_coordinates,
        plot_dataset_composition,
        plot_marginals,
        plot_pair_comparison,
        representation_stats,
        stratified_sample,
    )
    from infl_ens.training.setup import load_splits

    arm = _reference_arm(exp)
    cfg = arm.load()
    splits = load_splits(cfg)
    encoder = make_encoder(cfg)
    if bool((cfg.get("trait_space") or {}).get("cache", False)):
        _load_space(cfg)  # builds the cache when missing
        artifacts = load_cache_artifacts(cfg)
    else:
        bundle = build_safety_trait_space_bundle(splits, encoder, **_trait_space_build_kwargs(cfg))
        artifacts = artifacts_from_bundle(bundle)
    labels = list(artifacts.axis_labels)

    prompts, split_ids, split_names = stratified_sample(splits, 8000, int(cfg.get("seed", 0)))
    import numpy as np

    emb = np.asarray(encoder(prompts), dtype=float)
    legacy = legacy_coordinates(emb, artifacts.axes, artifacts.gammas)
    pre = _pre_normalizer_coordinates(emb, artifacts.axes)
    cdf_only = np.clip(artifacts.normalizer.transform(pre), 0.0, 1.0)
    has_stretch = not np.all(artifacts.gammas == 1.0)
    new = np.clip(1.0 - np.power(1.0 - cdf_only, artifacts.gammas), 0.0, 1.0) if has_stretch else cdf_only
    stats_legacy = representation_stats(legacy, labels)
    stats_new = representation_stats(new, labels)
    stats_cdf = representation_stats(cdf_only, labels) if has_stretch else None

    apply_paper_style()
    sub = out / "trait_repr"
    written: list[Path] = []
    written += _save(
        plot_marginals(
            legacy, new, labels, stats_legacy=stats_legacy, stats_new=stats_new,
            pre_stretch=cdf_only if has_stretch else None, stats_pre_stretch=stats_cdf,
            title=f"{exp.name}: trait marginals, clipped vs quantile",
        ),
        sub / "trait_marginals_old_vs_new", exp,
    )
    n_axes = len(labels)
    pairs = [(i, j) for i in range(n_axes) for j in range(i + 1, n_axes)][:4]
    written += _save(
        plot_pair_comparison(
            legacy, new, labels, pairs=pairs,
            title=f"{exp.name}: pairwise density, clipped (top) vs quantile (bottom)",
        ),
        sub / "trait_pairs_old_vs_new", exp,
    )
    written += _save(
        plot_dataset_composition(split_ids, split_names, title=f"{exp.name}: sampled prompts per benchmark"),
        sub / "dataset_composition", exp,
    )
    summary = {
        "config": str(arm.config_path),
        "axis_labels": labels,
        "n_prompts_sampled": len(prompts),
        "quantile_knots": int(artifacts.normalizer.n_knots),
        "normalizer_fit_n": int(artifacts.normalizer.fit_n),
        "coordinate_stretch_gammas": artifacts.gammas.tolist(),
        "legacy": stats_legacy,
        "new": stats_new,
        "cdf_only": stats_cdf,
    }
    summary_path = sub / "trait_repr_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written.append(summary_path)
    return written


def render_benchmark_space(exp: ExperimentConfig, out: Path) -> list[Path]:
    """Pairwise resource-density heatmaps with each arm's final positions."""
    from infl_ens.figures.benchmark_space import plot_pairwise_heatmaps

    arm = _reference_arm(exp)
    cfg = arm.load()
    space, splits = _load_space(cfg)
    labels = tuple(space.axis_labels or tuple(f"axis {i}" for i in range(space.L)))
    prompts = [p for split in splits for p in split.prompts]
    import numpy as np

    prompt_coords = space.project(prompts) if prompts else None
    positions: dict[str, np.ndarray] = {}
    try:
        for name, vec in _history(arm)[-1]["positions"].items():
            positions[name] = np.asarray(vec, dtype=float)
    except (FileNotFoundError, ValueError):
        pass

    apply_paper_style()
    fig = plot_pairwise_heatmaps(
        space.grid,
        space.weights,
        axis_labels=labels,
        prompt_coords=prompt_coords,
        positions=positions,
        title=f"{exp.name}: benchmark-space resource distribution",
    )
    return _save(fig, out / "benchmark_space", exp)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

FIGURES: dict[str, FigureSpec] = {
    spec.name: spec
    for spec in (
        FigureSpec("oracle_routing", "oracle vs pooled vs learned per arm (.tex)", render_oracle_routing),
        FigureSpec("arm_comparison", "cross-arm oracle/pooled/learned overlay (.tex)", render_arm_comparison),
        FigureSpec("pair_positions", "final pair positions over every axis pair", render_pair_positions),
        FigureSpec("within_pair", "within-pair separation over rounds", render_within_pair),
        FigureSpec("closed_loop_history", "trajectories + utility tracking per arm", render_closed_loop_history),
        FigureSpec("per_round_tables", "held-out NLL by pair at the eval rounds", render_per_round_tables),
        FigureSpec("cross_arm_report", "data matching, routing headline, pair stability", render_cross_arm_report),
        FigureSpec("family_scale_nll", "family x scale held-out NLL grid + table", render_family_scale_nll),
        FigureSpec("trait_representation", "clipped vs quantile trait marginals", render_trait_representation, True),
        FigureSpec("benchmark_space", "resource-density heatmaps with positions", render_benchmark_space, True),
    )
}


def cpu_figures() -> list[str]:
    """Names of the figures that need no encoder.

    :returns: Registry keys in display order.
    :rtype: list[str]
    """
    return [name for name, spec in FIGURES.items() if not spec.requires_gpu]


def render_all(
    exp: ExperimentConfig,
    *,
    only: Optional[Sequence[str]] = None,
    figures_dir: Optional[Path] = None,
    include_gpu: bool = False,
) -> dict[str, list[Path]]:
    """Render the figures of an experiment.

    :param exp: Parsed experiment.
    :type exp: ExperimentConfig
    :param only: Registry names to render; defaults to ``figures.include``
        of the experiment, or every CPU figure when that is empty.
    :type only: Sequence[str] | None
    :param figures_dir: Override the experiment's ``figures_dir``.
    :type figures_dir: pathlib.Path | None
    :param include_gpu: Also render figures flagged ``requires_gpu`` when
        they are part of the selection.
    :type include_gpu: bool
    :returns: Written paths per figure name (empty list when skipped).
    :rtype: dict[str, list[pathlib.Path]]
    :raises KeyError: For an unknown figure name.
    """
    names = list(only) if only else (list(exp.figures.include) or cpu_figures())
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise KeyError(f"unknown figure(s) {unknown}; known: {list(FIGURES)}")
    out = Path(figures_dir) if figures_dir is not None else Path(exp.figures_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: dict[str, list[Path]] = {}
    for name in names:
        spec = FIGURES[name]
        if spec.requires_gpu and not include_gpu:
            log.info("skipping %s (needs the encoder; pass include_gpu/--gpu)", name)
            written[name] = []
            continue
        try:
            paths = spec.render(exp, out)
        except (FileNotFoundError, ValueError) as exc:
            log.warning("skipping %s: %s", name, exc)
            written[name] = []
            continue
        for path in paths:
            log.info("wrote %s", path)
        written[name] = paths
    return written


__all__ = [
    "FIGURES",
    "FigureSpec",
    "cpu_figures",
    "render_all",
    "render_arm_comparison",
    "render_benchmark_space",
    "render_closed_loop_history",
    "render_cross_arm_report",
    "render_family_scale_nll",
    "render_oracle_routing",
    "render_pair_positions",
    "render_per_round_tables",
    "render_trait_representation",
    "render_within_pair",
]
