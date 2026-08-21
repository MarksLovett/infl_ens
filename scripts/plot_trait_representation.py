#!/usr/bin/env python3
"""Compare legacy clipped calibration against quantile normalization.

Shows how benchmark prompts are *represented* in the learned trait space
under the two normalization schemes:

- **legacy**: ``clip((w·e - lo)/(hi - lo), 0, 1)`` per axis, then the
  clipped residual rescale, then the concave stretch. This is the
  pre-quantile pipeline; moderate scores pile up against the box faces.
- **new**: unclipped affine scores → residualize → optional frozen linear
  transform → per-axis empirical CDF → stretch → clip. Marginals are
  near-uniform by construction.

Both representations are derived from **one** encode pass over the same
prompts, using the same learned axis directions, so the figure isolates
the normalization stage. Running the pipeline twice via
``--config-override trait_space.*`` would instead change the cache
fingerprint and force a second full re-encode.

Caveat: residualizer coefficients stored in the cache are fitted on
unclipped scores under the current code, so the legacy arm is a faithful
reconstruction of the legacy *normalization*, not a bit-exact replay of a
legacy build.

Run::

    PYTHONPATH=src python scripts/plot_trait_representation.py \\
        --config configs/benchmark/router/seven_axis_pair_merge_split.yaml \\
        --output-dir scripts/figures/trait_repr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infl_ens.data.benchmarks.safety_trait_space import (  # noqa: E402
    _pre_normalizer_coordinates,
    _raw_coordinate_matrix,
)
from infl_ens.data.trait_space_cache import (  # noqa: E402
    build_or_load_safety_trait_space,
    load_cache_artifacts,
    make_trait_space_encoder,
)
from infl_ens.evaluation.benchmarks import load_benchmark_splits  # noqa: E402
from infl_ens.vis.save import save_figure  # noqa: E402

_SATURATION_TOL = 1e-6


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _stratified_sample(
    splits: Sequence[Any],
    max_prompts: int,
    seed: int,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Sample prompts evenly across benchmarks.

    :param splits: Loaded benchmark splits.
    :type splits: Sequence[BenchmarkSplit]
    :param max_prompts: Total prompt budget across all splits.
    :type max_prompts: int
    :param seed: RNG seed for reproducible sampling.
    :type seed: int
    :returns: ``(prompts, split_ids, split_names)``.
    :rtype: tuple[list[str], numpy.ndarray, list[str]]
    """
    rng = np.random.default_rng(seed)
    per_split = max(1, max_prompts // max(len(splits), 1))
    prompts: list[str] = []
    ids: list[int] = []
    names = [s.name for s in splits]
    for idx, split in enumerate(splits):
        pool = list(split.prompts)
        if len(pool) > per_split:
            pick = rng.choice(len(pool), size=per_split, replace=False)
            pool = [pool[int(i)] for i in pick]
        prompts.extend(pool)
        ids.extend([idx] * len(pool))
    return prompts, np.asarray(ids, dtype=int), names


def _legacy_coordinates(
    embeddings: np.ndarray,
    axes: Sequence[Any],
    gammas: np.ndarray,
) -> np.ndarray:
    """Reconstruct the pre-quantile clipped coordinate pipeline.

    :param embeddings: Sentence embeddings, shape ``(N, D)``.
    :type embeddings: numpy.ndarray
    :param axes: Learned axes.
    :type axes: Sequence[LearnedAxis]
    :param gammas: Per-axis stretch exponents.
    :type gammas: numpy.ndarray
    :returns: Legacy coordinates in ``[0, 1]^L``.
    :rtype: numpy.ndarray
    """
    base = np.clip(_raw_coordinate_matrix(embeddings, axes), 0.0, 1.0)
    coords = base.copy()
    for j, axis in enumerate(axes):
        if axis.residual_coef is None or j == 0:
            continue
        pred = axis.residual_intercept + coords[:, :j] @ axis.residual_coef
        resid = base[:, j] - pred
        lo = axis.residual_lo if axis.residual_lo is not None else axis.lo
        hi = axis.residual_hi if axis.residual_hi is not None else axis.hi
        span = max(float(hi) - float(lo), 1e-12)
        coords[:, j] = np.clip((resid - float(lo)) / span, 0.0, 1.0)
    if not np.all(gammas == 1.0):
        coords = 1.0 - np.power(1.0 - np.clip(coords, 0.0, 1.0), gammas)
    return np.clip(coords, 0.0, 1.0)


def _ks_vs_uniform(column: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic of one column against U[0, 1]."""
    v = np.sort(np.asarray(column, dtype=float))
    n = v.shape[0]
    if n == 0:
        return float("nan")
    grid = np.arange(1, n + 1) / n
    return float(np.max(np.maximum(np.abs(grid - v), np.abs(v - (grid - 1.0 / n)))))


def _representation_stats(coords: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    """Per-axis saturation, uniformity, and correlation diagnostics.

    :param coords: Coordinates in ``[0, 1]^L``, shape ``(N, L)``.
    :type coords: numpy.ndarray
    :param labels: Axis names.
    :type labels: Sequence[str]
    :returns: JSON-safe summary statistics.
    :rtype: dict
    """
    x = np.asarray(coords, dtype=float)
    per_axis = []
    for j, name in enumerate(labels):
        col = x[:, j]
        at_zero = float(np.mean(col <= _SATURATION_TOL))
        at_one = float(np.mean(col >= 1.0 - _SATURATION_TOL))
        per_axis.append(
            {
                "axis": name,
                "frac_at_zero": at_zero,
                "frac_at_one": at_one,
                "frac_saturated": at_zero + at_one,
                "ks_vs_uniform": _ks_vs_uniform(col),
                "mean": float(col.mean()),
                "std": float(col.std()),
            },
        )
    corr = np.corrcoef(x, rowvar=False) if x.shape[0] > 1 else np.eye(x.shape[1])
    off = corr - np.diag(np.diag(corr))
    return {
        "n_prompts": int(x.shape[0]),
        "in_unit_box": bool(np.all((x >= 0.0) & (x <= 1.0))),
        "max_frac_saturated": max(a["frac_saturated"] for a in per_axis),
        "max_ks_vs_uniform": max(a["ks_vs_uniform"] for a in per_axis),
        "mean_abs_offdiag_corr": float(np.mean(np.abs(off))),
        "per_axis": per_axis,
    }


def plot_marginals(
    legacy: np.ndarray,
    new: np.ndarray,
    labels: Sequence[str],
    *,
    stats_legacy: dict[str, Any],
    stats_new: dict[str, Any],
    title: Optional[str] = None,
):
    """Per-axis marginal histograms, legacy row over new row.

    :param legacy: Legacy coordinates, shape ``(N, L)``.
    :type legacy: numpy.ndarray
    :param new: Quantile-normalized coordinates, shape ``(N, L)``.
    :type new: numpy.ndarray
    :param labels: Axis names.
    :type labels: Sequence[str]
    :param stats_legacy: Summary stats for the legacy arm.
    :type stats_legacy: dict
    :param stats_new: Summary stats for the new arm.
    :type stats_new: dict
    :param title: Optional figure suptitle.
    :type title: str | None
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n_axes = len(labels)
    fig, axarr = plt.subplots(
        2, n_axes, figsize=(2.6 * n_axes, 5.6), sharex=True, squeeze=False,
    )
    bins = np.linspace(0.0, 1.0, 41)
    rows = (
        ("legacy: clip to [0,1]", legacy, stats_legacy, "#d62728"),
        ("new: quantile CDF", new, stats_new, "#1f77b4"),
    )
    for row, (row_label, coords, stats, color) in enumerate(rows):
        for j, name in enumerate(labels):
            ax = axarr[row][j]
            ax.hist(
                coords[:, j], bins=bins, color=color, alpha=0.85,
                density=True, edgecolor="none",
            )
            ax.axhline(1.0, color="0.35", lw=0.9, ls="--", zorder=3)
            info = stats["per_axis"][j]
            ax.set_title(
                f"{name}\nsat={info['frac_saturated']:.1%}  KS={info['ks_vs_uniform']:.3f}",
                fontsize=8,
            )
            ax.set_xlim(-0.02, 1.02)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(row_label, fontsize=9)
    fig.suptitle(
        title or "Trait-space representation: clipped calibration vs quantile normalization",
        fontsize=11,
    )
    fig.text(
        0.5, 0.005,
        "dashed line = uniform density. sat = fraction of mass pinned at 0 or 1; "
        "KS = deviation from U[0,1].",
        ha="center", fontsize=8, color="0.3",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig


def plot_pair_comparison(
    legacy: np.ndarray,
    new: np.ndarray,
    labels: Sequence[str],
    *,
    pairs: Sequence[tuple[int, int]],
    bins: int = 48,
    title: Optional[str] = None,
):
    """Side-by-side 2-D empirical densities for selected axis pairs.

    :param legacy: Legacy coordinates, shape ``(N, L)``.
    :type legacy: numpy.ndarray
    :param new: Quantile-normalized coordinates, shape ``(N, L)``.
    :type new: numpy.ndarray
    :param labels: Axis names.
    :type labels: Sequence[str]
    :param pairs: Axis index pairs to render.
    :type pairs: Sequence[tuple[int, int]]
    :param bins: Histogram resolution per axis.
    :type bins: int
    :param title: Optional figure suptitle.
    :type title: str | None
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n_pairs = len(pairs)
    fig, axarr = plt.subplots(
        2, n_pairs, figsize=(2.9 * n_pairs, 6.0), squeeze=False,
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = (("legacy", legacy), ("new", new))
    for row, (row_label, coords) in enumerate(rows):
        for k, (i, j) in enumerate(pairs):
            ax = axarr[row][k]
            heat, _, _ = np.histogram2d(
                coords[:, i], coords[:, j], bins=(edges, edges),
            )
            total = heat.sum()
            if total > 0:
                heat = heat / total
            vmax = np.percentile(heat[heat > 0], 98) if np.any(heat > 0) else 1.0
            ax.imshow(
                heat.T, origin="lower", extent=(0, 1, 0, 1),
                aspect="equal", cmap="magma", vmin=0.0, vmax=max(vmax, 1e-12),
            )
            ax.set_xlabel(labels[i], fontsize=8)
            if k == 0:
                ax.set_ylabel(f"{row_label}\n{labels[j]}", fontsize=8)
            else:
                ax.set_ylabel(labels[j], fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(
        title or "Pairwise prompt density: clipped calibration (top) vs quantile (bottom)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_dataset_composition(
    split_ids: np.ndarray,
    split_names: Sequence[str],
    *,
    title: Optional[str] = None,
):
    """Sampled prompt counts per benchmark.

    :param split_ids: Per-prompt benchmark index, shape ``(N,)``.
    :type split_ids: numpy.ndarray
    :param split_names: Benchmark names.
    :type split_names: Sequence[str]
    :param title: Optional figure title.
    :type title: str | None
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    counts = [int(np.sum(split_ids == i)) for i in range(len(split_names))]
    fig, ax = plt.subplots(figsize=(1.3 * len(split_names) + 2.5, 3.6))
    ax.bar(range(len(split_names)), counts, color="#4c72b0")
    ax.set_xticks(range(len(split_names)))
    ax.set_xticklabels(split_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("prompts sampled", fontsize=9)
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=8)
    ax.set_title(title or "Sampled prompts per benchmark", fontsize=10)
    fig.tight_layout()
    return fig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, required=True, help="Router YAML config.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "scripts" / "figures" / "trait_repr",
    )
    p.add_argument("--max-prompts", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pair-bins", type=int, default=48)
    p.add_argument(
        "--max-pairs",
        type=int,
        default=4,
        help="How many axis pairs to render in the pairwise figure.",
    )
    p.add_argument("--title-prefix", type=str, default=None)
    p.add_argument("--dpi", type=int, default=220)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    cfg = _load_yaml(args.config)

    entries = []
    for entry in cfg.get("benchmarks", []):
        resolved = dict(entry)
        path = Path(str(entry["path"]))
        if not path.is_absolute():
            resolved["path"] = str(ROOT / path)
        entries.append(resolved)
    splits = load_benchmark_splits(entries)
    print(f"[repr] loaded {len(splits)} benchmark splits")

    # Builds and caches the trait space; this is the expensive encode.
    space = build_or_load_safety_trait_space(cfg, splits)
    print(f"[repr] trait space ready: L={space.L}, K={space.K}")

    artifacts = load_cache_artifacts(cfg)
    labels = list(artifacts.axis_labels)

    prompts, split_ids, split_names = _stratified_sample(
        splits, args.max_prompts, args.seed,
    )
    print(f"[repr] encoding {len(prompts)} sampled prompts (single pass)")
    encoder = make_trait_space_encoder(cfg)
    emb = np.asarray(encoder(prompts), dtype=float)

    legacy = _legacy_coordinates(emb, artifacts.axes, artifacts.gammas)
    pre = _pre_normalizer_coordinates(emb, artifacts.axes, artifacts.linear_transform)
    new = artifacts.normalizer.transform(pre)
    if not np.all(artifacts.gammas == 1.0):
        new = 1.0 - np.power(1.0 - np.clip(new, 0.0, 1.0), artifacts.gammas)
    new = np.clip(new, 0.0, 1.0)

    stats_legacy = _representation_stats(legacy, labels)
    stats_new = _representation_stats(new, labels)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.title_prefix or args.config.stem

    fig = plot_marginals(
        legacy, new, labels,
        stats_legacy=stats_legacy,
        stats_new=stats_new,
        title=f"{prefix}: trait marginals, clipped vs quantile",
    )
    written = save_figure(fig, out_dir / "trait_marginals_old_vs_new", dpi=args.dpi)
    print(f"[repr] wrote {[str(p) for p in written]}")

    n_axes = len(labels)
    all_pairs = [(i, j) for i in range(n_axes) for j in range(i + 1, n_axes)]
    pairs = all_pairs[: max(1, args.max_pairs)]
    fig = plot_pair_comparison(
        legacy, new, labels,
        pairs=pairs,
        bins=args.pair_bins,
        title=f"{prefix}: pairwise density, clipped (top) vs quantile (bottom)",
    )
    written = save_figure(fig, out_dir / "trait_pairs_old_vs_new", dpi=args.dpi)
    print(f"[repr] wrote {[str(p) for p in written]}")

    fig = plot_dataset_composition(
        split_ids, split_names, title=f"{prefix}: sampled prompts per benchmark",
    )
    written = save_figure(fig, out_dir / "dataset_composition", dpi=args.dpi)
    print(f"[repr] wrote {[str(p) for p in written]}")

    summary = {
        "config": str(args.config),
        "axis_labels": labels,
        "n_prompts_sampled": len(prompts),
        "quantile_knots": int(artifacts.normalizer.n_knots),
        "normalizer_fit_n": int(artifacts.normalizer.fit_n),
        "linear_transform": (
            artifacts.linear_transform.mode
            if artifacts.linear_transform is not None
            else None
        ),
        "coordinate_stretch_gammas": artifacts.gammas.tolist(),
        "legacy": stats_legacy,
        "new": stats_new,
    }
    summary_path = out_dir / "trait_repr_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[repr] wrote {summary_path}")

    print("\n=== representation summary ===")
    print(f"{'axis':<20} {'legacy sat':>11} {'new sat':>9} {'legacy KS':>10} {'new KS':>8}")
    for a, b in zip(stats_legacy["per_axis"], stats_new["per_axis"]):
        print(
            f"{a['axis']:<20} {a['frac_saturated']:>10.1%} "
            f"{b['frac_saturated']:>8.1%} {a['ks_vs_uniform']:>10.3f} "
            f"{b['ks_vs_uniform']:>8.3f}",
        )
    print(f"\nnew in_unit_box={stats_new['in_unit_box']} "
          f"max_ks={stats_new['max_ks_vs_uniform']:.3f} "
          f"max_sat={stats_new['max_frac_saturated']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
