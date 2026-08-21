#!/usr/bin/env python3
"""Check whether post-theory-pre agent positions sit on trait-density modes.

Loads GA (or other) run histories, rebuilds the trait space from the run
config, estimates resource modes via (1) KDE grid local maxima, (2) a
5-component GMM on the calibration corpus, and (3) per-benchmark centroids.
Reports distances, density ratios, and peak-assignment overlap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

from infl_ens.data.trait_space import TraitSpace, _kde_on_grid
from infl_ens.data.trait_space_cache import build_or_load_safety_trait_space
from infl_ens.evaluation.benchmarks import load_benchmark_splits

MERGE_GROUPS: list[tuple[str, list[str]]] = [
    ("merge-harm", ["clone-0", "clone-1"]),
    ("merge-hallucination", ["clone-2", "clone-3"]),
    ("merge-privacy", ["clone-4", "clone-5"]),
    ("merge-overrefusal", ["clone-6", "clone-7"]),
    ("merge-policy", ["clone-8", "clone-9"]),
]
AGENT_NAMES = [f"clone-{i}" for i in range(10)]


def _kde_at_points(
    corpus: np.ndarray,
    points: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Unnormalised KDE density at arbitrary query points.

    :param corpus: Calibration coordinates, shape ``(N, L)``.
    :type corpus: numpy.ndarray
    :param points: Query points, shape ``(M, L)``.
    :type points: numpy.ndarray
    :param bandwidth: Isotropic Gaussian bandwidth.
    :type bandwidth: float
    :returns: Density values, shape ``(M,)``.
    :rtype: numpy.ndarray
    """
    diffs = points[:, None, :] - corpus[None, :, :]
    sq = np.sum(diffs ** 2, axis=2)
    log_w = -0.5 * sq / (bandwidth ** 2)
    log_w_max = log_w.max(axis=1, keepdims=True)
    return np.exp(log_w - log_w_max).sum(axis=1)


def _find_grid_local_maxima(
    grid: np.ndarray,
    weights: np.ndarray,
    n_grid: int,
    L: int,
    *,
    min_weight_frac: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local maxima on a regular Cartesian KDE grid.

    :param grid: Flat grid points, shape ``(K, L)``.
    :type grid: numpy.ndarray
    :param weights: Normalised KDE weights, shape ``(K,)``.
    :type weights: numpy.ndarray
    :param n_grid: Points per axis.
    :type n_grid: int
    :param L: Trait dimensionality.
    :type L: int
    :param min_weight_frac: Drop peaks below this fraction of max weight.
    :type min_weight_frac: float
    :returns: ``(peak_positions, peak_weights)`` sorted by descending weight.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    shape = (n_grid,) * L
    tensor = weights.reshape(shape)
    w_max = float(tensor.max())
    cutoff = w_max * min_weight_frac
    peaks: list[tuple[np.ndarray, float]] = []
    for idx in np.ndindex(shape):
        val = float(tensor[idx])
        if val < cutoff:
            continue
        is_max = True
        for offset in np.ndindex(*tuple(3 for _ in range(L))):
            off = tuple(int(o) - 1 for o in offset)
            if all(o == 0 for o in off):
                continue
            nb = tuple(
                max(0, min(n_grid - 1, idx[d] + off[d]))
                for d in range(L)
            )
            if float(tensor[nb]) > val + 1e-15:
                is_max = False
                break
        if is_max:
            flat = int(np.ravel_multi_index(idx, shape))
            peaks.append((grid[flat], val))
    peaks.sort(key=lambda t: t[1], reverse=True)
    if not peaks:
        return np.zeros((0, L)), np.zeros(0)
    pos = np.stack([p[0] for p in peaks], axis=0)
    wts = np.array([p[1] for p in peaks], dtype=float)
    return pos, wts


def _benchmark_centroids_fast(
    splits: Sequence[Any],
    space: TraitSpace,
) -> tuple[np.ndarray, list[str]]:
    """Per-benchmark mean coordinate via ``space.project`` on prompts."""
    centroids: list[np.ndarray] = []
    names: list[str] = []
    for split in splits:
        coords = _project_corpus(space, list(split.prompts))
        centroids.append(coords.mean(axis=0))
        names.append(split.axis_name)
    return np.stack(centroids, axis=0), names


def _project_corpus(space: TraitSpace, prompts: list[str], *, batch: int = 512) -> np.ndarray:
    """Project calibration prompts to trait coordinates."""
    chunks: list[np.ndarray] = []
    for start in range(0, len(prompts), batch):
        block = prompts[start : start + batch]
        coords = np.asarray(space.project(block), dtype=float)
        if coords.ndim == 1:
            coords = coords[None, :]
        chunks.append(coords)
    return np.concatenate(chunks, axis=0)


def _pair_centers(positions: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Collapse merge pairs to one position per pair."""
    centers: list[np.ndarray] = []
    labels: list[str] = []
    for train_as, members in MERGE_GROUPS:
        idx = [AGENT_NAMES.index(m) for m in members]
        centers.append(positions[idx].mean(axis=0))
        labels.append(train_as)
    return np.stack(centers, axis=0), labels


def _extract_positions(hist: list[dict], phase: str) -> np.ndarray | None:
    """Pull ``(10, L)`` positions for a named phase from round-0 history."""
    ti = hist[0].get("theory_init", {})
    pre = ti.get("theory_pre", {})
    if phase == "post_theory_pre":
        end = pre.get("theory_pre_end")
        if end is not None:
            return np.asarray(end, dtype=float)
        end_pos = ti.get("theory_end")
        if end_pos is not None:
            return np.stack(
                [np.asarray(end_pos[n], dtype=float) for n in AGENT_NAMES],
                axis=0,
            )
    if phase == "pre_theory_pre":
        start = pre.get("theory_pre_initial")
        if start is not None:
            return np.asarray(start, dtype=float)
    if phase == "post_init":
        end_pos = hist[0].get("positions")
        if end_pos is not None:
            return np.stack(
                [np.asarray(end_pos[n], dtype=float) for n in AGENT_NAMES],
                axis=0,
            )
    return None


def _match_peaks(
    agents: np.ndarray,
    peaks: np.ndarray,
) -> dict[str, Any]:
    """Hungarian match agents to peaks; report distances and uniqueness."""
    if peaks.shape[0] == 0:
        return {"mean_dist": float("nan"), "max_dist": float("nan"), "pairs": []}
    dist = np.linalg.norm(
        agents[:, None, :] - peaks[None, :, :],
        axis=2,
    )
    n = min(agents.shape[0], peaks.shape[0])
    row_idx, col_idx = linear_sum_assignment(dist[:n, :n])
    pairs = [
        {
            "agent_idx": int(r),
            "peak_idx": int(c),
            "l2": float(dist[r, c]),
        }
        for r, c in zip(row_idx, col_idx, strict=True)
    ]
    dists = [p["l2"] for p in pairs]
    return {
        "mean_dist": float(np.mean(dists)),
        "max_dist": float(np.max(dists)),
        "pairs": pairs,
        "unique_peaks": len(set(col_idx)) == len(col_idx),
    }


def _analyze_phase(
    label: str,
    positions: np.ndarray,
    corpus: np.ndarray,
    bandwidth: float,
    grid_peaks: np.ndarray,
    gmm_means: np.ndarray,
    bench_centroids: np.ndarray,
) -> dict[str, Any]:
    """Summarise one position set against all mode estimates."""
    pairs, pair_labels = _pair_centers(positions)
    density = _kde_at_points(corpus, pairs, bandwidth)
    corpus_density = _kde_at_points(corpus, corpus, bandwidth)
    d_max = float(corpus_density.max())
    d_med = float(np.median(corpus_density))

    return {
        "phase": label,
        "pair_labels": pair_labels,
        "pair_positions": pairs.tolist(),
        "density_at_pairs": density.tolist(),
        "density_frac_of_max": (density / max(d_max, 1e-12)).tolist(),
        "density_percentile_vs_corpus": [
            float(np.mean(corpus_density <= d)) for d in density
        ],
        "match_grid_peaks": _match_peaks(pairs, grid_peaks),
        "match_gmm_means": _match_peaks(pairs, gmm_means),
        "match_benchmark_centroids": _match_peaks(pairs, bench_centroids),
        "mean_pairwise_l2": float(np.mean([
            np.linalg.norm(pairs[i] - pairs[j])
            for i in range(len(pairs))
            for j in range(i + 1, len(pairs))
        ])),
    }


def analyze_run(
    hist_path: Path,
    cfg_path: Path,
    *,
    analysis_n_grid: int = 8,
) -> dict[str, Any]:
    """Run peak-seeking diagnostics for one history file."""
    hist = json.loads(hist_path.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    splits = load_benchmark_splits(cfg.get("benchmarks", []))
    space = build_or_load_safety_trait_space(cfg, splits)
    L = space.L

    # Calibration corpus coordinates (same prompts used for KDE at build).
    cal_prompts = [p for s in splits for p in s.prompts]
    corpus = _project_corpus(space, cal_prompts)

    ts_cfg = cfg.get("trait_space", {})
    bandwidth = float(ts_cfg.get("kde_bandwidth", 0.08))

    # Finer grid for mode finding (training uses coarse n_grid=3).
    fine_axes = [np.linspace(0.0, 1.0, analysis_n_grid) for _ in range(L)]
    mesh = np.meshgrid(*fine_axes, indexing="ij")
    fine_grid = np.stack([m.ravel() for m in mesh], axis=1)
    fine_weights = _kde_on_grid(corpus, fine_grid, bandwidth)
    grid_peaks, peak_weights = _find_grid_local_maxima(
        fine_grid, fine_weights, analysis_n_grid, L,
    )

    gmm = GaussianMixture(
        n_components=5,
        covariance_type="full",
        random_state=0,
        n_init=5,
    )
    gmm.fit(corpus)
    gmm_means = gmm.means_

    bench_centroids, bench_names = _benchmark_centroids_fast(splits, space)

    phases: dict[str, np.ndarray | None] = {
        "post_init": _extract_positions(hist, "post_init"),
        "pre_theory_pre": _extract_positions(hist, "pre_theory_pre"),
        "post_theory_pre": _extract_positions(hist, "post_theory_pre"),
    }

    phase_results = []
    for name, pos in phases.items():
        if pos is None:
            continue
        phase_results.append(
            _analyze_phase(
                name, pos, corpus, bandwidth,
                grid_peaks, gmm_means, bench_centroids,
            ),
        )

    return {
        "history": str(hist_path),
        "config": str(cfg_path),
        "trait_L": L,
        "axis_labels": list(space.axis_labels or []),
        "kde_bandwidth": bandwidth,
        "analysis_n_grid": analysis_n_grid,
        "n_corpus": len(corpus),
        "n_grid_peaks": int(grid_peaks.shape[0]),
        "grid_peaks_top5": grid_peaks[:5].tolist(),
        "grid_peak_weights_top5": peak_weights[:5].tolist(),
        "gmm_means": gmm_means.tolist(),
        "benchmark_centroids": bench_centroids.tolist(),
        "benchmark_names": bench_names,
        "phases": phase_results,
    }


def _print_summary(result: dict[str, Any]) -> None:
    """Human-readable stdout summary."""
    print(f"\n=== {result['history']} ===")
    print(
        f"corpus N={result['n_corpus']}  "
        f"grid peaks={result['n_grid_peaks']}  "
        f"bandwidth={result['kde_bandwidth']}",
    )
    print(f"axis labels: {result['axis_labels']}")
    for phase in result["phases"]:
        print(f"\n  [{phase['phase']}] mean_pairwise={phase['mean_pairwise_l2']:.3f}")
        dens_pct = phase["density_percentile_vs_corpus"]
        print(
            f"    density percentile (per pair): "
            + ", ".join(f"{p:.2f}" for p in dens_pct),
        )
        for mode_name in (
            "match_grid_peaks",
            "match_gmm_means",
            "match_benchmark_centroids",
        ):
            m = phase[mode_name]
            print(
                f"    {mode_name}: mean_L2={m['mean_dist']:.3f} "
                f"max_L2={m['max_dist']:.3f} "
                f"unique={m.get('unique_peaks', '?')}",
            )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmark/router/attribution_2x2/ga_theory_pre.yaml",
    )
    parser.add_argument(
        "--history",
        default="results/attribution_2x2/ga_theory_pre/seed0/history.json",
    )
    parser.add_argument("--analysis-n-grid", type=int, default=8)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        metavar="HISTORY",
        help="Additional history paths (same config).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    histories = [Path(args.history)] + [Path(p) for p in args.extra]
    results = [
        analyze_run(h, cfg_path, analysis_n_grid=args.analysis_n_grid)
        for h in histories
    ]
    for r in results:
        _print_summary(r)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = results[0] if len(results) == 1 else {"runs": results}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
