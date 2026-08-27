"""Per-axis niche diagnostics: variance, ICA, and routing mid-mass.

Runs three pass/fail gates per benchmark axis on a router config:

1. **Variance** — supervised axis aligns with a PCA direction (unsupervised
   variance structure).
2. **ICA** — supervised axis aligns with an independent component.
3. **Mid-mass** — benchmark has distinct prompt cloud *and* its merge pair
   receives non-trivial proportional :math:`G` mass on the flat pool
   (``4 G(1-G)`` averaged over prompts).

Axes failing any gate are candidates for collapse; passing axes are candidates
for router-only fixes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from infl_ens.evaluation.routing_eval import (
    aggregate_clone_g_to_merge,
    load_final_positions,
    load_flat_partition_pool,
    parse_merge_groups,
    resolve_merge_adapters,
)
from infl_ens.inflgame.router.allocation import allocation_weights
from infl_ens.training.__main__ import _load_splits, _load_yaml, _make_trait_space, _sigma_from_cfg

DEFAULT_THRESHOLDS: dict[str, float] = {
    "structure_cosine": 0.15,
    "distinctness": 0.15,
    "mean_merge_g": 0.05,
}


@dataclass(frozen=True)
class AxisNicheResult:
    """Niche diagnostic for one benchmark axis.

    :param benchmark: Benchmark identifier (e.g. ``beavertails``).
    :type benchmark: str
    :param axis_name: Trait-space axis label.
    :type axis_name: str
    :param pca_cosine: Max absolute cosine with a PCA component.
    :type pca_cosine: float
    :param ica_cosine: Max absolute cosine with an ICA component.
    :type ica_cosine: float
    :param distinctness: NN-mixing niche distinctness in trait space.
    :type distinctness: float
    :param mid_mass_g: Mean ``4 G(1-G)`` for the axis merge on its prompts.
    :type mid_mass_g: float
    :param mean_merge_g: Mean merge-level :math:`G` on the axis prompts.
    :type mean_merge_g: float
    :param variance_pass: Whether PCA or ICA structure gate passed.
    :type variance_pass: bool
    :param ica_pass: Alias for structure gate (max PCA/ICA cosine).
    :type ica_pass: bool
    :param mid_mass_pass: Whether the mid-mass gate passed.
    :type mid_mass_pass: bool
    :param passes: Whether all three gates passed.
    :type passes: bool
  """

    benchmark: str
    axis_name: str
    pca_cosine: float
    ica_cosine: float
    distinctness: float
    mid_mass_g: float
    mean_merge_g: float
    variance_pass: bool
    ica_pass: bool
    mid_mass_pass: bool
    passes: bool


def _unit(v: np.ndarray) -> np.ndarray:
    """Return a unit-norm copy of ``v``.

    :param v: Input vector.
    :type v: numpy.ndarray
    :returns: Unit vector.
    :rtype: numpy.ndarray
    """
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _lda_direction(
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    shrinkage: float = 0.1,
) -> np.ndarray:
    """Shrinkage Fisher LDA direction.

    :param pos: Positive-class embeddings.
    :type pos: numpy.ndarray
    :param neg: Negative-class embeddings.
    :type neg: numpy.ndarray
    :param shrinkage: Covariance shrinkage in ``[0, 1]``.
    :type shrinkage: float
    :returns: Unit axis direction.
    :rtype: numpy.ndarray
    """
    mu_pos, mu_neg = pos.mean(axis=0), neg.mean(axis=0)
    d = pos.shape[1]
    cov_pos = np.cov(pos, rowvar=False) if len(pos) > 1 else np.zeros((d, d))
    cov_neg = np.cov(neg, rowvar=False) if len(neg) > 1 else np.zeros((d, d))
    sw = ((len(pos) - 1) * cov_pos + (len(neg) - 1) * cov_neg) / max(
        len(pos) + len(neg) - 2, 1,
    )
    tau = float(np.trace(sw) / d) if d else 0.0
    sw_reg = (1.0 - shrinkage) * sw + shrinkage * tau * np.eye(d)
    try:
        direction = np.linalg.solve(sw_reg, mu_pos - mu_neg)
    except np.linalg.LinAlgError:
        direction = mu_pos - mu_neg
    return _unit(direction)


def _pca(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centered PCA via thin SVD.

    :param x: Data matrix ``(N, D)``.
    :type x: numpy.ndarray
    :returns: ``(components, explained_variance_ratio)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    mean = x.mean(axis=0)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = (s ** 2) / max(len(x) - 1, 1)
    ratio = var / var.sum() if var.sum() > 0 else var
    return vt, ratio


def _fastica(x: np.ndarray, n_components: int, *, seed: int = 0) -> np.ndarray:
    """Deflationary FastICA (log-cosh), components in original space.

    :param x: Data matrix ``(N, D)``.
    :type x: numpy.ndarray
    :param n_components: Number of components.
    :type n_components: int
    :param seed: RNG seed.
    :type seed: int
    :returns: Unit rows ``(C, D)``.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    mean = x.mean(axis=0)
    xc = (x - mean).T
    d, n = xc.shape
    c = min(n_components, d)
    cov = (xc @ xc.T) / n
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:c]
    vals = np.clip(vals[order], 1e-10, None)
    vecs = vecs[:, order]
    whiten = np.diag(vals ** -0.5) @ vecs.T
    xw = whiten @ xc
    w_mat = np.zeros((c, c))
    for i in range(c):
        w = _unit(rng.standard_normal(c))
        for _ in range(300):
            ws = w @ xw
            g = np.tanh(ws)
            g_prime = 1.0 - g ** 2
            w_new = (xw * g).mean(axis=1) - g_prime.mean() * w
            for j in range(i):
                w_new -= (w_new @ w_mat[j]) * w_mat[j]
            w_new = _unit(w_new)
            if np.abs(np.abs(w_new @ w) - 1.0) < 1e-5:
                w = w_new
                break
            w = w_new
        w_mat[i] = w
    comps = w_mat @ whiten
    return np.stack([_unit(row) for row in comps], axis=0)


def _max_abs_cosine(axis: np.ndarray, components: np.ndarray) -> float:
    """Maximum absolute cosine between ``axis`` and component rows.

    :param axis: Supervised axis, shape ``(D,)``.
    :type axis: numpy.ndarray
    :param components: Component matrix, shape ``(K, D)``.
    :type components: numpy.ndarray
    :returns: Max absolute cosine.
    :rtype: float
    """
    if components.size == 0:
        return 0.0
    cos = np.abs(components @ _unit(axis))
    return float(cos.max())


def _nn_mixing(
    coords: np.ndarray,
    origins: np.ndarray,
    n_classes: int,
    *,
    k_nn: int = 15,
    seed: int = 0,
) -> np.ndarray:
    """Row-normalised k-NN origin mixing matrix.

    :param coords: Trait coordinates ``(N, L)``.
    :type coords: numpy.ndarray
    :param origins: Benchmark index per row.
    :type origins: numpy.ndarray
    :param n_classes: Number of benchmarks.
    :type n_classes: int
    :param k_nn: Neighbours per query.
    :type k_nn: int
    :param seed: RNG seed for tie breaks.
    :type seed: int
    :returns: Mixing matrix ``(K, K)``.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    n = coords.shape[0]
    k = min(k_nn, n - 1)
    dists = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(dists, np.inf)
    nn_idx = np.argpartition(dists, kth=k, axis=1)[:, :k]
    mix = np.zeros((n_classes, n_classes), dtype=float)
    for i in range(n):
        neigh = nn_idx[i]
        order = neigh[np.argsort(dists[i, neigh] + 1e-9 * rng.random(len(neigh)))]
        for j in order:
            mix[origins[i], origins[j]] += 1.0
    row_sum = mix.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    return mix / row_sum


def _distinctness(nn_mix: np.ndarray, k: int) -> float:
    """Niche distinctness for row ``k`` of an NN-mixing matrix.

    :param nn_mix: Row-normalised mixing matrix.
    :type nn_mix: numpy.ndarray
    :param k: Row index.
    :type k: int
    :returns: Distinctness in ``[0, 1]``.
    :rtype: float
    """
    n = nn_mix.shape[0]
    row = nn_mix[k]
    self_mix = float(row[k])
    off = np.delete(row, k)
    mean_off = float(off.mean()) if off.size else 0.0
    denom = max(1.0 - 1.0 / n, 1e-12)
    return (self_mix - mean_off) / denom


def _benchmark_to_merge(
    benchmark: str,
    clone_to_merge: Mapping[str, str],
    merge_name_map: Mapping[str, str],
) -> str | None:
    """Map a benchmark id to its own-axis merge adapter name.

    :param benchmark: Benchmark identifier.
    :type benchmark: str
    :param clone_to_merge: Clone to config merge mapping.
    :type clone_to_merge: Mapping[str, str]
    :param merge_name_map: Config merge to resolved merge mapping.
    :type merge_name_map: Mapping[str, str]
    :returns: Resolved merge name, if known.
    :rtype: str | None
    """
    from infl_ens.evaluation.specialist_tables import AXIS_SPECIALIST

    spec = AXIS_SPECIALIST.get(benchmark)
    if spec is None:
        if benchmark == "jbb_behaviors":
            spec = "merge-jailbreak"
        else:
            return None
    return merge_name_map.get(spec, spec)


def run_axis_niche_diagnostic(
    *,
    router_config: Path,
    repo_root: Path,
    history_path: Path | None = None,
    merge_run_dir: Path | None = None,
    partition: str = "test",
    max_eval_records: int | None = 1000,
    seed: int = 0,
    threshold: float = 0.5,
    shrinkage: float = 0.1,
    n_ica: int = 8,
    k_nn: int = 15,
    thresholds: Mapping[str, float] | None = None,
) -> list[AxisNicheResult]:
    """Run variance / ICA / mid-mass gates on every benchmark axis.

    :param router_config: Router YAML path.
    :type router_config: pathlib.Path
    :param repo_root: Repository root.
    :type repo_root: pathlib.Path
    :param history_path: Optional closed-loop history for routing mid-mass.
    :type history_path: pathlib.Path | None
    :param merge_run_dir: Merge run dir (needed with ``history_path``).
    :type merge_run_dir: pathlib.Path | None
    :param partition: Data partition for routing mid-mass.
    :type partition: str
    :param max_eval_records: Per-benchmark cap for routing analysis.
    :type max_eval_records: int | None
    :param seed: RNG seed.
    :type seed: int
    :param threshold: Positive/negative score split for LDA axes.
    :type threshold: float
    :param shrinkage: LDA shrinkage.
    :type shrinkage: float
    :param n_ica: ICA components.
    :type n_ica: int
    :param k_nn: Neighbours for mixing matrix.
    :type k_nn: int
    :param thresholds: Override default pass thresholds.
    :type thresholds: Mapping[str, float] | None
    :returns: One result per loaded benchmark split.
    :rtype: list[AxisNicheResult]
    """
    thr = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        thr.update(thresholds)

    cfg = _load_yaml(router_config)
    splits = _load_splits(cfg)
    from infl_ens.data.encoders import make_encoder

    encoder = make_encoder(cfg)

    blocks: list[np.ndarray] = []
    origin_blocks: list[np.ndarray] = []
    bench_names: list[str] = []
    axis_names: list[str] = []
    axes: list[np.ndarray] = []
    for k, split in enumerate(splits):
        emb = np.asarray(encoder(list(split.prompts)), dtype=float)
        blocks.append(emb)
        origin_blocks.append(np.full(len(split.prompts), k, dtype=int))
        bench_names.append(split.name)
        axis_names.append(split.axis_name or split.name)
        labels = (split.scores >= threshold).astype(int)
        pos, neg = emb[labels == 1], emb[labels == 0]
        if len(pos) < 2 or len(neg) < 2:
            axes.append(np.zeros(emb.shape[1]))
        else:
            axes.append(_lda_direction(pos, neg, shrinkage=shrinkage))
    emb_all = np.concatenate(blocks, axis=0)
    origins = np.concatenate(origin_blocks, axis=0)
    axes_arr = np.stack(axes, axis=0)

    pca_comps, _ = _pca(emb_all)
    ica_comps = _fastica(emb_all, n_ica, seed=seed)
    space = _make_trait_space(cfg, splits)
    sup_coords = (emb_all - emb_all.mean(axis=0)) @ axes_arr.T
    nn_mix = _nn_mixing(sup_coords, origins, len(splits), k_nn=k_nn, seed=seed)

    g_merge_by_bench: dict[str, np.ndarray] = {}
    if history_path is not None and merge_run_dir is not None:
        cl = cfg.get("closed_loop", {})
        clone_to_merge, config_merge_names = parse_merge_groups(cl)
        agent_names = [a["name"] for a in cfg["agents"]]
        rnd = json.loads(history_path.read_text(encoding="utf-8"))[-1]["round"]
        merge_names, merge_name_map = resolve_merge_adapters(
            merge_run_dir, int(rnd), config_merge_names,
        )
        prompts, _, bench_labels = load_flat_partition_pool(
            cfg,
            repo_root=repo_root,
            partition=partition,
            max_eval_records=max_eval_records,
            seed=seed,
        )
        positions = load_final_positions(history_path, agent_names)
        coords = np.asarray(space.project(prompts), dtype=float)
        sigma = _sigma_from_cfg(cfg, len(agent_names), space)
        cov = float(sigma) ** 2 * np.eye(space.L)
        g_clone = allocation_weights(positions, coords, cov)
        g_merge = aggregate_clone_g_to_merge(
            g_clone, agent_names, clone_to_merge, merge_names, merge_name_map,
        )
        for bench in bench_names:
            merge = _benchmark_to_merge(bench, clone_to_merge, merge_name_map)
            if merge is None or merge not in merge_names:
                continue
            j = merge_names.index(merge)
            mask = np.array([b == bench for b in bench_labels])
            if mask.any():
                g_merge_by_bench[bench] = g_merge[j, mask]

    results: list[AxisNicheResult] = []
    for k, bench in enumerate(bench_names):
        pca_cos = _max_abs_cosine(axes_arr[k], pca_comps)
        ica_cos = _max_abs_cosine(axes_arr[k], ica_comps)
        distinct = _distinctness(nn_mix, k)
        g_vec = g_merge_by_bench.get(bench)
        if g_vec is not None and g_vec.size:
            mid_mass = float(np.mean(4.0 * g_vec * (1.0 - g_vec)))
            mean_g = float(g_vec.mean())
        else:
            mid_mass = 0.0
            mean_g = 0.0

        variance_pass = max(pca_cos, ica_cos) >= thr["structure_cosine"]
        ica_pass = variance_pass
        mid_mass_pass = (
            distinct >= thr["distinctness"]
            and mean_g >= thr["mean_merge_g"]
        )
        results.append(
            AxisNicheResult(
                benchmark=bench,
                axis_name=axis_names[k],
                pca_cosine=pca_cos,
                ica_cosine=ica_cos,
                distinctness=distinct,
                mid_mass_g=mid_mass,
                mean_merge_g=mean_g,
                variance_pass=variance_pass,
                ica_pass=ica_pass,
                mid_mass_pass=mid_mass_pass,
                passes=variance_pass and mid_mass_pass,
            ),
        )
    return results


def format_niche_markdown(results: Sequence[AxisNicheResult]) -> str:
    """Render niche diagnostic table as markdown.

    :param results: Per-axis niche results.
    :type results: Sequence[AxisNicheResult]
    :returns: Markdown text.
    :rtype: str
    """
    lines = [
        "## Axis niche gates (variance / ICA / mid-mass)",
        "",
        "| Benchmark | PCA | ICA | distinct | mid-mass G | mean G | struct | mid | pass |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for r in results:
        mark = "✓" if r.passes else "✗"
        struct = "✓" if r.variance_pass else "✗"
        mid = "✓" if r.mid_mass_pass else "✗"
        lines.append(
            f"| {r.benchmark} | {r.pca_cosine:.3f} | {r.ica_cosine:.3f} | "
            f"{r.distinctness:.3f} | {r.mid_mass_g:.3f} | {r.mean_merge_g:.3f} | "
            f"{struct} | {mid} | {mark} |",
        )
    passed = [r.benchmark for r in results if r.passes]
    failed = [r.benchmark for r in results if not r.passes]
    lines.extend([
        "",
        f"**Pass:** {', '.join(passed) or '(none)'}",
        f"**Fail:** {', '.join(failed) or '(none)'}",
        "",
    ])
    return "\n".join(lines)


def niche_results_to_dict(results: Sequence[AxisNicheResult]) -> dict[str, Any]:
    """Serialize niche results to JSON-safe dict.

    :param results: Per-axis results.
    :type results: Sequence[AxisNicheResult]
    :returns: Summary dict with ``axes`` list and pass/fail partitions.
    :rtype: dict[str, Any]
    """
    return {
        "passing_axes": [r.benchmark for r in results if r.passes],
        "failing_axes": [r.benchmark for r in results if not r.passes],
        "axes": [
            {
                "benchmark": r.benchmark,
                "axis_name": r.axis_name,
                "pca_cosine": r.pca_cosine,
                "ica_cosine": r.ica_cosine,
                "distinctness": r.distinctness,
                "mid_mass_g": r.mid_mass_g,
                "mean_merge_g": r.mean_merge_g,
                "variance_pass": r.variance_pass,
                "ica_pass": r.ica_pass,
                "mid_mass_pass": r.mid_mass_pass,
                "passes": r.passes,
            }
            for r in results
        ],
    }
