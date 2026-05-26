"""Diagnose KDE-smoothed resource density vs the actual prompt scatter.

When the closed-loop SFT trajectory and the theoretical Nash diverge,
the most common cause is that the resource density :math:`B(b)` used
by the theory includes mass in regions where no actual prompt
embeddings project. This script makes the gap quantitative.

It supports three density modes (``--density-mode``):

- ``kde``: the original KDE-smoothed :math:`B(b)` from
  :func:`build_safety_trait_space` (controlled by
  ``trait_space.kde_bandwidth``).
- ``empirical``: a 2-D histogram of actual prompt projections on the
  same :math:`n_{\\text{grid}}\\times n_{\\text{grid}}` lattice. Mass
  goes only where prompts actually project; no extrapolation.
- ``both``: run the full pipeline twice, once per density, and produce
  a 2-panel comparison figure. The console table prints both
  side-by-side so the bandwidth-induced ghost-density story is
  visible at a glance.

Other knobs:

- ``--config-override`` repeats with dotted ``KEY=VAL`` syntax to
  override any config field (e.g. ``trait_space.kde_bandwidth=0.04``)
  without editing the YAML.
- ``--radius`` controls the neighbourhood size used to count actual
  prompts near each theory-NE / SFT-end position.
- ``--theory-steps`` / ``--theory-lr`` control the gradient-ascent
  recomputation of theory NE for each density.

Run with::

    python scripts/diagnose_trait_support.py \\
        --config configs/benchmark/router/<...>.yaml \\
        --history results/<...>/history.json \\
        --output-stem scripts/figures/<...>/trait_support \\
        --density-mode both

When ``--density-mode both`` is set, the script writes
``<stem>_kde.{pdf,png}``, ``<stem>_empirical.{pdf,png}``, and
``<stem>_compare.{pdf,png}`` (the 2-panel figure).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any

import numpy as np

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from infl_ens.data.benchmarks import (  # noqa: E402
    BenchmarkSplit,
    build_safety_trait_space,
    load_beavertails,
    load_halueval,
)
from infl_ens.data.encoders import SentenceTransformerEncoder  # noqa: E402
from infl_ens.data.trait_space import TraitSpace  # noqa: E402
from infl_ens.inflgame.router import RouterAgent  # noqa: E402
from infl_ens.training.router_training import (  # noqa: E402
    RouterTrainingConfig,
    train_router_positions,
)
from infl_ens.utils.resource import gaussian_stability_threshold  # noqa: E402


# -----------------------------------------------------------------------------
# Config helpers (mirror infl_ens.training.__main__)
# -----------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file via PyYAML.

    :param path: Path to the config file.
    :type path: pathlib.Path
    :returns: Parsed configuration mapping.
    :rtype: dict
    """
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    """Apply dotted ``KEY=VAL`` overrides in place; JSON-decode values.

    :param cfg: Mutable config dictionary.
    :type cfg: dict
    :param overrides: List of dotted ``KEY=VAL`` strings.
    :type overrides: list[str]
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

    :param cfg: Parsed YAML config.
    :type cfg: dict
    :returns: List of loaded benchmark splits.
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
                path, tasks=entry.get("tasks"), max_records=max_records,
            ))
        else:  # pragma: no cover
            raise ValueError(f"unknown benchmark kind {kind!r}")
    return splits


def _build_trait_space(cfg: dict[str, Any], splits: list[BenchmarkSplit]) -> TraitSpace:
    """Reconstruct the closed-loop's KDE trait space.

    :param cfg: Parsed YAML config (post-override).
    :type cfg: dict
    :param splits: Loaded benchmark splits.
    :type splits: list[BenchmarkSplit]
    :returns: KDE-smoothed trait space.
    :rtype: TraitSpace
    """
    ts_cfg = cfg.get("trait_space", {})
    encoder = SentenceTransformerEncoder(
        model_name=ts_cfg.get(
            "encoder", "sentence-transformers/all-MiniLM-L6-v2",
        ),
    )
    return build_safety_trait_space(
        splits, encoder,
        n_grid=int(ts_cfg.get("n_grid", 32)),
        kde_bandwidth=ts_cfg.get("kde_bandwidth"),
        threshold=float(ts_cfg.get("threshold", 0.5)),
    )


def _resolve_sigma(
    cfg: dict[str, Any], n_agents: int, space: TraitSpace,
) -> float:
    """Resolve :math:`\\sigma` from the config.

    :param cfg: Parsed YAML config.
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


# -----------------------------------------------------------------------------
# Density construction
# -----------------------------------------------------------------------------

def _empirical_weights_on_grid(
    grid: np.ndarray, coords: np.ndarray, smoothing_cells: int = 1,
) -> np.ndarray:
    """Construct an empirical resource histogram on the trait grid.

    The grid is assumed to be the standard regular ``n_grid x n_grid``
    lattice over :math:`[0, 1]^2` that :func:`build_safety_trait_space`
    produces. We:

    1. Detect the grid spacing from the unique axis values.
    2. Bin each prompt projection into the nearest grid cell.
    3. Optionally apply a single-pass 3x3 (or larger, controlled by
       ``smoothing_cells``) box smoothing to avoid hard zeros that
       would freeze the gradient ascent at low-density cells.
    4. Normalise to sum to one.

    The result is mass-conservative against the actual prompt
    distribution: no mass exists outside the data support modulo the
    small box smoothing.

    :param grid: Trait-space grid, shape ``(K, 2)`` where
        :math:`K = n_{\\text{grid}}^2`.
    :type grid: numpy.ndarray
    :param coords: Projected prompt coordinates, shape ``(M, 2)``.
    :type coords: numpy.ndarray
    :param smoothing_cells: Half-width of the box-smoothing kernel in
        grid cells. ``0`` disables smoothing; ``1`` (default) applies a
        3x3 box.
    :type smoothing_cells: int
    :returns: Resource mass per grid cell, shape ``(K,)``, summing to
        one.
    :rtype: numpy.ndarray
    """
    n_per_axis = int(np.sqrt(grid.shape[0]))
    if n_per_axis * n_per_axis != grid.shape[0]:  # pragma: no cover
        raise ValueError(
            f"empirical density requires a square grid; got {grid.shape[0]} cells"
        )

    axis0 = np.unique(grid[:, 0])
    axis1 = np.unique(grid[:, 1])
    # Bin edges from the cell centres (cell width inferred from axis spacing).
    dx = float(axis0[1] - axis0[0]) if axis0.size > 1 else 1.0 / n_per_axis
    dy = float(axis1[1] - axis1[0]) if axis1.size > 1 else 1.0 / n_per_axis
    edges_x = np.concatenate([[axis0[0] - dx / 2.0], axis0 + dx / 2.0])
    edges_y = np.concatenate([[axis1[0] - dy / 2.0], axis1 + dy / 2.0])

    # numpy histogram2d's first dim is x; we want the grid order to match
    # how build_safety_trait_space reshape (n_per_axis, n_per_axis) lays
    # things out. Cross-check against grid[:, 0].reshape(n,n) ordering.
    H, _, _ = np.histogram2d(
        coords[:, 0], coords[:, 1], bins=(edges_x, edges_y),
    )                                                  # (n_per_axis, n_per_axis)

    if smoothing_cells > 0:
        k = 2 * int(smoothing_cells) + 1
        kernel = np.ones((k, k), dtype=float)
        # Convolve manually to avoid scipy dependency.
        pad = smoothing_cells
        H_padded = np.pad(H, pad, mode="edge")
        smoothed = np.zeros_like(H)
        for i in range(k):
            for j in range(k):
                smoothed += H_padded[i:i + H.shape[0], j:j + H.shape[1]]
        smoothed /= float(kernel.sum())
        H = smoothed

    # Confirm the reshape order matches grid layout: rebuild grid the
    # same way and check the first few cells agree to within float tol.
    test_grid = np.stack(
        np.meshgrid(axis0, axis1, indexing="ij"), axis=-1,
    ).reshape(-1, 2)
    if not np.allclose(test_grid, grid):
        # Fallback: re-index H to match the actual grid ordering.
        rebuilt = np.zeros(grid.shape[0])
        for idx, (gx, gy) in enumerate(grid):
            i = int(np.argmin(np.abs(axis0 - gx)))
            j = int(np.argmin(np.abs(axis1 - gy)))
            rebuilt[idx] = H[i, j]
        weights = rebuilt
    else:
        weights = H.reshape(-1)

    total = weights.sum()
    if total <= 0:  # pragma: no cover
        raise ValueError(
            "empirical weights summed to zero; no prompts in any grid cell"
        )
    return weights / total


def _build_empirical_space(
    kde_space: TraitSpace, coords: np.ndarray, smoothing_cells: int,
) -> TraitSpace:
    """Return a TraitSpace clone with weights replaced by the empirical histogram.

    Preserves the projector, encoder, grid, and any other attributes;
    only ``weights`` is replaced. The resource-weighted ``mean`` is a
    computed ``@property`` on :class:`TraitSpace` (not a stored field),
    so it is automatically recomputed from the new weights — no
    explicit re-assignment is needed.

    :class:`TraitSpace` is a frozen dataclass in the current codebase,
    so direct attribute assignment raises ``FrozenInstanceError``. We
    use :func:`dataclasses.replace` to construct a new instance with
    the swapped ``weights`` field, and fall back to ``__new__`` plus
    ``object.__setattr__`` only if the dataclass route fails (e.g.
    future implementations that change ``TraitSpace`` to a plain
    class).

    :param kde_space: Original KDE-smoothed trait space.
    :type kde_space: TraitSpace
    :param coords: Projected prompt coordinates, shape ``(M, 2)``.
    :type coords: numpy.ndarray
    :param smoothing_cells: Box-smoothing half-width in grid cells.
    :type smoothing_cells: int
    :returns: New trait space with empirical weights.
    :rtype: TraitSpace
    """
    new_weights = _empirical_weights_on_grid(
        kde_space.grid, coords, smoothing_cells=smoothing_cells,
    )

    # ``mean`` is a computed @property on TraitSpace, so we deliberately
    # do NOT pass it to replace() (it isn't a dataclass field) and do NOT
    # try to overwrite it on the resulting instance (would raise
    # "property has no setter"). It re-evaluates from `weights` on
    # access.
    try:
        return _dc_replace(kde_space, weights=new_weights)
    except TypeError:
        # Fallback: non-dataclass TraitSpace variant. Build a shallow
        # copy via __new__ and only set the field; do not touch `mean`.
        space = kde_space.__class__.__new__(kde_space.__class__)
        for k, v in kde_space.__dict__.items():
            object.__setattr__(space, k, v)
        object.__setattr__(space, "weights", new_weights)
        return space


# -----------------------------------------------------------------------------
# Diagnostic primitives
# -----------------------------------------------------------------------------

def _empirical_density_count(
    points: np.ndarray, query: np.ndarray, radius: float,
) -> int:
    """Count actual prompt projections inside an L2 ball around ``query``.

    :param points: Projected prompt coordinates, shape ``(M, 2)``.
    :type points: numpy.ndarray
    :param query: Single query point, shape ``(2,)``.
    :type query: numpy.ndarray
    :param radius: Neighbourhood radius (trait units).
    :type radius: float
    :returns: Number of prompts within ``radius`` of ``query``.
    :rtype: int
    """
    diff = points - query[None, :]
    dist = np.linalg.norm(diff, axis=-1)
    return int((dist <= radius).sum())


def _mass_at(
    grid: np.ndarray, weights: np.ndarray, query: np.ndarray,
) -> float:
    """Return the resource mass at the grid cell nearest ``query``.

    :param grid: Trait grid, shape ``(K, 2)``.
    :type grid: numpy.ndarray
    :param weights: Per-cell resource weights summing to one.
    :type weights: numpy.ndarray
    :param query: Query coordinate, shape ``(2,)``.
    :type query: numpy.ndarray
    :returns: Nearest-cell mass.
    :rtype: float
    """
    idx = int(np.argmin(np.linalg.norm(grid - query[None, :], axis=-1)))
    return float(weights[idx])


def _re_init_agents_from_history(
    history: list[dict[str, Any]],
) -> list[RouterAgent]:
    """Re-create agents at their round-0 positions from ``history.json``.

    :param history: Loaded ``history.json`` as a list of round records.
    :type history: list[dict]
    :returns: Agents initialised at the saved round-0 positions.
    :rtype: list[RouterAgent]
    """
    return [
        RouterAgent(name=name, position=np.asarray(pos, dtype=float))
        for name, pos in history[0]["positions"].items()
    ]


# -----------------------------------------------------------------------------
# Per-mode analysis
# -----------------------------------------------------------------------------

def _run_one_mode(
    *,
    mode: str,
    space: TraitSpace,
    coords: np.ndarray,
    history: list[dict[str, Any]],
    cfg: dict[str, Any],
    radius: float,
    theory_steps: int,
    theory_lr: float,
) -> dict[str, Any]:
    """Run the diagnostic pipeline for one density mode.

    Recomputes :math:`\\sigma` (via the stability threshold, which
    depends on ``space.weights`` — so it can differ between KDE and
    empirical modes), runs ``train_router_positions`` to recover the
    theoretical Nash for this density, and tabulates the per-agent
    density/count metrics at both theory NE and SFT end.

    :param mode: ``'kde'`` or ``'empirical'``; used in labels only.
    :type mode: str
    :param space: Trait space with the relevant weights already set.
    :type space: TraitSpace
    :param coords: Projected prompt coordinates, shape ``(M, 2)``.
    :type coords: numpy.ndarray
    :param history: Loaded ``history.json``.
    :type history: list[dict]
    :param cfg: Parsed YAML config (post-override).
    :type cfg: dict
    :param radius: Neighbourhood radius for the empirical prompt count.
    :type radius: float
    :param theory_steps: Gradient-ascent steps for theory NE.
    :type theory_steps: int
    :param theory_lr: Learning rate for theory NE.
    :type theory_lr: float
    :returns: Dictionary with keys ``mode``, ``sigma``, ``stability_threshold``,
        ``theory_positions``, ``per_agent`` (per-agent summary metrics).
    :rtype: dict
    """
    agents = _re_init_agents_from_history(history)
    n_agents = len(agents)
    sigma = _resolve_sigma(cfg, n_agents, space)
    s0 = gaussian_stability_threshold(n_agents, space.grid, space.weights)
    print(
        f"[{mode}] computing theory NE at sigma={sigma:.4f}  "
        f"(sigma_0* = {s0:.4f}, mean = {space.mean})"
    )

    rt_cfg = RouterTrainingConfig(
        sigma=sigma,
        learning_rate=theory_lr,
        n_steps=theory_steps,
        tol=1e-8,
        clip_to_box=True,
    )
    train_router_positions(space, agents, rt_cfg, seed=int(cfg.get("seed", 0)))
    theory_positions = {a.name: a.position.copy() for a in agents}

    start_positions = {
        name: np.asarray(pos, dtype=float)
        for name, pos in history[0]["positions"].items()
    }
    sft_end_positions = {
        name: np.asarray(pos, dtype=float)
        for name, pos in history[-1]["positions"].items()
    }

    per_agent: dict[str, dict[str, Any]] = {}
    for name in theory_positions:
        nxe = theory_positions[name]
        sfx = sft_end_positions[name]
        per_agent[name] = {
            "theory_ne": nxe.tolist(),
            "theory_ne_mass": _mass_at(space.grid, space.weights, nxe),
            "theory_ne_n_prompts": _empirical_density_count(
                coords, nxe, radius,
            ),
            "sft_end": sfx.tolist(),
            "sft_end_mass": _mass_at(space.grid, space.weights, sfx),
            "sft_end_n_prompts": _empirical_density_count(
                coords, sfx, radius,
            ),
            "start": start_positions[name].tolist(),
            "l2_theory_minus_sft": float(np.linalg.norm(nxe - sfx)),
        }

    return {
        "mode": mode,
        "sigma": sigma,
        "stability_threshold": float(s0),
        "resource_mean": space.mean.tolist(),
        "theory_positions": {k: v.tolist() for k, v in theory_positions.items()},
        "per_agent": per_agent,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _plot_one(
    *,
    ax: Any,
    space: TraitSpace,
    scatter_coords: np.ndarray,
    result: dict[str, Any],
    axis_labels: list[str],
    radius: float,
    title: str,
) -> None:
    """Plot one density panel: contour + scatter + agent markers.

    :param ax: Matplotlib axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param space: Trait space for this density mode (KDE or empirical).
    :type space: TraitSpace
    :param scatter_coords: Subsampled prompt projections, shape ``(M', 2)``.
    :type scatter_coords: numpy.ndarray
    :param result: Output of :func:`_run_one_mode` for this density.
    :type result: dict
    :param axis_labels: X- and Y-axis labels.
    :type axis_labels: list[str]
    :param radius: Empirical-count radius (for the title line).
    :type radius: float
    :param title: Panel title (full).
    :type title: str
    """
    import matplotlib.pyplot as plt

    n_grid_per_axis = int(np.sqrt(space.grid.shape[0]))
    gx = space.grid[:, 0].reshape(n_grid_per_axis, n_grid_per_axis)
    gy = space.grid[:, 1].reshape(n_grid_per_axis, n_grid_per_axis)
    gz = space.weights.reshape(n_grid_per_axis, n_grid_per_axis)

    cs = ax.contourf(gx, gy, gz, levels=20, cmap="viridis", alpha=0.55)
    plt.colorbar(cs, ax=ax, label="resource mass B(b)")

    ax.scatter(
        scatter_coords[:, 0], scatter_coords[:, 1],
        s=2.5, c="white", alpha=0.18, label="actual prompts",
    )
    ax.scatter(
        [space.mean[0]], [space.mean[1]],
        marker="P", s=180, c="black", edgecolors="white", linewidths=1.4,
        label=r"$\mathbb{E}_B[b]$",
    )

    cmap = plt.get_cmap("tab10")
    for i, (name, s) in enumerate(result["per_agent"].items()):
        c = cmap(i % 10)
        x0 = np.asarray(s["start"])
        ne = np.asarray(s["theory_ne"])
        sfx = np.asarray(s["sft_end"])
        ax.plot([x0[0], sfx[0]], [x0[1], sfx[1]],
                "-", color=c, alpha=0.4, linewidth=1.2)
        ax.scatter([x0[0]], [x0[1]], marker="o", s=70, facecolors="none",
                   edgecolors=c, linewidths=1.6, zorder=5)
        ax.scatter([sfx[0]], [sfx[1]], marker="*", s=190, c=[c],
                   edgecolors="black", linewidths=0.7, zorder=6,
                   label=f"{name}  SFT end")
        ax.scatter([ne[0]], [ne[1]], marker="X", s=160, c=[c],
                   edgecolors="black", linewidths=0.7, zorder=6,
                   label=f"{name}  theory NE")

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=6, framealpha=0.85, ncol=2)


def _save_single(
    *,
    space: TraitSpace,
    scatter_coords: np.ndarray,
    result: dict[str, Any],
    output_stem: str,
    title: str,
    axis_labels: list[str],
    radius: float,
) -> None:
    """Render and save one density panel as ``<stem>.{pdf,png}``.

    :param space: Trait space for this density mode.
    :type space: TraitSpace
    :param scatter_coords: Subsampled prompt projections.
    :type scatter_coords: numpy.ndarray
    :param result: Per-mode result from :func:`_run_one_mode`.
    :type result: dict
    :param output_stem: Output stem; ``.pdf`` and ``.png`` are written.
    :type output_stem: str
    :param title: Figure title.
    :type title: str
    :param axis_labels: X- and Y-axis labels.
    :type axis_labels: list[str]
    :param radius: Empirical-count radius.
    :type radius: float
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 7.5))
    _plot_one(
        ax=ax, space=space, scatter_coords=scatter_coords, result=result,
        axis_labels=axis_labels, radius=radius, title=title,
    )
    fig.tight_layout()
    out = Path(output_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_stem}.{{pdf,png}}")


def _save_compare(
    *,
    spaces: dict[str, TraitSpace],
    scatter_coords: np.ndarray,
    results: dict[str, dict[str, Any]],
    output_stem: str,
    title_lines: dict[str, str],
    axis_labels: list[str],
    radius: float,
) -> None:
    """Render and save the 2-panel KDE-vs-empirical comparison figure.

    :param spaces: Mapping ``{'kde': space, 'empirical': space}``.
    :type spaces: dict[str, TraitSpace]
    :param scatter_coords: Subsampled prompt projections.
    :type scatter_coords: numpy.ndarray
    :param results: Mapping mode -> per-mode result.
    :type results: dict[str, dict]
    :param output_stem: Output stem; ``_compare.{pdf,png}`` is written.
    :type output_stem: str
    :param title_lines: Mapping mode -> panel title string.
    :type title_lines: dict[str, str]
    :param axis_labels: X- and Y-axis labels.
    :type axis_labels: list[str]
    :param radius: Empirical-count radius (for the title).
    :type radius: float
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.0))
    _plot_one(
        ax=axes[0], space=spaces["kde"], scatter_coords=scatter_coords,
        result=results["kde"], axis_labels=axis_labels,
        radius=radius, title=title_lines["kde"],
    )
    _plot_one(
        ax=axes[1], space=spaces["empirical"], scatter_coords=scatter_coords,
        result=results["empirical"], axis_labels=axis_labels,
        radius=radius, title=title_lines["empirical"],
    )
    fig.tight_layout()
    out_path = f"{output_stem}_compare"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_path}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}.{{pdf,png}}")


# -----------------------------------------------------------------------------
# Console table
# -----------------------------------------------------------------------------

def _print_table(result: dict[str, Any], n_prompts: int, radius: float) -> None:
    """Print the per-agent diagnostic table for one density mode.

    :param result: Per-mode result from :func:`_run_one_mode`.
    :type result: dict
    :param n_prompts: Total prompt count (for the header line).
    :type n_prompts: int
    :param radius: Empirical-count radius (for the header line).
    :type radius: float
    """
    print()
    print(
        f"[{result['mode']}]  σ = {result['sigma']:.4f}   "
        f"σ₀* = {result['stability_threshold']:.4f}   "
        f"n_prompts = {n_prompts}   radius = {radius}"
    )
    print(
        f"{'agent':<10} "
        f"{'theory NE':>20} {'B(NE)':>9} {'#prompts':>10}  "
        f"{'SFT end':>20} {'B(end)':>9} {'#prompts':>10}  "
        f"{'L2':>7}"
    )
    for name, s in result["per_agent"].items():
        ne = s["theory_ne"]
        se = s["sft_end"]
        print(
            f"{name:<10} "
            f"({ne[0]:>6.3f},{ne[1]:>6.3f})   "
            f"{s['theory_ne_mass']:>9.5f} {s['theory_ne_n_prompts']:>10d}   "
            f"({se[0]:>6.3f},{se[1]:>6.3f})   "
            f"{s['sft_end_mass']:>9.5f} {s['sft_end_n_prompts']:>10d}   "
            f"{s['l2_theory_minus_sft']:>7.3f}"
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Optional argument vector for testing; defaults to
        ``sys.argv[1:]``.
    :type argv: list[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose KDE-smoothed vs empirical resource density to explain "
            "SFT-vs-theory position-update gaps."
        ),
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to the closed-loop YAML config.",
    )
    parser.add_argument(
        "--history", required=True, type=Path,
        help="Path to the closed loop's history.json.",
    )
    parser.add_argument(
        "--output-stem", required=True,
        help=(
            "Output stem. In 'kde' / 'empirical' mode, <stem>.{pdf,png} is "
            "written. In 'both' mode, <stem>_kde.{pdf,png}, "
            "<stem>_empirical.{pdf,png}, and <stem>_compare.{pdf,png} are "
            "written."
        ),
    )
    parser.add_argument(
        "--summary-json", default=None,
        help="Optional path to write a JSON summary.",
    )
    parser.add_argument(
        "--axis-labels", nargs=2, default=["dim 0", "dim 1"],
        help="X- and Y-axis labels for the trait-space plot.",
    )
    parser.add_argument(
        "--title", default="",
        help="Optional figure title suffix.",
    )
    parser.add_argument(
        "--radius", type=float, default=0.05,
        help="Neighbourhood radius (trait units) for the empirical count.",
    )
    parser.add_argument(
        "--max-scatter", type=int, default=8000,
        help="Maximum number of prompts to scatter on the plot.",
    )
    parser.add_argument(
        "--theory-steps", type=int, default=5000,
        help="Gradient-ascent steps for theory NE.",
    )
    parser.add_argument(
        "--theory-lr", type=float, default=5e-3,
        help="Learning rate for theory NE.",
    )
    parser.add_argument(
        "--density-mode", choices=("kde", "empirical", "both"), default="kde",
        help=(
            "Resource density to use for theory NE: 'kde' (the original "
            "smoothed B(b)), 'empirical' (2-D histogram of actual prompt "
            "projections on the same grid), or 'both' (run both and "
            "produce a side-by-side comparison figure)."
        ),
    )
    parser.add_argument(
        "--empirical-smoothing-cells", type=int, default=1,
        help=(
            "Box-smoothing half-width (in grid cells) for the empirical "
            "histogram. 0 disables smoothing; 1 (default) applies a 3x3 "
            "box to avoid hard zeros that would freeze the gradient ascent."
        ),
    )
    parser.add_argument(
        "--config-override", action="append", default=[],
        metavar="KEY=VAL",
        help="Repeatable dotted-key override (JSON-decoded).",
    )
    args = parser.parse_args(argv)

    cfg = _load_yaml(args.config)
    _apply_overrides(cfg, args.config_override)

    with args.history.open("r", encoding="utf-8") as fh:
        history = json.load(fh)

    splits = _load_splits(cfg)
    kde_space = _build_trait_space(cfg, splits)
    all_prompts = [p for s in splits for p in s.prompts]
    print(f"projecting {len(all_prompts)} prompts ...")
    coords = kde_space.project(all_prompts)
    if coords.shape[0] > args.max_scatter:
        idx = np.random.default_rng(0).choice(
            coords.shape[0], size=args.max_scatter, replace=False,
        )
        scatter_coords = coords[idx]
    else:
        scatter_coords = coords

    modes_to_run = (
        ["kde"] if args.density_mode == "kde"
        else ["empirical"] if args.density_mode == "empirical"
        else ["kde", "empirical"]
    )

    spaces: dict[str, TraitSpace] = {}
    results: dict[str, dict[str, Any]] = {}
    for mode in modes_to_run:
        if mode == "kde":
            space = kde_space
        else:
            space = _build_empirical_space(
                kde_space, coords,
                smoothing_cells=args.empirical_smoothing_cells,
            )
        spaces[mode] = space
        results[mode] = _run_one_mode(
            mode=mode,
            space=space,
            coords=coords,
            history=history,
            cfg=cfg,
            radius=args.radius,
            theory_steps=args.theory_steps,
            theory_lr=args.theory_lr,
        )

    # ---- Figures ----
    bw = cfg.get("trait_space", {}).get("kde_bandwidth", None)
    bw_str = f"{bw}" if bw is not None else "auto"
    title_lines = {
        "kde": (
            f"KDE B(b)  (bw = {bw_str},  σ = {results.get('kde', {}).get('sigma', 0):.3f})"
            + (f"\n{args.title}" if args.title else "")
        ),
        "empirical": (
            f"empirical B(b)  (smoothing = {args.empirical_smoothing_cells} cells,  "
            f"σ = {results.get('empirical', {}).get('sigma', 0):.3f})"
            + (f"\n{args.title}" if args.title else "")
        ),
    }

    if args.density_mode == "both":
        # Single-panel files for each mode + a side-by-side compare.
        _save_single(
            space=spaces["kde"], scatter_coords=scatter_coords,
            result=results["kde"],
            output_stem=f"{args.output_stem}_kde",
            title=title_lines["kde"], axis_labels=args.axis_labels,
            radius=args.radius,
        )
        _save_single(
            space=spaces["empirical"], scatter_coords=scatter_coords,
            result=results["empirical"],
            output_stem=f"{args.output_stem}_empirical",
            title=title_lines["empirical"], axis_labels=args.axis_labels,
            radius=args.radius,
        )
        _save_compare(
            spaces=spaces, scatter_coords=scatter_coords, results=results,
            output_stem=args.output_stem, title_lines=title_lines,
            axis_labels=args.axis_labels, radius=args.radius,
        )
    else:
        only = modes_to_run[0]
        _save_single(
            space=spaces[only], scatter_coords=scatter_coords,
            result=results[only],
            output_stem=args.output_stem,
            title=title_lines[only], axis_labels=args.axis_labels,
            radius=args.radius,
        )

    # ---- Console tables ----
    for mode in modes_to_run:
        _print_table(results[mode], n_prompts=len(all_prompts), radius=args.radius)

    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "config_overrides": args.config_override,
                    "kde_bandwidth": bw,
                    "empirical_smoothing_cells": args.empirical_smoothing_cells,
                    "radius": args.radius,
                    "n_prompts_total": len(all_prompts),
                    "modes": {m: results[m] for m in modes_to_run},
                },
                fh,
                indent=2,
            )
        print(f"wrote {args.summary_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
