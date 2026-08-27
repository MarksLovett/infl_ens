"""Final trait-space positions of each merge pair, and within-pair separation.

Works for any closed-loop run that used ``sft_merge_groups`` (soft or hard
routing, any trait-space dimension ``L``).  Inputs are the ``positions``
of ``history.json`` records and the ``(train_as, members)`` merge groups;
the functions here are pure (records in, :class:`~matplotlib.figure.Figure`
out) and :mod:`infl_ens.figures.render` does the reading.

``plot_final_positions``
    All :math:`\\binom{L}{2}` axis-pair projections of one round.  One
    marker per clone, coloured by merge group, with a segment joining the
    members of each group.  A pair that is still co-located shows as a
    single point; a pair that came apart shows a visible segment, which is
    the point of the figure: co-location under the theory-matched update
    is a prediction, not something the update enforces.

``plot_within_pair``
    Within-pair L2 distance per group across rounds (log scale), the audit
    of that prediction over the whole run.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure

MergeGroups = Sequence[tuple[str, Sequence[str]]]


def merge_groups_from_config(cfg: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    """Read literal merge groups from a resolved run config.

    :param cfg: Resolved config (``closed_loop.sft_merge_groups`` expanded).
    :type cfg: Mapping
    :returns: ``(train_as, members)`` pairs, or an empty list.
    :rtype: list[tuple[str, list[str]]]
    """
    groups = (cfg.get("closed_loop") or {}).get("sft_merge_groups")
    if isinstance(groups, list) and groups and isinstance(groups[0], Mapping):
        return [(str(g["train_as"]), [str(n) for n in g["names"]]) for g in groups]
    return []


def merge_groups_from_history(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[str]]]:
    """Fall back to the ``pair_members`` logged by soft-pair runs.

    :param records: History records.
    :type records: Sequence[Mapping]
    :returns: ``(train_as, members)`` pairs, or an empty list.
    :rtype: list[tuple[str, list[str]]]
    """
    members = records[0].get("pair_members") if records else None
    if isinstance(members, Mapping) and members:
        return [(str(k), [str(n) for n in v]) for k, v in sorted(members.items())]
    return []


def positions_of(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Clone name to position vector for one round.

    :param record: One history record.
    :type record: Mapping
    :returns: Name to ``(L,)`` array.
    :rtype: dict[str, numpy.ndarray]
    """
    return {name: np.asarray(vec, dtype=float) for name, vec in record["positions"].items()}


def within_pair_series(
    records: Sequence[Mapping[str, Any]],
    groups: MergeGroups,
) -> dict[str, list[float]]:
    """Within-pair L2 per group per round, recomputed from logged positions.

    Recomputed rather than read from ``agent_geometry`` so the figure is
    valid for runs whose geometry block is absent or shaped differently.

    :param records: History records.
    :type records: Sequence[Mapping]
    :param groups: Merge groups.
    :type groups: Sequence[tuple[str, Sequence[str]]]
    :returns: Group name to one distance per round (``nan`` when a member
        is missing).
    :rtype: dict[str, list[float]]
    """
    out: dict[str, list[float]] = {}
    for name, members in groups:
        series: list[float] = []
        for rec in records:
            pos = positions_of(rec)
            pts = [pos[m] for m in members if m in pos]
            if len(pts) < 2:
                series.append(float("nan"))
                continue
            series.append(float(np.linalg.norm(pts[0] - pts[1])))
        out[name] = series
    return out


def plot_final_positions(
    record: Mapping[str, Any],
    groups: MergeGroups,
    *,
    axis_labels: Sequence[str] = (),
    title: str = "run",
) -> Figure:
    """Scatter one round's positions over every axis pair, grouping members.

    :param record: History record to plot (normally the last).
    :type record: Mapping
    :param groups: Merge groups.
    :type groups: Sequence[tuple[str, Sequence[str]]]
    :param axis_labels: Trait-axis names (padded with ``axis i``).
    :type axis_labels: Sequence[str]
    :param title: Figure title prefix.
    :type title: str
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    pos = positions_of(record)
    dim = len(next(iter(pos.values())))
    labels = list(axis_labels) + [f"axis {i}" for i in range(len(axis_labels), dim)]
    pairs = list(combinations(range(dim), 2)) or [(0, 0)]
    ncols = min(4, len(pairs)) or 1
    nrows = int(np.ceil(len(pairs) / ncols))
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % 10) for i, (name, _) in enumerate(groups)}

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows), squeeze=False)
    for ax_idx, (i, j) in enumerate(pairs):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        for name, members in groups:
            present = [pos[m] for m in members if m in pos]
            if not present:
                continue
            pts = np.stack(present, axis=0)
            ax.plot(
                pts[:, i], pts[:, j],
                marker="o", markersize=6, linewidth=1.6,
                color=colors[name], alpha=0.9,
                label=name if ax_idx == 0 else None,
            )
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel(labels[i], fontsize=8)
        ax.set_ylabel(labels[j], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.5)
    for spare in range(len(pairs), nrows * ncols):
        axes[spare // ncols][spare % ncols].axis("off")

    handles, hlabels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, hlabels, loc="lower center", ncol=min(len(handles), 7), fontsize=8, frameon=False)
    fig.suptitle(f"{title} — final positions (round {record.get('round', '?')})", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig


def plot_within_pair(
    records: Sequence[Mapping[str, Any]],
    groups: MergeGroups,
    *,
    title: str = "run",
) -> Figure:
    """Within-pair L2 distance per group across rounds.

    :param records: History records.
    :type records: Sequence[Mapping]
    :param groups: Merge groups.
    :type groups: Sequence[tuple[str, Sequence[str]]]
    :param title: Figure title prefix.
    :type title: str
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    rounds = [int(r.get("round", k)) for k, r in enumerate(records)]
    cmap = plt.get_cmap("tab10")
    series_by_group = within_pair_series(records, groups)

    positive = [v for s in series_by_group.values() for v in s if np.isfinite(v) and v > 0.0]
    # A log axis cannot show an exact zero. Clamp to one decade below the
    # smallest real separation rather than to a fixed epsilon, which would
    # otherwise stretch the axis over a meaningless ~16 decades.
    use_log = bool(positive)
    floor = (min(positive) / 10.0) if positive else 0.0

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for idx, (name, _members) in enumerate(groups):
        vals = [(max(v, floor) if use_log else v) for v in series_by_group[name]]
        ax.plot(rounds, vals, marker="o", markersize=3.5, linewidth=1.4, color=cmap(idx % 10), label=name)
    if use_log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor / 2.0)
        note = "partners separate — sampling noise in the per-clone step"
    else:
        ax.set_ylim(-1.0, 1.0)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        note = "identically zero at every round — pairs persist unforced"
    ax.set_xlabel("round")
    ax.set_ylabel(r"within-pair $\|x_i - x_j\|_2$")
    ax.grid(alpha=0.3, linewidth=0.5)
    if groups:
        ax.legend(ncol=min(len(groups), 4), fontsize=8, frameon=False)
    ax.set_title(f"{title} — within-pair separation ({note})", fontsize=10)
    fig.tight_layout()
    return fig


__all__ = [
    "merge_groups_from_config",
    "merge_groups_from_history",
    "plot_final_positions",
    "plot_within_pair",
    "positions_of",
    "within_pair_series",
]
