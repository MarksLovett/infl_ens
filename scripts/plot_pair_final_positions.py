#!/usr/bin/env python3
"""Final trait-space positions of each merge pair, plus within-pair separation.

Works for any closed-loop run that used ``sft_merge_groups`` (soft or hard
routing, any trait-space dimension ``L``). Reads only artifacts the trainer
already writes:

- ``<run>/history.json``          -- per-round ``positions`` and ``agent_geometry``
- ``<run>/resolved_config.yaml``  -- ``closed_loop.sft_merge_groups`` expanded
  to literal member lists (falls back to ``pair_members`` in the history when
  the resolved config is absent).

Two figures:

``<stem>_final_positions``
    All :math:`\\binom{L}{2}` axis-pair projections of the final round. One
    marker per clone, coloured by its merge group, with a segment joining the
    members of each group. A pair that is still co-located shows as a single
    point; a pair that came apart shows a visible segment, which is the point
    of the figure -- co-location under the theory-matched update is a
    prediction, not something the update enforces.

``<stem>_within_pair``
    Within-pair L2 distance per group across rounds (log scale), the audit of
    that prediction over the whole run.

Example::

    python scripts/plot_pair_final_positions.py \\
        --run-dir results/seven_axis_soft_topk3_pairs/seed0 \\
        --output-stem scripts/figures/three_arm/soft_topk3 \\
        --axis-labels harm hallucination jailbreak privacy overrefusal injection policy \\
        --title "Soft top-3 pairs"
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infl_ens.vis.save import save_figure  # noqa: E402


def _load_history(run_dir: Path) -> list[dict[str, Any]]:
    """Read ``history.json``.

    :raises FileNotFoundError: If the history is missing.
    """
    path = run_dir / "history.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"{path} holds no rounds")
    return records


def _load_groups(
    run_dir: Path,
    records: Sequence[dict[str, Any]],
) -> list[tuple[str, list[str]]]:
    """Resolve merge groups as ``(train_as, members)``.

    Prefers ``resolved_config.yaml`` (written for every arm); falls back to
    the soft-pairs ``pair_members`` field in the history.

    :raises ValueError: If no group definition can be found.
    """
    cfg_path = run_dir / "resolved_config.yaml"
    if cfg_path.is_file():
        try:
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            groups = (cfg.get("closed_loop") or {}).get("sft_merge_groups")
            if isinstance(groups, list) and groups:
                return [
                    (str(g["train_as"]), [str(n) for n in g["names"]])
                    for g in groups
                ]
        except ImportError:  # pragma: no cover - PyYAML absent
            pass
    members = records[0].get("pair_members")
    if isinstance(members, dict) and members:
        return [(str(k), [str(n) for n in v]) for k, v in sorted(members.items())]
    raise ValueError(
        f"no merge groups found in {cfg_path} or history pair_members; "
        "was this run configured with sft_merge_groups?"
    )


def _positions(record: dict[str, Any]) -> dict[str, np.ndarray]:
    """Clone name to position vector for one round."""
    return {
        name: np.asarray(vec, dtype=float)
        for name, vec in record["positions"].items()
    }


def plot_final_positions(
    record: dict[str, Any],
    groups: Sequence[tuple[str, list[str]]],
    *,
    axis_labels: Sequence[str],
    title: str,
    output_stem: Path,
) -> list[Path]:
    """Scatter the final positions over every axis pair, grouping members.

    :returns: Written image paths.
    :rtype: list[pathlib.Path]
    """
    import matplotlib.pyplot as plt

    pos = _positions(record)
    dim = len(next(iter(pos.values())))
    labels = list(axis_labels) + [
        f"axis {i}" for i in range(len(axis_labels), dim)
    ]
    pairs = list(combinations(range(dim), 2))
    ncols = min(4, len(pairs)) or 1
    nrows = int(np.ceil(len(pairs) / ncols))
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % 10) for i, (name, _) in enumerate(groups)}

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows), squeeze=False,
    )
    for ax_idx, (i, j) in enumerate(pairs):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        for name, members in groups:
            pts = np.stack(
                [pos[m] for m in members if m in pos], axis=0,
            ) if any(m in pos for m in members) else None
            if pts is None or pts.size == 0:
                continue
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
        fig.legend(
            handles, hlabels, loc="lower center",
            ncol=min(len(handles), 7), fontsize=8, frameon=False,
        )
    fig.suptitle(
        f"{title} — final positions (round {record.get('round', '?')})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return save_figure(fig, output_stem)


def plot_within_pair(
    records: Sequence[dict[str, Any]],
    groups: Sequence[tuple[str, list[str]]],
    *,
    title: str,
    output_stem: Path,
) -> list[Path]:
    """Within-pair L2 distance per group across rounds.

    Distances are recomputed from the logged positions rather than read from
    ``agent_geometry``, so the figure is valid for runs whose geometry block
    is absent or shaped differently.

    :returns: Written image paths.
    :rtype: list[pathlib.Path]
    """
    import matplotlib.pyplot as plt

    rounds = [int(r.get("round", k)) for k, r in enumerate(records)]
    cmap = plt.get_cmap("tab10")

    series_by_group: dict[str, list[float]] = {}
    for name, members in groups:
        series: list[float] = []
        for rec in records:
            pos = _positions(rec)
            pts = [pos[m] for m in members if m in pos]
            if len(pts) < 2:
                series.append(float("nan"))
                continue
            series.append(
                float(np.linalg.norm(np.asarray(pts[0]) - np.asarray(pts[1])))
            )
        series_by_group[name] = series

    positive = [
        v for s in series_by_group.values() for v in s
        if np.isfinite(v) and v > 0.0
    ]
    # A log axis cannot show an exact zero. Clamp to one decade below the
    # smallest real separation rather than to a fixed epsilon, which would
    # otherwise stretch the axis over a meaningless ~16 decades.
    use_log = bool(positive)
    floor = (min(positive) / 10.0) if positive else 0.0

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for idx, (name, _members) in enumerate(groups):
        vals = [
            (max(v, floor) if use_log else v) for v in series_by_group[name]
        ]
        ax.plot(
            rounds, vals, marker="o", markersize=3.5, linewidth=1.4,
            color=cmap(idx % 10), label=name,
        )
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
    ax.legend(ncol=min(len(groups), 4), fontsize=8, frameon=False)
    ax.set_title(f"{title} — within-pair separation ({note})", fontsize=10)
    fig.tight_layout()
    return save_figure(fig, output_stem)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Closed-loop run root.")
    parser.add_argument(
        "--output-stem", required=True,
        help="Stem; '_final_positions' and '_within_pair' are appended.",
    )
    parser.add_argument(
        "--axis-labels", nargs="*", default=[], help="Trait axis names.",
    )
    parser.add_argument("--title", default="run", help="Figure title.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    stem = Path(args.output_stem)
    if not stem.is_absolute():
        stem = ROOT / stem

    records = _load_history(run_dir)
    groups = _load_groups(run_dir, records)
    written: list[Path] = []
    written += plot_final_positions(
        records[-1], groups,
        axis_labels=args.axis_labels, title=args.title,
        output_stem=stem.with_name(stem.name + "_final_positions"),
    )
    written += plot_within_pair(
        records, groups, title=args.title,
        output_stem=stem.with_name(stem.name + "_within_pair"),
    )
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
