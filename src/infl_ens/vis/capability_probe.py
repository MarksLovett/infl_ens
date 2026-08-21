"""Capability-probe figures (pure: arrays in, Figure out)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from infl_ens.evaluation.capability_probe import cross_batch_margin
from infl_ens.vis.save import BboxInches, PathLike, save_figure


def plot_probe(
    records: list[dict],
    history: list[dict],
    names: list[str],
    *,
    title: Optional[str] = None,
    output_stem: Optional[PathLike] = None,
    save_formats: Sequence[str] = ("pdf", "png"),
    save_dpi: int = 200,
    save_bbox_inches: BboxInches = "tight",
):
    """Render Tier 1 (SFT loss curves) and Tier 3 (cross-NLL matrix + margin).

    :param records: Output of :func:`infl_ens.evaluation.capability_probe.probe_run`.
    :type records: list[dict]
    :param history: Loaded ``history.json`` records.
    :type history: list[dict]
    :param names: Agent names.
    :type names: list[str]
    :param title: Optional figure suptitle.
    :type title: str | None
    :param output_stem: If set, write ``.<format>`` files under this stem.
    :type output_stem: str | pathlib.Path | None
    :param save_formats: Extensions to write when ``output_stem`` is set.
    :type save_formats: Sequence[str]
    :param save_dpi: DPI for raster exports.
    :type save_dpi: int
    :param save_bbox_inches: ``bbox_inches`` passed to :func:`save_figure`.
    :type save_bbox_inches: str | None
    :returns: Matplotlib figure handle.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n_agents = len(names)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(n_agents, 3)))

    rounds = sorted({rec["round"] for rec in records})
    name_to_idx = {n: i for i, n in enumerate(names)}
    nll_mat = np.full((len(rounds), n_agents, n_agents), np.nan)
    for rec in records:
        ri = rounds.index(rec["round"])
        i = name_to_idx[rec["agent_i"]]
        j = name_to_idx[rec["agent_j"]]
        nll_mat[ri, i, j] = rec["nll"]

    margins = cross_batch_margin(records)

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, :])
    any_curve = False
    for i, name in enumerate(names):
        rs: list[int] = []
        means: list[float] = []
        mins: list[float] = []
        maxes: list[float] = []
        n_steps_per_round: list[int] = []
        for rec in history:
            r_idx = int(rec["round"])
            logs = rec.get("agent_sft_logs", {}).get(name, [])
            losses = [float(e["loss"]) for e in logs if "loss" in e]
            if not losses:
                for e in logs:
                    if "train_loss" in e:
                        losses.append(float(e["train_loss"]))
                        break
            if losses:
                rs.append(r_idx)
                means.append(float(np.mean(losses)))
                mins.append(float(np.min(losses)))
                maxes.append(float(np.max(losses)))
                n_steps_per_round.append(len(losses))
        if rs:
            any_curve = True
            ax.plot(
                rs,
                means,
                "-o",
                color=colors[i],
                lw=1.8,
                mfc="white",
                mec=colors[i],
                label=f"{name} (≈{np.mean(n_steps_per_round):.0f} steps/rd)",
            )
            ax.fill_between(rs, mins, maxes, color=colors[i], alpha=0.18, linewidth=0)
    ax.set_xlabel("round")
    ax.set_ylabel("train loss (per-round mean; band = min/max within round)")
    if any_curve:
        ax.set_title(
            "Tier 1  —  per-round SFT training loss "
            "(legend shows mean optimiser steps per round)",
        )
        ax.legend(loc="best", fontsize=8, frameon=True)
    else:
        ax.set_title(
            "Tier 1  —  no per-agent loss records found "
            "(set sft.logging_steps to 1 in your config)",
        )
    ax.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    final_mat = nll_mat[-1]
    if not np.all(np.isnan(final_mat)):
        vmin, vmax = np.nanmin(final_mat), np.nanmax(final_mat)
        im = ax2.imshow(final_mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        for ii in range(n_agents):
            for jj in range(n_agents):
                v = final_mat[ii, jj]
                if np.isnan(v):
                    continue
                txt = f"{v:.3f}"
                ax2.text(
                    jj,
                    ii,
                    txt,
                    ha="center",
                    va="center",
                    color="white" if v < (vmin + vmax) / 2 else "black",
                    fontsize=9,
                )
        plt.colorbar(im, ax=ax2, label="mean NLL / token")
    ax2.set_xticks(range(n_agents))
    ax2.set_xticklabels([n.replace("clone-", "c") for n in names])
    ax2.set_yticks(range(n_agents))
    ax2.set_yticklabels([n.replace("clone-", "c") for n in names])
    ax2.set_xlabel("agent j's batch")
    ax2.set_ylabel("agent i's model")
    ax2.set_title(f"Tier 3a  —  cross-NLL at round {rounds[-1]}")

    ax3 = fig.add_subplot(gs[1, 1])
    rs = sorted(margins.keys())
    d_means = [margins[r]["diag_mean"] for r in rs]
    o_means = [margins[r]["off_mean"] for r in rs]
    ax3.plot(rs, d_means, "-o", color="tab:blue", lw=1.6, label="diag (own batch)")
    ax3.plot(rs, o_means, "-s", color="tab:red", lw=1.6, label="off-diag (others' batches)")
    ax3.set_xlabel("round")
    ax3.set_ylabel("mean NLL / token")
    ax3.set_title("Tier 3b  —  fit on own vs others' batches")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=9)

    ax4 = fig.add_subplot(gs[1, 2])
    margin_vals = [margins[r]["margin"] for r in rs]
    ax4.bar(rs, margin_vals, color="tab:green", edgecolor="black", linewidth=0.5)
    for r_, m in zip(rs, margin_vals):
        ax4.text(
            r_,
            m + (0.001 if m >= 0 else -0.003),
            f"{m:.3f}",
            ha="center",
            fontsize=8,
            va="bottom" if m >= 0 else "top",
        )
    ax4.axhline(0, color="black", lw=0.8)
    ax4.set_xlabel("round")
    ax4.set_ylabel(r"$\mu(r)$ = NLL(others) − NLL(own)")
    ax4.set_title("Tier 3c  —  specialisation margin\n(positive = specialised)")
    ax4.grid(True, axis="y", alpha=0.3)

    if title is not None:
        fig.suptitle(title, fontsize=12)

    if output_stem is not None:
        save_figure(
            fig,
            output_stem,
            formats=save_formats,
            dpi=save_dpi,
            bbox_inches=save_bbox_inches,
        )

    return fig
