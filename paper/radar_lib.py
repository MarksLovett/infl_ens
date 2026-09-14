"""Polar (radar) primitives for the pair-position figures.

A pair's position is a point in the seven-dimensional trait space, all seven
coordinates in [0, 1]. A radar shows all seven at once, which a scatter of axis
pairs cannot: the shape *is* the specialisation. Nothing else in the project
draws polar axes, so these are local helpers rather than a reuse of
``infl_ens.figures``.

Colour is never the only channel here. Seven categorical hues cannot all be
distinguished under the common colour-vision deficiencies, so every hue is
paired with a text cue -- the dominant spoke label is bolded in the same colour
-- and overlays are separated by line style as well.
"""
from __future__ import annotations

import numpy as np

from figures_lib import AXES_SHORT

N_AXES = len(AXES_SHORT)
ANGLES = np.linspace(0.0, 2.0 * np.pi, N_AXES, endpoint=False)
_CLOSED = np.concatenate([ANGLES, ANGLES[:1]])


def radar_axes(fig, spec, *, title: str = "", label_size: float = 4.6):
    """One polar panel with the seven trait axes as spokes.

    :param fig: Target figure.
    :param spec: A ``GridSpec`` cell.
    :param title: Panel title, placed clear of the north spoke label.
    :param label_size: Spoke label size; panels are ~1 inch, so this is small.
    :returns: The configured polar axes.
    """
    ax = fig.add_subplot(spec, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(ANGLES)
    ax.set_xticklabels(AXES_SHORT, fontsize=label_size)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.5])
    ax.set_yticklabels([])
    ax.grid(lw=0.35)
    ax.tick_params(pad=-4)
    ax.spines["polar"].set_linewidth(0.5)
    if title:
        # The north spoke label sits where a default title would land.
        ax.set_title(title, fontsize=6.2, pad=6)
    return ax


def draw_polygon(ax, v, *, color, ls="-", lw=1.0, fill_alpha=0.0, zorder=3):
    """Draw one seven-coordinate position as a closed polygon."""
    closed = np.concatenate([np.asarray(v, dtype=float), np.asarray(v[:1], dtype=float)])
    ax.plot(_CLOSED, closed, color=color, ls=ls, lw=lw, zorder=zorder)
    if fill_alpha > 0:
        ax.fill(_CLOSED, closed, color=color, alpha=fill_alpha, lw=0, zorder=zorder - 1)


def emphasise_spoke(ax, axis_idx: int, color: str) -> None:
    """Bold the label of the axis a pair specialises on.

    This is the text cue that makes the per-pair hue redundant rather than
    load-bearing.
    """
    if not (0 <= axis_idx < N_AXES):
        return
    label = ax.get_xticklabels()[axis_idx]
    label.set_color(color)
    label.set_fontweight("bold")
