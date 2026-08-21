"""Optional figure export helpers for :mod:`infl_ens.vis`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Sequence, Union

from matplotlib.figure import Figure

PathLike = Union[str, Path]
BboxInches = Union[Literal["tight"], str, None]


def save_figure(
    fig: Figure,
    output_stem: PathLike,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 200,
    bbox_inches: BboxInches = "tight",
) -> list[Path]:
    """Write a figure to disk under ``output_stem.<ext>``.

    :param fig: Figure to save.
    :type fig: matplotlib.figure.Figure
    :param output_stem: Filename stem without extension; parent dirs are created.
    :type output_stem: str | pathlib.Path
    :param formats: File extensions to write, e.g. ``("pdf", "png")``.
    :type formats: Sequence[str]
    :param dpi: Resolution for raster formats.
    :type dpi: int
    :param bbox_inches: Passed to :meth:`matplotlib.figure.Figure.savefig`.
        Use ``None`` to skip tight cropping.
    :type bbox_inches: str | None
    :returns: Paths written, in ``formats`` order.
    :rtype: list[pathlib.Path]
    """
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in formats:
        path = stem.with_suffix(f".{ext.lstrip('.')}")
        kwargs: dict = {"dpi": dpi}
        if bbox_inches is not None:
            kwargs["bbox_inches"] = bbox_inches
        fig.savefig(path, **kwargs)
        written.append(path)
    return written
