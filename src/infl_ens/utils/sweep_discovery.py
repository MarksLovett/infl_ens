"""Discover closed-loop runs under sigma×seed sweep directories."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import numpy as np

SIGMA_DIR_RE = re.compile(r"^sigma(?P<val>[0-9.]+)$", re.IGNORECASE)
ROUND_DIR_RE = re.compile(r"^r(?P<val>\d+)$", re.IGNORECASE)
SEED_DIR_RE = re.compile(r"^seed(?P<val>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class RunCell:
    """One trained run in a nested sweep grid.

    :param group_slug: Directory name, e.g. ``sigma0.5`` or ``r20``.
    :type group_slug: str
    :param group_value: Parsed sweep coordinate (σ fraction or ``n_rounds``).
    :type group_value: float
    :param group_kind: ``sigma`` or ``round``.
    :type group_kind: str
    :param seed: Training RNG seed.
    :type seed: int
    :param run_dir: Directory containing ``history.json``.
    :type run_dir: pathlib.Path
    """

    group_slug: str
    group_value: float
    group_kind: str
    seed: int
    run_dir: Path

    @property
    def sigma_slug(self) -> str:
        """Backward-compatible alias when ``group_kind=='sigma'``."""
        return self.group_slug

    @property
    def sigma_fraction(self) -> float:
        """Backward-compatible alias when ``group_kind=='sigma'``."""
        return self.group_value


def load_history(path: Path) -> list[dict]:
    """Load a ``history.json`` file.

    :param path: Path to the history file.
    :type path: pathlib.Path
    :returns: Per-round records.
    :rtype: list[dict]
    """
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def agent_order(records: Sequence[dict]) -> list[str]:
    """Agent names in first-round order.

    :param records: History records.
    :type records: Sequence[dict]
    :returns: Ordered agent names.
    :rtype: list[str]
    """
    return list(records[0]["positions"].keys())


def position_tensor(records: Sequence[dict], names: Sequence[str]) -> np.ndarray:
    """Stack positions to ``(T, N, L)``.

    :param records: History records.
    :type records: Sequence[dict]
    :param names: Agent order.
    :type names: Sequence[str]
    :returns: Position tensor.
    :rtype: numpy.ndarray
    """
    return np.stack(
        [
            np.stack([np.asarray(r["positions"][n], dtype=float) for n in names])
            for r in records
        ],
        axis=0,
    )


def discover_group_seed_runs(
    root: Path,
    *,
    layout: str = "auto",
) -> list[RunCell]:
    """Find sweep cells under ``sigma*/seed*`` or ``r*/seed*``.

    :param root: Sweep results root.
    :type root: pathlib.Path
    :param layout: ``auto``, ``sigma_seed``, or ``round_seed``.
    :type layout: str
    :returns: Discovered run cells sorted by group then seed.
    :rtype: list[RunCell]
    """
    if layout == "auto":
        sigma_cells = _discover_group_seed(root, SIGMA_DIR_RE, "sigma")
        if sigma_cells:
            return sigma_cells
        return _discover_group_seed(root, ROUND_DIR_RE, "round")
    if layout == "sigma_seed":
        return _discover_group_seed(root, SIGMA_DIR_RE, "sigma")
    if layout == "round_seed":
        return _discover_group_seed(root, ROUND_DIR_RE, "round")
    raise ValueError(f"unknown layout {layout!r}")


def _discover_group_seed(
    root: Path,
    pattern: re.Pattern[str],
    kind: str,
) -> list[RunCell]:
    cells: list[RunCell] = []
    if not root.is_dir():
        return cells
    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir():
            continue
        m = pattern.match(group_dir.name)
        if not m:
            continue
        group_val = float(m.group("val"))
        for seed_dir in sorted(group_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            sm = SEED_DIR_RE.match(seed_dir.name)
            if not sm:
                continue
            hist = seed_dir / "history.json"
            if not hist.is_file():
                continue
            cells.append(
                RunCell(
                    group_slug=group_dir.name,
                    group_value=group_val,
                    group_kind=kind,
                    seed=int(sm.group("val")),
                    run_dir=seed_dir,
                )
            )
    return cells


def discover_flat_sweep_runs(root: Path, mode: str) -> list[dict]:
    """Find flat sweep runs ``seed*`` / ``sigma*`` / ``kde*`` under ``root``.

    :param root: Sweep root directory.
    :type root: pathlib.Path
    :param mode: One of ``seeds``, ``sigma``, ``kde``.
    :type mode: str
    :returns: Run descriptors with ``slug``, ``value``, ``history_path``,
        ``theory_path``.
    :rtype: list[dict]
    """
    prefix = {"seeds": "seed", "sigma": "sigma", "kde": "kde"}[mode]
    runs: list[dict] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        history = d / "history.json"
        if not history.exists():
            continue
        raw = d.name[len(prefix):]
        try:
            value: float = float(raw) if mode != "seeds" else float(int(raw))
        except ValueError:
            continue
        theory = d / "theory_vs_sft.json"
        runs.append({
            "slug": d.name,
            "value": value,
            "history_path": history,
            "theory_path": theory if theory.exists() else None,
        })
    runs.sort(key=lambda r: r["value"])
    return runs


def iter_sigma_seed_histories(root: Path) -> Iterator[dict]:
    """Yield sweep cells that contain ``history.json``.

    Walks ``root/sigma*/seed*/history.json``.

    :param root: Sweep root directory.
    :type root: pathlib.Path
    :yields: Dicts with keys ``sigma``, ``seed``, ``history_path``,
        ``history``.
    :rtype: Iterator[dict]
    """
    if not root.is_dir():
        return
    for sigma_dir in sorted(root.iterdir()):
        if not sigma_dir.is_dir():
            continue
        sm = SIGMA_DIR_RE.match(sigma_dir.name)
        if not sm:
            continue
        sigma = float(sm.group("val"))
        for seed_dir in sorted(sigma_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            sd = SEED_DIR_RE.match(seed_dir.name)
            hist_path = seed_dir / "history.json"
            if not sd or not hist_path.is_file():
                continue
            history = json.loads(hist_path.read_text(encoding="utf-8"))
            yield {
                "sigma": sigma,
                "seed": int(sd.group("val")),
                "history_path": hist_path,
                "history": history,
            }


def final_positions(history: list[dict]) -> tuple[list[str], np.ndarray]:
    """Extract sorted agent names and final positions from a history.

    :param history: Loaded ``history.json`` records.
    :type history: list[dict]
    :returns: ``(names, positions)`` with ``positions`` shape ``(N, L)``.
    :rtype: tuple[list[str], numpy.ndarray]
    :raises ValueError: If the history is empty or missing positions.
    """
    if not history:
        raise ValueError("history is empty")
    last = history[-1]
    if "positions" not in last:
        raise ValueError("history is missing positions")
    names = sorted(last["positions"].keys())
    pos = np.stack(
        [np.asarray(last["positions"][name], dtype=float) for name in names],
    )
    return names, pos


def collect_final_layout_labels(
    root: Path,
    *,
    classify_fn: Callable[[np.ndarray], str],
    spread_fn: Callable[[np.ndarray], float],
) -> list[dict]:
    """Classify final layouts for every completed sigma×seed run.

    :param root: Sweep root directory.
    :type root: pathlib.Path
    :param classify_fn: Layout classifier, e.g.
        :func:`infl_ens.training.pool_dynamics.classify_layout`.
    :type classify_fn: Callable[[numpy.ndarray], str]
    :param spread_fn: Pairwise spread metric, e.g.
        :func:`infl_ens.training.pool_dynamics.pairwise_spread`.
    :type spread_fn: Callable[[numpy.ndarray], float]
    :returns: One row per completed run with ``sigma``, ``seed``,
        ``spread``, and ``label``.
    :rtype: list[dict]
    """
    rows: list[dict] = []
    for cell in iter_sigma_seed_histories(root):
        _, pos = final_positions(cell["history"])
        rows.append({
            "sigma": cell["sigma"],
            "seed": cell["seed"],
            "spread": spread_fn(pos),
            "label": classify_fn(pos),
        })
    return rows


def discover_sigma_seed_history_paths(root: Path) -> list[tuple[float, int, Path]]:
    """Find ``(sigma_fraction, seed, history_path)`` under flat or variant layouts.

    :param root: Sweep directory (``sigma*/seed*`` or ``variant/sigma*/seed*``).
    :type root: pathlib.Path
    :returns: Sorted list of discoveries.
    :rtype: list[tuple[float, int, pathlib.Path]]
    """
    out: list[tuple[float, int, Path]] = []

    def scan(base: Path) -> None:
        if not base.is_dir():
            return
        for sigma_dir in sorted(base.iterdir()):
            if not sigma_dir.is_dir():
                continue
            sm = SIGMA_DIR_RE.match(sigma_dir.name)
            if not sm:
                continue
            sigma = float(sm.group("val"))
            for seed_dir in sorted(sigma_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                sd = SEED_DIR_RE.match(seed_dir.name)
                if not sd:
                    continue
                hist = seed_dir / "history.json"
                if hist.is_file():
                    out.append((sigma, int(sd.group("val")), hist))

    scan(root)
    if not out:
        for variant in sorted(root.iterdir()):
            if variant.is_dir():
                scan(variant)
    return sorted(out)
