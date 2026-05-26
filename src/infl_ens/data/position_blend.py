"""EMA blend toward a trait-space centroid after corpus projection.

Pure NumPy helpers used by :meth:`infl_ens.inflgame.router.RouterAgent
.update_position_from_corpus` and closed-loop trainers. Lives under
:mod:`infl_ens.data` so :mod:`infl_ens.inflgame` does not import
:mod:`infl_ens.utils`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

VALID_BLEND_MODES = frozenset({"static", "cap_linf", "cap_l2", "trust_box"})


def parse_position_step(cfg: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Normalise a ``closed_loop.position_step`` config block.

    :param cfg: Raw config mapping or ``None`` (static step).
    :type cfg: dict | None
    :returns: Dict with keys ``mode``, ``step_cap``, ``blend_max``,
        ``clip_to_box``.
    :rtype: dict
    """
    if not cfg:
        return {
            "mode": "static",
            "step_cap": 0.05,
            "blend_max": None,
            "clip_to_box": False,
        }
    if isinstance(cfg, str):
        return {
            "mode": cfg,
            "step_cap": 0.05,
            "blend_max": None,
            "clip_to_box": False,
        }
    return {
        "mode": str(cfg.get("mode", "static")),
        "step_cap": float(cfg.get("step_cap", 0.05)),
        "blend_max": cfg.get("blend_max"),
        "clip_to_box": bool(cfg.get("clip_to_box", False)),
    }


def effective_blend(
    current: np.ndarray,
    target: np.ndarray,
    *,
    blend: float = 0.5,
    mode: str = "static",
    step_cap: float = 0.05,
    blend_max: Optional[float] = None,
    eps: float = 1e-12,
) -> float:
    """Effective EMA coefficient for one position update.

    :param current: Position before the update, shape ``(L,)``.
    :type current: numpy.ndarray
    :param target: Corpus centroid (or weighted centroid), shape ``(L,)``.
    :type target: numpy.ndarray
    :param blend: Default / ceiling blend (from ``closed_loop.blend``).
    :type blend: float
    :param mode: Step policy — ``static``, ``cap_linf``, ``cap_l2``,
        or ``trust_box`` (per-coordinate cap plus box feasibility).
    :type mode: str
    :param step_cap: Maximum allowed step size for capped modes
        (L∞ for ``cap_linf`` / ``trust_box``, L2 for ``cap_l2``).
    :type step_cap: float
    :param blend_max: Optional ceiling; defaults to *blend*.
    :type blend_max: float | None
    :param eps: Threshold below which *current* ≈ *target*.
    :type eps: float
    :returns: Blend coefficient in ``[0, 1]``.
    :rtype: float
    :raises ValueError: If *mode* is unknown or *blend* is outside ``[0, 1]``.
    """
    if mode not in VALID_BLEND_MODES:
        raise ValueError(
            f"mode must be one of {sorted(VALID_BLEND_MODES)}, got {mode!r}"
        )
    bmax = float(blend if blend_max is None else blend_max)
    if not 0.0 <= bmax <= 1.0:
        raise ValueError(f"blend_max must be in [0, 1], got {bmax}")
    if mode == "static":
        return bmax

    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    delta = target - current

    if mode == "cap_linf":
        m = float(np.max(np.abs(delta)))
        if m < eps:
            return 0.0
        return float(np.clip(min(bmax, step_cap / m), 0.0, 1.0))

    if mode == "cap_l2":
        n = float(np.linalg.norm(delta))
        if n < eps:
            return 0.0
        return float(np.clip(min(bmax, step_cap / n), 0.0, 1.0))

    # trust_box: cap L∞ step, then shrink so the update stays in [0, 1]^L.
    m = float(np.max(np.abs(delta)))
    if m < eps:
        return 0.0
    beta = min(bmax, step_cap / m)
    for _ in range(24):
        pos = (1.0 - beta) * current + beta * target
        if np.all(pos >= -eps) and np.all(pos <= 1.0 + eps):
            break
        beta *= 0.5
    return float(np.clip(beta, 0.0, 1.0))


def apply_position_update(
    current: np.ndarray,
    target: np.ndarray,
    *,
    blend: float = 0.5,
    position_step: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, float]:
    """Apply one EMA position update with optional adaptive blend.

    :param current: Position before update.
    :type current: numpy.ndarray
    :param target: Target centroid.
    :type target: numpy.ndarray
    :param blend: Base blend from config.
    :type blend: float
    :param position_step: Optional ``closed_loop.position_step`` block.
    :type position_step: dict | None
    :returns: ``(new_position, effective_blend)``.
    :rtype: tuple[numpy.ndarray, float]
    """
    ps = parse_position_step(position_step)
    beta = effective_blend(
        current,
        target,
        blend=blend,
        mode=ps["mode"],
        step_cap=ps["step_cap"],
        blend_max=ps["blend_max"],
    )
    new_pos = (1.0 - beta) * current + beta * target
    if ps["clip_to_box"]:
        new_pos = np.clip(new_pos, 0.0, 1.0)
    return new_pos, beta
