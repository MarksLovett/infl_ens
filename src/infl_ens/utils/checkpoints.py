"""Checkpoint directory utilities."""

from __future__ import annotations

import shutil
from pathlib import Path


def _round_index(name: str) -> int | None:
    """Parse ``round-NN`` directory names.

    :param name: Directory basename.
    :type name: str
    :returns: Round index or ``None``.
    :rtype: int | None
    """
    if not name.startswith("round-"):
        return None
    suffix = name.split("-", 1)[1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def prune_intermediate_adapters(
    results_root: Path,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Delete intermediate ``round-NN`` LoRA dirs, keeping only the final round.

    Walks ``<results_root>/**/agents/<agent>/round-NN/``. Flat layouts
    (``agents/<agent>/adapter_model.safetensors`` with no ``round-*``
    children) are left unchanged.

    :param results_root: Top-level ``results/`` directory.
    :type results_root: pathlib.Path
    :param dry_run: If ``True``, report deletions without removing files.
    :type dry_run: bool
    :returns: Counts with keys ``agents_scanned``, ``round_dirs_removed``,
        ``adapters_before``, ``adapters_after``.
    :rtype: dict[str, int]
    """
    agents_dirs = sorted(p for p in results_root.rglob("agents") if p.is_dir())
    adapters_before = sum(
        1 for _ in results_root.rglob("adapter_model.safetensors")
    )

    agents_scanned = 0
    round_dirs_removed = 0

    for agents_dir in agents_dirs:
        for agent_dir in sorted(p for p in agents_dir.iterdir() if p.is_dir()):
            round_dirs = [
                p for p in agent_dir.iterdir()
                if p.is_dir() and _round_index(p.name) is not None
            ]
            if not round_dirs:
                continue

            agents_scanned += 1
            max_round = max(_round_index(p.name) for p in round_dirs)
            for rd in round_dirs:
                r_idx = _round_index(rd.name)
                if r_idx is None or r_idx == max_round:
                    continue
                round_dirs_removed += 1
                if dry_run:
                    print(f"would remove  {rd}")
                else:
                    shutil.rmtree(rd)

    adapters_after = sum(
        1 for _ in results_root.rglob("adapter_model.safetensors")
    )
    return {
        "agents_scanned": agents_scanned,
        "round_dirs_removed": round_dirs_removed,
        "adapters_before": adapters_before,
        "adapters_after": adapters_after,
    }
