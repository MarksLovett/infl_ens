"""Mean and variance of final trait positions across seeds.

Reads ``<root>/sigma*/seed*/history.json`` (or ``<root>/<variant>/sigma*/...``)
and reports per-clone mean ± std for each trait axis.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_SIGMA_RE = re.compile(r"^sigma(?P<val>[0-9.]+)$", re.IGNORECASE)
_SEED_RE = re.compile(r"^seed(?P<val>\d+)$", re.IGNORECASE)


def _discover_histories(root: Path) -> list[tuple[float, int, Path]]:
    """Find ``(sigma_fraction, seed, path)`` triples under *root*.

    :param root: Sweep directory (flat ``sigma*/seed*`` or one variant subdir).
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
            sm = _SIGMA_RE.match(sigma_dir.name)
            if not sm:
                continue
            sigma = float(sm.group("val"))
            for seed_dir in sorted(sigma_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                sd = _SEED_RE.match(seed_dir.name)
                if not sd:
                    continue
                hist = seed_dir / "history.json"
                if hist.is_file():
                    out.append((sigma, int(sd.group("val")), hist))

    # Flat layout: root/sigma*/seed*
    scan(root)
    # Variant layout: root/variant/sigma*/seed* — only if flat found nothing
    if not out:
        for variant in sorted(root.iterdir()):
            if variant.is_dir():
                scan(variant)
    return sorted(out)


def aggregate_positions(
    entries: list[tuple[float, int, Path]],
) -> dict[float, dict]:
    """Compute mean/std of final positions per sigma.

    :param entries: ``(sigma_fraction, seed, history_path)`` list.
    :type entries: list[tuple[float, int, pathlib.Path]]
    :returns: Nested stats dict keyed by sigma fraction.
    :rtype: dict
    """
    by_sigma: dict[float, list[tuple[int, dict[str, list[float]]]]] = {}
    for sigma, seed, path in entries:
        with path.open(encoding="utf-8") as fh:
            records = json.load(fh)
        positions = records[-1]["positions"]
        by_sigma.setdefault(sigma, []).append((seed, positions))

    result: dict[float, dict] = {}
    for sigma, runs in sorted(by_sigma.items()):
        names = list(runs[0][1].keys())
        n = len(runs)
        spreads: list[float] = []
        per_agent: dict[str, dict] = {}
        for _, pos in runs:
            arr = np.stack([np.asarray(pos[name], dtype=float) for name in names])
            if len(names) >= 2:
                spreads.append(float(np.mean([
                    np.linalg.norm(arr[i] - arr[j])
                    for i in range(len(names))
                    for j in range(i + 1, len(names))
                ])))
        for name in names:
            arr = np.array([np.asarray(pos[name], dtype=float) for _, pos in runs])
            per_agent[name] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0, ddof=1 if n > 1 else 0).tolist(),
            }
        result[sigma] = {
            "n_runs": n,
            "seeds": [s for s, _ in runs],
            "spread_mean": float(np.mean(spreads)),
            "spread_std": float(np.std(spreads, ddof=1 if n > 1 else 0)),
            "agents": per_agent,
        }
    return result


def _print_report(stats: dict[float, dict], axis_labels: tuple[str, str]) -> None:
    """Print human-readable tables.

    :param stats: Output of :func:`aggregate_positions`.
    :type stats: dict
    :param axis_labels: Trait axis names.
    :type axis_labels: tuple[str, str]
    """
    for sigma, block in sorted(stats.items()):
        print(f"\n{'=' * 60}")
        print(f"  sigma_fraction = {sigma:g}   n = {block['n_runs']} seeds")
        print(f"  pairwise spread: {block['spread_mean']:.4f} ± {block['spread_std']:.4f}")
        print(f"  seeds: {block['seeds']}")
        print(f"{'=' * 60}")
        print(f"{'clone':<10} {axis_labels[0]+' (mean±std)':<22} {axis_labels[1]+' (mean±std)':<22}")
        print("-" * 60)
        for name, ag in block["agents"].items():
            m0, m1 = ag["mean"]
            s0, s1 = ag["std"]
            print(
                f"{name:<10} {m0:.4f}±{s0:.4f}          "
                f"{m1:.4f}±{s1:.4f}"
            )


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional CLI args.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument(
        "--axis-labels", nargs=2, default=["harm", "hallucination"],
    )
    args = p.parse_args(argv)

    entries = _discover_histories(args.root)
    if not entries:
        print(f"no histories under {args.root}", file=sys.stderr)
        return 1

    stats = aggregate_positions(entries)
    _print_report(stats, (args.axis_labels[0], args.axis_labels[1]))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
