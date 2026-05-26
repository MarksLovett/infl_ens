"""Post-process a completed sigma sweep into per-sigma figure subfolders.

Discovers every ``sigma*`` directory under a sweep root, verifies each
run's ``history.json`` is well-formed and complete, then for each sigma
runs (in order):

1. ``scripts/plot_closed_loop_history.py`` → trajectory + utility tracking
2. ``scripts/compare_theory_vs_sft.py``    → theory vs SFT comparison
3. ``scripts/probe_sft_capability.py``     → cross-perplexity probe

After all per-sigma post-processing, runs ``scripts/plot_sweep.py
--mode sigma`` to aggregate the whole sweep into a single comparison
figure, then prints a cross-sigma table of final-round specialisation
margin.

This script exists because:

- Bash heredocs interact badly with multi-line paste in some terminals,
  which has caused real failures in this project's workflow.
- ``run_sweep.sh`` slugs use ``printf '%g'`` and strip trailing zeros
  (``sigma_fraction=1.0`` becomes slug ``sigma1``), but figure
  subfolders look cleaner with ``sigma_1.0``. Maintaining the asymmetric
  mapping in bash is error-prone.
- Auto-discovery of completed runs is more robust than passing an
  explicit sigma list — the script handles whatever subset of the sweep
  finished.
- A single Python entry point cleanly supports ``--verify-only`` and
  ``--skip-probe`` shortcuts for quick iterative use.

Typical invocation::

    python scripts/sigma_sweep_post.py \\
        --sweep-root results/sweep_sigma_r20_strategic_long \\
        --fig-root scripts/figures/sigma_sweep_r20 \\
        --config configs/benchmark/router/safety_truth_n4_r20_strategic_long.yaml

For the cumulative-LoRA sigma sweep::

    python scripts/sigma_sweep_post.py \\
        --sweep-root results/sweep_sigma_r20_strategic_long_cum \\
        --fig-root scripts/figures/sigma_sweep_r20_cum \\
        --config configs/benchmark/router/safety_truth_n4_r20_strategic_long_cum.yaml

To check what's on disk without running anything expensive::

    python scripts/sigma_sweep_post.py --sweep-root <dir> --verify-only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# -----------------------------------------------------------------------------
# Discovery and verification
# -----------------------------------------------------------------------------

_SLUG_RE = re.compile(r"^sigma([0-9]+(?:\.[0-9]+)?)$")


def parse_sigma_from_slug(slug: str) -> float:
    """Extract the sigma value from a ``run_sweep.sh`` sigma-mode slug.

    The launcher writes slugs of the form ``sigma{printf '%g' val}`` so
    e.g. ``0.25`` → ``"sigma0.25"`` and ``1.0`` → ``"sigma1"``.

    :param slug: Slug string from a sweep directory name.
    :type slug: str
    :returns: Parsed sigma value.
    :rtype: float
    :raises ValueError: If the slug doesn't match the expected format.
    """
    m = _SLUG_RE.match(slug)
    if m is None:
        raise ValueError(f"not a sigma-sweep slug: {slug!r}")
    return float(m.group(1))


def fig_subfolder_name(sigma: float) -> str:
    """Return the figures-subfolder name for a given sigma value.

    Uses Python's default float string repr so ``1.0`` stays
    ``"sigma_1.0"`` rather than collapsing to ``"sigma_1"``. This is the
    opposite convention from ``run_sweep.sh`` slugs, intentionally —
    figure subfolders are human-facing and benefit from consistent
    formatting; sweep slugs match the legacy bash launcher.

    :param sigma: Sigma value.
    :type sigma: float
    :returns: Subfolder name (without leading directories).
    :rtype: str
    """
    return f"sigma_{sigma}"


@dataclass
class RunStatus:
    """Verification result for a single sigma run.

    :ivar slug: Sweep slug (e.g. ``"sigma0.25"``).
    :ivar sigma: Parsed sigma_fraction value.
    :ivar run_dir: Run directory on disk.
    :ivar history_exists: Whether ``history.json`` is present.
    :ivar rounds_completed: Number of round records in ``history.json``.
    :ivar n_sft_steps_round0: SFT loss entries for clone-0 round 0.
    :ivar agents_with_data: Number of agents that have any SFT logs in
        round 0.
    :ivar incomplete: True if anything looks wrong with this run.
    :ivar note: Human-readable diagnostic message.
    """

    slug: str
    sigma: float
    run_dir: Path
    history_exists: bool
    rounds_completed: int
    n_sft_steps_round0: int
    agents_with_data: int
    incomplete: bool
    note: str


def discover_runs(sweep_root: Path) -> list[RunStatus]:
    """Walk ``sweep_root`` and verify every ``sigma*`` subdirectory.

    :param sweep_root: Sweep root path (e.g. ``results/sweep_sigma_*``).
    :type sweep_root: pathlib.Path
    :returns: Run statuses, sorted by sigma value.
    :rtype: list[RunStatus]
    """
    if not sweep_root.exists():
        raise FileNotFoundError(f"sweep root not found: {sweep_root}")
    runs: list[RunStatus] = []
    for child in sorted(sweep_root.iterdir()):
        if not child.is_dir():
            continue
        try:
            sigma = parse_sigma_from_slug(child.name)
        except ValueError:
            continue
        runs.append(_verify_run(child, child.name, sigma))
    runs.sort(key=lambda r: r.sigma)
    return runs


def _verify_run(run_dir: Path, slug: str, sigma: float) -> RunStatus:
    """Inspect a single run directory.

    :param run_dir: Run directory.
    :type run_dir: pathlib.Path
    :param slug: Sweep slug.
    :type slug: str
    :param sigma: Parsed sigma value.
    :type sigma: float
    :returns: Verification result.
    :rtype: RunStatus
    """
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        return RunStatus(
            slug=slug, sigma=sigma, run_dir=run_dir,
            history_exists=False, rounds_completed=0,
            n_sft_steps_round0=0, agents_with_data=0,
            incomplete=True, note="no history.json",
        )
    try:
        with hist_path.open("r", encoding="utf-8") as fh:
            hist = json.load(fh)
    except json.JSONDecodeError as exc:
        return RunStatus(
            slug=slug, sigma=sigma, run_dir=run_dir,
            history_exists=True, rounds_completed=0,
            n_sft_steps_round0=0, agents_with_data=0,
            incomplete=True, note=f"history.json malformed: {exc}",
        )
    n_rounds = len(hist)
    if n_rounds == 0:
        return RunStatus(
            slug=slug, sigma=sigma, run_dir=run_dir,
            history_exists=True, rounds_completed=0,
            n_sft_steps_round0=0, agents_with_data=0,
            incomplete=True, note="empty history",
        )
    # Inspect round 0 for SFT log density and agent coverage
    r0 = hist[0]
    sft_logs = r0.get("agent_sft_logs", {})
    n_loss_c0 = sum(
        1 for e in sft_logs.get("clone-0", []) if "loss" in e
    )
    n_agents_with_data = sum(
        1 for entries in sft_logs.values() if any("loss" in e for e in entries)
    )
    note_parts: list[str] = []
    incomplete = False
    if n_rounds < 2:
        note_parts.append(f"only {n_rounds} round(s)")
        incomplete = True
    if n_loss_c0 == 0:
        note_parts.append("no SFT loss entries on clone-0 round 0")
        incomplete = True
    elif n_loss_c0 < 4:
        note_parts.append(f"only {n_loss_c0} SFT loss entries (very short SFT)")
    note = "; ".join(note_parts) if note_parts else "ok"
    return RunStatus(
        slug=slug, sigma=sigma, run_dir=run_dir,
        history_exists=True, rounds_completed=n_rounds,
        n_sft_steps_round0=n_loss_c0,
        agents_with_data=n_agents_with_data,
        incomplete=incomplete, note=note,
    )


def print_verify_table(runs: list[RunStatus]) -> None:
    """Print a status table to stdout.

    :param runs: Run statuses from :func:`discover_runs`.
    :type runs: list[RunStatus]
    """
    print()
    print("=== verification ===")
    print(f"{'slug':<14}{'sigma':<8}{'rounds':<10}"
          f"{'sft_r0':<10}{'agents':<10}{'note'}")
    print("-" * 74)
    for r in runs:
        print(f"{r.slug:<14}{r.sigma:<8.4g}{r.rounds_completed:<10}"
              f"{r.n_sft_steps_round0:<10}{r.agents_with_data:<10}{r.note}")
    bad = [r.slug for r in runs if r.incomplete]
    if bad:
        print(f"\nINCOMPLETE: {', '.join(bad)}")
    else:
        print(f"\nAll {len(runs)} runs look complete.")


# -----------------------------------------------------------------------------
# Subprocess wrappers
# -----------------------------------------------------------------------------

def _run(cmd: list[str], *, label: str) -> bool:
    """Run a subprocess and report success.

    :param cmd: Command argv list.
    :type cmd: list[str]
    :param label: Short label for log output.
    :type label: str
    :returns: True if the command exited 0, False otherwise.
    :rtype: bool
    """
    print(f"  [{label}] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [{label}] FAILED with exit code {result.returncode}")
        return False
    return True


def post_process_run(
    status: RunStatus,
    fig_dir: Path,
    config_path: Path,
    *,
    axis_labels: tuple[str, ...] = ("harm", "hallucination"),
    max_prompts: int = 128,
    run_trajectory: bool = True,
    run_theory: bool = True,
    run_probe: bool = True,
    title_prefix: str = "",
) -> dict[str, bool]:
    """Run trajectory, theory-vs-SFT, and probe scripts for one sigma.

    :param status: Verification result for the run.
    :type status: RunStatus
    :param fig_dir: Per-sigma figure subfolder (created if missing).
    :type fig_dir: pathlib.Path
    :param config_path: Base YAML config for theory-vs-SFT comparison.
    :type config_path: pathlib.Path
    :param axis_labels: Axis labels for trajectory plots.
    :type axis_labels: tuple[str, ...]
    :param max_prompts: ``--max-prompts`` forwarded to the probe.
    :type max_prompts: int
    :param run_trajectory: Whether to run the trajectory plot.
    :type run_trajectory: bool
    :param run_theory: Whether to run the theory-vs-SFT comparison.
    :type run_theory: bool
    :param run_probe: Whether to run the (expensive) capability probe.
    :type run_probe: bool
    :param title_prefix: Prepended to per-figure titles.
    :type title_prefix: str
    :returns: Per-step success map keyed by ``"trajectory"``,
        ``"theory_vs_sft"``, ``"probe"``.
    :rtype: dict[str, bool]
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    rundir = status.run_dir
    history = rundir / "history.json"
    base = f"sigma_fraction={status.sigma:g}, 20 rounds"
    suffix = f" — {base}" if not title_prefix else f" — {title_prefix} {base}"
    out: dict[str, bool] = {}
    if run_trajectory:
        out["trajectory"] = _run(
            [
                "python", "scripts/plot_closed_loop_history.py",
                "--history", str(history),
                "--axis-labels", *axis_labels,
                "--title", f"Trajectory{suffix}",
                "--output-stem", str(fig_dir / "trajectory"),
            ],
            label="trajectory",
        )
    if run_theory:
        out["theory_vs_sft"] = _run(
            [
                "python", "scripts/compare_theory_vs_sft.py",
                "--config", str(config_path),
                "--history", str(history),
                "--axis-labels", *axis_labels,
                "--title", f"Theory vs SFT{suffix}",
                "--output-stem", str(fig_dir / "theory_vs_sft"),
                "--summary-json", str(rundir / "theory_vs_sft.json"),
                # Critical: each sigma run must compute theory at its OWN
                # sigma, not the base config's sigma_fraction (which is a
                # single fixed value across the whole sweep). Without
                # this override every per-sigma theory_vs_sft.json
                # contains the same NE positions, which then propagates
                # into plot_sweep's per-panel theory overlay.
                "--sigma-fraction-override", str(status.sigma),
            ],
            label="theory_vs_sft",
        )
    if run_probe:
        out["probe"] = _run(
            [
                "python", "scripts/probe_sft_capability.py",
                "--run-dir", str(rundir),
                "--base-sft-dir", str(rundir / "agents"),
                "--max-prompts", str(max_prompts),
                "--output-stem", str(fig_dir / "probe"),
                "--title", f"Capability probe{suffix}",
            ],
            label="probe",
        )
    return out


def aggregate_sweep(
    sweep_root: Path, fig_root: Path, *, title: str, with_theory: bool = True,
) -> bool:
    """Run :mod:`scripts.plot_sweep` to make the cross-sigma comparison.

    :param sweep_root: Sweep results root.
    :type sweep_root: pathlib.Path
    :param fig_root: Per-sweep figures root (aggregate goes in a
        nested ``aggregate/`` subfolder).
    :type fig_root: pathlib.Path
    :param title: Aggregate-figure title.
    :type title: str
    :param with_theory: Pass ``--with-theory`` to ``plot_sweep.py``.
    :type with_theory: bool
    :returns: True on success.
    :rtype: bool
    """
    agg_dir = fig_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    stem = agg_dir / fig_root.name  # e.g. sigma_sweep_r20
    cmd = [
        "python", "scripts/plot_sweep.py",
        "--root", str(sweep_root),
        "--mode", "sigma",
        "--title", title,
        "--output-stem", str(stem),
    ]
    if with_theory:
        cmd.append("--with-theory")
    return _run(cmd, label="aggregate")


# -----------------------------------------------------------------------------
# Margin table
# -----------------------------------------------------------------------------

def print_margin_table(runs: list[RunStatus], fig_root: Path) -> None:
    """Print a cross-sigma table of final-round specialisation margins.

    Reads per-sigma probe CSVs from ``fig_root/sigma_X.Y/probe.csv`` and
    computes ``margin = mean NLL(others) − mean NLL(own)`` at the last
    round logged.

    :param runs: Run statuses (used only for the sigma list).
    :type runs: list[RunStatus]
    :param fig_root: Per-sweep figures root.
    :type fig_root: pathlib.Path
    """
    print()
    print("=== cross-sigma final-round specialisation margin ===")
    print(f"{'sigma':<8}{'final_margin':<16}{'diag_NLL':<14}{'off_NLL':<14}")
    print("-" * 52)
    for r in runs:
        csv_path = fig_root / fig_subfolder_name(r.sigma) / "probe.csv"
        if not csv_path.exists():
            print(f"{r.sigma:<8.4g}(no probe csv at {csv_path})")
            continue
        try:
            with csv_path.open("r", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except (OSError, csv.Error) as exc:
            print(f"{r.sigma:<8.4g}(failed to read probe csv: {exc})")
            continue
        if not rows:
            print(f"{r.sigma:<8.4g}(probe csv empty)")
            continue
        last = max(int(x["round"]) for x in rows)
        diag = [
            float(x["nll"]) for x in rows
            if x["agent_i"] == x["agent_j"] and int(x["round"]) == last
        ]
        off = [
            float(x["nll"]) for x in rows
            if x["agent_i"] != x["agent_j"] and int(x["round"]) == last
        ]
        if not diag or not off:
            print(f"{r.sigma:<8.4g}(incomplete probe)")
            continue
        diag_m, off_m = float(np.mean(diag)), float(np.mean(off))
        print(f"{r.sigma:<8.4g}{off_m - diag_m:<16.4f}"
              f"{diag_m:<14.4f}{off_m:<14.4f}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    :returns: Parser.
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="Post-process a completed sigma sweep into unique "
                    "per-sigma figure subfolders, aggregate, and report "
                    "specialisation margins.",
    )
    p.add_argument("--sweep-root", type=Path, required=True,
                   help="Sweep results root (contains sigma*/ subdirs).")
    p.add_argument("--fig-root", type=Path, default=None,
                   help="Figures output root. Defaults to "
                        "scripts/figures/<sweep-root-basename>.")
    p.add_argument("--config", type=Path, default=None,
                   help="Base YAML config for theory-vs-SFT comparison. "
                        "Required unless --skip-theory.")
    p.add_argument("--axis-labels", type=str, nargs="+",
                   default=["harm", "hallucination"],
                   help="Axis labels for trajectory plots.")
    p.add_argument("--max-prompts", type=int, default=128,
                   help="--max-prompts forwarded to probe_sft_capability.py.")
    p.add_argument("--title-prefix", type=str, default="",
                   help="Prepended to per-figure titles.")
    p.add_argument("--aggregate-title", type=str, default=None,
                   help="Title for the aggregate cross-sigma figure.")
    p.add_argument("--verify-only", action="store_true",
                   help="Print verification table and exit. No expensive work.")
    p.add_argument("--skip-trajectory", action="store_true",
                   help="Skip the trajectory plot step.")
    p.add_argument("--skip-theory", action="store_true",
                   help="Skip the theory_vs_sft step.")
    p.add_argument("--skip-probe", action="store_true",
                   help="Skip the (expensive) capability probe step.")
    p.add_argument("--skip-aggregate", action="store_true",
                   help="Skip the final plot_sweep.py aggregation.")
    p.add_argument("--continue-on-incomplete", action="store_true",
                   help="Process incomplete runs anyway (default: abort).")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional CLI argv override.
    :type argv: list[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    args = _build_parser().parse_args(argv)
    sweep_root: Path = args.sweep_root.resolve()
    fig_root: Path = (
        args.fig_root or Path("scripts/figures") / sweep_root.name
    ).resolve()

    runs = discover_runs(sweep_root)
    if not runs:
        print(f"no sigma* subdirectories found under {sweep_root}",
              file=sys.stderr)
        return 1
    print_verify_table(runs)
    if args.verify_only:
        return 0

    incomplete_runs = [r for r in runs if r.incomplete]
    if incomplete_runs and not args.continue_on_incomplete:
        print(f"\naborting because {len(incomplete_runs)} run(s) are "
              "incomplete. Pass --continue-on-incomplete to process anyway.",
              file=sys.stderr)
        return 2

    if not args.skip_theory and args.config is None:
        print("--config is required unless --skip-theory", file=sys.stderr)
        return 1

    print()
    print(f"=== post-processing {len(runs)} sigma values ===")
    print(f"fig_root: {fig_root}")
    fig_root.mkdir(parents=True, exist_ok=True)

    any_failed = False
    for r in runs:
        if r.incomplete and not args.continue_on_incomplete:
            continue
        print(f"\n--- sigma={r.sigma:g} (slug={r.slug}) ---")
        result = post_process_run(
            r,
            fig_root / fig_subfolder_name(r.sigma),
            args.config or Path("/dev/null"),  # only used if not skip_theory
            axis_labels=tuple(args.axis_labels),
            max_prompts=args.max_prompts,
            run_trajectory=not args.skip_trajectory,
            run_theory=not args.skip_theory,
            run_probe=not args.skip_probe,
            title_prefix=args.title_prefix,
        )
        if not all(result.values()):
            any_failed = True

    if not args.skip_aggregate:
        print()
        print("--- aggregating sweep ---")
        title = args.aggregate_title or (
            f"Sigma sweep — {sweep_root.name}"
        )
        if not aggregate_sweep(sweep_root, fig_root, title=title):
            any_failed = True

    if not args.skip_probe:
        print_margin_table(runs, fig_root)

    print()
    print(f"figures under: {fig_root}")
    return 0 if not any_failed else 3


if __name__ == "__main__":
    raise SystemExit(main()) 