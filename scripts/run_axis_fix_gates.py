#!/usr/bin/env python3
"""Niche diagnostic + flat routing gates for split fix tracks.

Runs the three-way niche gate (variance / ICA / mid-mass) on the baseline
seven-axis run, then evaluates flat test-pool route-then-score for one or
more candidate runs. Each fix track is gated independently on the withheld
test partition:

- **collapse** — ``seven_axis_collapse_dead_axes`` (5 axes, dead benchmarks
  dropped).
- **router_improve** — ``seven_axis_router_improve_split`` (strategic routing
  on surviving niche-passing geometry).

Example::

    python scripts/run_axis_fix_gates.py \\
        --baseline-run results/seven_axis_pair_merge_split/seed0 \\
        --baseline-baseline-run results/seven_axis_baseline_replay_split/seed0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infl_ens.evaluation.axis_niche import (  # noqa: E402
    format_niche_markdown,
    niche_results_to_dict,
    run_axis_niche_diagnostic,
)
from infl_ens.evaluation.routing_eval import (  # noqa: E402
    format_headline_markdown,
    report_to_dict,
    run_flat_routing_eval,
)


def _gate_passes(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> bool:
    """Return whether candidate beats baseline on headline expected-G NLL.

    :param baseline: Flat routing metrics for baseline run.
    :type baseline: dict[str, float]
    :param candidate: Flat routing metrics for candidate run.
    :type candidate: dict[str, float]
    :returns: ``True`` if candidate expected routing NLL is lower.
    :rtype: bool
    """
    key = "learned_routing_expected_nll"
    b = float(baseline.get(key, baseline.get("learned_routing_nll", 1e9)))
    c = float(candidate.get(key, candidate.get("learned_routing_nll", 1e9)))
    return c < b


def _flat_from_report(report_dict: dict[str, Any]) -> dict[str, float]:
    """Extract flat metrics from a routing report dict.

    :param report_dict: Output of :func:`report_to_dict`.
    :type report_dict: dict[str, Any]
    :returns: Flat metric mapping.
    :rtype: dict[str, float]
    """
    return report_dict["flat"]


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--router-config",
        default="configs/benchmark/router/seven_axis_pair_merge_split.yaml",
    )
    parser.add_argument(
        "--baseline-run",
        default="results/seven_axis_pair_merge_split/seed0",
    )
    parser.add_argument(
        "--baseline-baseline-run",
        default="results/seven_axis_baseline_replay_split/seed0",
    )
    parser.add_argument(
        "--collapse-run",
        default=None,
        help="Optional collapse-track merge run directory.",
    )
    parser.add_argument(
        "--collapse-config",
        default="configs/benchmark/router/seven_axis_collapse_dead_axes.yaml",
    )
    parser.add_argument(
        "--collapse-baseline-run",
        default=None,
    )
    parser.add_argument(
        "--router-improve-run",
        default=None,
        help="Optional router-improve merge run directory.",
    )
    parser.add_argument(
        "--router-improve-config",
        default="configs/benchmark/router/seven_axis_router_improve_split.yaml",
    )
    parser.add_argument(
        "--router-improve-baseline-run",
        default=None,
    )
    parser.add_argument("--partition", default="test")
    parser.add_argument("--max-eval-records", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="results/seven_axis_fix_gates",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Only run niche diagnostic (no GPU scoring).",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    baseline_run = Path(args.baseline_run)
    router_config = Path(args.router_config)
    niche_results = run_axis_niche_diagnostic(
        router_config=router_config,
        repo_root=ROOT,
        history_path=baseline_run / "history.json",
        merge_run_dir=baseline_run,
        partition=args.partition,
        max_eval_records=args.max_eval_records,
        seed=args.seed,
    )
    niche_md = format_niche_markdown(niche_results)
    niche_json = niche_results_to_dict(niche_results)
    (out / "niche_diagnostic.md").write_text(niche_md, encoding="utf-8")
    (out / "niche_diagnostic.json").write_text(
        json.dumps(niche_json, indent=2), encoding="utf-8",
    )
    print(niche_md)

    summary: dict[str, Any] = {
        "niche": niche_json,
        "tracks": {},
    }

    if args.skip_scoring:
        (out / "gate_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8",
        )
        print(f"\nwrote {out / 'gate_summary.json'} (scoring skipped)")
        return 0

    baseline_report = run_flat_routing_eval(
        router_config=router_config,
        history_path=baseline_run / "history.json",
        merge_run_dir=baseline_run,
        baseline_run_dir=Path(args.baseline_baseline_run),
        repo_root=ROOT,
        partition=args.partition,
        max_eval_records=args.max_eval_records,
        seed=args.seed,
    )
    baseline_dict = report_to_dict(baseline_report)
    (out / "baseline_routing.json").write_text(
        json.dumps(baseline_dict, indent=2), encoding="utf-8",
    )
    print(format_headline_markdown(baseline_report))

    def _eval_track(
        name: str,
        run_dir: Path | None,
        config_path: str,
        baseline_dir: Path | None,
    ) -> None:
        if run_dir is None or not run_dir.is_dir():
            return
        hist = run_dir / "history.json"
        if not hist.is_file():
            print(f"skip {name}: missing {hist}")
            return
        bl = baseline_dir or run_dir.parent / f"{run_dir.name}_baseline"
        report = run_flat_routing_eval(
            router_config=Path(config_path),
            history_path=hist,
            merge_run_dir=run_dir,
            baseline_run_dir=bl,
            repo_root=ROOT,
            partition=args.partition,
            max_eval_records=args.max_eval_records,
            seed=args.seed,
        )
        rep_dict = report_to_dict(report)
        (out / f"{name}_routing.json").write_text(
            json.dumps(rep_dict, indent=2), encoding="utf-8",
        )
        flat_b = _flat_from_report(baseline_dict)
        flat_c = _flat_from_report(rep_dict)
        passed = _gate_passes(flat_b, flat_c)
        summary["tracks"][name] = {
            "run_dir": str(run_dir),
            "flat": flat_c,
            "delta_vs_baseline_expected": (
                flat_c.get("learned_routing_expected_nll", 0)
                - flat_b.get("learned_routing_expected_nll", 0)
            ),
            "gate_pass": passed,
        }
        print(f"\n=== {name} track ===")
        print(format_headline_markdown(report))
        print(f"gate pass (expected G < baseline): {passed}")

    _eval_track(
        "collapse",
        Path(args.collapse_run) if args.collapse_run else None,
        args.collapse_config,
        Path(args.collapse_baseline_run) if args.collapse_baseline_run else None,
    )
    _eval_track(
        "router_improve",
        Path(args.router_improve_run) if args.router_improve_run else None,
        args.router_improve_config,
        Path(args.router_improve_baseline_run)
        if args.router_improve_baseline_run
        else None,
    )

    (out / "gate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(f"\nwrote {out / 'gate_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
