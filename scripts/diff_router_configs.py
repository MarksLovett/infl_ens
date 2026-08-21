#!/usr/bin/env python3
"""Diff two router YAML configs; only allowlisted keys may differ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infl_ens.training.__main__ import _load_yaml  # noqa: E402

DEFAULT_ALLOWLIST = frozenset({
    "output_dir",
    "closed_loop.fixed_positions",
    "closed_loop.sft.output_dir",
})


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict to dot-path keys."""
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, path))
    else:
        out[prefix] = d
    return out


def diff_configs(
    reference: Path,
    candidate: Path,
    *,
    allowlist: frozenset[str] = DEFAULT_ALLOWLIST,
) -> dict[str, Any]:
    """Return structural diff; fail if any non-allowlisted key differs."""
    ref = _flatten(_load_yaml(reference))
    cand = _flatten(_load_yaml(candidate))
    all_keys = sorted(set(ref) | set(cand))
    allowed_diffs: dict[str, dict[str, Any]] = {}
    forbidden_diffs: dict[str, dict[str, Any]] = {}
    missing: dict[str, str] = {}

    for key in all_keys:
        in_ref = key in ref
        in_cand = key in cand
        if not in_ref:
            missing[key] = "missing_in_reference"
            if key not in allowlist:
                forbidden_diffs[key] = {"reference": None, "candidate": cand[key]}
            else:
                allowed_diffs[key] = {"reference": None, "candidate": cand[key]}
            continue
        if not in_cand:
            missing[key] = "missing_in_candidate"
            if key not in allowlist:
                forbidden_diffs[key] = {"reference": ref[key], "candidate": None}
            else:
                allowed_diffs[key] = {"reference": ref[key], "candidate": None}
            continue
        if ref[key] != cand[key]:
            entry = {"reference": ref[key], "candidate": cand[key]}
            if key in allowlist:
                allowed_diffs[key] = entry
            else:
                forbidden_diffs[key] = entry

    ok = len(forbidden_diffs) == 0
    return {
        "ok": ok,
        "reference": str(reference),
        "candidate": str(candidate),
        "allowlist": sorted(allowlist),
        "allowed_diffs": allowed_diffs,
        "forbidden_diffs": forbidden_diffs,
        "missing": missing,
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        help="Extra allowlisted dot-path (repeatable)",
    )
    args = parser.parse_args()

    allowlist = DEFAULT_ALLOWLIST | frozenset(args.allow)
    result = diff_configs(
        Path(args.reference),
        Path(args.candidate),
        allowlist=allowlist,
    )
    print("=== config diff (allowlist) ===")
    print(f"reference: {result['reference']}")
    print(f"candidate: {result['candidate']}")
    print(f"allowlist: {result['allowlist']}")
    if result["allowed_diffs"]:
        print("\n--- allowed diffs ---")
        for key, vals in sorted(result["allowed_diffs"].items()):
            print(f"{key}:")
            print(f"  ref:  {vals['reference']!r}")
            print(f"  cand: {vals['candidate']!r}")
    if result["forbidden_diffs"]:
        print("\n--- FORBIDDEN diffs ---")
        for key, vals in sorted(result["forbidden_diffs"].items()):
            print(f"{key}:")
            print(f"  ref:  {vals['reference']!r}")
            print(f"  cand: {vals['candidate']!r}")
    print(f"\nok={result['ok']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
