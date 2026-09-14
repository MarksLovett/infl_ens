"""Score every expert on the VALIDATION partition and persist the matrix.

The routing stage only ever scores the test partition, so the per-example
expert-NLL matrix exists for test and nowhere else. Fitting a router *on* the
experts' losses needs a held-out split the router did not see at evaluation
time, which is what validation is for.

This runs exactly the same scoring path as the routing stage
(``run_flat_routing_eval``) with ``partition="val"``, and writes the arrays into
``<run_dir>/val_arrays/``. It touches no pipeline code and overwrites no
existing diagnostics -- the test-partition results are left alone.

    python scripts/score_val_matrix.py --run results/seven_axis_theory_eq/seed0 \
        --generalist results/seven_axis_3arm_generalist/seed0

Cost is roughly half a test pass (2,836 validation prompts against 5,671 test),
so about 8 minutes per seven-pair arm on the A100.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from infl_ens.evaluation.routing_eval import (  # noqa: E402
    report_to_dict,
    run_flat_routing_eval,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="arm run dir, e.g. results/X/seed0")
    ap.add_argument("--generalist", required=True, help="pooled baseline run dir")
    ap.add_argument("--partition", default="val")
    ap.add_argument("--max-records", type=int, default=1000)
    args = ap.parse_args()

    run = (REPO / args.run).resolve()
    gen = (REPO / args.generalist).resolve()
    resolved = run / "resolved_config.yaml"
    for p in (resolved, run / "history.json", gen):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    cfg = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    sft = cfg.get("sft", {}) or {}
    eval_block = cfg.get("eval", {}) or {}

    # Last trained round, same convention the routing stage uses.
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    last = len(history) - 1

    # Partition-scoped so a --partition test run cannot clobber the
    # validation matrices, which live in val_arrays/.
    out_dir = run / f"{args.partition}_arrays"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"scoring {args.partition} for {run.parent.name} "
          f"(round {last}, base {sft.get('base_model')})", flush=True)
    report = run_flat_routing_eval(
        router_config=resolved,
        history_path=run / "history.json",
        merge_run_dir=run,
        baseline_run_dir=gen,
        repo_root=REPO,
        partition=args.partition,
        max_eval_records=args.max_records,
        seed=int(cfg.get("seed", 0)),
        round_idx=last,
        base_model=str(sft.get("base_model")),
        max_seq_length=int(sft.get("max_seq_length", 1024)),
        forward_batch_size=int(eval_block.get("forward_batch_size", 8)),
        save_arrays_dir=out_dir,
    )
    (out_dir / "diagnostics.json").write_text(
        json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")
    for f in sorted(out_dir.glob("*.npy")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
