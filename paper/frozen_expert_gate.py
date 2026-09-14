"""Can a learned combiner beat the game router, with the experts held frozen?

The experts are never retrained here. Only the rule that combines them changes,
and every learned rule is fitted on the VALIDATION expert-NLL matrix and scored
on TEST -- so nothing is fitted on the numbers it is judged by.

Combiners
    game              the learned allocation G (what the paper reports)
    stacking          global simplex weights fit on validation NLL
                      (``fit_simplex_stacking``) -- one weight vector for all
                      prompts, no per-prompt features, no geometry
    best-single       the single expert with the lowest validation NLL,
                      used for every test prompt
    val-permutation   G with the expert permutation chosen on validation
    uniform           1/K
    oracle            per-prompt best expert (unachievable ceiling)

``stacking`` is the sharpest of these: it sees the full expert-NLL vector on a
held-out split and is free to weight the experts however it likes, but it cannot
condition on the prompt. If it matches the game router, the routing is not
buying anything a fixed weighting could not.

    python frozen_expert_gate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
sys.path.insert(0, str(REPO / "src"))

from infl_ens.evaluation.routers import (  # noqa: E402
    fit_simplex_stacking,
    validation_selected_permutation,
)
from infl_ens.evaluation.routing_eval import mixture_nll  # noqa: E402

ARMS = [
    (2, "2 pairs, naive", "seven_axis_2pair_sigma05"),
    (2, "2 pairs, equilibrium", "seven_axis_theory_eq_2pair"),
    (3, "3 pairs, naive", "seven_axis_3pair_sigma05"),
    (3, "3 pairs, equilibrium", "seven_axis_theory_eq_3pair"),
    (4, "4 pairs, naive", "seven_axis_4pair_sigma05"),
    (4, "4 pairs, equilibrium", "seven_axis_theory_eq_4pair"),
    (5, "5 pairs, naive", "seven_axis_5pair_sigma05"),
    (5, "5 pairs, equilibrium", "seven_axis_theory_eq_5pair"),
    (6, "6 pairs, naive", "seven_axis_6pair_sigma05"),
    (6, "6 pairs, equilibrium", "seven_axis_theory_eq_6pair"),
    (7, "7 pairs, naive", "seven_axis_soft_full_pairs"),
    (7, "7 pairs, equilibrium", "seven_axis_theory_eq"),
    (9, "9 pairs, naive", "seven_axis_9pair_sigma05"),
    (9, "9 pairs, equilibrium", "seven_axis_theory_eq_9pair"),
]


def load_split(run: str, sub: str):
    d = REPO / "results" / run / "seed0" / sub
    try:
        return (np.load(d / "merge_nll.npy"),
                np.load(d / "g_merge.npy").T,
                np.load(d / "token_counts.npy"))
    except FileNotFoundError:
        return None


def score(nll, ntok, w):
    return float((w * nll).sum(axis=1).mean()), float(mixture_nll(nll, ntok, w).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(PAPER / "data" / "frozen_expert_gate.json"))
    args = ap.parse_args()
    store = {}

    for kk, label, run in ARMS:
        val = load_split(run, "val_arrays")
        test = load_split(run, "")
        if val is None or test is None:
            print(f"{label}: missing {'val' if val is None else 'test'} matrices")
            continue
        v_nll, v_w, v_tok = val
        t_nll, t_w, t_tok = test
        pooled = json.loads(
            (REPO / "results" / run / "seed0" /
             "routing_ensemble_diagnostics.json").read_text())["flat"]["pooled_nll"]
        m, k = t_nll.shape

        combiners: dict[str, np.ndarray] = {"game": t_w}

        # Global simplex weights fit on validation; broadcast to every test row.
        stack = fit_simplex_stacking(v_nll, v_tok)
        combiners["stacking"] = np.broadcast_to(stack, (m, k)).copy()

        # Best single expert by validation mean NLL.
        best = int(v_nll.mean(axis=0).argmin())
        one = np.zeros((m, k)); one[:, best] = 1.0
        combiners["best-single"] = one

        # Expert permutation of G selected on validation. Signature is
        # (base_weights, expert_nll) and it returns (permutation, summary).
        # K! grows fast, so skip the search once it stops being cheap.
        if k <= 8:
            perm, _summary = validation_selected_permutation(v_w, v_nll)
            combiners["val-permutation"] = t_w[:, list(perm)]
        else:
            print(f"  ({label}: permutation search skipped, {k}! too large)")

        combiners["uniform"] = np.full((m, k), 1.0 / k)
        oracle = np.zeros((m, k)); oracle[np.arange(m), t_nll.argmin(axis=1)] = 1.0
        combiners["oracle"] = oracle

        print(f"\n{label}   ({run}: val {v_nll.shape}, test {t_nll.shape})")
        print(f"  {'combiner':16s} {'expected':>10s} {'mixture':>10s} "
              f"{'gain(exp)':>10s} {'gain(mix)':>10s}")
        row = {}
        for name, w in combiners.items():
            e, mx = score(t_nll, t_tok, w)
            row[name] = {"expected": e, "mixture": mx,
                         "gain_expected": pooled - e, "gain_mixture": pooled - mx}
            print(f"  {name:16s} {e:10.4f} {mx:10.4f} "
                  f"{pooled - e:+10.4f} {pooled - mx:+10.4f}")
        row["stacking_weights"] = [round(float(x), 4) for x in stack]
        row["best_single_expert"] = best
        store[run] = {"label": label, "pairs": kk, "pooled": pooled, "combiners": row}

        g, s = row["game"], row["stacking"]
        print(f"  -> game beats fitted stacking by "
              f"{s['expected'] - g['expected']:+.4f} expected, "
              f"{s['mixture'] - g['mixture']:+.4f} mixture")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(store, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
