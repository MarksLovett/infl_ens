"""Can an ordinary predictive router beat the game router?

This is the strongest form of the frozen-expert question. Earlier combiners
(global stacking, best-single) could not condition on the prompt, so losing to
the game router only showed that *some* per-prompt signal exists. A linear gate
over trait coordinates conditions on exactly what the game router conditions on
-- the prompt's position in trait space -- and is fitted directly on the
validation expert-NLL matrix, which is strictly more supervision than the game
router ever sees. If it still loses, the geometry is not merely informative, it
is close to the best use of that information.

Everything is fitted on VALIDATION and scored on TEST. The experts are frozen
throughout; only the combination rule changes.

Gates
    game            the learned allocation G
    linear-cls      linear router, classification objective on argmin expert
    linear-soft     linear router, soft targets from the full NLL vector
    linear-reg      linear router, regression on the NLL vector
    stacking        global simplex weights (no per-prompt conditioning)
    uniform, oracle bounds

    python feature_gate.py
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
    fit_linear_router, fit_simplex_stacking,
)
from infl_ens.evaluation.routing_eval import mixture_nll  # noqa: E402

ARMS = [
    (2, "2 pairs, naive", "seven_axis_2pair_sigma05"),
    (2, "2 pairs, equilibrium", "seven_axis_theory_eq_2pair"),
    (7, "7 pairs, naive", "seven_axis_soft_full_pairs"),
    (7, "7 pairs, equilibrium", "seven_axis_theory_eq"),
    (9, "9 pairs, naive", "seven_axis_9pair_sigma05"),
    (9, "9 pairs, equilibrium", "seven_axis_theory_eq_9pair"),
]


def load(run: str, part: str):
    d = REPO / "results" / run / "seed0" / f"{part}_arrays"
    try:
        return dict(nll=np.load(d / "merge_nll.npy"),
                    w=np.load(d / "g_merge.npy").T,
                    tok=np.load(d / "token_counts.npy"),
                    x=np.load(d / "coords.npy"))
    except FileNotFoundError:
        return None


def score(nll, tok, w):
    return float((w * nll).sum(axis=1).mean()), float(mixture_nll(nll, tok, w).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(PAPER / "data" / "feature_gate.json"))
    args = ap.parse_args()
    store = {}

    for k, label, run in ARMS:
        v, t = load(run, "val"), load(run, "test")
        if v is None or t is None:
            print(f"{label}: missing {'val' if v is None else 'test'} arrays")
            continue
        pooled = json.loads((REPO / "results" / run / "seed0" /
                             "routing_ensemble_diagnostics.json"
                             ).read_text())["flat"]["pooled_nll"]
        m, kk = t["nll"].shape
        gates = {"game": t["w"]}

        # The router exposes weights(), not predict(). fit_best_linear_router
        # needs benchmark labels, which are not persisted, so sweep temperature
        # by hand and select on a held-out slice of validation instead.
        cut = len(v["x"]) // 2
        for obj in ("classification", "soft", "regression"):
            best_w, best_v = None, None
            for temp in (0.03, 0.1, 0.3, 1.0):
                r = fit_linear_router(v["x"][:cut], v["nll"][:cut],
                                      objective=obj, temperature=temp, seed=0)
                held = float((r.weights(v["x"][cut:]) * v["nll"][cut:]).sum(1).mean())
                if best_v is None or held < best_v:
                    best_v, best_w = held, temp
            r = fit_linear_router(v["x"], v["nll"], objective=obj,
                                  temperature=best_w, seed=0)
            gates[f"linear-{obj[:3]}"] = r.weights(t["x"])

        gates["stacking"] = np.broadcast_to(
            fit_simplex_stacking(v["nll"], v["tok"]), (m, kk)).copy()
        gates["uniform"] = np.full((m, kk), 1.0 / kk)
        orc = np.zeros((m, kk)); orc[np.arange(m), t["nll"].argmin(axis=1)] = 1.0
        gates["oracle"] = orc

        print(f"\n{label}   (val {v['nll'].shape}, test {t['nll'].shape})")
        print(f"  {'gate':14s} {'gain(exp)':>10s} {'gain(mix)':>10s}")
        row = {}
        for name, w in gates.items():
            e, mx = score(t["nll"], t["tok"], w)
            row[name] = {"gain_expected": pooled - e, "gain_mixture": pooled - mx}
            print(f"  {name:14s} {pooled - e:+10.4f} {pooled - mx:+10.4f}")
        store[run] = {"label": label, "pairs": k, "pooled": pooled, "gates": row}

        best_learned = max((n for n in row if n.startswith("linear")),
                           key=lambda n: row[n]["gain_mixture"], default=None)
        if best_learned:
            d = row["game"]["gain_mixture"] - row[best_learned]["gain_mixture"]
            print(f"  -> game beats best learned gate ({best_learned}) by "
                  f"{d:+.4f} mixture")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(store, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
