"""Same-expert routing nulls: does the learned geometry do any work?

Holds the expert family completely fixed and swaps only the rule that combines
them. If a null matches the game router, then seven adapters are useful under
almost any combination rule and the geometry is not what is buying the gain.

Every policy here is computed offline from the saved ``(M, K)`` per-example NLL
matrix and the ``(K, M)`` router weights -- no GPU, no retraining, no re-scoring.
That is the point: these are the cheapest possible checks on the central claim.

Policies
    game            the learned allocation G (what the paper reports)
    shuffled        G with expert identities permuted -- same weight
                    *distribution*, wrong assignment. Averaged over permutations.
    uniform-mix     every expert weighted 1/K
    random-hard     one expert per prompt, drawn uniformly. Averaged over draws.
    game-hard       argmax of G, one expert per prompt
    oracle          per-prompt best expert (ceiling, not achievable)

Each is scored two ways, because the distinction matters and the paper's
headline uses the first:
    expected   sum_p w_p NLL_p          -- committing to one expert
    mixture    -1/n log sum_p w_p P_p   -- an actual ensemble

    python routing_nulls.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
sys.path.insert(0, str(REPO / "src"))

from infl_ens.evaluation.routing_eval import mixture_nll  # noqa: E402

# Both initialisations at every capacity we have matrices for, so the nulls can
# be read as a curve in K and as a naive-vs-equilibrium contrast at each K.
ARMS = [(k, f"{k} pairs, {tag}", run) for k, tag, run in [
    (2, "naive", "seven_axis_2pair_sigma05"),
    (2, "equilibrium", "seven_axis_theory_eq_2pair"),
    (3, "naive", "seven_axis_3pair_sigma05"),
    (3, "equilibrium", "seven_axis_theory_eq_3pair"),
    (4, "naive", "seven_axis_4pair_sigma05"),
    (4, "equilibrium", "seven_axis_theory_eq_4pair"),
    (5, "naive", "seven_axis_5pair_sigma05"),
    (5, "equilibrium", "seven_axis_theory_eq_5pair"),
    (6, "naive", "seven_axis_6pair_sigma05"),
    (6, "equilibrium", "seven_axis_theory_eq_6pair"),
    (7, "naive", "seven_axis_soft_full_pairs"),
    (7, "equilibrium", "seven_axis_theory_eq"),
    (9, "naive", "seven_axis_9pair_sigma05"),
    (9, "equilibrium", "seven_axis_theory_eq_9pair"),
]]


def load(run: str):
    d = REPO / "results" / run / "seed0"
    try:
        nll = np.load(d / "merge_nll.npy")          # (M, K)
        g = np.load(d / "g_merge.npy")              # (K, M)
        ntok = np.load(d / "token_counts.npy")      # (M,)
    except FileNotFoundError:
        return None
    diag = json.loads((d / "routing_ensemble_diagnostics.json").read_text())["flat"]
    return nll, g.T, ntok, float(diag["pooled_nll"])


def one_hot(idx: np.ndarray, k: int) -> np.ndarray:
    w = np.zeros((len(idx), k))
    w[np.arange(len(idx)), idx] = 1.0
    return w


def policies(nll: np.ndarray, w: np.ndarray, rng: np.random.Generator):
    """Yield (name, weights) for each combination rule. Weights are (M, K)."""
    m, k = nll.shape
    yield "game", w
    yield "uniform-mix", np.full((m, k), 1.0 / k)
    yield "game-hard", one_hot(w.argmax(axis=1), k)
    yield "oracle", one_hot(nll.argmin(axis=1), k)

    # Shuffled identities: keep each row's weight *multiset*, permute which
    # expert gets which weight. Average over all permutations for small K,
    # otherwise a random sample -- one draw would be noise.
    perms = list(itertools.permutations(range(k)))
    if len(perms) > 200:
        perms = [tuple(rng.permutation(k)) for _ in range(200)]
    perms = [p for p in perms if list(p) != list(range(k))]
    yield "shuffled", ("avg", [w[:, list(p)] for p in perms])

    # Random hard routing: one uniformly-drawn expert per prompt.
    yield "random-hard", ("avg", [one_hot(rng.integers(0, k, m), k)
                                  for _ in range(200)])


def score(nll, ntok, w):
    """(expected, mixture) mean NLL for one weight matrix."""
    return float((w * nll).sum(axis=1).mean()), float(mixture_nll(nll, ntok, w).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(PAPER / "data" / "routing_nulls.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    store = {}

    for kk, label, run in ARMS:
        got = load(run)
        if got is None:
            print(f"{label}: no saved matrices")
            continue
        nll, w, ntok, pooled = got
        print(f"\n{label}   ({run}, {nll.shape[0]} prompts x {nll.shape[1]} experts)")
        print(f"  {'policy':14s} {'expected':>10s} {'mixture':>10s}   "
              f"{'gain(exp)':>10s} {'gain(mix)':>10s}")
        row = {}
        for name, spec in policies(nll, w, rng):
            if isinstance(spec, tuple):
                e, mx = np.mean([score(nll, ntok, x) for x in spec[1]], axis=0)
            else:
                e, mx = score(nll, ntok, spec)
            row[name] = {"expected": e, "mixture": mx,
                         "gain_expected": pooled - e, "gain_mixture": pooled - mx}
            print(f"  {name:14s} {e:10.4f} {mx:10.4f}   "
                  f"{pooled - e:+10.4f} {pooled - mx:+10.4f}")
        store[run] = {"label": label, "pairs": kk, "pooled": pooled, "policies": row}

        g, s = row["game"], row["shuffled"]
        print(f"  -> geometry is worth {s['expected'] - g['expected']:+.4f} expected, "
              f"{s['mixture'] - g['mixture']:+.4f} mixture (vs shuffled identities)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(store, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
