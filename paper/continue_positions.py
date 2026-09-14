"""Would more rounds have carried the pairs to the theory equilibrium?

The closed loop's position update never looks at an adapter. For soft routing
with ``routing_weight='G'`` it is

.. math::

    \\hat x_i \\;=\\; \\frac{\\sum_m G_i(b_m)\\,(1-G_i(b_m))\\, b_m}
                          {\\sum_m G_i(b_m)\\,(1-G_i(b_m))},
    \\qquad
    x_i \\leftarrow (1-\\beta)\\,x_i + \\beta\\,\\hat x_i ,

which depends only on the prompt coordinates and on :math:`\\sigma`. So the
entire position trajectory can be continued on CPU from where training stopped,
with no training, no GPU and no adapters.

That makes a sharp test available. The update's fixed point, :math:`\\hat x_i =
x_i`, is *algebraically the same condition* as :math:`\\nabla_{x_i} u_i = 0`
under isotropic :math:`\\Sigma`, since

.. math::

    \\nabla_{x_i} u_i \\;=\\; \\sigma^{-2} \\sum_m G_i (1-G_i)\\,(b_m - x_i).

So continuing the loop against the empirical prompt cloud should converge to an
equilibrium of the game *under the empirical measure* -- the "empirical B"
solve -- and not to the approximate initialisation the runs started from. If the
twelve trained rounds simply ran out of road, this shows where they were headed.

    python continue_positions.py --rounds 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(PAPER))

OUT = PAPER / "data" / "position_continuation.json"

RUNS = [
    ("reach 0.05", "seven_axis_sigma005_full"),
    ("reach 0.50", "seven_axis_soft_full_pairs"),
    ("reach 0.70", "seven_axis_sigma07_full"),
    ("2 pairs", "seven_axis_2pair_sigma05"),
    ("5 pairs", "seven_axis_5pair_sigma05"),
]


def allocation(pos: np.ndarray, coords: np.ndarray, sigma: float) -> np.ndarray:
    """Clone-level ``G``, shape ``(N, M)``. Softmax over agents, as shipped."""
    pos_sq = np.einsum("nl,nl->n", pos, pos)
    c_sq = np.einsum("ml,ml->m", coords, coords)
    d2 = pos_sq[:, None] - 2.0 * (pos @ coords.T) + c_sq[None, :]
    np.maximum(d2, 0.0, out=d2)
    log_f = -0.5 * d2 / (sigma * sigma)
    log_f -= log_f.max(axis=0, keepdims=True)
    f = np.exp(log_f)
    return f / f.sum(axis=0, keepdims=True)


def continue_loop(pos: np.ndarray, coords: np.ndarray, sigma: float,
                  rounds: int, blend: float = 0.5,
                  tol: float = 1e-10) -> tuple[np.ndarray, list[float], int]:
    """Iterate the theory-matched position update with no training.

    Uses the whole partition each round rather than a sampled batch, i.e. the
    expectation of the update the loop actually applies. That removes the batch
    noise the trained trajectory carries and isolates the fixed point.
    """
    pos = pos.copy()
    steps: list[float] = []
    used = 0
    for used in range(1, rounds + 1):
        g = allocation(pos, coords, sigma)
        mass = g * (1.0 - g)                          # (N, M), not renormalised
        denom = mass.sum(axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-300)
        target = (mass @ coords) / denom
        new = np.clip((1.0 - blend) * pos + blend * target, 0.0, 1.0)
        step = float(np.abs(new - pos).max())
        pos = new
        steps.append(step)
        if step < tol:
            break
    return pos, steps, used


def pair_positions(pos: np.ndarray) -> np.ndarray:
    """Collapse co-located clone pairs to one row each (clones 2i, 2i+1)."""
    return 0.5 * (pos[0::2] + pos[1::2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--support", type=int, default=0)
    args = ap.parse_args()

    from figures_lib import match_pairs, prompt_coords, true_theory

    coords = prompt_coords()
    if args.support and args.support < len(coords):
        idx = np.random.default_rng(0).choice(len(coords), args.support, replace=False)
        coords = coords[np.sort(idx)]

    store: dict[str, dict] = {}
    for label, run in RUNS:
        hist_path = REPO / "results" / run / "seed0" / "history.json"
        cfg_path = REPO / "results" / run / "seed0" / "resolved_config.yaml"
        if not hist_path.is_file():
            print(f"  {label}: no history")
            continue
        records = json.loads(hist_path.read_text(encoding="utf-8"))
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        names = [a["name"] for a in cfg["agents"]]
        blend = float(cfg["closed_loop"].get("blend", 0.5))

        trained = np.array([records[-1]["positions"][n] for n in names], dtype=float)
        tt = true_theory(run)
        if tt is None:
            print(f"  {label}: no empirical solve cached")
            continue
        sigma = float(tt["sigma_absolute"])

        final, steps, used = continue_loop(trained, coords, sigma,
                                           args.rounds, blend=blend)

        trained_p = pair_positions(trained)
        final_p = pair_positions(final)
        emp = tt["positions"]
        theory_init = np.array(
            [records[0]["theory_init"]["pair_positions"][k]
             for k in sorted(records[0]["theory_init"]["pair_positions"],
                             key=lambda s: s)], dtype=float)

        def dist(a, b):
            perm, _ = match_pairs(a, b)
            return float(np.linalg.norm(a - b[perm], axis=1).mean())

        row = {
            "run": run, "label": label, "sigma": sigma, "blend": blend,
            "rounds_used": used, "converged": steps[-1] < 1e-10,
            "last_step": steps[-1],
            "moved_from_trained": float(
                np.linalg.norm(final_p - trained_p, axis=1).mean()),
            "trained_to_empirical": dist(trained_p, emp),
            "continued_to_empirical": dist(final_p, emp),
            "trained_to_approx": dist(trained_p, theory_init),
            "continued_to_approx": dist(final_p, theory_init),
        }
        store[run] = row
        print(f"  {label:12s} rounds={used:4d} conv={row['converged']}"
              f"  trained->emp {row['trained_to_empirical']:.3f}"
              f" -> continued->emp {row['continued_to_empirical']:.3f}"
              f"   (moved {row['moved_from_trained']:.3f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(store, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
