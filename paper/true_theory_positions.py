"""Re-solve the theory initialisation against the empirical resource measure.

Why this exists
---------------
``_kde_on_grid`` (``src/infl_ens/data/trait_space.py``) subtracts a per-grid-point
max for log-space stability and never multiplies it back, so the weights it
returns are not a KDE -- a single prompt yields a *uniform* vector regardless of
where it sits. The shipped ``B(b)`` is therefore close to a uniform prior over
the ``{0, 0.5, 1}^7`` lattice (per-axis variance 0.159, anisotropy 1.29) rather
than a model of the prompt cloud (0.084, anisotropy 3.10).

Every trained run in this project used that approximate ``B``. Nothing here
re-runs or invalidates them: training always drew its position updates from real
batches, so only the *starting point* and the σ₀\\* normalisation were affected.
What this script adds is the comparison that was missing -- where the solve
would have put the pairs if it had seen the real distribution.

The solve is CPU-only and takes a few minutes per arm. Results are cached to
``paper/data/true_theory_positions.json`` so the figures never pay for it.

    python true_theory_positions.py            # solve anything not yet cached
    python true_theory_positions.py --force    # re-solve everything
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(PAPER))

from infl_ens.data.trait_space import TraitSpace  # noqa: E402
from infl_ens.training.agent_init import (  # noqa: E402
    init_agents_theory_gradient_paired,
)
from infl_ens.utils.resource import gaussian_stability_threshold  # noqa: E402

# The shipped inner loop costs ~445 ms/step at N=14 against 28k support points
# (~2 h per arm): it applies an isotropic Sigma with a 7x7 LAPACK solve per
# support point, builds an (N, K, L) temporary to contract away, and computes
# the allocation twice per step so it can fill a diagnostic the caller discards.
# fast_solve does the same arithmetic without those three, and is checked
# against the reference to machine precision before anything is solved. src/ is
# untouched, so no experiment changes.
import infl_ens.training.pool_dynamics as _pool  # noqa: E402
from fast_solve import (  # noqa: E402
    assert_matches_reference, fast_train_router_positions,
)

CACHE = REPO / "data" / "trait_space_cache" / "3b42c68a8dd334c5"
OUT = PAPER / "data" / "true_theory_positions.json"

# The arms the radars draw. Everything is split 0.
RUNS = [
    "seven_axis_sigma005_full",
    "seven_axis_soft_full_pairs",
    "seven_axis_sigma07_full",
    "seven_axis_2pair_sigma05",
    "seven_axis_5pair_sigma05",
]


def empirical_space(coords: np.ndarray, labels) -> TraitSpace:
    """The prompt cloud itself as the resource measure, uniformly weighted.

    The lattice is only a quadrature rule for :math:`\\int B(b) G(b)\\,db`; the
    prompts *are* the distribution, so using them as the support removes an
    approximation rather than adding one. Corollary 8 is stated for a general
    ``Sigma_B``, so the threshold formula is unaffected.
    """
    def _no_project(_queries):
        raise RuntimeError("the offline theory solve never projects queries")

    return TraitSpace(
        grid=np.ascontiguousarray(coords, dtype=float),
        weights=np.full(len(coords), 1.0 / len(coords)),
        project=_no_project,
        axis_labels=tuple(labels) if labels else None,
    )


def solve_one(run: str, coords: np.ndarray, lattice, labels) -> dict:
    """Theory positions for one arm under the empirical measure.

    The absolute sigma is held at whatever the run actually used -- fraction
    times the *lattice* threshold -- because the experiments are not being
    changed. Only the measure the solve optimises against differs.
    """
    cfg_path = REPO / "results" / run / "seed0" / "resolved_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cl = cfg.get("closed_loop", {})
    # ``agents``, ``sigma_fraction`` and ``seed`` live at the top level; the
    # solve takes the whole config, not the closed_loop block.
    n_agents = len(cfg["agents"])
    seed = int(cfg.get("seed", 0))

    grid_l, w_l = lattice
    s0_lattice = gaussian_stability_threshold(n_agents, grid_l, w_l)
    s0_empirical = gaussian_stability_threshold(
        n_agents, coords, np.full(len(coords), 1.0 / len(coords)))
    frac = float(cfg.get("sigma_fraction", 0.8))
    sigma_abs = frac * max(s0_lattice, 1e-3)

    space = empirical_space(coords, labels)
    t0 = time.perf_counter()
    _agents, meta = init_agents_theory_gradient_paired(
        cfg, space, sigma=sigma_abs, seed=seed,
        init_noise=float(cl.get("init_noise", 0.0) or 0.0),
        theory_cfg=cl.get("theory_gradient"),
    )
    elapsed = time.perf_counter() - t0

    pair_pos = meta.get("pair_positions") or {}
    return {
        "run": run, "seed": "seed0", "n_agents": n_agents,
        "sigma_fraction": frac,
        "sigma_absolute": sigma_abs,
        "sigma0_lattice": float(s0_lattice),
        "sigma0_empirical": float(s0_empirical),
        "pair_positions": {k: list(map(float, v)) for k, v in pair_pos.items()},
        "pair_dominant_axis": meta.get("pair_dominant_axis", {}),
        "converged": bool(meta.get("converged", False)),
        "n_steps": meta.get("n_steps"),
        "support": int(len(coords)),
        "solve_seconds": round(elapsed, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--support", type=int, default=0,
                    help="Subsample the prompt cloud to this many points (0 = all).")
    args = ap.parse_args()

    sys.path.insert(0, str(PAPER))
    from figures_lib import prompt_coords

    delta = assert_matches_reference()
    print(f"fast solver verified against the reference: max |diff| = {delta:.2e}")
    _pool.train_router_positions = fast_train_router_positions

    z = np.load(CACHE / "arrays.npz")
    lattice = (z["grid"], z["weights"])
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    labels = [a["name"] for a in manifest.get("axes", [])] or None

    coords = prompt_coords()
    if args.support and args.support < len(coords):
        idx = np.random.default_rng(0).choice(len(coords), args.support, replace=False)
        coords = coords[np.sort(idx)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    store = json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else {}

    for run in args.runs:
        if run in store and not args.force:
            print(f"  {run}: cached ({store[run]['solve_seconds']}s)")
            continue
        print(f"  {run}: solving against {len(coords)} prompts ...", flush=True)
        store[run] = solve_one(run, coords, lattice, labels)
        print(f"      {len(store[run]['pair_positions'])} pairs,"
              f" sigma={store[run]['sigma_absolute']:.4f},"
              f" {store[run]['solve_seconds']}s")
        OUT.write_text(json.dumps(store, indent=2), encoding="utf-8")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
