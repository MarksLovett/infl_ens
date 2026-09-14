"""Write the arm config that starts training at the theory equilibrium.

The question this sets up: the closed loop normally starts from a solve against
an approximate resource measure and then moves very little (see
``continue_positions.py`` -- twelve rounds ends well short of a fixed point, and
running the update to convergence barely changes the positions). So what happens
if training *starts* at an actual equilibrium of the game under the real prompt
distribution?

Everything else is held at the sigma-0.50 seed-0 arm's settings: same split, same
absolute sigma, same twelve rounds, same soft dense routing. Only the starting
positions differ, so the contrast against ``seven_axis_soft_full_pairs/seed0`` is
a clean one-factor comparison, and the existing seed-0 pooled generalist stays
data-matched and is not retrained.

Clones are placed co-located in pairs, which is a fixed point of the dynamics
(identical positions give identical allocations, hence identical updates), and
``sft_merge_groups`` is pinned to match that pairing explicitly rather than being
inferred from a solve that no longer runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
sys.path.insert(0, str(PAPER))

# Each entry: the arm whose settings and equilibrium are copied, and the slug the
# new arm/experiment are written under.
SOURCES = {
    "theory_eq": "seven_axis_soft_full_pairs",          # 7 pairs, the standard arm
    "theory_eq_2pair": "seven_axis_2pair_sigma05",
    "theory_eq_3pair": "seven_axis_3pair_sigma05",
    "theory_eq_4pair": "seven_axis_4pair_sigma05",
    "theory_eq_5pair": "seven_axis_5pair_sigma05",
    "theory_eq_6pair": "seven_axis_6pair_sigma05",
    "theory_eq_8pair": "seven_axis_8pair_sigma05",
    "theory_eq_9pair": "seven_axis_9pair_sigma05",
}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="theory_eq", choices=sorted(SOURCES),
                    help="which arm to generate")
    args = ap.parse_args()
    slug = args.name
    SOURCE_RUN = SOURCES[slug]
    ARM = REPO / "configs" / "arms" / f"soft_{slug}_seed0.yaml"
    EXPERIMENT = REPO / "configs" / "experiments" / f"seven_axis_{slug}.yaml"
    RESULTS = f"results/seven_axis_{slug}"
    from continue_positions import allocation, continue_loop, pair_positions
    from figures_lib import prompt_coords, true_theory

    tt = true_theory(SOURCE_RUN)
    if tt is None:
        raise SystemExit("run true_theory_positions.py first")
    pos = np.asarray(tt["positions"], dtype=float)      # (P, L)
    n_pairs, L = pos.shape
    sigma = float(tt["sigma_absolute"])

    # The gradient-ascent solve stops at its 8000-step cap without meeting tol,
    # so its endpoint is near an equilibrium but not on one. Polish it with the
    # position-update fixed-point iteration, which has the SAME fixed points
    # (x_hat = x is algebraically grad u = 0) but converges in ~100 iterations
    # instead of tens of thousands of small gradient steps. The starting
    # position has to actually be an equilibrium or the experiment is not
    # testing what it claims to.
    coords = prompt_coords()

    def grad_norm(pair_pos):
        clones = np.repeat(pair_pos, 2, axis=0)
        g = allocation(clones, coords, sigma)
        m = g * (1.0 - g)
        gr = (m @ coords - m.sum(axis=1, keepdims=True) * clones) / sigma ** 2
        return float(np.linalg.norm(gr, axis=1).mean())

    before = grad_norm(pos)
    clones = np.repeat(pos, 2, axis=0)
    polished, steps, used = continue_loop(clones, coords, sigma, 4000,
                                          blend=0.5, tol=1e-13)
    pos = pair_positions(polished)
    after = grad_norm(pos)
    moved = float(np.linalg.norm(pair_positions(clones) - pos, axis=1).mean())
    print(f"polish: |grad u| {before:.4g} -> {after:.4g} in {used} iterations "
          f"(moved {moved:.4f}, last step {steps[-1]:.2e})")
    if after > 1e-3:
        raise SystemExit(f"polished start is still not an equilibrium: {after:.4g}")
    frac = float(tt["sigma_fraction"])
    tt = dict(tt, converged=True, polish_iterations=used,
              polish_grad_before=before, polish_grad_after=after)

    lines = [
        "# Training from the theory equilibrium, seed 0.",
        "#",
        f"# One factor moves against {SOURCE_RUN}/seed0: where the",
        "# clones start. Everything else -- split, absolute sigma, twelve rounds,",
        "# fully dense soft routing, loss weighting -- is identical, so the two",
        "# arms share a test pool and the existing seed-0 generalist is still",
        "# data-matched and is NOT retrained.",
        "#",
        "# The positions below are an equilibrium of the influencer game solved",
        f"# against the empirical prompt cloud ({tt['support']} prompts, uniform",
        "# weight) rather than against the lattice KDE. That solve is CPU-only;",
        "# see paper/true_theory_positions.py. Pinning the result here keeps it",
        "# in the resolved config instead of being recomputed per run.",
        "#",
        f"#   sigma (absolute)      {sigma:.6f}",
        f"#   sigma_0* lattice      {tt['sigma0_lattice']:.6f}   (what the runs quote against)",
        f"#   sigma_0* empirical    {tt['sigma0_empirical']:.6f}",
        f"#   pairs                 {n_pairs}  ({2 * n_pairs} clones, co-located)",
        f"#   grad-norm at start    {tt['polish_grad_after']:.3e}" f"   (polished from {tt['polish_grad_before']:.3e} in {tt['polish_iterations']} fixed-point iterations)",
        "",
        "includes:",
        "  - _closed_loop_base.yaml",
        "",
        f"output_dir: {RESULTS}/seed0",
        "",
        f"sigma_fraction: {frac:.2f}",
        "",
        "# An explicit list replaces the base's `agents: {pairs_from_axes: true}`",
        "# mapping, which always expands to N = 2L = 14 regardless of pair count.",
        f"# This arm needs N = {2 * n_pairs}, one clone pair per routing pair.",
        "agents:",
        "closed_loop:",
        "  routing_mode: soft",
        f"  soft_top_k: {n_pairs}                     # == number of pairs: fully dense",
        "  soft_loss: weighted",
        "  init_mode: given",
        "  init_noise: 0.0",
        "  init_positions_source: >-",
        "    influencer-game equilibrium under the empirical prompt measure,",
        f"    solved offline from {SOURCE_RUN}/seed0 settings",
        "",
        "  # Clone 2i and 2i+1 share pair i's position. Co-location is a fixed",
        "  # point of the update, so the pairs stay merged without being forced.",
        "  init_positions:",
    ]
    agent_lines = [f"  - name: clone-{c}" for c in range(2 * n_pairs)]
    at = lines.index("agents:") + 1
    lines[at:at] = agent_lines + [""]
    for p in range(n_pairs):
        row = ", ".join(f"{v:.10f}" for v in pos[p])
        for c in (2 * p, 2 * p + 1):
            lines.append(f"    clone-{c}: [{row}]")

    lines += [
        "",
        "  # Pinned to match the co-location above, since no solve runs to infer it.",
        "  sft_merge_groups:",
    ]
    for p in range(n_pairs):
        lines.append(f"    - train_as: pair-{p}")
        lines.append(f"      names: [clone-{2 * p}, clone-{2 * p + 1}]")
    lines.append("")

    ARM.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {ARM}")

    EXPERIMENT.write_text("\n".join([
        "# Does starting at the theory equilibrium change the outcome?",
        "#",
        f"# The specialist arm is identical to {SOURCE_RUN}/seed0 except for",
        "# its starting positions, which are an equilibrium of the game under the",
        "# empirical prompt measure. Compare its routing NLL against:",
        "#   - the pooled generalist (role: generalist below, NOT retrained)",
        f"#   - {SOURCE_RUN}/seed0, the same arm started from the",
        "#     approximate lattice solve",
        "# All three share the seed-0 split and the 5,671-prompt test pool.",
        "#",
        f"#   python -m infl_ens.pipeline --config configs/experiments/seven_axis_{slug}.yaml",
        f"name: seven_axis_{slug}",
        f"results_dir: {RESULTS}",
        f"figures_dir: figures/seven_axis_{slug}",
        "",
        "arms:",
        f"  - name: {slug}",
        '    label: "Theory equilibrium start"',
        f'    title: "Soft dense routing, {n_pairs} pairs, started at the empirical-B equilibrium"',
        "    role: specialist",
        f"    config: ../arms/soft_{slug}_seed0.yaml",
        "  - name: generalist",
        '    label: "Pooled generalist"',
        '    title: "Pooled generalist (existing seed-0 replay, not retrained)"',
        "    role: generalist",
        "    config: ../arms/generalist_replay.yaml",
        "",
        "stages: [manifest, train, perround, routing, figures]",
        "",
    ]), encoding="utf-8")
    print(f"wrote {EXPERIMENT}")


if __name__ == "__main__":
    main()
