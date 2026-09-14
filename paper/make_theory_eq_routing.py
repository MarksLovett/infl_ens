"""Theory-equilibrium variants of the routing ladder.

The five routing designs (dense soft, top-3 soft, top-3 unit, sampled top-3,
hard k=1) differ only in how prompts are assigned and weighted during training.
None of that enters the theory solve, which is gradient ascent on
:math:`u_i = \\sum_b B(b) G_i(b)` over clone positions. All five arms are N = 14
at ``sigma_fraction`` 0.50, so they share one equilibrium -- the same one
``soft_theory_eq_seed0.yaml`` already pins.

This lifts that exact position block rather than re-solving, so every arm in the
ladder starts from byte-identical coordinates and the only thing that moves is
the routing rule.

One asymmetry worth recording: under hard routing the centroid mass is
:math:`(1-G_i)` on the prompts actually routed, because the sampling step
supplies the other :math:`G_i` factor. In expectation that is the same
:math:`G_i(1-G_i)` estimator soft routing computes in closed form, so the fixed
point is unchanged -- but it is a higher-variance estimator, so the hard arm
should wander further from the equilibrium than the soft ones. That drift is a
measurement, not a defect.

    python make_theory_eq_routing.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
ARMS = REPO / "configs" / "arms"
SOURCE = ARMS / "soft_theory_eq_seed0.yaml"

# slug -> (human label, closed_loop overrides carrying the routing design)
LADDER = {
    "theory_eq_soft_topk3": (
        "Soft top-3, share-weighted",
        ["  routing_mode: soft", "  soft_top_k: 3", "  soft_loss: weighted"],
    ),
    "theory_eq_topk3_unit": (
        "Soft top-3, unit loss",
        ["  routing_mode: soft", "  soft_top_k: 3", "  soft_loss: unit"],
    ),
    "theory_eq_hard_topk3": (
        "Sampled top-3, unit loss",
        ["  routing_mode: soft", "  soft_top_k: 3",
         "  soft_select: sample", "  soft_loss: unit"],
    ),
    "theory_eq_hard_matched": (
        "Hard k=1 (one categorical draw)",
        ["  routing_mode: hard"],
    ),
}


def position_block(text: str) -> list[str]:
    """The init_positions + sft_merge_groups lines from the 7-pair arm, verbatim."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "init_positions:")
    return lines[start:]


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"missing {SOURCE}; run make_theory_eq_arm.py first")
    src = SOURCE.read_text(encoding="utf-8")
    block = position_block(src)
    sigma = re.search(r"sigma \(absolute\)\s+([0-9.]+)", src)
    grad = re.search(r"grad-norm at start\s+([0-9.e+-]+)", src)
    frac = re.search(r"^sigma_fraction: ([0-9.]+)", src, re.M)

    for slug, (label, overrides) in LADDER.items():
        arm = ARMS / f"soft_{slug}_seed0.yaml"
        lines = [
            f"# Routing ladder at the theory equilibrium: {label}. Seed 0.",
            "#",
            "# Starts from the SAME equilibrium as soft_theory_eq_seed0.yaml -- the",
            "# position block below is lifted from it verbatim, not re-solved. The",
            "# theory solve is gradient ascent on the allocation and never consults",
            "# the routing rule, and all five ladder arms are N = 14 at the same",
            "# sigma, so one equilibrium serves them all. Only the routing design",
            "# moves, which is what makes this a one-factor comparison against",
            f"# the corresponding arm of seven_axis_3arm.",
            "#",
            f"#   sigma (absolute)      {sigma.group(1) if sigma else '?'}",
            f"#   grad-norm at start    {grad.group(1) if grad else '?'}",
            "#   pairs                 7  (14 clones, co-located)",
        ]
        if "hard" in slug:
            lines += [
                "#",
                "# NOTE: hard routing's centroid uses (1-G_i) on routed prompts; the",
                "# sampling step supplies the other G_i factor, so in expectation it",
                "# is the same estimator and the equilibrium is unchanged. It is",
                "# higher variance, so expect more drift from the start than the",
                "# soft arms showed (0.013 mean L2 over twelve rounds).",
            ]
        lines += [
            "",
            "includes:",
            "  - _closed_loop_base.yaml",
            "",
            f"output_dir: results/seven_axis_{slug}/seed0",
            "",
            f"sigma_fraction: {frac.group(1) if frac else '0.50'}",
            "",
            "closed_loop:",
        ] + overrides + [
            "  init_mode: given",
            "  init_noise: 0.0",
            "  init_positions_source: >-",
            "    influencer-game equilibrium under the empirical prompt measure;",
            "    identical to seven_axis_theory_eq/seed0",
            "",
        ] + block
        arm.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {arm.name}")

    exp = REPO / "configs" / "experiments" / "seven_axis_theory_eq_routing.yaml"
    rows = []
    for slug, (label, _o) in LADDER.items():
        rows += [
            f"  - name: {slug}",
            f'    label: "{label}"',
            f'    title: "{label}, started at the empirical-B equilibrium"',
            "    role: specialist",
            f"    config: ../arms/soft_{slug}_seed0.yaml",
        ]
    exp.write_text("\n".join([
        "# The routing ladder, started at the theory equilibrium instead of the",
        "# approximate lattice solve.",
        "#",
        "# Compare each arm against its counterpart in seven_axis_3arm (same",
        "# routing design, approximate init). The dense soft arm is already done",
        "# as seven_axis_theory_eq, so it is not repeated here.",
        "#",
        "# All four share one equilibrium and one test pool, and reuse the",
        "# existing seed-0 generalist, which is NOT retrained.",
        "#",
        "#   python -m infl_ens.pipeline --config configs/experiments/seven_axis_theory_eq_routing.yaml",
        "name: seven_axis_theory_eq_routing",
        "results_dir: results/seven_axis_theory_eq_routing",
        "figures_dir: figures/seven_axis_theory_eq_routing",
        "",
        "arms:",
    ] + rows + [
        "  - name: generalist",
        '    label: "Pooled generalist"',
        '    title: "Pooled generalist (existing seed-0 replay, not retrained)"',
        "    role: generalist",
        "    config: ../arms/generalist_replay.yaml",
        "",
        "stages: [manifest, train, perround, routing, figures]",
        "",
    ]), encoding="utf-8")
    print(f"  wrote {exp.name}")


if __name__ == "__main__":
    main()
