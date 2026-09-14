"""Theory-equilibrium variants of the reach sweep and the base-model sweep.

Two families, which need different handling:

**Reach (sigma).** Each arm has its own ``sigma_fraction``, and the equilibrium
depends on sigma, so every one needs its own solve. Those come from
``true_theory_positions.py``; this script polishes each to a true fixed point
before writing it.

**Base model.** Every model arm shares the trait space (the encoder is
Qwen3-Embedding-8B regardless of which model is being fine-tuned), N = 14, and
``sigma_fraction`` 0.50 -- so they all share *one* equilibrium, the same one the
7-pair arm already uses. No solve is needed; the position block is lifted
verbatim, which also guarantees the model comparison is not confounded by
differing start points.

Each arm reuses its existing pooled generalist, which is NOT retrained.

    python make_theory_eq_sweep.py            # write everything
    python make_theory_eq_sweep.py --family sigma
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
ARMS = REPO / "configs" / "arms"
EXPS = REPO / "configs" / "experiments"
SEVEN_PAIR = ARMS / "soft_theory_eq_seed0.yaml"
sys.path.insert(0, str(PAPER))

# slug -> source run whose sigma and equilibrium are copied
SIGMA = {
    "theory_eq_sigma005": "seven_axis_sigma005_full",
    "theory_eq_sigma01": "seven_axis_sigma01_full",
    "theory_eq_sigma02": "seven_axis_sigma02_full",
    "theory_eq_sigma03": "seven_axis_sigma03_full",
    "theory_eq_sigma04": "seven_axis_sigma04_full",
    "theory_eq_sigma045": "seven_axis_sigma045_full",
    "theory_eq_sigma055": "seven_axis_sigma055_full",
    "theory_eq_sigma06": "seven_axis_sigma06_full",
    "theory_eq_sigma07": "seven_axis_sigma07_full",
}

# slug -> (label, model include, generalist arm). All share the 7-pair equilibrium.
MODELS = {
    "theory_eq_3b": ("Qwen2.5-3B", "../models/qwen2_5_3b_instruct.yaml",
                     "generalist_replay_3b.yaml"),
    "theory_eq_7b": ("Qwen2.5-7B", "../models/qwen2_5_7b_instruct.yaml",
                     "generalist_replay_7b.yaml"),
    "theory_eq_llama1b": ("Llama-3.2-1B", "../models/llama_3_2_1b_instruct.yaml",
                          "generalist_replay_llama1b.yaml"),
    "theory_eq_gemma3_1b": ("Gemma-3-1B", "../models/gemma_3_1b_it.yaml",
                            "generalist_replay_gemma3_1b.yaml"),
}


def position_block(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    i = next(j for j, l in enumerate(lines) if l.strip() == "init_positions:")
    return lines[i:]


def polished_block(run: str) -> tuple[list[str], float, float, float]:
    """Solve-derived positions for one sigma arm, polished to grad ~ 0."""
    from continue_positions import allocation, continue_loop, pair_positions
    from figures_lib import prompt_coords, true_theory

    tt = true_theory(run)
    if tt is None:
        raise SystemExit(f"no cached solve for {run}; run true_theory_positions.py")
    coords = prompt_coords()
    sigma = float(tt["sigma_absolute"])
    pos = np.asarray(tt["positions"], dtype=float)

    def grad_norm(pp):
        cl = np.repeat(pp, 2, axis=0)
        g = allocation(cl, coords, sigma)
        m = g * (1.0 - g)
        gr = (m @ coords - m.sum(axis=1, keepdims=True) * cl) / sigma ** 2
        return float(np.linalg.norm(gr, axis=1).mean())

    before = grad_norm(pos)
    polished, _steps, used = continue_loop(np.repeat(pos, 2, axis=0), coords,
                                           sigma, 4000, blend=0.5, tol=1e-13)
    pos = pair_positions(polished)
    after = grad_norm(pos)
    if after > 1e-3:
        raise SystemExit(f"{run}: start is not an equilibrium ({after:.3g})")
    print(f"    polish |grad u| {before:.3g} -> {after:.3g} in {used} iters")

    block = ["  init_positions:"]
    for p in range(len(pos)):
        row = ", ".join(f"{v:.10f}" for v in pos[p])
        for c in (2 * p, 2 * p + 1):
            block.append(f"    clone-{c}: [{row}]")
    block += ["", "  sft_merge_groups:"]
    for p in range(len(pos)):
        block += [f"    - train_as: pair-{p}",
                  f"      names: [clone-{2 * p}, clone-{2 * p + 1}]"]
    return block, sigma, float(tt["sigma_fraction"]), after


def write_experiment(slug: str, label: str, generalist: str) -> None:
    (EXPS / f"seven_axis_{slug}.yaml").write_text("\n".join([
        f"# {label} at the theory equilibrium. Seed 0.",
        "#",
        "# One-factor contrast against the corresponding approximate-init arm:",
        "# only the starting positions differ. Same split, same test pool, and",
        f"# the existing generalist ({generalist}) is reused, NOT retrained.",
        "#",
        f"#   python -m infl_ens.pipeline --config configs/experiments/seven_axis_{slug}.yaml",
        f"name: seven_axis_{slug}",
        f"results_dir: results/seven_axis_{slug}",
        f"figures_dir: figures/seven_axis_{slug}",
        "",
        "arms:",
        f"  - name: {slug}",
        f'    label: "{label}"',
        f'    title: "{label}, started at the empirical-B equilibrium"',
        "    role: specialist",
        f"    config: ../arms/soft_{slug}_seed0.yaml",
        "  - name: generalist",
        '    label: "Pooled generalist"',
        '    title: "Pooled generalist (existing replay, not retrained)"',
        "    role: generalist",
        f"    config: ../arms/{generalist}",
        "",
        "stages: [manifest, train, perround, routing, figures]",
        "",
    ]), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=("sigma", "models", "all"), default="all")
    args = ap.parse_args()

    if args.family in ("sigma", "all"):
        print("reach sweep (own equilibrium per sigma):")
        for slug, run in SIGMA.items():
            print(f"  {slug}")
            block, sigma, frac, grad = polished_block(run)
            (ARMS / f"soft_{slug}_seed0.yaml").write_text("\n".join([
                f"# Reach {frac:.2f} at the theory equilibrium. Seed 0.",
                "#",
                f"# One factor moves against {run}/seed0: the starting positions.",
                "# The equilibrium is solved at THIS arm's sigma against the",
                "# empirical prompt cloud, then polished to a fixed point.",
                "#",
                f"#   sigma (absolute)      {sigma:.6f}",
                f"#   grad-norm at start    {grad:.3e}",
                "#   pairs                 7  (14 clones, co-located)",
                "",
                "includes:",
                "  - _closed_loop_base.yaml",
                "",
                f"output_dir: results/seven_axis_{slug}/seed0",
                "",
                f"sigma_fraction: {frac:.2f}",
                "",
                "closed_loop:",
                "  routing_mode: soft",
                "  soft_top_k: 7",
                "  soft_loss: weighted",
                "  init_mode: given",
                "  init_noise: 0.0",
                "  init_positions_source: >-",
                f"    influencer-game equilibrium at sigma {sigma:.6f} under the",
                "    empirical prompt measure",
                "",
            ] + block) + "\n", encoding="utf-8")
            write_experiment(slug, f"Reach {frac:.2f}", "generalist_replay.yaml")

    if args.family in ("models", "all"):
        block = position_block(SEVEN_PAIR)
        src = SEVEN_PAIR.read_text(encoding="utf-8")
        sigma = re.search(r"sigma \(absolute\)\s+([0-9.]+)", src)
        print("base models (all share the 7-pair equilibrium):")
        for slug, (label, include, generalist) in MODELS.items():
            print(f"  {slug}")
            (ARMS / f"soft_{slug}_seed0.yaml").write_text("\n".join([
                f"# {label} at the theory equilibrium. Seed 0.",
                "#",
                "# The trait space is built by the Qwen3-Embedding-8B encoder",
                "# regardless of which model is fine-tuned, and every model arm is",
                "# N = 14 at sigma_fraction 0.50 -- so they all share ONE",
                "# equilibrium, lifted verbatim from soft_theory_eq_seed0.yaml.",
                "# Identical start points mean the model comparison cannot be",
                "# confounded by differing initial geometry.",
                "#",
                f"#   sigma (absolute)      {sigma.group(1) if sigma else '?'}",
                "#   pairs                 7  (14 clones, co-located)",
                "",
                "includes:",
                "  - _closed_loop_base.yaml",
                f"  - {include}",
                "",
                f"output_dir: results/seven_axis_{slug}/seed0",
                "",
                "sigma_mode: stability_fraction",
                "sigma_fraction: 0.5",
                "",
                "closed_loop:",
                "  routing_mode: soft",
                "  soft_top_k: 7",
                "  soft_loss: weighted",
                "  init_mode: given",
                "  init_noise: 0.0",
                "  init_positions_source: >-",
                "    influencer-game equilibrium under the empirical prompt measure;",
                "    identical to seven_axis_theory_eq/seed0",
                "",
            ] + block) + "\n", encoding="utf-8")
            write_experiment(slug, label, generalist)


if __name__ == "__main__":
    main()
