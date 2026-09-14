"""Second data split for the reach sweep at the theory equilibrium.

Why this split and not another experiment: the σ stability result -- equilibrium
arms spanning 0.0027 SD across reach against the naive arms' 0.0089 -- is
exactly the size of seed-0's split noise, measured at −0.016 to +0.013 against
replicate means. It is the claim most in need of a second split, and it is also
the cheapest to replicate, because the ten *naive* dseed1 arms already exist.
Only the equilibrium side has to be trained.

What carries over and what does not:

- **Positions are unchanged.** The equilibrium is solved against the full
  28,355-prompt trait cloud, which does not depend on how that cloud is
  partitioned. Reusing the seed-0 equilibrium keeps exactly one factor moving
  between the two splits.
- **The manifest, generalist and test pool all change**, which is the point.
  ``data/splits/seven_axis_soft_dseed1.json`` and
  ``results/seven_axis_generalist_dseed1/seed0`` already exist from the
  replicate sweep, so no baseline work is needed.
- **Absolute sigma is unchanged**: fraction times the lattice threshold, which
  depends on the trait space rather than the split.

    python make_theory_eq_dseed1.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
ARMS = REPO / "configs" / "arms"
EXPS = REPO / "configs" / "experiments"
sys.path.insert(0, str(PAPER))

# seed-0 arm to lift positions from  ->  new dseed1 slug
SWEEP = {
    "soft_theory_eq_sigma005_seed0": "theory_eq_sigma005_d1",
    "soft_theory_eq_sigma01_seed0": "theory_eq_sigma01_d1",
    "soft_theory_eq_sigma02_seed0": "theory_eq_sigma02_d1",
    "soft_theory_eq_sigma03_seed0": "theory_eq_sigma03_d1",
    "soft_theory_eq_sigma04_seed0": "theory_eq_sigma04_d1",
    "soft_theory_eq_sigma045_seed0": "theory_eq_sigma045_d1",
    "soft_theory_eq_seed0": "theory_eq_sigma05_d1",
    "soft_theory_eq_sigma055_seed0": "theory_eq_sigma055_d1",
    "soft_theory_eq_sigma06_seed0": "theory_eq_sigma06_d1",
    "soft_theory_eq_sigma07_seed0": "theory_eq_sigma07_d1",
}

MANIFEST = "data/splits/seven_axis_soft_dseed1.json"
GENERALIST = "generalist_replay_dseed1.yaml"


def main() -> None:
    for src_name, slug in SWEEP.items():
        src = (ARMS / f"{src_name}.yaml").read_text(encoding="utf-8")
        # The 7-pair arm predates the explicit sigma_fraction line and inherits
        # 0.5 from the base, so read the RESOLVED value rather than the text.
        sys.path.insert(0, str(REPO / "src"))
        from infl_ens.config import load_config
        frac_val = float(load_config(ARMS / f"{src_name}.yaml",
                                     validate=False).get("sigma_fraction", 0.5))
        frac = type("M", (), {"group": staticmethod(lambda _i: f"{frac_val:.2f}")})()
        sigma = re.search(r"sigma \(absolute\)\s+([0-9.]+)", src)
        i = src.splitlines().index("  init_positions:")
        block = src.splitlines()[i:]

        header = [
            f"# Reach {frac.group(1)} at the theory equilibrium, DATA SPLIT 1.",
            "#",
            "# Replicate of the seed-0 arm on an independent partition. The",
            "# positions are lifted verbatim from the seed-0 arm: the equilibrium",
            "# is solved against the whole trait cloud, which does not depend on",
            "# how that cloud is split, so reusing it keeps one factor moving.",
            "#",
            "# The split, generalist and test pool all change. Absolute sigma does",
            "# not -- it is the fraction times the lattice threshold, a property",
            "# of the trait space rather than the partition.",
            "#",
            f"#   sigma (absolute)      {sigma.group(1) if sigma else '?'}",
            "#   pairs                 7  (14 clones, co-located)",
            "",
            "includes:",
            "  - _closed_loop_base.yaml",
            "",
            f"output_dir: results/seven_axis_{slug}/seed0",
            "",
            "seed: 1",
            f"sigma_fraction: {frac.group(1)}",
            "",
            "data_split:",
            "  seed: 1",
            f"  manifest: {MANIFEST}",
            "",
            "agents:",
        ] + [f"  - name: clone-{c}" for c in range(14)] + [
            "",
            "closed_loop:",
            "  routing_mode: soft",
            "  soft_top_k: 7",
            "  soft_loss: weighted",
            "  init_mode: given",
            "  init_noise: 0.0",
            "  init_positions_source: >-",
            "    influencer-game equilibrium under the empirical prompt measure;",
            f"    identical to {src_name}",
            "",
        ]
        (ARMS / f"soft_{slug}_seed0.yaml").write_text(
            "\n".join(header + block) + "\n", encoding="utf-8")

        (EXPS / f"seven_axis_{slug}.yaml").write_text("\n".join([
            f"# Reach {frac.group(1)} at the theory equilibrium, data split 1.",
            "#",
            f"# Contrast against seven_axis_sigma{frac.group(1).replace('0.','').ljust(3,'0')[:3]}"
            "_full/dseed1 (naive init, same split),",
            "# which already exists from the replicate sweep. The dseed1 generalist",
            "# is reused and NOT retrained.",
            "#",
            f"#   python -m infl_ens.pipeline --config configs/experiments/seven_axis_{slug}.yaml",
            f"name: seven_axis_{slug}",
            f"results_dir: results/seven_axis_{slug}",
            f"figures_dir: figures/seven_axis_{slug}",
            "",
            "arms:",
            f"  - name: {slug}",
            f'    label: "Reach {frac.group(1)}, split 1"',
            f'    title: "Reach {frac.group(1)} at the empirical-B equilibrium, data split 1"',
            "    role: specialist",
            f"    config: ../arms/soft_{slug}_seed0.yaml",
            "  - name: generalist",
            '    label: "Pooled generalist (split 1)"',
            '    title: "Pooled generalist for data split 1 (existing replay, not retrained)"',
            "    role: generalist",
            f"    config: ../arms/{GENERALIST}",
            "",
            "stages: [manifest, train, perround, routing, figures]",
            "",
        ]), encoding="utf-8")
        print(f"  wrote soft_{slug}_seed0.yaml + seven_axis_{slug}.yaml")


if __name__ == "__main__":
    main()
