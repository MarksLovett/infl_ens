"""Guards for the paper figure pipeline.

Everything here runs against a synthetic results tree, so the tests need no GPU,
no trained adapters and no access to the training host.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))

pytest.importorskip("matplotlib")


def _diagnostics(pooled: float, learned: float, oracle: float) -> dict:
    return {
        "flat": {
            "pooled_nll": pooled,
            "learned_routing_nll": learned,
            "learned_routing_expected_nll": learned,
            "learned_routing_argmax_nll": learned - 0.01,
            "strategic_routing_expected_nll": learned + 0.005,
            "oracle_routing_nll": oracle,
            "routing_agreement_argmax": 0.7,
            "n_prompts": 100,
            "round": 11,
        },
        "per_benchmark": {
            "beavertails": {
                "n": 50, "pooled_nll": pooled, "learned_expected_nll": learned,
                "oracle_nll": oracle,
            },
        },
    }


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """A results tree with one scored split, one decoy and one unscored split."""
    import figures_lib

    results = tmp_path / "results"
    run = results / "seven_axis_demo"
    for sub, scored in (("seed0", True), ("dseed1", True),
                        ("seed1", True),        # run-seed replicate: must be ignored
                        ("dseed9", False)):     # queued but not yet scored
        (run / sub).mkdir(parents=True)
        if scored:
            (run / sub / "routing_ensemble_diagnostics.json").write_text(
                json.dumps(_diagnostics(2.0, 1.94, 1.88)), encoding="utf-8")
    monkeypatch.setattr(figures_lib, "RESULTS", results)
    return results


def test_seed_discovery_takes_data_splits_only(tree):
    """seed1 is a run-seed replicate on split 0, not an independent split."""
    from figures_lib import seeds_for

    found = seeds_for("seven_axis_demo")
    assert found == ["seed0", "dseed1"], found


def test_unscored_split_is_skipped_until_it_has_diagnostics(tree):
    """A queued replicate must not appear as a hole in a mean."""
    from figures_lib import seeds_for

    assert "dseed9" not in seeds_for("seven_axis_demo")


def test_contrast_is_within_split(tree):
    from figures_lib import contrast

    c = contrast(_diagnostics(2.0, 1.94, 1.88)["flat"])
    assert c["gain"] == pytest.approx(0.06)
    assert c["envelope"] == pytest.approx(0.12)
    assert c["capture"] == pytest.approx(0.5)


def test_summarise_reports_sample_sd_and_zero_for_one_run():
    from figures_lib import summarise

    mean, sd, n = summarise([1.0, 2.0, 3.0])
    assert (mean, n) == (2.0, 3)
    assert sd == pytest.approx(1.0)
    assert summarise([5.0]) == (5.0, 0.0, 1)


def test_match_pairs_recovers_a_known_permutation():
    """Pair index is not an identity across runs; matching must undo the shuffle."""
    from figures_lib import match_pairs

    rng = np.random.default_rng(0)
    ref = rng.random((7, 7))
    shuffle = np.array([3, 0, 6, 1, 5, 2, 4])
    perm, cost = match_pairs(ref, ref[shuffle])
    assert cost == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_array_equal(ref[shuffle][perm], ref)


def test_match_pairs_on_itself_is_the_identity():
    from figures_lib import match_pairs

    ref = np.random.default_rng(1).random((5, 7))
    perm, cost = match_pairs(ref, ref)
    np.testing.assert_array_equal(perm, np.arange(5))
    assert cost == pytest.approx(0.0, abs=1e-9)


def test_every_registered_figure_is_callable():
    """The registry is the contract `make_figures.py --only` validates against."""
    import make_figures

    assert len(make_figures.FIGURES) >= 13
    for name, fn in make_figures.FIGURES.items():
        assert callable(fn), name


def test_retired_figures_stay_retired():
    """Two figures were cut for being unreadable; nothing should resurrect them.

    ``gain_envelope`` plotted a part-to-whole decomposition whose message is now
    one column of ``sigma_table``, and its own labels collided. ``convergence``
    carried a single fact, which moved into the methods prose.
    """
    import make_figures

    retired = {"gain_envelope", "convergence"}
    assert not (retired & set(make_figures.FIGURES))


def test_manuscript_figures_are_still_built():
    """The artifact dropped fig_replication; main.tex still includes it.

    Dropping a figure from the walkthrough must not stop it being generated, or
    the next latexmk run fails on a missing file.
    """
    import re

    import make_figures

    tex = PAPER / "main.tex"
    if not tex.is_file():
        pytest.skip("manuscript not present")
    needed = set(re.findall(r"includegraphics\[[^\]]*\]\{fig_([a-z_]+)\.pdf\}",
                            tex.read_text(encoding="utf-8")))
    missing = needed - set(make_figures.FIGURES)
    assert not missing, f"main.tex includes figures nothing builds: {sorted(missing)}"


def test_every_figure_says_which_way_is_better():
    """A reader cannot tell a good NLL from a bad one without being told."""
    from figures_lib import better
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_ylabel("mean NLL")
    better(ax, "y", "down")
    assert "better" in ax.get_ylabel()
    down = ax.get_ylabel()
    better(ax, "y", "down")                 # idempotent: no doubled suffix
    assert ax.get_ylabel() == down
    plt.close(fig)


def test_artifact_contents_match_its_sections():
    """The table of contents and the sections must not drift apart.

    Cutting a section is two edits, and forgetting the second leaves a dead
    anchor in the contents list -- which is exactly what happened when
    ``r-conv`` was removed.
    """
    import re

    template = PAPER / "artifact_template.html"
    if not template.is_file():
        pytest.skip("artifact template not present")
    text = template.read_text(encoding="utf-8")

    linked = re.findall(r'<li><a href="#([a-z0-9-]+)">', text)
    sections = re.findall(r'<section id="([a-z0-9-]+)">', text)
    assert linked == sections, (
        f"contents lists {linked} but sections are {sections}")


def test_artifact_template_only_references_built_figures():
    """A placeholder with no PNG behind it fails the build, so catch it here."""
    import re

    template = PAPER / "artifact_template.html"
    if not template.is_file():
        pytest.skip("artifact template not present")
    import make_figures

    names = {"fig_" + n for n in make_figures.FIGURES}
    referenced = set(re.findall(r"\{\{fig:([a-z_]+)\}\}", template.read_text(encoding="utf-8")))
    unknown = {n for n in referenced if "fig_" + n not in names}
    assert not unknown, f"template references unbuilt figures: {sorted(unknown)}"


def test_drift_from_theory_init_is_inward_and_measurable():
    """Pins the claims the artifact makes about theory-init drift.

    The walkthrough asserts three things about how far pairs leave the solve:
    the move is real, it contracts the configuration, and part of it is the pair
    retreating along its own dominant axis. All three are stated with numbers,
    so they need a guard that fails when a rebuild would make them false.
    """
    from figures_lib import drift_stats

    stats = [s for s in (drift_stats(r, b) for r, b in (
        ("seven_axis_soft_full_pairs", "seed0"),
        ("seven_axis_sigma05_full", "dseed1"),
        ("seven_axis_sigma05_full", "dseed2"),
        ("seven_axis_sigma05_full", "dseed3"),
    )) if s is not None]
    if not stats:
        pytest.skip("no scored sigma-0.50 runs pulled locally")

    for s in stats:
        # A real displacement, not a rounding error, in a box of diagonal sqrt(7).
        assert 0.05 < s["mean_l2"] < 0.6, s
        # Training never spreads the configuration further than the solve did.
        assert s["radius_final"] < s["radius_theory"], s
        # Part of the move is the specialist backing off its own axis.
        assert s["own_axis_drift"] < 0, s
        assert 0 <= s["kept_dominant"] <= s["n_pairs"]
