"""Every figure builder returns a Figure / TeX / table on synthetic inputs.

The plot functions are pure (records or arrays in, Figure out); this module
also drives :func:`infl_ens.figures.render.render_all` over a synthetic
experiment so the artifact-reading layer is exercised end to end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from infl_ens.experiment import (  # noqa: E402
    ArmSpec,
    EvalSettings,
    ExperimentConfig,
    FigureSettings,
    SmokeSettings,
)
from infl_ens.figures import (  # noqa: E402
    plot_benchmark_nll_comparison,
    plot_dataset_composition,
    plot_final_positions,
    plot_history,
    plot_marginals,
    plot_pair_comparison,
    plot_pairwise_heatmaps,
    plot_within_pair,
)
from infl_ens.figures.cross_arm_report import build_cross_arm_report, data_matching  # noqa: E402
from infl_ens.figures.per_round_tables import (  # noqa: E402
    build_per_round_tables,
    eval_rows_cover,
    pivot_per_round,
)
from infl_ens.figures.pgf_tex import arm_comparison_tex, oracle_routing_tex  # noqa: E402
from infl_ens.figures.render import FIGURES, cpu_figures, render_all  # noqa: E402
from infl_ens.figures.trait_representation import representation_stats  # noqa: E402

GROUPS = [("pair-0", ["clone-0", "clone-1"]), ("pair-1", ["clone-2", "clone-3"])]
BENCHES = ["beavertails", "halueval", "ai4privacy", "orbench", "prompt_injection", "do_not_answer"]


def _history(n_rounds: int = 3, dim: int = 2, separate: bool = False) -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    base = rng.random((4, dim))
    base[1] = base[0]
    base[3] = base[2]
    records = []
    for r in range(n_rounds):
        pos = base + 0.02 * r
        if separate:
            pos[1] += 0.01 * (r + 1)
        records.append({
            "round": r,
            "positions": {f"clone-{i}": pos[i].tolist() for i in range(4)},
            "u_grid": (np.ones(4) / 4).tolist(),
            "u_pool": (np.ones(4) / 4).tolist(),
            "observed_share": (np.ones(4) / 4).tolist(),
            "strategic_share_pool": (np.ones(4) / 4).tolist(),
            "agent_geometry": {"within_merge_l2": {"pair-0": 0.0, "pair-1": 0.01 * r}},
            "batch_prompts": [f"prompt {r} {k}" for k in range(5)],
            "pair_members": {name: members for name, members in GROUPS},
        })
    return records


def _report(offset: float = 0.0) -> dict[str, Any]:
    per = {
        b: {
            "n": 100,
            "pooled_nll": 1.2 + offset,
            "learned_expected_nll": 1.1 + offset,
            "learned_argmax_nll": 1.15 + offset,
            "oracle_nll": 1.0 + offset,
        }
        for b in BENCHES
    }
    return {
        "flat": {
            "pooled_nll": 1.2 + offset,
            "learned_routing_expected_nll": 1.1 + offset,
            "oracle_routing_nll": 1.0 + offset,
            "n_prompts": 600,
            "round": 11,
        },
        "per_benchmark": per,
    }


def _eval_rows() -> list[dict[str, Any]]:
    return [
        {"round": r, "agent": a, "benchmark": b, "mean_nll": 1.0 + 0.1 * r + 0.01 * i, "n_tokens": 10}
        for r in (4, 11)
        for i, a in enumerate(("pair-0", "pair-1"))
        for b in BENCHES
    ]


def test_pair_figures_return_figures() -> None:
    records = _history()
    assert isinstance(plot_final_positions(records[-1], GROUPS, axis_labels=("a", "b"), title="t"), Figure)
    assert isinstance(plot_within_pair(records, GROUPS, title="t"), Figure)
    assert isinstance(plot_within_pair(_history(separate=True), GROUPS, title="t"), Figure)


def test_final_positions_handles_seven_axes() -> None:
    records = _history(dim=7)
    fig = plot_final_positions(records[-1], GROUPS, axis_labels=("harm", "hallucination"), title="7d")
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 21


def test_plot_history_projects_higher_dimensions() -> None:
    assert isinstance(plot_history(_history(dim=2), axis_labels=("a", "b")), Figure)
    assert isinstance(plot_history(_history(dim=7), axis_labels=("a", "b", "c")), Figure)


def test_trait_representation_figures() -> None:
    rng = np.random.default_rng(1)
    legacy = np.clip(rng.normal(0.5, 0.4, (200, 3)), 0, 1)
    new = rng.random((200, 3))
    labels = ["harm", "hallucination", "privacy"]
    stats_legacy = representation_stats(legacy, labels)
    stats_new = representation_stats(new, labels)
    assert stats_legacy["max_frac_saturated"] > stats_new["max_frac_saturated"]
    assert isinstance(
        plot_marginals(legacy, new, labels, stats_legacy=stats_legacy, stats_new=stats_new), Figure,
    )
    assert isinstance(plot_pair_comparison(legacy, new, labels, pairs=[(0, 1), (1, 2)]), Figure)
    assert isinstance(plot_dataset_composition(np.array([0, 0, 1, 2]), labels), Figure)


def test_nll_bar_and_heatmaps() -> None:
    fig = plot_benchmark_nll_comparison(
        ["beavertails", "halueval"],
        {"beavertails": "Harm", "halueval": "Hallucination"},
        {"beavertails": 1.2, "halueval": 1.1},
        {"beavertails": {"clone-0": 1.0}, "halueval": {"clone-0": 1.05}},
        base_label="Base (toy)",
    )
    assert isinstance(fig, Figure)
    grid = np.array([[x, y] for x in np.linspace(0, 1, 5) for y in np.linspace(0, 1, 5)])
    weights = np.ones(len(grid)) / len(grid)
    fig = plot_pairwise_heatmaps(grid, weights, axis_labels=("a", "b"), prompt_coords=grid)
    assert isinstance(fig, Figure)


def test_pgf_writers_emit_axes() -> None:
    single = oracle_routing_tex(_report(), experiment_label="Arm & co")
    assert single.count("\\begin{axis}") == 2
    assert "Arm \\& co" in single
    multi = arm_comparison_tex([("Soft k=3", _report()), ("Hard", _report(0.05))])
    assert multi.count("\\addlegendentry") == 3 + 2 + 2
    with pytest.raises(ValueError):
        arm_comparison_tex([])


def test_per_round_pivot_and_outputs(tmp_path: Path) -> None:
    rows = _eval_rows()
    rounds, agents, table = pivot_per_round(rows, rounds=[4, 11])
    assert rounds == [4, 11] and agents == ["pair-0", "pair-1"]
    assert table[11]["pair-1"] == pytest.approx(2.11)
    with pytest.raises(ValueError, match="not in the report"):
        pivot_per_round(rows, rounds=[4, 7])

    eval_dir = tmp_path / "eval_val"
    eval_dir.mkdir()
    (eval_dir / "eval_results.json").write_text(json.dumps({"results": rows}), encoding="utf-8")
    assert eval_rows_cover(eval_dir, [4, 11])
    assert not eval_rows_cover(eval_dir, [4, 5])
    written = build_per_round_tables(eval_dir, tmp_path / "tables" / "pivot", label="t", rounds=[4, 11])
    assert set(written) == {"csv", "md", "tex", "json"}
    payload = json.loads(written["json"].read_text(encoding="utf-8"))
    assert payload["delta_first_to_last"]["pair-0"] == pytest.approx(0.7)


def test_data_matching_flags_mismatch() -> None:
    same = _history()
    different = _history()
    different[1]["batch_prompts"] = ["other"]
    assert data_matching([("a", same), ("b", _history())])["all_identical"]
    assert not data_matching([("a", same), ("b", different)])["all_identical"]


def _fake_experiment(tmp_path: Path, *, with_reports: bool = True) -> ExperimentConfig:
    arms = []
    for name, offset in (("soft", 0.0), ("hard", 0.05)):
        run = tmp_path / "results" / name
        run.mkdir(parents=True)
        (run / "history.json").write_text(json.dumps(_history(12)), encoding="utf-8")
        if with_reports:
            (run / "routing_ensemble_diagnostics.json").write_text(json.dumps(_report(offset)), encoding="utf-8")
        eval_dir = run / "eval_val"
        eval_dir.mkdir()
        (eval_dir / "eval_results.json").write_text(json.dumps({"results": _eval_rows()}), encoding="utf-8")
        cfg = tmp_path / f"{name}.yaml"
        cfg.write_text(f"task: closed_loop\noutput_dir: {run.as_posix()}\n", encoding="utf-8")
        arms.append(ArmSpec(name=name, label=name, title=name.title(), role="specialist",
                            config_path=cfg, output_dir=run))
    gen_run = tmp_path / "results" / "generalist"
    gen_run.mkdir()
    gen_cfg = tmp_path / "gen.yaml"
    gen_cfg.write_text(f"task: baseline_replay\noutput_dir: {gen_run.as_posix()}\n", encoding="utf-8")
    arms.append(ArmSpec(name="generalist", label="Pooled", title="Pooled", role="generalist",
                        config_path=gen_cfg, output_dir=gen_run))
    return ExperimentConfig(
        name="fake", path=tmp_path / "exp.yaml",
        results_dir=tmp_path / "results" / "fake", figures_dir=tmp_path / "figures",
        arms=tuple(arms), stages=("figures",),
        eval=EvalSettings(perround_rounds=(4, "final")),
        figures=FigureSettings(axis_labels=("harm", "hallucination"), formats=("png",), compile_tex="never"),
        smoke=SmokeSettings(),
    )


def test_render_all_writes_every_cpu_figure(tmp_path: Path) -> None:
    exp = _fake_experiment(tmp_path)
    written = render_all(exp, only=cpu_figures())
    assert set(written) == set(cpu_figures())
    for name, paths in written.items():
        assert paths, name
        for path in paths:
            assert path.is_file(), path
    assert (tmp_path / "figures" / "arm_comparison.tex").is_file()
    assert (tmp_path / "figures" / "soft_final_positions.png").is_file()
    assert (tmp_path / "figures" / "cross_analysis.md").is_file()
    assert (tmp_path / "results" / "soft" / "tables" / "pair_nll_by_round.json").is_file()


def test_render_all_skips_missing_inputs_and_gpu_figures(tmp_path: Path) -> None:
    exp = _fake_experiment(tmp_path, with_reports=False)
    written = render_all(exp, only=["oracle_routing", "pair_positions", "trait_representation"])
    assert written["oracle_routing"] == []
    assert written["trait_representation"] == []
    assert written["pair_positions"]
    with pytest.raises(KeyError):
        render_all(exp, only=["nope"])
    assert FIGURES["trait_representation"].requires_gpu


def test_cross_arm_report_reads_run_dirs(tmp_path: Path) -> None:
    exp = _fake_experiment(tmp_path)
    render_all(exp, only=["per_round_tables"])
    report, md = build_cross_arm_report(
        [(a.label, a.run_dir) for a in exp.specialists], generalist_run_dir=exp.generalist.run_dir,
    )
    assert report["data_matching"]["all_identical"]
    assert report["arms"]["soft"]["routing"]["learned_minus_pooled"] == pytest.approx(-0.1)
    assert report["arms"]["soft"]["per_round"]["n_pairs"] == 2
    assert "## 2. Routing headline" in md
