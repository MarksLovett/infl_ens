"""The model scale-family sweep: configs, pairing, formatting and the figure.

Covers the four moving parts added for the 3 family x 3 scale sweep:

- the nine specialist cells keep the pinned trait-space cache fingerprint
  and differ only in ``sft.base_model`` / ``output_dir``;
- the experiment loads with nine specialist + nine generalist arms and each
  specialist pairs with its same ``(family, scale)`` generalist;
- ``make_chat_formatter`` uses the tokenizer's chat template and falls back
  to the Qwen formatting when none is present (no transformers needed);
- the ``family_scale_nll`` figure builder and table writer produce output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infl_ens.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CELL_DIR = ROOT / "configs" / "arms" / "scale_family"
EXPERIMENT = ROOT / "configs" / "experiments" / "scale_family_sweep.yaml"
EXPECTED_FINGERPRINT = "3b42c68a8dd334c5"

CELLS = sorted(p for p in CELL_DIR.glob("*.yaml") if not p.name.startswith("_"))
SPECIALISTS = [p for p in CELLS if not p.stem.endswith("_gen")]
GENERALISTS = [p for p in CELLS if p.stem.endswith("_gen")]


def test_nine_specialist_and_nine_generalist_files() -> None:
    assert len(SPECIALISTS) == 9
    assert len(GENERALISTS) == 9
    # every specialist has a matching <cell>_gen generalist
    spec_names = {p.stem for p in SPECIALISTS}
    gen_names = {p.stem[: -len("_gen")] for p in GENERALISTS}
    assert spec_names == gen_names


@pytest.mark.parametrize("path", SPECIALISTS, ids=[p.stem for p in SPECIALISTS])
def test_specialist_keeps_the_cached_fingerprint(path: Path) -> None:
    from infl_ens.data.trait_space_cache import trait_space_fingerprint

    cfg = load_config(path)
    assert cfg["task"] == "closed_loop"
    assert trait_space_fingerprint(cfg) == EXPECTED_FINGERPRINT


def test_specialists_differ_only_in_base_model_and_output() -> None:
    resolved = {p.stem: load_config(p) for p in SPECIALISTS}
    reference = resolved["qwen_1b"]
    base_models = set()
    outputs = set()
    for name, cfg in resolved.items():
        base_models.add(cfg["sft"]["base_model"])
        outputs.add(cfg["output_dir"])
        # every non-model, non-output block matches the reference cell
        for key in ("benchmarks", "trait_space", "data_split", "agents",
                    "closed_loop", "seed", "sigma_fraction"):
            assert cfg[key] == reference[key], (name, key)
        # the sft block matches except for the base model
        sft = {k: v for k, v in cfg["sft"].items() if k != "base_model"}
        ref_sft = {k: v for k, v in reference["sft"].items() if k != "base_model"}
        assert sft == ref_sft, name
    assert len(base_models) == 9
    assert len(outputs) == 9


def test_generalist_points_at_its_specialist_history() -> None:
    for path in GENERALISTS:
        cfg = load_config(path)
        cell = path.stem[: -len("_gen")]
        assert cfg["task"] == "baseline_replay"
        assert cfg["baseline_replay"]["agent_name"] == "pooled-baseline"
        assert cfg["history_path"] == f"results/scale_family/{cell}/seed0/history.json"
        # the generalist reuses the specialist cell's base model
        assert cfg["sft"]["base_model"] == load_config(CELL_DIR / f"{cell}.yaml")["sft"]["base_model"]


def test_experiment_loads_and_pairs_each_cell() -> None:
    from infl_ens.experiment import load_experiment

    exp = load_experiment(EXPERIMENT)
    assert len(exp.specialists) == 9
    assert len(exp.generalists) == 9
    for spec in exp.specialists:
        gen = exp.generalist_for(spec)
        assert gen is not None
        assert gen.cell == spec.cell
        assert gen.role == "generalist"
    # families x scales cover the full 3 x 3 grid
    cells = {(s.family, s.scale) for s in exp.specialists}
    assert len(cells) == 9
    assert {f for f, _ in cells} == {"Qwen2.5", "Llama-3.x", "Gemma"}
    assert {s for _, s in cells} == {"1b", "3b", "8b"}


class _StubTokenizer:
    """Minimal tokenizer exposing a chat template for formatter tests."""

    def __init__(self, has_template: bool) -> None:
        self.chat_template = "TEMPLATE" if has_template else None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        body = " | ".join(f"{m['role']}:{m['content']}" for m in messages)
        return f"[{body}]" + ("<gen>" if add_generation_prompt else "")


def test_make_chat_formatter_uses_template_and_falls_back() -> None:
    from infl_ens.training.sft_training import _format_chat, make_chat_formatter

    templated = make_chat_formatter(_StubTokenizer(has_template=True))
    assert templated("hi", "there") == "[user:hi | assistant:there]"
    assert templated("hi", None) == "[user:hi]<gen>"

    fallback = make_chat_formatter(_StubTokenizer(has_template=False))
    assert fallback is _format_chat
    assert fallback("hi", "there") == _format_chat("hi", "there")


def test_family_scale_figure_and_table(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from infl_ens.figures.scale_family import (
        CellNLL,
        plot_family_scale_nll,
        write_family_scale_table,
    )

    families = ["Qwen2.5", "Llama-3.x", "Gemma"]
    scales = ["1b", "3b", "8b"]
    cells = [
        CellNLL(f, s, learned_nll=1.0 + 0.1 * i, pooled_nll=1.2, oracle_nll=0.9)
        for i, f in enumerate(families)
        for s in scales
    ]
    fig = plot_family_scale_nll(cells, families=families, scales=scales, title="t")
    assert isinstance(fig, Figure)

    written = write_family_scale_table(
        cells, tmp_path / "fs", families=families, scales=scales, label="t",
    )
    assert set(written) == {"csv", "md", "tex", "json"}
    for path in written.values():
        assert path.is_file()

    with pytest.raises(ValueError):
        plot_family_scale_nll([])
