"""The layered arm configs must keep the trait-space cache fingerprint.

The GPU host holds a 28k-prompt encode under
``data/trait_space_cache/3b42c68a8dd334c5``.  That fingerprint hashes the
resolved ``benchmarks`` list and ``trait_space`` block, so any edit to the
shared fragments that changes either forces a multi-hour re-encode.  This
test pins the contract for every arm of the experiment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infl_ens.config import load_config

ROOT = Path(__file__).resolve().parents[1]
ARMS_DIR = ROOT / "configs" / "arms"
ARMS = sorted(p for p in ARMS_DIR.glob("*.yaml") if not p.name.startswith("_"))
SPECIALISTS = [p for p in ARMS if p.name != "generalist_replay.yaml"]

EXPECTED_FINGERPRINT = "3b42c68a8dd334c5"

EXPECTED_BENCHMARKS = [
    {"kind": "beavertails", "path": "data/beavertails/30k_train.jsonl", "max_records": 5000},
    {"kind": "halueval", "path": "data/halueval", "tasks": ["qa", "dialogue"], "max_records": 5000},
    {"kind": "jbb_behaviors", "path": "data/jbb_behaviors", "include_benign": True, "max_records": None},
    {
        "kind": "ai4privacy",
        "path": "data/ai4privacy",
        "score_mode": "density",
        "english_only": True,
        "max_records": 5000,
    },
    {
        "kind": "orbench",
        "path": "data/orbench",
        "configs": ["or-bench-80k", "or-bench-toxic"],
        "max_records": 5000,
    },
    {"kind": "prompt_injection", "path": "data/prompt_injection", "max_records": 5000},
    {"kind": "do_not_answer", "path": "data/do_not_answer", "include_benign": True, "max_records": 5000},
]

EXPECTED_TRAIT_SPACE = {
    "encoder": "drawais/Qwen3-Embedding-8B-AWQ-INT4",
    "encoder_batch_size": 32,
    "cache": True,
    "cache_dir": "data/trait_space_cache",
    "n_grid": 3,
    "kde_bandwidth": 0.08,
    "threshold": 0.5,
    "coordinate_residualize": True,
    "mode_alignment_weight": 0.35,
    "mode_alignment_weights": {
        "jailbreak": 1.0,
        "privacy": 1.0,
        "overrefusal": 1.0,
        "injection": 1.0,
        "policy_violation": 1.0,
    },
    "coordinate_stretch_gamma": 1.0,
}

ROUTING_KEYS = {"routing_mode", "soft_top_k", "soft_loss", "soft_select"}


def test_all_arms_are_present() -> None:
    names = {p.stem for p in ARMS}
    assert names == {
        "soft_full_pairs",
        "soft_topk3_pairs",
        "topk3_unit_pairs",
        "hard_topk3_pairs",
        "hard_pairs_matched",
        "generalist_replay",
    }


@pytest.mark.parametrize("path", ARMS, ids=[p.stem for p in ARMS])
def test_arm_resolves_to_the_cached_fingerprint(path: Path) -> None:
    from infl_ens.data.trait_space_cache import trait_space_fingerprint

    cfg = load_config(path)
    assert cfg["benchmarks"] == EXPECTED_BENCHMARKS
    assert cfg["trait_space"] == EXPECTED_TRAIT_SPACE
    assert isinstance(cfg["trait_space"]["encoder"], str)
    assert trait_space_fingerprint(cfg) == EXPECTED_FINGERPRINT


@pytest.mark.parametrize("path", ARMS, ids=[p.stem for p in ARMS])
def test_encoder_block_agrees_with_trait_space_encoder(path: Path) -> None:
    cfg = load_config(path)
    assert cfg["encoder"]["model_name"] == cfg["trait_space"]["encoder"]


def test_specialist_arms_differ_only_in_routing_and_output() -> None:
    """Data-matching precondition: same seed, manifest, init and LoRA settings."""
    resolved = {p.stem: load_config(p) for p in SPECIALISTS}
    reference = resolved["soft_topk3_pairs"]
    ref_cl = {k: v for k, v in reference["closed_loop"].items() if k not in ROUTING_KEYS}
    outputs = set()
    for name, cfg in resolved.items():
        outputs.add(cfg["output_dir"])
        cl = {k: v for k, v in cfg["closed_loop"].items() if k not in ROUTING_KEYS}
        assert cl == ref_cl, name
        for key in ("benchmarks", "trait_space", "data_split", "sft", "agents", "seed", "sigma_fraction"):
            assert cfg[key] == reference[key], (name, key)
    assert len(outputs) == len(resolved)


def test_routing_knobs_match_the_design_table() -> None:
    resolved = {p.stem: load_config(p)["closed_loop"] for p in SPECIALISTS}
    assert resolved["soft_full_pairs"]["routing_mode"] == "soft"
    assert resolved["soft_full_pairs"]["soft_top_k"] == 7
    assert resolved["soft_full_pairs"]["soft_loss"] == "weighted"
    assert resolved["soft_topk3_pairs"]["soft_top_k"] == 3
    assert resolved["soft_topk3_pairs"]["soft_loss"] == "weighted"
    assert resolved["topk3_unit_pairs"]["soft_top_k"] == 3
    assert resolved["topk3_unit_pairs"]["soft_loss"] == "unit"
    assert resolved["hard_topk3_pairs"]["soft_select"] == "sample"
    assert resolved["hard_topk3_pairs"]["soft_loss"] == "unit"
    assert resolved["hard_pairs_matched"]["routing_mode"] == "hard"
    assert "soft_select" not in resolved["topk3_unit_pairs"]
