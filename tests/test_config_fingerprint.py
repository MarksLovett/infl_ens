"""The layered arm configs must keep the trait-space cache fingerprint.

The GPU host holds a 28k-prompt encode under
``data/trait_space_cache/3b42c68a8dd334c5``.  That fingerprint hashes the
resolved ``benchmarks`` list and ``trait_space`` block, so any edit to the
shared fragments that changes either forces a multi-hour re-encode.  This
test pins the contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infl_ens.config import load_config
from infl_ens.data.trait_space_cache import trait_space_fingerprint

ROOT = Path(__file__).resolve().parents[1]
ARMS = [
    ROOT / "configs" / "arms" / "soft_topk3_pairs.yaml",
    ROOT / "configs" / "arms" / "hard_pairs_matched.yaml",
    ROOT / "configs" / "arms" / "generalist_replay.yaml",
]

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


@pytest.mark.parametrize("path", ARMS, ids=[p.stem for p in ARMS])
def test_arm_resolves_to_the_cached_fingerprint(path: Path) -> None:
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
    soft = load_config(ARMS[0])
    hard = load_config(ARMS[1])
    assert soft["output_dir"] != hard["output_dir"]
    routing_keys = {"routing_mode", "soft_top_k", "soft_loss"}
    soft_cl = {k: v for k, v in soft["closed_loop"].items() if k not in routing_keys}
    hard_cl = {k: v for k, v in hard["closed_loop"].items() if k not in routing_keys}
    assert soft_cl == hard_cl
    assert soft["closed_loop"]["routing_mode"] == "soft"
    assert hard["closed_loop"]["routing_mode"] == "hard"
    for key in ("benchmarks", "trait_space", "data_split", "sft", "agents", "seed", "sigma_fraction"):
        assert soft[key] == hard[key]
