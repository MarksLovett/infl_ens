"""Config-driven encoder selection (no torch required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from infl_ens.config import ConfigError, load_config
from infl_ens.data.encoders import encoder_kwargs_from_config, make_encoder

ROOT = Path(__file__).resolve().parents[1]
QWEN_PRESET = ROOT / "configs" / "encoders" / "qwen3_embedding_8b_awq.yaml"
BGE_PRESET = ROOT / "configs" / "encoders" / "bge_large_en_v1_5.yaml"


def test_default_preset_yaml_names_qwen3_awq() -> None:
    """The project preset selects the Qwen3-Embedding-8B AWQ checkpoint."""
    cfg = load_config(QWEN_PRESET)
    assert cfg["trait_space"]["encoder"] == "drawais/Qwen3-Embedding-8B-AWQ-INT4"
    assert cfg["encoder"]["model_name"] == cfg["trait_space"]["encoder"]
    assert encoder_kwargs_from_config(cfg) == {
        "model_name": "drawais/Qwen3-Embedding-8B-AWQ-INT4",
        "pooling": "last_token",
        "padding_side": "left",
        "max_length": 512,
        "normalize": True,
        "torch_dtype": "auto",
        "device_map": "auto",
        "batch_size": 32,
    }


def test_alternative_preset_is_a_valid_hf_encoder_config() -> None:
    cfg = load_config(BGE_PRESET)
    kwargs = encoder_kwargs_from_config(cfg)
    assert kwargs["model_name"] == "BAAI/bge-large-en-v1.5"
    assert kwargs["pooling"] == "cls"
    assert kwargs["device_map"] is None


def test_make_encoder_forwards_resolved_kwargs() -> None:
    cfg = load_config(QWEN_PRESET)
    with patch("infl_ens.data.encoders.HuggingFaceEncoder") as encoder_cls:
        make_encoder(cfg)
    encoder_cls.assert_called_once_with(**encoder_kwargs_from_config(cfg))


def test_legacy_string_form_promotes_encoder_batch_size() -> None:
    cfg = {"trait_space": {"encoder": "org/model", "encoder_batch_size": 4}}
    assert encoder_kwargs_from_config(cfg) == {"model_name": "org/model", "batch_size": 4}


def test_legacy_mapping_form_is_forwarded_verbatim() -> None:
    inline = {
        "model_name": "org/model",
        "batch_size": 4,
        "max_length": 1024,
        "pooling": "mean",
        "device_map": None,
        "padding_side": "right",
    }
    cfg = {"trait_space": {"encoder": dict(inline), "encoder_batch_size": 99}}
    assert encoder_kwargs_from_config(cfg) == inline


def test_make_encoder_requires_model_name() -> None:
    with pytest.raises(ConfigError, match="no encoder model selected"):
        encoder_kwargs_from_config({})
    with pytest.raises(ConfigError, match="no encoder model selected"):
        encoder_kwargs_from_config({"encoder": {"pooling": "mean"}})


def test_make_encoder_rejects_name_mismatch() -> None:
    cfg = {"trait_space": {"encoder": "org/a"}, "encoder": {"model_name": "org/b"}}
    with pytest.raises(ConfigError, match="disagrees"):
        encoder_kwargs_from_config(cfg)


def test_make_encoder_rejects_bad_encoder_type() -> None:
    with pytest.raises(ConfigError, match="string or mapping"):
        encoder_kwargs_from_config({"trait_space": {"encoder": 3}})
