"""Offline tests for direct Hugging Face embedding extraction."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from infl_ens.data.encoders import DEFAULT_ENCODER_MODEL, HuggingFaceEncoder
from infl_ens.data.trait_space_cache import make_trait_space_encoder


class _FakeTokenizer:
    """Tokenizer stub that emits a left-padded two-item batch."""

    pad_token: str | None = None
    eos_token = "<eos>"

    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        """Return a deterministic left-padded token batch.

        :param texts: Input texts.
        :type texts: list[str]
        :returns: Token IDs and attention masks.
        :rtype: dict[str, torch.Tensor]
        """
        assert texts == ["short", "long"]
        return {
            "input_ids": torch.tensor([[0, 1, 2], [3, 4, 5]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        }


class _FakeModel:
    """Minimal ``AutoModel`` substitute with fixed token hidden states."""

    config = SimpleNamespace(hidden_size=2)
    device = torch.device("cpu")

    def to(self, device: object) -> _FakeModel:
        """Record model placement and return this model.

        :param device: Requested device.
        :type device: object
        :returns: This fake model.
        :rtype: _FakeModel
        """
        self.to_device = device
        return self

    def eval(self) -> _FakeModel:
        """Match the Hugging Face evaluation-mode API.

        :returns: This fake model.
        :rtype: _FakeModel
        """
        self.evaluated = True
        return self

    def __call__(self, **_: torch.Tensor) -> SimpleNamespace:
        """Return distinguishable token representations.

        :returns: Object exposing ``last_hidden_state``.
        :rtype: types.SimpleNamespace
        """
        return SimpleNamespace(
            last_hidden_state=torch.tensor(
                [
                    [[99.0, 99.0], [1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0], [9.0, 12.0]],
                ],
            ),
        )


def test_qwen_default_is_the_requested_awq_model() -> None:
    """The project default is the requested Qwen3 AWQ embedding checkpoint."""
    assert DEFAULT_ENCODER_MODEL == "drawais/Qwen3-Embedding-8B-AWQ-INT4"


def test_encoder_uses_left_padded_final_token_pooling() -> None:
    """Qwen-style left padding selects each batch row's final token."""
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as load_tok,
        patch("transformers.AutoModel.from_pretrained", return_value=model) as load_model,
    ):
        encoder = HuggingFaceEncoder(
            "example/qwen-embedding",
            device_map=None,
        )
        embeddings = encoder(["short", "long"])

    assert load_tok.call_args.kwargs["padding_side"] == "left"
    assert load_model.call_args.kwargs["trust_remote_code"] is False
    assert model.to_device == "cpu"
    expected = np.asarray([[0.6, 0.8], [0.6, 0.8]], dtype=np.float32)
    assert np.allclose(embeddings, expected)


def test_encoder_leaves_device_mapped_model_in_place() -> None:
    """AWQ model loading delegates placement to Transformers' device map."""
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModel.from_pretrained", return_value=model) as load_model,
    ):
        HuggingFaceEncoder("example/qwen-embedding", device_map="auto")

    assert load_model.call_args.kwargs["device_map"] == "auto"
    assert not hasattr(model, "to_device")


def test_encoder_factory_accepts_full_hugging_face_configuration() -> None:
    """The trait-space factory forwards model-specific encoder settings."""
    cfg = {
        "trait_space": {
            "encoder": {
                "model_name": "drawais/Qwen3-Embedding-8B-AWQ-INT4",
                "batch_size": 4,
                "max_length": 1024,
                "pooling": "last_token",
                "device_map": "auto",
                "padding_side": "left",
            },
        },
    }
    with patch("infl_ens.data.trait_space_cache.HuggingFaceEncoder") as encoder_cls:
        make_trait_space_encoder(cfg)

    encoder_cls.assert_called_once_with(**cfg["trait_space"]["encoder"])
