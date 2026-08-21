"""Hugging Face text encoders for trait-space construction.

The trait-space pipeline uses a single embedding backend:
:class:`HuggingFaceEncoder`.  It loads an encoder-compatible model directly
with :mod:`transformers`, pools token hidden states, and returns a NumPy
embedding matrix.  Keeping tokenisation, pooling, normalisation, and model
selection in one place makes the geometry used to build a trait space the
same geometry used later to project prompts during routing.

The default model is the project's quantized
``drawais/Qwen3-Embedding-8B-AWQ-INT4`` encoder.  Its Qwen-recommended
left-padding and final-token pooling are used by default.  Any Hugging Face
model supported by ``AutoModel`` can be selected through the
``trait_space.encoder`` configuration.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np

DEFAULT_ENCODER_MODEL = "drawais/Qwen3-Embedding-8B-AWQ-INT4"
PoolingStrategy = Literal["mean", "cls", "last_token"]
_POOLING_STRATEGIES = frozenset({"mean", "cls", "last_token"})
_DTYPE_NAMES = frozenset({"auto", "float32", "float16", "bfloat16"})


class HuggingFaceEncoder:
    """Encode text directly with a Hugging Face ``AutoModel``.

    The model must expose ``last_hidden_state`` (as standard encoder models
    do).  Qwen3-Embedding uses left padding plus ``last_token`` pooling, so
    those are the defaults.  Embeddings are L2-normalised by default so trait
    directions are not driven by sentence length or vector norm.

    :param model_name: Hugging Face model identifier or local model path.
    :type model_name: str
    :param device: Torch device such as ``'cuda'``, ``'cuda:0'``, or
        ``'cpu'``.  Used only when ``device_map`` is ``None``.  ``None``
        selects CUDA when available and CPU otherwise.
    :type device: str | None
    :param batch_size: Number of texts encoded per forward pass.
    :type batch_size: int
    :param max_length: Maximum number of tokens retained per text.
    :type max_length: int
    :param pooling: Token-pooling strategy: attention-mask ``'mean'``, first
        token ``'cls'``, or final non-padding token ``'last_token'``.
    :type pooling: {'mean', 'cls', 'last_token'}
    :param normalize: Whether to L2-normalise each output embedding.
    :type normalize: bool
    :param torch_dtype: Model-weight dtype: ``'auto'``, ``'float32'``,
        ``'float16'``, or ``'bfloat16'``.  ``'auto'`` lets Transformers use
        the checkpoint's declared dtype.
    :type torch_dtype: str
    :param device_map: Transformers device-placement strategy.  ``'auto'``
        is the default because the AWQ model is large and quantized; it lets
        Accelerate place it without a subsequent unsupported ``.to(...)``.
        Set to ``None`` to place an unquantized model on ``device`` directly.
    :type device_map: str | None
    :param padding_side: Batch-padding side.  Qwen3 embedding models require
        ``'left'`` with final-token pooling.
    :type padding_side: {'left', 'right'}
    :param attn_implementation: Optional Transformers attention backend, for
        example ``'flash_attention_2'`` when it is installed.
    :type attn_implementation: str | None
    :param trust_remote_code: Whether to permit custom model code from the
        model repository.  Defaults to ``False`` for reproducibility and
        safety.
    :type trust_remote_code: bool
    :raises ImportError: If ``torch`` or ``transformers`` is not installed.
    :raises ValueError: If an option is invalid or the selected model lacks
        the token-hidden-state interface required for embedding.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ENCODER_MODEL,
        *,
        device: Optional[str] = None,
        batch_size: int = 2,
        max_length: int = 512,
        pooling: PoolingStrategy = "last_token",
        normalize: bool = True,
        torch_dtype: str = "auto",
        device_map: Optional[str] = "auto",
        padding_side: Literal["left", "right"] = "left",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must be a non-empty Hugging Face identifier")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if max_length < 1:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if pooling not in _POOLING_STRATEGIES:
            raise ValueError(
                "pooling must be one of "
                f"{sorted(_POOLING_STRATEGIES)}, got {pooling!r}",
            )
        if torch_dtype not in _DTYPE_NAMES:
            raise ValueError(
                "torch_dtype must be one of "
                f"{sorted(_DTYPE_NAMES)}, got {torch_dtype!r}",
            )
        if padding_side not in ("left", "right"):
            raise ValueError(
                f"padding_side must be 'left' or 'right', got {padding_side!r}",
            )

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "HuggingFaceEncoder requires torch and transformers; install "
                "them with `pip install 'infl_ens[ml]'`."
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.pooling: PoolingStrategy = pooling
        self.normalize = bool(normalize)
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.padding_side = padding_side
        self.attn_implementation = attn_implementation
        self.trust_remote_code = bool(trust_remote_code)

        load_kwargs = {"trust_remote_code": self.trust_remote_code}
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side=self.padding_side,
            **load_kwargs,
        )
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token is None:
                raise ValueError(
                    f"tokenizer for {model_name!r} has neither pad_token nor eos_token; "
                    "cannot batch variable-length texts",
                )
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs = dict(load_kwargs)
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        self._model = AutoModel.from_pretrained(model_name, **model_kwargs)
        if device_map is None:
            self._model.to(self.device)
        self._model.eval()
        self._input_device = getattr(self._model, "device", self.device)

        hidden_size = getattr(self._model.config, "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError(
                f"model {model_name!r} does not expose a positive config.hidden_size; "
                "choose an encoder model supported by AutoModel",
            )
        self.embedding_dimension = hidden_size

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of texts into a two-dimensional NumPy array.

        :param texts: Texts to encode.  The order is preserved.
        :type texts: Sequence[str]
        :returns: Array of shape ``(len(texts), embedding_dimension)``.
        :rtype: numpy.ndarray
        :raises TypeError: If any input item is not a string.
        :raises ValueError: If the model does not return ``last_hidden_state``.
        """
        batch_texts = list(texts)
        if any(not isinstance(text, str) for text in batch_texts):
            raise TypeError("HuggingFaceEncoder expects a sequence of strings")
        if not batch_texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        torch = self._torch
        blocks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(batch_texts), self.batch_size):
                tokenized = self._tokenizer(
                    batch_texts[start:start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                model_inputs = {
                    name: value.to(self._input_device) for name, value in tokenized.items()
                }
                outputs = self._model(**model_inputs)
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    raise ValueError(
                        f"model {self.model_name!r} did not return last_hidden_state; "
                        "choose an encoder model supported by AutoModel",
                    )
                pooled = self._pool(hidden, model_inputs["attention_mask"])
                if self.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                blocks.append(pooled.float().cpu().numpy())
        return np.concatenate(blocks, axis=0)

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """Encode ``texts``; shorthand for :meth:`encode`.

        :param texts: Texts to encode.
        :type texts: Sequence[str]
        :returns: Embedding matrix.
        :rtype: numpy.ndarray
        """
        return self.encode(texts)

    def _pool(self, hidden: object, attention_mask: object) -> object:
        """Pool token hidden states according to the configured strategy.

        :param hidden: Model hidden states of shape ``(B, T, D)``.
        :type hidden: torch.Tensor
        :param attention_mask: Token mask of shape ``(B, T)``.
        :type attention_mask: torch.Tensor
        :returns: Pooled embeddings of shape ``(B, D)``.
        :rtype: torch.Tensor
        """
        torch = self._torch
        if self.pooling == "cls":
            return hidden[:, 0, :]
        if self.pooling == "last_token":
            if bool(attention_mask[:, -1].all().item()):
                return hidden[:, -1, :]
            last_index = attention_mask.sum(dim=1).to(dtype=torch.long) - 1
            last_index = last_index.clamp(min=0)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[rows, last_index, :]

        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
