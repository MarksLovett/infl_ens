"""Sentence-embedding wrappers for trait-space construction.

The router's :class:`infl_ens.data.trait_space.TraitSpace` takes a generic
``encoder: Callable[[Sequence[str]], np.ndarray]``. This module provides two
production-grade implementations:

- :class:`SentenceTransformerEncoder`: thin wrapper around the
  ``sentence-transformers`` library. Recommended default; the
  ``all-MiniLM-L6-v2`` model is small (~22M params), fast on CPU, and gives
  reasonable cosine geometry for short prompts.
- :class:`HuggingFaceEncoder`: mean-pooled hidden states from any HF
  causal/encoder model. Useful when you want the *agent's own* embeddings
  to define the trait space (e.g. Qwen2.5-1.5B), at higher cost.

Both classes are callable so they can be passed directly to
:func:`infl_ens.data.trait_space.build_trait_space`.

.. note::
    These imports are deferred so that the wider ``inflai`` package can be
    imported on machines without ``sentence-transformers`` / ``torch`` /
    ``transformers`` installed. ``ImportError`` is raised only on construction.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class SentenceTransformerEncoder:
    """Sentence-Transformers backend for :class:`TraitSpace` encoders.

    :param model_name: HuggingFace identifier of the sentence-transformer
        model. Defaults to ``'sentence-transformers/all-MiniLM-L6-v2'``,
        which is the standard small/fast pick.
    :type model_name: str
    :param device: Optional torch device string (``'cuda'``, ``'cpu'``,
        ``'cuda:0'``, etc.). If ``None``, sentence-transformers picks
        automatically.
    :type device: str | None
    :param batch_size: Encoder batch size.
    :type batch_size: int
    :param normalize: Whether to L2-normalise outputs at encode time.
        :func:`infl_ens.data.trait_space.build_trait_space` normalises again
        internally; setting ``True`` here just avoids the extra pass.
    :type normalize: bool
    :raises ImportError: If ``sentence-transformers`` is not installed.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - import-only error path
            raise ImportError(
                "SentenceTransformerEncoder requires sentence-transformers; "
                "install it with `pip install sentence-transformers`."
            ) from exc
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = int(batch_size)
        self._normalize = bool(normalize)
        self.model_name = model_name

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of texts.

        :param texts: Texts to embed.
        :type texts: Sequence[str]
        :returns: Embedding matrix, shape ``(len(texts), D)``.
        :rtype: numpy.ndarray
        """
        emb = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=float)


class HuggingFaceEncoder:
    """Mean-pooled HuggingFace-model embeddings for trait-space construction.

    Use this when you want the trait space to live in the agent's own
    representation geometry (e.g. embedding queries with Qwen2.5-1.5B's
    final hidden states). Slower and more memory-hungry than a dedicated
    sentence-transformer, but model-aware.

    :param model_name: HuggingFace model identifier (e.g.
        ``'Qwen/Qwen2.5-1.5B-Instruct'``).
    :type model_name: str
    :param device: Torch device string. ``None`` → auto.
    :type device: str | None
    :param batch_size: Tokenisation/forward batch size.
    :type batch_size: int
    :param max_length: Max tokens per input. Longer inputs are truncated.
    :type max_length: int
    :param dtype: Torch dtype string (``'float16'``, ``'bfloat16'``,
        ``'float32'``). Defaults to ``'float32'`` to avoid CPU/GPU dtype
        mismatches on small models.
    :type dtype: str
    :raises ImportError: If ``transformers`` / ``torch`` are not installed.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 256,
        dtype: str = "float32",
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HuggingFaceEncoder requires transformers and torch; "
                "install with `pip install transformers torch`."
            ) from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
        self._model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch_dtype,
        )
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        self._batch_size = int(batch_size)
        self._max_length = int(max_length)
        self.model_name = model_name

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of texts via attention-mask mean pooling.

        :param texts: Texts to embed.
        :type texts: Sequence[str]
        :returns: Embedding matrix, shape ``(len(texts), D)``.
        :rtype: numpy.ndarray
        """
        torch = self._torch
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self._batch_size):
                batch = list(texts[i:i + self._batch_size])
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**enc).last_hidden_state  # (B, T, D)
                mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                outs.append(pooled.float().cpu().numpy())
        return np.concatenate(outs, axis=0)
