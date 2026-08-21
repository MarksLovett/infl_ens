"""Perplexity-style metrics for adapter evaluation on benchmark corpora."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from infl_ens.data.benchmarks import BenchmarkSplit


def format_chat_example(prompt: str, response: Optional[str]) -> str:
    """Format one (prompt, response) pair as a Qwen2.5 chat string.

    Delegates to :func:`infl_ens.training.sft_training._format_chat` so eval
    NLL matches the SFT training objective byte-for-byte.

    :param prompt: User prompt.
    :type prompt: str
    :param response: Optional assistant response.
    :type response: str | None
    :returns: Single training/eval string.
    :rtype: str
    """
    from infl_ens.training.sft_training import _format_chat

    return _format_chat(prompt, response)


def split_to_texts(
    split: BenchmarkSplit,
    *,
    formatting_func: Optional[Callable[[str, Optional[str]], str]] = None,
) -> list[str]:
    """Convert a :class:`BenchmarkSplit` to chat-formatted strings.

    :param split: Loaded benchmark split.
    :type split: BenchmarkSplit
    :param formatting_func: Optional ``(prompt, response) -> str`` hook.
        Defaults to :func:`format_chat_example`.
    :type formatting_func: Callable[[str, str | None], str] | None
    :returns: One formatted string per row.
    :rtype: list[str]
    """
    fmt = formatting_func or format_chat_example
    if split.responses:
        return [
            fmt(p, r if r else None)
            for p, r in zip(split.prompts, split.responses)
        ]
    return [fmt(p, None) for p in split.prompts]


def mean_token_nll(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    device,
) -> tuple[float, int, int]:
    """Compute mean per-token negative log-likelihood over ``texts``.

    :param model: HF causal LM (optionally with a PEFT adapter).
    :type model: transformers.PreTrainedModel
    :param tokenizer: Tokenizer aligned with ``model``.
    :type tokenizer: transformers.PreTrainedTokenizer
    :param texts: Pre-formatted strings (e.g. from :func:`split_to_texts`).
    :type texts: Sequence[str]
    :param max_length: Maximum sequence length after tokenisation.
    :type max_length: int
    :param batch_size: Forward-pass micro-batch size.
    :type batch_size: int
    :param device: Torch device.
    :type device: torch.device
    :returns: Tuple ``(mean_nll_per_token, total_tokens, n_examples)``.
    :rtype: tuple[float, int, int]
    """
    import torch

    if not texts:
        return float("nan"), 0, 0

    total_nll = 0.0
    total_tokens = 0
    n_examples = 0
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        with torch.no_grad():
            out = model(**enc, labels=labels)
        n_tokens = int((labels != -100).sum().item())
        if n_tokens == 0:
            continue
        total_nll += float(out.loss.item()) * n_tokens
        total_tokens += n_tokens
        n_examples += len(chunk)

    if total_tokens == 0:
        return float("nan"), 0, n_examples
    return total_nll / total_tokens, total_tokens, n_examples
