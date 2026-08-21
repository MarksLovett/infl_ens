"""Probe whether closed-loop SFT changes model capability across rounds.

Computes per-round cross-perplexity matrices and cross-batch specialisation
margins from ``history.json`` and saved per-round adapters.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _load_probe_history(run_dir: Path) -> list[dict]:
    """Load ``history.json`` and validate per-round probe fields.

    :param run_dir: Closed-loop run directory.
    :type run_dir: pathlib.Path
    :returns: Per-round records.
    :rtype: list[dict]
    :raises ValueError: If required fields are missing.
    """
    path = run_dir / "history.json"
    with path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    if not records:
        raise ValueError(f"{path} is empty")
    required = {"round", "agent_prompts", "agent_sft_logs"}
    missing = required - set(records[0])
    if missing:
        raise ValueError(
            f"{path} is missing required fields {missing}. "
            "Re-run with closed_loop.save_per_round: true so the dispatcher "
            "logs agent_prompts / agent_sft_logs per round.",
        )
    return records


def _agent_names(records: Sequence[dict]) -> list[str]:
    """Return agent names in their canonical order.

    :param records: Loaded history.
    :type records: Sequence[dict]
    :returns: Agent names.
    :rtype: list[str]
    """
    return list(records[0]["positions"].keys())


def _adapter_path_for_round(
    run_dir: Path,
    agent: str,
    round_idx: int,
    base_sft_dir: Path,
    *,
    n_history_rounds: int,
) -> Optional[Path]:
    """Locate a per-round adapter directory if present.

    Tries the per-round layout (``<sft>/<agent>/round-NN``) first, then the
    flat layout (``<sft>/<agent>``) used by old runs.

    :param run_dir: Closed-loop run root.
    :type run_dir: pathlib.Path
    :param agent: Agent name.
    :type agent: str
    :param round_idx: Round number.
    :type round_idx: int
    :param base_sft_dir: ``closed_loop.sft.output_dir`` from the original
        config (usually ``<run_dir>/agents``).
    :type base_sft_dir: pathlib.Path
    :param n_history_rounds: Total rounds in ``history.json`` (for last-round fallback).
    :type n_history_rounds: int
    :returns: Path to the adapter dir, or ``None`` if not found.
    :rtype: pathlib.Path | None
    """
    candidates = [
        base_sft_dir / agent / f"round-{round_idx:02d}",
        run_dir / "agents" / agent / f"round-{round_idx:02d}",
    ]
    if round_idx == n_history_rounds - 1:
        candidates.extend([base_sft_dir / agent, run_dir / "agents" / agent])
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    return None


def _format_chat(prompt: str, response: Optional[str]) -> str:
    """Match :func:`infl_ens.training.sft_training._format_chat`.

    :param prompt: User prompt.
    :type prompt: str
    :param response: Optional assistant response.
    :type response: str | None
    :returns: Chat-formatted string used during training.
    :rtype: str
    """
    if response:
        return (
            "<|im_start|>user\n" + prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n" + response + "<|im_end|>"
        )
    return "<|im_start|>user\n" + prompt + "<|im_end|>"


def _compute_nll_batch(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    device,
) -> tuple[float, int]:
    """Compute mean per-token NLL over pre-formatted strings.

    :param model: HF causal LM (with adapter loaded).
    :type model: transformers.PreTrainedModel
    :param tokenizer: Tokenizer.
    :type tokenizer: transformers.PreTrainedTokenizer
    :param texts: Chat-formatted training strings.
    :type texts: Sequence[str]
    :param max_length: Max tokens per example.
    :type max_length: int
    :param batch_size: Forward-pass batch size.
    :type batch_size: int
    :param device: Torch device.
    :type device: torch.device
    :returns: Tuple ``(mean_nll_per_token, total_tokens)``.
    :rtype: tuple[float, int]
    """
    import torch

    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start:start + batch_size])
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
    if total_tokens == 0:
        return float("nan"), 0
    return total_nll / total_tokens, total_tokens


def _load_base(model_name: str):
    """Load the base causal LM once.

    :param model_name: HuggingFace model id.
    :type model_name: str
    :returns: Tuple ``(model, tokenizer, device)``.
    :rtype: tuple
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (
        device.type == "cuda" and torch.cuda.is_bf16_supported()
    ) else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:  # pragma: no cover - old transformers
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _wrap_with_adapter(base_model, adapter_path: Path):
    """Return a PeftModel wrapping ``base_model`` with the given adapter.

    :param base_model: Base causal LM.
    :type base_model: transformers.PreTrainedModel
    :param adapter_path: Directory containing ``adapter_model.safetensors``.
    :type adapter_path: pathlib.Path
    :returns: ``PeftModel`` ready for inference.
    :rtype: peft.PeftModel
    """
    from peft import PeftModel
    return PeftModel.from_pretrained(base_model, str(adapter_path))


def probe_run(
    run_dir: Path,
    base_sft_dir: Path,
    base_model: str,
    *,
    rounds: Optional[Sequence[int]] = None,
    max_prompts_per_batch: Optional[int] = None,
    max_seq_length: int = 1024,
    forward_batch_size: int = 8,
) -> list[dict]:
    """Compute the cross-perplexity matrix per round.

    :param run_dir: Closed-loop run directory.
    :type run_dir: pathlib.Path
    :param base_sft_dir: ``closed_loop.sft.output_dir`` used in the run.
    :type base_sft_dir: pathlib.Path
    :param base_model: HF base-model identifier.
    :type base_model: str
    :param rounds: Optional subset of round indices to probe. ``None``
        means all rounds present in ``history.json``.
    :type rounds: Sequence[int] | None
    :param max_prompts_per_batch: Optional cap on prompts per agent batch.
    :type max_prompts_per_batch: int | None
    :param max_seq_length: Tokeniser truncation length.
    :type max_seq_length: int
    :param forward_batch_size: Inference batch size.
    :type forward_batch_size: int
    :returns: Per-(round, i, j) records.
    :rtype: list[dict]
    """
    records = _load_probe_history(run_dir)
    names = _agent_names(records)
    n_history_rounds = len(records)

    target_rounds = (
        sorted({int(r["round"]) for r in records}) if rounds is None
        else sorted(set(int(r) for r in rounds))
    )

    print(f"loading base model: {base_model}")
    base_model_obj, tokenizer, device = _load_base(base_model)
    rng = np.random.default_rng(0)

    results: list[dict] = []
    for r in target_rounds:
        rec = next((x for x in records if int(x["round"]) == r), None)
        if rec is None:
            continue
        prompts_by_agent = rec.get("agent_prompts", {})
        responses_by_agent = rec.get("agent_responses", {})

        texts_by_agent: dict[str, list[str]] = {}
        for j_name in names:
            p_j = list(prompts_by_agent.get(j_name, []))
            r_j = list(responses_by_agent.get(j_name, []))
            if max_prompts_per_batch is not None and len(p_j) > max_prompts_per_batch:
                idx = rng.choice(len(p_j), size=max_prompts_per_batch, replace=False)
                p_j = [p_j[k] for k in idx]
                r_j = [r_j[k] for k in idx] if r_j else []
            r_j = r_j if r_j and any(r_j) else [None] * len(p_j)
            texts_by_agent[j_name] = [_format_chat(p, resp) for p, resp in zip(p_j, r_j)]

        print(f"\n=== round {r} ===")
        for i_name in names:
            adapter = _adapter_path_for_round(
                run_dir,
                i_name,
                r,
                base_sft_dir,
                n_history_rounds=n_history_rounds,
            )
            if adapter is None:
                print(f"  [{i_name}] no adapter found for round {r}; skipping")
                continue
            print(f"  [{i_name}] loading adapter from {adapter}")
            try:
                model = _wrap_with_adapter(base_model_obj, adapter)
            except Exception as exc:
                print(f"  [{i_name}] failed to load adapter: {exc}")
                continue
            model.eval()

            for j_name in names:
                texts = texts_by_agent[j_name]
                if not texts:
                    continue
                nll, n_tok = _compute_nll_batch(
                    model,
                    tokenizer,
                    texts,
                    max_length=max_seq_length,
                    batch_size=forward_batch_size,
                    device=device,
                )
                results.append({
                    "round": r,
                    "agent_i": i_name,
                    "agent_j": j_name,
                    "nll": nll,
                    "n_prompts": len(texts),
                    "n_tokens": n_tok,
                })
                tag = "*" if i_name == j_name else " "
                print(
                    f"   {tag} {i_name} on {j_name}: NLL = {nll:.4f}  "
                    f"({len(texts)} prompts, {n_tok} tokens)",
                )

            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def cross_batch_margin(records: list[dict]) -> dict[int, dict]:
    """Compute the round-wise cross-batch specialisation margin.

    :param records: Output of :func:`probe_run`.
    :type records: list[dict]
    :returns: ``{round: {"diag_mean", "off_mean", "margin"}}``.
    :rtype: dict
    """
    by_round: dict[int, list[dict]] = {}
    for rec in records:
        by_round.setdefault(rec["round"], []).append(rec)
    out: dict[int, dict] = {}
    for r, recs in by_round.items():
        diag = [x["nll"] for x in recs if x["agent_i"] == x["agent_j"]]
        off = [x["nll"] for x in recs if x["agent_i"] != x["agent_j"]]
        if not diag or not off:
            continue
        d_mean = float(np.mean(diag))
        o_mean = float(np.mean(off))
        out[r] = {
            "diag_mean": d_mean,
            "off_mean": o_mean,
            "margin": o_mean - d_mean,
            "n_diag": len(diag),
            "n_off": len(off),
        }
    return out


def write_probe_csv(records: list[dict], path: Path) -> None:
    """Write per-(round, i, j) NLL records to CSV.

    :param records: Output of :func:`probe_run`.
    :type records: list[dict]
    :param path: Output CSV path.
    :type path: pathlib.Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "agent_i", "agent_j", "nll", "n_prompts", "n_tokens"])
        for rec in records:
            w.writerow([
                rec["round"],
                rec["agent_i"],
                rec["agent_j"],
                f"{rec['nll']:.6f}",
                rec["n_prompts"],
                rec["n_tokens"],
            ])
