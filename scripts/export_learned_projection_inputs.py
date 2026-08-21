"""Export prompt embeddings and per-prompt per-agent NLL for the learned-projection prototype.

Builds the ``(M, D)`` embedding matrix and ``(M, N)`` NLL matrix expected by
:mod:`scripts.prototype_learned_trait_projection` from round-39 specialist
adapters on the four safety benchmarks.

Example (on doob)::

    python scripts/export_learned_projection_inputs.py \\
        --run-dir results/ai4privacy_fixed_theory_specialists_r40/seed0 \\
        --round 39 \\
        --output-dir results/learned_projection_inputs/seed0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from infl_ens.data import SentenceTransformerEncoder
from infl_ens.evaluation.adapters import load_adapter_model, load_base_causal_lm
from infl_ens.evaluation.benchmarks import load_benchmark_splits, subsample_split
from infl_ens.evaluation.metrics import split_to_texts


BENCHMARKS = [
    {"kind": "beavertails", "path": "data/beavertails/30k_train.jsonl",
     "max_records": 5000},
    {"kind": "halueval", "path": "data/halueval",
     "tasks": ["qa", "dialogue"], "max_records": 5000},
    {"kind": "toxicchat", "path": "data/toxicchat",
     "score_mode": "jailbreaking", "human_annotated_only": False,
     "max_records": 5000},
    {"kind": "ai4privacy", "path": "data/ai4privacy",
     "score_mode": "density", "english_only": True, "max_records": 5000},
]


def _per_example_nll(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    device,
) -> np.ndarray:
    """Mean per-token NLL for each example, shape ``(len(texts),)``.

    :param model: HF causal LM (optionally with PEFT adapter).
    :type model: transformers.PreTrainedModel
    :param tokenizer: Tokenizer aligned with ``model``.
    :type tokenizer: transformers.PreTrainedTokenizer
    :param texts: Pre-formatted chat strings.
    :type texts: Sequence[str]
    :param max_length: Maximum sequence length.
    :type max_length: int
    :param batch_size: Forward micro-batch size.
    :type batch_size: int
    :param device: Torch device.
    :type device: torch.device
    :returns: Per-example mean token NLL.
    :rtype: numpy.ndarray
    """
    import torch

    out = np.empty(len(texts), dtype=float)
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
            logits = model(**enc).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(len(chunk), -1)
        mask = shift_labels != -100
        denom = mask.sum(dim=1).clamp(min=1)
        per_ex = (per_token * mask).sum(dim=1) / denom
        out[start : start + len(chunk)] = per_ex.float().cpu().numpy()
    return out


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    :param argv: Optional argument vector.
    :type argv: list[str] | None
    :returns: Exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Closed-loop run root with agents/clone-*.")
    parser.add_argument("--round", type=int, default=39,
                        help="Adapter round index (default: 39).")
    parser.add_argument("--agents", nargs="+",
                        default=["clone-0", "clone-1", "clone-2", "clone-3"],
                        help="Agent subdirectory names.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-eval-records", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--forward-batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    splits = load_benchmark_splits(BENCHMARKS)
    eval_splits = [
        subsample_split(s, args.max_eval_records, seed=args.seed)
        for s in splits
    ]
    prompts: list[str] = []
    benchmark_ids: list[int] = []
    for k, split in enumerate(eval_splits):
        prompts.extend(split.prompts)
        benchmark_ids.extend([k] * len(split.prompts))
    texts = []
    for split in eval_splits:
        texts.extend(split_to_texts(split))

    encoder = SentenceTransformerEncoder(args.encoder)
    emb = np.asarray(encoder(prompts), dtype=float)

    base, tokenizer, device = load_base_causal_lm(args.base_model)
    nll_cols = []
    for agent in args.agents:
        adapter_dir = args.run_dir / "agents" / agent / f"round-{args.round:02d}"
        model = load_adapter_model(base, adapter_dir)
        try:
            nll_cols.append(
                _per_example_nll(
                    model, tokenizer, texts,
                    max_length=args.max_seq_length,
                    batch_size=args.forward_batch_size,
                    device=device,
                )
            )
        finally:
            import torch
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    nll = np.stack(nll_cols, axis=1)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "prompt_embeddings.npy"
    nll_path = out_dir / "per_agent_nll.npy"
    np.save(emb_path, emb)
    np.save(nll_path, nll)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir.resolve()),
        "round": args.round,
        "agents": args.agents,
        "encoder": args.encoder,
        "n_prompts": len(prompts),
        "n_agents": len(args.agents),
        "benchmark_ids": benchmark_ids,
        "benchmarks": BENCHMARKS,
        "max_eval_records": args.max_eval_records,
        "seed": args.seed,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    oracle = float(nll.min(axis=1).mean())
    het = float(nll.mean() - oracle)
    print(f"wrote {emb_path}  shape={emb.shape}")
    print(f"wrote {nll_path}  shape={nll.shape}")
    print(f"wrote {meta_path}")
    print(f"heterogeneity gap (preview): {het:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
