"""Probe whether closed-loop SFT actually changes model capability.

Reads a closed-loop run directory and asks two questions per round:

**Tier 1 — does the training loss decrease?** Pulls the per-step SFT loss
history that the closed-loop dispatcher logs into ``history.json`` under
``agent_sft_logs`` and plots one loss curve per agent per round. If the
loss is flat the LoRA step isn't learning anything regardless of routing
rule.

**Tier 3 — do the agents actually differentiate?** For each saved
per-round adapter, computes the cross-perplexity matrix

.. math::

    P[i, j, r] \\;=\\; \\text{mean NLL of agent }i\\text{'s round-}r\\text{ model}
    \\text{ on agent }j\\text{'s round-}r\\text{ training prompts.}

If specialisation is real, the diagonal entries :math:`P[i, i, r]` should
be lower than the off-diagonals :math:`P[i, j, r]` for :math:`j \\neq i`.
The headline summary number is the **cross-batch margin** averaged across
all off-diagonal pairs:

.. math::

    \\mu(r) \\;=\\; \\text{mean}_{i \\neq j}\\,P[i, j, r] \\;-\\;
                  \\text{mean}_i\\,P[i, i, r].

Positive :math:`\\mu(r)` means agents have specialised to their own batches.
Near-zero :math:`\\mu(r)` means SFT-driven capability change is small enough
that the model is essentially interchangeable across agents — i.e., the
position dynamics measured by the rest of the pipeline are pure
routing-centroid geometry rather than capability drift.

Run with::

    python scripts/probe_sft_capability.py \\
        --run-dir results/safety_truth_n4_r10_strategic \\
        --output-stem scripts/figures/probe_strategic_seed0

Requirements
------------
- ``closed_loop.save_per_round: true`` was set in the original run config
  (otherwise only the last round's adapter is available).
- ``transformers``, ``peft``, ``torch`` installed in the active env.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = ROOT / "scripts" / "figures"


# -----------------------------------------------------------------------------
# Data structures and helpers
# -----------------------------------------------------------------------------

def _load_history(run_dir: Path) -> list[dict]:
    """Load ``history.json`` and validate it has the per-round fields we need.

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
            "logs agent_prompts / agent_sft_logs per round."
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


def _adapter_path(run_dir: Path, agent: str, round_idx: int, base_sft_dir: Path) -> Optional[Path]:
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
    :returns: Path to the adapter dir, or ``None`` if not found.
    :rtype: pathlib.Path | None
    """
    candidates = [
        base_sft_dir / agent / f"round-{round_idx:02d}",
        run_dir / "agents" / agent / f"round-{round_idx:02d}",
    ]
    if round_idx == len(_HISTORY_LEN_HACK) - 1:
        # The last round's adapter is always saved at the flat path too
        # (sft_train_agent's default behaviour). Fall back to that for runs
        # that pre-date the per-round flag.
        candidates.extend([base_sft_dir / agent, run_dir / "agents" / agent])
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    return None


# Module-level mutable used by _adapter_path to detect "last round" fallback.
# Set by probe_run() before any path resolution.
_HISTORY_LEN_HACK: list = []


# -----------------------------------------------------------------------------
# NLL computation
# -----------------------------------------------------------------------------

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
    """Compute mean per-token NLL over a sequence of pre-formatted strings.

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
        # Mask pad tokens so they don't contribute to the NLL.
        labels[enc["attention_mask"] == 0] = -100
        with torch.no_grad():
            out = model(**enc, labels=labels)
        # HF returns mean cross-entropy over non-ignored tokens; recover
        # the sum by multiplying back.
        n_tokens = int((labels != -100).sum().item())
        if n_tokens == 0:
            continue
        total_nll += float(out.loss.item()) * n_tokens
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("nan"), 0
    return total_nll / total_tokens, total_tokens


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def _load_base(model_name: str):
    """Load the base causal LM once.

    Tries ``dtype=`` (transformers >= 5) then ``torch_dtype=`` (older).

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


# -----------------------------------------------------------------------------
# Probe core
# -----------------------------------------------------------------------------

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
    :param max_prompts_per_batch: Optional cap on the number of prompts
        used per agent batch (subsample for speed). ``None`` uses all.
    :type max_prompts_per_batch: int | None
    :param max_seq_length: Tokeniser truncation length.
    :type max_seq_length: int
    :param forward_batch_size: Inference batch size.
    :type forward_batch_size: int
    :returns: Per-(round, i, j) records.
    :rtype: list[dict]
    """
    records = _load_history(run_dir)
    names = _agent_names(records)
    global _HISTORY_LEN_HACK
    _HISTORY_LEN_HACK = records  # used by _adapter_path's last-round fallback

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

        # Build agent_j's batch text for each j
        texts_by_agent: dict[str, list[str]] = {}
        for j_name in names:
            p_j = list(prompts_by_agent.get(j_name, []))
            r_j = list(responses_by_agent.get(j_name, []))
            if max_prompts_per_batch is not None and len(p_j) > max_prompts_per_batch:
                idx = rng.choice(len(p_j), size=max_prompts_per_batch, replace=False)
                p_j = [p_j[k] for k in idx]
                r_j = [r_j[k] for k in idx] if r_j else []
            r_j = r_j if r_j and any(r_j) else [None] * len(p_j)
            texts_by_agent[j_name] = [_format_chat(p, r) for p, r in zip(p_j, r_j)]

        print(f"\n=== round {r} ===")
        for i_name in names:
            adapter = _adapter_path(run_dir, i_name, r, base_sft_dir)
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
                    model, tokenizer, texts,
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
                print(f"   {tag} {i_name} on {j_name}: NLL = {nll:.4f}  "
                      f"({len(texts)} prompts, {n_tok} tokens)")

            # Free the adapter wrapper but keep the base model resident.
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_probe(
    records: list[dict],
    history: list[dict],
    names: list[str],
    *,
    title: Optional[str] = None,
):
    """Render Tier 1 (SFT loss curves) and Tier 3 (cross-NLL matrix + margin).

    :param records: Output of :func:`probe_run`.
    :type records: list[dict]
    :param history: Loaded ``history.json`` records.
    :type history: list[dict]
    :param names: Agent names.
    :type names: list[str]
    :param title: Optional figure suptitle.
    :type title: str | None
    :returns: Matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n_agents = len(names)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(n_agents, 3)))

    # Reshape records into a (T, N, N) NLL tensor for plotting.
    rounds = sorted({rec["round"] for rec in records})
    name_to_idx = {n: i for i, n in enumerate(names)}
    nll_mat = np.full((len(rounds), n_agents, n_agents), np.nan)
    for rec in records:
        ri = rounds.index(rec["round"])
        i = name_to_idx[rec["agent_i"]]
        j = name_to_idx[rec["agent_j"]]
        nll_mat[ri, i, j] = rec["nll"]

    margins = cross_batch_margin(records)

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # --- (a) Tier 1: per-agent SFT loss curves -------------------------------
    # X-axis is round number (not cumulative step) so that agents with
    # different routing shares still contribute one point per round and
    # all curves span the full run. The shaded band shows min/max loss
    # within the round; the solid line is the round-mean. Asymmetric
    # routing (e.g. winners in a (2, 2) equilibrium getting ~2× the queries
    # of losers) shows up as a wider band on the high-throughput agents
    # but otherwise leaves all four curves visible from round 0 to N-1.
    ax = fig.add_subplot(gs[0, :])
    any_curve = False
    for i, name in enumerate(names):
        rs: list[int] = []
        means: list[float] = []
        mins: list[float] = []
        maxes: list[float] = []
        n_steps_per_round: list[int] = []
        for rec in history:
            r_idx = int(rec["round"])
            logs = rec.get("agent_sft_logs", {}).get(name, [])
            losses = [float(e["loss"]) for e in logs if "loss" in e]
            if not losses:
                # Fallback: trl's final summary if logging_steps was coarse.
                for e in logs:
                    if "train_loss" in e:
                        losses.append(float(e["train_loss"]))
                        break
            if losses:
                rs.append(r_idx)
                means.append(float(np.mean(losses)))
                mins.append(float(np.min(losses)))
                maxes.append(float(np.max(losses)))
                n_steps_per_round.append(len(losses))
        if rs:
            any_curve = True
            ax.plot(rs, means, "-o", color=colors[i], lw=1.8, mfc="white",
                    mec=colors[i], label=f"{name} (≈{np.mean(n_steps_per_round):.0f} steps/rd)")
            ax.fill_between(rs, mins, maxes, color=colors[i], alpha=0.18,
                            linewidth=0)
    ax.set_xlabel("round")
    ax.set_ylabel("train loss (per-round mean; band = min/max within round)")
    if any_curve:
        ax.set_title("Tier 1  —  per-round SFT training loss "
                     "(legend shows mean optimiser steps per round)")
        ax.legend(loc="best", fontsize=8, frameon=True)
    else:
        ax.set_title("Tier 1  —  no per-agent loss records found "
                     "(set sft.logging_steps to 1 in your config)")
    ax.grid(True, alpha=0.3)

    # --- (b) Tier 3 (left): cross-NLL heatmap at final round -----------------
    ax2 = fig.add_subplot(gs[1, 0])
    final_mat = nll_mat[-1]
    if not np.all(np.isnan(final_mat)):
        vmin, vmax = np.nanmin(final_mat), np.nanmax(final_mat)
        im = ax2.imshow(final_mat, cmap="viridis", vmin=vmin, vmax=vmax,
                        aspect="auto")
        for ii in range(n_agents):
            for jj in range(n_agents):
                v = final_mat[ii, jj]
                if np.isnan(v):
                    continue
                txt = f"{v:.3f}"
                ax2.text(jj, ii, txt, ha="center", va="center",
                         color="white" if v < (vmin + vmax) / 2 else "black",
                         fontsize=9)
        plt.colorbar(im, ax=ax2, label="mean NLL / token")
    ax2.set_xticks(range(n_agents))
    ax2.set_xticklabels([n.replace("clone-", "c") for n in names])
    ax2.set_yticks(range(n_agents))
    ax2.set_yticklabels([n.replace("clone-", "c") for n in names])
    ax2.set_xlabel("agent j's batch")
    ax2.set_ylabel("agent i's model")
    ax2.set_title(f"Tier 3a  —  cross-NLL at round {rounds[-1]}")

    # --- (c) Tier 3 (middle): diagonal vs off-diagonal over rounds -----------
    ax3 = fig.add_subplot(gs[1, 1])
    rs = sorted(margins.keys())
    d_means = [margins[r]["diag_mean"] for r in rs]
    o_means = [margins[r]["off_mean"] for r in rs]
    ax3.plot(rs, d_means, "-o", color="tab:blue", lw=1.6, label="diag (own batch)")
    ax3.plot(rs, o_means, "-s", color="tab:red", lw=1.6, label="off-diag (others' batches)")
    ax3.set_xlabel("round")
    ax3.set_ylabel("mean NLL / token")
    ax3.set_title("Tier 3b  —  fit on own vs others' batches")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=9)

    # --- (d) Tier 3 (right): cross-batch margin per round --------------------
    ax4 = fig.add_subplot(gs[1, 2])
    margin_vals = [margins[r]["margin"] for r in rs]
    bars = ax4.bar(rs, margin_vals, color="tab:green",
                   edgecolor="black", linewidth=0.5)
    for r_, m in zip(rs, margin_vals):
        ax4.text(r_, m + (0.001 if m >= 0 else -0.003),
                 f"{m:.3f}", ha="center", fontsize=8,
                 va="bottom" if m >= 0 else "top")
    ax4.axhline(0, color="black", lw=0.8)
    ax4.set_xlabel("round")
    ax4.set_ylabel(r"$\mu(r)$ = NLL(others) − NLL(own)")
    ax4.set_title("Tier 3c  —  specialisation margin\n(positive = specialised)")
    ax4.grid(True, axis="y", alpha=0.3)

    if title is not None:
        fig.suptitle(title, fontsize=12)
    return fig


# -----------------------------------------------------------------------------
# CSV
# -----------------------------------------------------------------------------

def _write_csv(records: list[dict], path: Path) -> None:
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
                rec["round"], rec["agent_i"], rec["agent_j"],
                f"{rec['nll']:.6f}", rec["n_prompts"], rec["n_tokens"],
            ])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    :returns: Configured parser.
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="Probe SFT capability change across rounds via "
                    "train-loss curves and cross-perplexity."
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Closed-loop run directory containing history.json.")
    p.add_argument("--base-sft-dir", type=Path, default=None,
                   help="closed_loop.sft.output_dir from the original run "
                        "config. Defaults to <run-dir>/agents.")
    p.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HuggingFace model id for the base LM.")
    p.add_argument("--rounds", type=int, nargs="*", default=None,
                   help="Subset of rounds to probe (default: all).")
    p.add_argument("--max-prompts", type=int, default=64,
                   help="Cap on prompts per agent batch (subsample for speed).")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--forward-batch-size", type=int, default=8)
    p.add_argument("--output-stem", type=Path, default=None,
                   help="Output stem for CSV/PDF/PNG. Defaults to "
                        "scripts/figures/probe_<run-name>.")
    p.add_argument("--title", type=str, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    :param argv: Optional CLI argument vector.
    :type argv: list[str] | None
    :returns: Process exit code.
    :rtype: int
    """
    args = _build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    base_sft = (args.base_sft_dir or (run_dir / "agents")).resolve()

    records = probe_run(
        run_dir,
        base_sft_dir=base_sft,
        base_model=args.base_model,
        rounds=args.rounds,
        max_prompts_per_batch=args.max_prompts,
        max_seq_length=args.max_seq_length,
        forward_batch_size=args.forward_batch_size,
    )
    if not records:
        print("error: probe produced no records (no per-round adapters found?)",
              file=sys.stderr)
        return 1

    history = _load_history(run_dir)
    names = _agent_names(history)

    # Print headline summary
    margins = cross_batch_margin(records)
    print(f"\n{'round':>6} {'diag NLL':>12} {'off NLL':>12} {'margin':>10}")
    print("-" * 44)
    for r in sorted(margins):
        m = margins[r]
        print(f"{r:>6} {m['diag_mean']:>12.4f} {m['off_mean']:>12.4f} "
              f"{m['margin']:>10.4f}")
    last_r = max(margins)
    print(f"\nFinal-round margin = {margins[last_r]['margin']:.4f}")
    if margins[last_r]["margin"] < 0.005:
        print("  → SFT-driven specialisation is negligible "
              "(< 0.5% of an NLL unit).")
    elif margins[last_r]["margin"] < 0.05:
        print("  → SFT-driven specialisation is small but non-zero.")
    else:
        print("  → SFT-driven specialisation is substantial.")

    stem = args.output_stem or (FIGS_DIR / f"probe_{run_dir.name}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    _write_csv(records, csv_path)
    print(f"\nwrote {csv_path}")

    fig = plot_probe(records, history, names, title=args.title)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
