"""Resolve and load saved LoRA adapters for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

PathLike = Union[str, Path]

_ADAPTER_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def is_adapter_dir(path: Path) -> bool:
    """Return whether ``path`` looks like a PEFT LoRA checkpoint directory.

    :param path: Candidate adapter directory.
    :type path: pathlib.Path
    :returns: ``True`` if any standard adapter weight file exists.
    :rtype: bool
    """
    if not path.is_dir():
        return False
    return any((path / name).exists() for name in _ADAPTER_FILES)


def latest_round_dir(agent_dir: Path, round_idx: int) -> Optional[Path]:
    """Return the adapter directory in effect for ``agent_dir`` at ``round_idx``.

    Adapters are cumulative: an agent that received no data in a round
    keeps the weights of its last trained round. The directory in effect at
    round ``r`` is therefore ``round-r`` when it exists, else the newest
    ``round-NN`` with ``NN < r``.

    :param agent_dir: ``<run>/agents/<name>``.
    :type agent_dir: pathlib.Path
    :param round_idx: Requested round index.
    :type round_idx: int
    :returns: The adapter directory, or ``None`` when the agent has no
        adapter at or before ``round_idx``.
    :rtype: pathlib.Path | None
    """
    exact = agent_dir / f"round-{round_idx:02d}"
    if is_adapter_dir(exact):
        return exact
    if not agent_dir.is_dir():
        return None
    best: Optional[tuple[int, Path]] = None
    for child in agent_dir.iterdir():
        if not child.name.startswith("round-") or not is_adapter_dir(child):
            continue
        try:
            idx = int(child.name.split("-", 1)[1])
        except ValueError:
            continue
        if idx <= round_idx and (best is None or idx > best[0]):
            best = (idx, child)
    return best[1] if best is not None else None


def resolve_adapter_dir(path: PathLike) -> Path:
    """Validate and return an adapter directory.

    :param path: Directory containing ``adapter_model.safetensors`` or
        ``adapter_model.bin``.
    :type path: str | pathlib.Path
    :returns: Resolved absolute path.
    :rtype: pathlib.Path
    :raises FileNotFoundError: If ``path`` is missing or not an adapter dir.
    """
    p = Path(path).resolve()
    if not is_adapter_dir(p):
        raise FileNotFoundError(
            f"no LoRA adapter weights found under {p} "
            f"(expected one of {_ADAPTER_FILES})"
        )
    return p


@dataclass(frozen=True)
class AdapterRef:
    """A discovered adapter checkpoint under a closed-loop run.

    :param agent: Agent name (e.g. ``clone-0``).
    :type agent: str
    :param round: Training round index, or ``None`` for the flat
        ``agents/<agent>/`` layout.
    :type round: int | None
    :param path: Directory containing LoRA weights.
    :type path: pathlib.Path
    """

    agent: str
    round: Optional[int]
    path: Path


def discover_adapters(
    run_dir: PathLike,
    *,
    agents: Optional[Sequence[str]] = None,
    rounds: Optional[Sequence[int]] = None,
) -> list[AdapterRef]:
    """List saved adapters under ``<run_dir>/agents/``.

    Supports per-round layouts (``agents/<name>/round-NN``) and the flat
    layout (``agents/<name>/``) used when ``save_per_round`` is false.

    :param run_dir: Closed-loop or SFT run root.
    :type run_dir: str | pathlib.Path
    :param agents: Optional subset of agent names. ``None`` scans every
        child of ``agents/``.
    :type agents: Sequence[str] | None
    :param rounds: Optional subset of round indices for per-round dirs.
        ``None`` includes every ``round-*`` subdirectory plus flat dirs.
        When given, each requested round resolves to the adapter *in
        effect* at that round (:func:`latest_round_dir`): an agent not
        trained in round ``r`` is reported at ``r`` with the newest earlier
        adapter, so a per-round table keeps every agent in every column.
    :type rounds: Sequence[int] | None
    :returns: Discovered adapters sorted by agent then round.
    :rtype: list[AdapterRef]
    """
    root = Path(run_dir).resolve()
    agents_root = root / "agents"
    if not agents_root.is_dir():
        return []

    want_agents = set(agents) if agents is not None else None
    want_rounds = sorted(set(int(r) for r in rounds)) if rounds is not None else None
    found: list[AdapterRef] = []

    for agent_dir in sorted(agents_root.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent = agent_dir.name
        if want_agents is not None and agent not in want_agents:
            continue

        if want_rounds is not None:
            for r_idx in want_rounds:
                rd = latest_round_dir(agent_dir, r_idx)
                if rd is not None:
                    found.append(AdapterRef(agent=agent, round=r_idx, path=rd))
            continue

        round_dirs = sorted(
            p for p in agent_dir.iterdir()
            if p.is_dir() and p.name.startswith("round-")
        )
        if round_dirs:
            for rd in round_dirs:
                try:
                    r_idx = int(rd.name.split("-", 1)[1])
                except (IndexError, ValueError):
                    continue
                if is_adapter_dir(rd):
                    found.append(AdapterRef(agent=agent, round=r_idx, path=rd))
        elif is_adapter_dir(agent_dir):
            found.append(AdapterRef(agent=agent, round=None, path=agent_dir))

    return found


def load_base_causal_lm(
    base_model: str,
    *,
    universal_adapter_dir: Optional[PathLike] = None,
):
    """Load a base causal LM and tokenizer for inference.

    Heavy dependencies are imported lazily. Dtype selection prefers
    bfloat16 on CUDA and float32 elsewhere.

    When ``universal_adapter_dir`` is given, that LoRA is merged into the
    base weights **once**, here, in float32 before the cast (mirroring the
    training-side merge in
    :func:`infl_ens.training.sft_training.merge_frozen_adapter`), so every
    adapter later attached with :func:`load_adapter_model` sits on top of
    ``base + universal``. Merging in this function rather than in
    :func:`load_adapter_model` keeps the universal delta from being applied
    once per domain adapter.

    :param base_model: HuggingFace model id.
    :type base_model: str
    :param universal_adapter_dir: Optional frozen universal LoRA to merge
        into the base.
    :type universal_adapter_dir: str | pathlib.Path | None
    :returns: Tuple ``(model, tokenizer, device)``.
    :rtype: tuple
    :raises ImportError: If ``torch`` or ``transformers`` are missing.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    load_dtype = torch.float32 if universal_adapter_dir is not None else dtype
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=load_dtype)
    except TypeError:  # pragma: no cover - old transformers
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=load_dtype,
        )
    if universal_adapter_dir is not None:
        model = merge_universal_adapter(model, universal_adapter_dir, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def merge_universal_adapter(model, adapter_dir: PathLike, *, dtype):
    """Fold a LoRA adapter into ``model`` and cast the result to ``dtype``.

    :param model: Causal LM (float32 recommended) to merge into.
    :type model: transformers.PreTrainedModel
    :param adapter_dir: LoRA checkpoint directory.
    :type adapter_dir: str | pathlib.Path
    :param dtype: Torch dtype of the returned model.
    :type dtype: torch.dtype
    :returns: Plain (non-PEFT) merged model.
    :rtype: transformers.PreTrainedModel
    :raises FileNotFoundError: If ``adapter_dir`` holds no adapter weights.
    """
    from peft import PeftModel

    path = resolve_adapter_dir(adapter_dir)
    wrapped = PeftModel.from_pretrained(model, str(path))
    merged = wrapped.merge_and_unload()
    return merged.to(dtype)


def load_adapter_model(base_model, adapter_dir: PathLike):
    """Wrap ``base_model`` with a PEFT adapter from disk.

    :param base_model: Base causal LM from :func:`load_base_causal_lm`.
    :type base_model: transformers.PreTrainedModel
    :param adapter_dir: LoRA checkpoint directory.
    :type adapter_dir: str | pathlib.Path
    :returns: ``PeftModel`` in eval mode on the same device as the base.
    :rtype: peft.PeftModel
    """
    from peft import PeftModel

    path = resolve_adapter_dir(adapter_dir)
    wrapped = PeftModel.from_pretrained(base_model, str(path))
    wrapped.eval()
    return wrapped
