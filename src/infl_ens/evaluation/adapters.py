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
    :type rounds: Sequence[int] | None
    :returns: Discovered adapters sorted by agent then round.
    :rtype: list[AdapterRef]
    """
    root = Path(run_dir).resolve()
    agents_root = root / "agents"
    if not agents_root.is_dir():
        return []

    want_agents = set(agents) if agents is not None else None
    want_rounds = set(int(r) for r in rounds) if rounds is not None else None
    found: list[AdapterRef] = []

    for agent_dir in sorted(agents_root.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent = agent_dir.name
        if want_agents is not None and agent not in want_agents:
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
                if want_rounds is not None and r_idx not in want_rounds:
                    continue
                if is_adapter_dir(rd):
                    found.append(AdapterRef(agent=agent, round=r_idx, path=rd))
        elif is_adapter_dir(agent_dir):
            if want_rounds is None:
                found.append(AdapterRef(agent=agent, round=None, path=agent_dir))

    return found


def load_base_causal_lm(base_model: str):
    """Load a base causal LM and tokenizer for inference.

    Heavy dependencies are imported lazily. Dtype selection mirrors
    :mod:`infl_ens.evaluation.capability_probe`.

    :param base_model: HuggingFace model id.
    :type base_model: str
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
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    except TypeError:  # pragma: no cover - old transformers
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype,
        )
    model.to(device)
    model.eval()
    return model, tokenizer, device


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
