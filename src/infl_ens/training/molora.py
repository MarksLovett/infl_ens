"""MoLoRA: a jointly trained mixture of low-rank experts (Zadouri et al., 2023).

The standard PEFT-MoE baseline for the influencer-game specialists.  Every
target linear layer :math:`W` of a frozen base model receives :math:`E`
rank-:math:`r` LoRA experts :math:`(A_e, B_e)` and a token-level linear
gate :math:`W_g`.  With :math:`s = \\mathrm{softmax}(W_g x)` the layer
computes

.. math::

    y \\;=\\; Wx \\;+\\; \\frac{\\alpha}{r} \\sum_{e=1}^{E} s_e\\, B_e A_e x .

Routing is **dense** (all experts active, weighted by the gate) and
learned jointly with the experts on the ordinary SFT loss; there is no
external router and no game.  With ``n_experts = 7`` and
``expert_rank = 16`` the expert weights match the parameter budget of the
seven rank-16 pair adapters, so the comparison isolates the routing and
training rule rather than capacity.

This module holds model code only: the wrapper, the injection helper, the
parity/diagnostic helpers and the save/load pair.  Training lives in
:mod:`infl_ens.training.molora_replay`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, Union

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn

PathLike = Union[str, Path]

#: Weight and config file names written by :func:`save_molora`.
MOLORA_WEIGHTS_FILE = "molora_weights.safetensors"
MOLORA_CONFIG_FILE = "molora_config.json"


@dataclass
class MoLoRAConfig:
    """Hyperparameters of the mixture of low-rank experts.

    :param n_experts: Number of LoRA experts :math:`E` per target module.
    :type n_experts: int
    :param expert_rank: LoRA rank :math:`r` of every expert.
    :type expert_rank: int
    :param expert_alpha: LoRA alpha; the expert output is scaled by
        ``expert_alpha / expert_rank`` (``32 / 16 = 2`` matches the
        specialists).
    :type expert_alpha: int
    :param expert_dropout: Dropout applied to the layer input before the
        expert path (the gate sees the same dropped input).
    :type expert_dropout: float
    :param target_modules: Names of the ``nn.Linear`` sub-modules to wrap.
    :type target_modules: tuple[str, ...]
    :param balance_coef: Weight of the Switch-style load-balance term added
        to the training loss.  ``0.0`` (default, the MoLoRA paper setting)
        disables it; the term is still logged.
    :type balance_coef: float
    """

    n_experts: int = 7
    expert_rank: int = 16
    expert_alpha: int = 32
    expert_dropout: float = 0.0
    target_modules: tuple[str, ...] = field(default_factory=lambda: (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ))
    balance_coef: float = 0.0

    def __post_init__(self) -> None:
        if self.n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {self.n_experts}")
        if self.expert_rank < 1:
            raise ValueError(f"expert_rank must be >= 1, got {self.expert_rank}")
        if self.expert_alpha <= 0:
            raise ValueError(f"expert_alpha must be > 0, got {self.expert_alpha}")
        if not 0.0 <= self.expert_dropout < 1.0:
            raise ValueError(f"expert_dropout must be in [0, 1), got {self.expert_dropout}")
        if self.balance_coef < 0.0:
            raise ValueError(f"balance_coef must be >= 0, got {self.balance_coef}")
        self.target_modules = tuple(str(m) for m in self.target_modules)


def _torch() -> tuple[Any, Any, Any]:
    """Import torch lazily so the package stays importable without it."""
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:  # pragma: no cover - environment-level
        raise ImportError("infl_ens.training.molora requires torch") from exc
    return torch, nn, F


def _make_molora_linear_class() -> type:
    """Build :class:`MoLoRALinear` against the installed ``torch.nn``."""
    torch, nn, F = _torch()

    class MoLoRALinear(nn.Module):
        """One frozen ``nn.Linear`` plus :math:`E` gated LoRA experts.

        Parameters (all trainable, kept in float32 like PEFT adapters):

        - ``lora_A``: ``(E, r, in_features)``, Kaiming-uniform initialised.
        - ``lora_B``: ``(E, out_features, r)``, zero initialised so the
          wrapped layer starts identical to the base.
        - ``gate``: ``(E, in_features)``, zero initialised so the gate starts
          uniform (each expert weighted :math:`1/E`).

        After every forward pass ``last_gate_usage`` holds the mean gate
        probability per expert over the tokens of that batch and
        ``last_balance`` the Switch-style balance term
        :math:`E \\sum_e \\bar{s}_e^2` (equal to 1 at uniform usage, up to
        :math:`E` under collapse).  The balance term keeps its graph so a
        trainer can add ``balance_coef * last_balance`` to the loss.

        :param base: The linear layer to wrap; frozen in place.
        :type base: torch.nn.Linear
        :param cfg: Expert / gate hyperparameters.
        :type cfg: MoLoRAConfig
        """

        def __init__(self, base: "nn.Linear", cfg: MoLoRAConfig) -> None:
            super().__init__()
            if not isinstance(base, nn.Linear):
                raise TypeError(f"MoLoRALinear wraps nn.Linear, got {type(base).__name__}")
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.n_experts = int(cfg.n_experts)
            self.rank = int(cfg.expert_rank)
            self.scaling = float(cfg.expert_alpha) / float(cfg.expert_rank)
            self.in_features = int(base.in_features)
            self.out_features = int(base.out_features)
            device = base.weight.device
            e, r = self.n_experts, self.rank
            self.lora_A = nn.Parameter(
                torch.empty(e, r, self.in_features, device=device, dtype=torch.float32),
            )
            self.lora_B = nn.Parameter(
                torch.zeros(e, self.out_features, r, device=device, dtype=torch.float32),
            )
            self.gate = nn.Parameter(
                torch.zeros(e, self.in_features, device=device, dtype=torch.float32),
            )
            self.dropout = (
                nn.Dropout(cfg.expert_dropout) if cfg.expert_dropout > 0.0 else nn.Identity()
            )
            self.reset_expert_parameters()
            self.last_gate_usage: "torch.Tensor | None" = None
            self.last_balance: "torch.Tensor | None" = None

        def reset_expert_parameters(self) -> None:
            """Kaiming-uniform ``lora_A`` (as PEFT does), zero ``lora_B`` / ``gate``."""
            import math

            for a in self.lora_A:
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            nn.init.zeros_(self.gate)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            result = self.base(x)
            xd = self.dropout(x).to(self.lora_A.dtype)
            e, r = self.n_experts, self.rank
            # Gate probabilities per token: (..., E).
            probs = F.softmax(F.linear(xd, self.gate).float(), dim=-1)
            # Expert inner activations: (..., E, r).
            h = torch.einsum("...i,eri->...er", xd, self.lora_A)
            gated = (h * probs.to(h.dtype).unsqueeze(-1)).reshape(*h.shape[:-2], e * r)
            # One matmul against B concatenated over experts: (out, E*r).
            b_cat = self.lora_B.permute(1, 0, 2).reshape(self.out_features, e * r)
            mixed = F.linear(gated, b_cat) * self.scaling
            # Diagnostics: mean usage per expert and the balance term.
            usage = probs.reshape(-1, e).mean(dim=0)
            self.last_gate_usage = usage.detach()
            self.last_balance = float(e) * (usage * usage).sum()
            return result + mixed.to(result.dtype)

        def extra_repr(self) -> str:
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"n_experts={self.n_experts}, rank={self.rank}, scaling={self.scaling}"
            )

    return MoLoRALinear


@lru_cache(maxsize=1)
def molora_linear_class() -> type:
    """Return the (lazily built, cached) :class:`MoLoRALinear` class.

    Built on first use so this module imports without ``torch``.

    :returns: The wrapper class bound to the installed ``torch``.
    :rtype: type
    """
    return _make_molora_linear_class()


def inject_molora(model: "nn.Module", cfg: MoLoRAConfig) -> list["nn.Module"]:
    """Freeze ``model`` and wrap every target ``nn.Linear`` with MoLoRA experts.

    Modules are matched on the last component of their qualified name
    (``model.layers.3.self_attn.q_proj`` matches ``q_proj``), the same rule
    PEFT uses for ``target_modules``.  The model is mutated in place; only
    the new expert and gate tensors are trainable afterwards.

    :param model: A causal LM (or any module tree) to wrap.
    :type model: torch.nn.Module
    :param cfg: Expert / gate hyperparameters.
    :type cfg: MoLoRAConfig
    :returns: The injected wrappers in module-tree order.
    :rtype: list[torch.nn.Module]
    :raises ValueError: If no module matched ``cfg.target_modules``.
    """
    _, nn, _ = _torch()
    cls = molora_linear_class()
    for p in model.parameters():
        p.requires_grad_(False)

    targets = set(cfg.target_modules)
    to_wrap: list[tuple[str, str, "nn.Linear"]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, cls):
            continue
        parent_name, _, attr = name.rpartition(".")
        if attr in targets:
            to_wrap.append((parent_name, attr, module))
    if not to_wrap:
        raise ValueError(
            f"no nn.Linear matched target_modules {sorted(targets)}"
        )

    wrapped: list["nn.Module"] = []
    for parent_name, attr, module in to_wrap:
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapper = cls(module, cfg)
        setattr(parent, attr, wrapper)
        wrapped.append(wrapper)
    return wrapped


def molora_modules(model: "nn.Module") -> list["nn.Module"]:
    """Every :class:`MoLoRALinear` in ``model``, in module-tree order.

    :param model: A module tree previously passed to :func:`inject_molora`.
    :type model: torch.nn.Module
    :returns: The wrappers.
    :rtype: list[torch.nn.Module]
    """
    cls = molora_linear_class()
    return [m for m in model.modules() if isinstance(m, cls)]


def molora_state_dict(model: "nn.Module") -> dict[str, "torch.Tensor"]:
    """Expert and gate tensors only, keyed by qualified parameter name.

    :param model: A module tree previously passed to :func:`inject_molora`.
    :type model: torch.nn.Module
    :returns: ``{name: tensor}`` for ``lora_A`` / ``lora_B`` / ``gate``.
    :rtype: dict[str, torch.Tensor]
    """
    cls = molora_linear_class()
    out: dict[str, Any] = {}
    for name, module in model.named_modules():
        if isinstance(module, cls):
            for key in ("lora_A", "lora_B", "gate"):
                out[f"{name}.{key}" if name else key] = getattr(module, key).detach().cpu()
    return out


def molora_param_count(model: "nn.Module") -> dict[str, int]:
    """Trainable parameter counts for the parity report.

    :param model: A module tree previously passed to :func:`inject_molora`.
    :type model: torch.nn.Module
    :returns: ``{"experts": ..., "gate": ..., "total": ...}``.
    :rtype: dict[str, int]
    """
    experts = 0
    gate = 0
    for m in molora_modules(model):
        experts += m.lora_A.numel() + m.lora_B.numel()
        gate += m.gate.numel()
    return {"experts": int(experts), "gate": int(gate), "total": int(experts + gate)}


def gate_usage(modules: Sequence["nn.Module"]) -> list[float] | None:
    """Mean gate probability per expert, averaged over ``modules``.

    :param modules: Wrappers that have run at least one forward pass.
    :type modules: Sequence[torch.nn.Module]
    :returns: Length-``E`` list, or ``None`` if no module has statistics.
    :rtype: list[float] | None
    """
    torch, _, _ = _torch()
    stats = [m.last_gate_usage for m in modules if m.last_gate_usage is not None]
    if not stats:
        return None
    return torch.stack([s.float().cpu() for s in stats]).mean(dim=0).tolist()


def balance_loss(modules: Sequence["nn.Module"]) -> "torch.Tensor | None":
    """Mean Switch-style balance term over ``modules`` (keeps the graph).

    :param modules: Wrappers that have run a forward pass in this step.
    :type modules: Sequence[torch.nn.Module]
    :returns: Scalar tensor, or ``None`` if no module has statistics.
    :rtype: torch.Tensor | None
    """
    torch, _, _ = _torch()
    terms = [m.last_balance for m in modules if m.last_balance is not None]
    if not terms:
        return None
    return torch.stack(terms).mean()


def save_molora(model: "nn.Module", cfg: MoLoRAConfig, out_dir: PathLike, *,
                base_model: str | None = None) -> Path:
    """Write the expert/gate tensors and the config to ``out_dir``.

    :param model: Injected model.
    :type model: torch.nn.Module
    :param cfg: The config used for injection.
    :type cfg: MoLoRAConfig
    :param out_dir: Destination directory (created if missing).
    :type out_dir: str | pathlib.Path
    :param base_model: Optional base-model id recorded in the config file.
    :type base_model: str | None
    :returns: The directory written.
    :rtype: pathlib.Path
    """
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.contiguous() for k, v in molora_state_dict(model).items()}
    save_file(tensors, str(out / MOLORA_WEIGHTS_FILE))
    meta: dict[str, Any] = asdict(cfg)
    meta["target_modules"] = list(cfg.target_modules)
    meta["base_model"] = base_model
    meta["format"] = "molora"
    (out / MOLORA_CONFIG_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def is_molora_dir(path: PathLike) -> bool:
    """Whether ``path`` holds a MoLoRA checkpoint written by :func:`save_molora`.

    :param path: Candidate directory.
    :type path: str | pathlib.Path
    :returns: ``True`` if both the weights and the config file exist.
    :rtype: bool
    """
    p = Path(path)
    return (p / MOLORA_WEIGHTS_FILE).is_file() and (p / MOLORA_CONFIG_FILE).is_file()


def load_molora_config(adapter_dir: PathLike) -> MoLoRAConfig:
    """Read the :class:`MoLoRAConfig` stored in a checkpoint directory.

    :param adapter_dir: Directory written by :func:`save_molora`.
    :type adapter_dir: str | pathlib.Path
    :returns: The parsed config.
    :rtype: MoLoRAConfig
    """
    meta = json.loads((Path(adapter_dir) / MOLORA_CONFIG_FILE).read_text(encoding="utf-8"))
    fields = {k: meta[k] for k in (
        "n_experts", "expert_rank", "expert_alpha", "expert_dropout",
        "target_modules", "balance_coef",
    ) if k in meta}
    return MoLoRAConfig(**fields)


def load_molora(model: "nn.Module", adapter_dir: PathLike) -> "nn.Module":
    """Inject experts into ``model`` and load a saved checkpoint into them.

    :param model: A freshly loaded base model (mutated in place).
    :type model: torch.nn.Module
    :param adapter_dir: Directory written by :func:`save_molora`.
    :type adapter_dir: str | pathlib.Path
    :returns: ``model`` with the experts loaded, in eval mode.
    :rtype: torch.nn.Module
    :raises FileNotFoundError: If the directory is not a MoLoRA checkpoint.
    :raises ValueError: If the checkpoint keys do not match the injected model.
    """
    from safetensors.torch import load_file

    p = Path(adapter_dir)
    if not is_molora_dir(p):
        raise FileNotFoundError(f"{p} is not a MoLoRA checkpoint")
    cfg = load_molora_config(p)
    inject_molora(model, cfg)
    tensors = load_file(str(p / MOLORA_WEIGHTS_FILE))
    expected = set(molora_state_dict(model))
    got = set(tensors)
    if expected != got:
        missing = sorted(expected - got)[:5]
        extra = sorted(got - expected)[:5]
        raise ValueError(
            f"MoLoRA checkpoint mismatch: missing {missing} extra {extra}"
        )
    model.load_state_dict(tensors, strict=False)
    model.eval()
    return model


__all__ = [
    "MOLORA_CONFIG_FILE",
    "MOLORA_WEIGHTS_FILE",
    "MoLoRAConfig",
    "balance_loss",
    "gate_usage",
    "inject_molora",
    "is_molora_dir",
    "load_molora",
    "load_molora_config",
    "molora_linear_class",
    "molora_modules",
    "molora_param_count",
    "molora_state_dict",
    "save_molora",
]
