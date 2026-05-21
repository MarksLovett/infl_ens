"""Training subpackage: router-position training and LoRA SFT.

Public surface:

- :class:`RouterTrainingConfig`, :func:`train_router_positions` from
  :mod:`infl_ens.training.router_training`.
- :class:`SFTTrainingConfig`, :func:`sft_train_agent` from
  :mod:`infl_ens.training.sft_training` (lazy: importing this module
  does not import the SFT stack).

Both trainers are reachable through the single CLI
``python -m infl_ens.training`` (see :mod:`infl_ens.training.__main__`).
"""

from __future__ import annotations

from infl_ens.training.router_training import (
    RouterTrainingConfig,
    train_router_positions,
)


def __getattr__(name: str):
    """Lazy import for the SFT trainer to avoid pulling in torch/transformers."""
    if name in ("SFTTrainingConfig", "sft_train_agent"):
        from infl_ens.training import sft_training
        return getattr(sft_training, name)
    raise AttributeError(name)


__all__ = [
    "RouterTrainingConfig",
    "SFTTrainingConfig",
    "sft_train_agent",
    "train_router_positions",
]
