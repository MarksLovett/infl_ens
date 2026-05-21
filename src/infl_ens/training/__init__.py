"""Training entry points and trainers.

Per AGENTS.md §4 rule 1 there is a single training CLI:

.. code-block:: bash

   python -m infl_ens.training --config <path>

which dispatches on the config's ``task`` field. The router trainer
(gradient ascent on agent positions) is exported *eagerly* because it
has no heavy dependencies. The SFT helpers are exported *lazily* — they
appear in ``dir(infl_ens.training)`` and can be imported by name, but
the underlying :mod:`infl_ens.training.sft_training` module (which pulls
in :mod:`torch`, :mod:`transformers`, :mod:`peft`, :mod:`trl`) is only
loaded the first time the attribute is accessed.

Public surface
--------------

Eager:

- :class:`RouterTrainingConfig`, :func:`train_router_positions` from
  :mod:`infl_ens.training.router_training`.

Lazy:

- :class:`SFTTrainingConfig`, :func:`sft_train_agent` from
  :mod:`infl_ens.training.sft_training`.

Both trainers are reachable through the single CLI; see
:mod:`infl_ens.training.__main__` for the dispatch table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from infl_ens.training.router_training import (
    RouterTrainingConfig,
    train_router_positions,
)

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers / IDEs
    from infl_ens.training.sft_training import (
        SFTTrainingConfig,
        sft_train_agent,
    )

#: Names that are resolved through :func:`__getattr__` on first access.
_LAZY_SFT_NAMES: frozenset[str] = frozenset({"SFTTrainingConfig", "sft_train_agent"})

__all__ = [
    "RouterTrainingConfig",
    "SFTTrainingConfig",
    "sft_train_agent",
    "train_router_positions",
]


def __getattr__(name: str) -> Any:
    """Resolve lazy SFT exports on first access.

    Importing :mod:`infl_ens.training.sft_training` pulls in
    :mod:`torch` and :mod:`transformers`; this proxy defers that cost
    until the attribute is actually requested, then caches it in the
    module globals so subsequent accesses are O(1).

    :param name: Attribute name being looked up on the module.
    :type name: str
    :returns: The requested attribute from
              :mod:`infl_ens.training.sft_training`.
    :rtype: Any
    :raises AttributeError: If ``name`` is not a recognised lazy export.
    """
    if name in _LAZY_SFT_NAMES:
        from infl_ens.training import sft_training
        value = getattr(sft_training, name)
        globals()[name] = value  # cache for subsequent accesses
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy SFT names to :func:`dir` and shell tab-completion.

    :returns: Sorted list of public attribute names, including the lazy
              SFT exports that have not yet been materialised.
    :rtype: list[str]
    """
    return sorted(set(globals()) | _LAZY_SFT_NAMES)
