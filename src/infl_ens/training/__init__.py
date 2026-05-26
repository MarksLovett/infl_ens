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

Per-example loss + centroid weighting
-------------------------------------

:func:`sft_train_agent` accepts two optional keyword-only arrays:

- ``sample_weights``: per-prompt loss weights. When provided, the
  trainer switches to a pre-tokenised dataset path and uses a custom
  collator
  (:class:`infl_ens.training.sft_training._WeightedLMCollator`) plus a
  :class:`trl.SFTTrainer` subclass whose ``compute_loss`` emits the
  weight-normalised mean of per-example response-token cross-entropy.
- ``eval_weights``: forwarded as ``scores`` to
  :meth:`infl_ens.inflgame.router.RouterAgent.update_position_from_corpus`
  so the post-SFT trait-space position is the weighted centroid of the
  routed corpus.

The two are **independent**: you can weight the loss but not the
centroid, or weight the centroid but not the loss. The closed-loop
dispatcher uses this independence to expose the ``'position_only'``
mode that gives gradient-matched position drift without paying the LoRA
ESS cost.

The unweighted path (``sample_weights=None``) is preserved
byte-for-byte: no behaviour change for existing callers. Implementation
details (``_WeightedLMCollator``, ``_build_weighted_sft_trainer_class``)
are private to :mod:`infl_ens.training.sft_training`.

Closed-loop dispatcher knobs
----------------------------

The ``closed_loop`` task in :mod:`infl_ens.training.__main__` exposes
three orthogonal "rule" knobs in the YAML config:

- ``closed_loop.routing_weight``: ``'G'`` (canonical, Lovett & Fu 2024)
  or ``'G_times_1mG'`` (strategic).
- ``closed_loop.loss_reweight``: where to apply the per-query weight
  :math:`w_m = 1 - G_i(\\mathbf{x}, b_m)`:

  - ``null``: nowhere (uniform loss + uniform centroid).
  - ``'one_minus_G'``: both the SFT loss AND the centroid update.
  - ``'position_only'``: centroid update ONLY; SFT loss is unit-weight.

- ``closed_loop.save_per_round``: per-round adapter archiving for
  :mod:`scripts.probe_sft_capability`.
- ``closed_loop.position_step``: adaptive EMA blend for trait-space
  position updates (:mod:`infl_ens.utils.position_step`).

The matrix of gradient-aligned modes is:

+------------------+----------------------+--------------------------------+
| routing_weight   | loss_reweight        | meaning                        |
+==================+======================+================================+
| ``G``            | ``null``             | canonical naive (no            |
|                  |                      | gradient correction)           |
+------------------+----------------------+--------------------------------+
| ``G_times_1mG``  | ``null``             | strategic routing              |
|                  |                      | (:math:`\\approx \\nabla u_i`)  |
+------------------+----------------------+--------------------------------+
| ``G``            | ``one_minus_G``      | full reweight                  |
|                  |                      | (:math:`= \\nabla u_i`         |
|                  |                      | exactly; reduced LoRA ESS)     |
+------------------+----------------------+--------------------------------+
| ``G``            | ``position_only``    | decoupled reweight             |
|                  |                      | (:math:`= \\nabla u_i`         |
|                  |                      | in position; full LoRA ESS;    |
|                  |                      | LoRA sees canonical corpus     |
|                  |                      | including strongholds)         |
+------------------+----------------------+--------------------------------+
| ``G_times_1mG``  | ``one_minus_G`` /    | rejected (double-counts the    |
|                  | ``position_only``    | (1-G) factor)                  |
+------------------+----------------------+--------------------------------+

See :mod:`scripts.compare_reweighted_drift` for the numerical alignment
check that distinguishes the "≈" and "=" cases, and
:mod:`scripts.compare_routing_ess` for the per-round ESS gap diagnostic
that motivates the ``position_only`` mode.
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
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy SFT names to :func:`dir` and shell tab-completion.

    :returns: Sorted list of public attribute names.
    :rtype: list[str]
    """
    return sorted(set(globals()) | _LAZY_SFT_NAMES)
