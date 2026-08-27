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

Theory-vs-SFT comparison helpers live in
:mod:`infl_ens.training.theory_vs_sft` (not re-exported at package import
time; import by submodule to avoid pulling trait-space builders eagerly).

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
dispatcher uses this independence to give gradient-matched position drift
by default without paying the LoRA ESS cost.

The unweighted path (``sample_weights=None``) is preserved
byte-for-byte: no behaviour change for existing callers. Implementation
details (``_WeightedLMCollator``, ``_build_weighted_sft_trainer_class``)
are private to :mod:`infl_ens.training.sft_training`.

Closed-loop dispatcher knobs
----------------------------

The ``closed_loop`` task in :mod:`infl_ens.training.__main__` exposes
four orthogonal "rule" knobs in the YAML config:

- ``closed_loop.routing_weight``: ``'G'`` (canonical, Lovett & Fu 2024)
  or ``'G_times_1mG'`` (strategic).
- ``closed_loop.routing_mode``: ``'hard'`` (sample one agent per query)
  or ``'soft'`` (assign each query to its top-``soft_top_k`` agents;
  ``closed_loop.soft_loss`` = ``'weighted'`` share-weighted loss or
  ``'unit'`` unit-weight "top-k winners").
- ``closed_loop.position_update``: centroid mass of the position step.
  ``'theory_matched'`` (**default**) makes the expected trait-space drift
  proportional to the strategic gradient coefficient :math:`G_i(1-G_i)`
  in every routing mode — ``(1 - G_i)`` under hard canonical routing,
  uniform under strategic routing, and the dense :math:`G_i(1-G_i)` mass
  over the whole batch under soft routing
  (:func:`infl_ens.inflgame.router.allocation.matched_centroid_mass`),
  independent of ``soft_top_k``. ``'naive'`` keeps the historical
  uninstrumented centroid as an ablation arm.
- ``closed_loop.loss_reweight``: loss-side weighting under hard routing —
  ``null`` (unit) or ``'one_minus_G'`` (:math:`w_m = 1 - G_i(\\mathbf{x},
  b_m)` on the SFT loss). ``'position_only'`` is a deprecated alias for
  ``null`` + ``position_update: theory_matched``.
- ``closed_loop.save_per_round``: per-round adapter archiving for
  :mod:`infl_ens.evaluation.capability_probe`.
- ``closed_loop.position_step``: adaptive EMA blend for trait-space
  position updates (:mod:`infl_ens.training.position_step`).

The matrix of gradient-aligned modes (``position_update: theory_matched``
unless stated) is:

+--------------+------------------+----------------------+--------------------------------+
| routing_mode | routing_weight   | loss_reweight        | meaning                        |
+==============+==================+======================+================================+
| ``hard``     | ``G``            | ``null``             | unit loss, ``(1-G)`` centroid  |
|              |                  |                      | (:math:`= \\nabla u_i` in      |
|              |                  |                      | position; full LoRA ESS)       |
+--------------+------------------+----------------------+--------------------------------+
| ``hard``     | ``G_times_1mG``  | ``null``             | strategic routing              |
|              |                  |                      | (:math:`\\approx \\nabla u_i`)  |
+--------------+------------------+----------------------+--------------------------------+
| ``hard``     | ``G``            | ``one_minus_G``      | full reweight                  |
|              |                  |                      | (:math:`= \\nabla u_i`         |
|              |                  |                      | exactly; reduced LoRA ESS)     |
+--------------+------------------+----------------------+--------------------------------+
| ``soft``     | ``G``            | ``null``             | dense ``G(1-G)`` centroid over |
|              |                  |                      | the whole batch                |
|              |                  |                      | (:math:`= \\nabla u_i` exactly, |
|              |                  |                      | any ``soft_top_k``)            |
+--------------+------------------+----------------------+--------------------------------+
| ``hard``     | ``G``            | ``null`` + ``naive`` | canonical naive (uniform       |
|              |                  |                      | centroid; no correction)       |
+--------------+------------------+----------------------+--------------------------------+
| ``soft``     | ``G``            | ``null`` + ``naive`` | renormalised-share centroid    |
|              |                  |                      | (NOT gradient matched)         |
+--------------+------------------+----------------------+--------------------------------+
| ``hard``     | ``G_times_1mG``  | ``one_minus_G``      | rejected (double-counts the    |
|              |                  |                      | (1-G) factor)                  |
+--------------+------------------+----------------------+--------------------------------+

See :mod:`infl_ens.inflgame.router.verification` for the numerical alignment
check that distinguishes the "≈" and "=" cases (including the soft
top-``k`` columns), and :mod:`scripts.compare_routing_ess` for the per-round
ESS gap diagnostic that motivates decoupling the loss from the centroid.
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
