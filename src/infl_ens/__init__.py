"""Influencer-game router and learning algorithms for small language models.

This package implements the *influencer's game* (Lovett & Fu, 2024) and
extends it to align small language models (SLMs) as the learning agents.
The proportional-allocation rule

.. math::

   G_i(\\mathbf{x}, b) \\;=\\; \\frac{f_i(x_i, b)}{\\sum_{j=1}^{N} f_j(x_j, b)}

routes queries to candidate models via a multivariate-Gaussian influence
kernel on a trait space constructed automatically from a calibration
corpus.

Subpackages are imported lazily — ``import infl_ens`` does *not* pull in
:mod:`torch`, :mod:`transformers`, or any other heavy dependency. Import
the specific subpackage you need:

- :mod:`infl_ens.config`     — layered YAML loading shared by every CLI
- :mod:`infl_ens.experiment` — experiment files (arms, stages, analysis settings)
- :mod:`infl_ens.data`       — trait spaces, encoders, benchmark loaders
- :mod:`infl_ens.inflgame`   — game environment (router, allocation math)
- :mod:`infl_ens.training`   — the closed loop and the pooled replay
- :mod:`infl_ens.evaluation` — adapter scoring and route-then-score
- :mod:`infl_ens.figures`    — figures and tables of an experiment
- :mod:`infl_ens.pipeline`   — the end-to-end experiment runner
- :mod:`infl_ens.utils`      — cross-cutting helpers

The experiment runner is ``python -m infl_ens.pipeline --config <experiment>``;
the per-stage CLIs are ``infl_ens.training``, ``infl_ens.evaluation`` and
``infl_ens.figures`` (see AGENTS.md rule 1).

References
----------
Lovett, M. & Fu, X. (2024). *Learning Dynamics of the Influencer's Game
in Resource Landscapes.*
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
