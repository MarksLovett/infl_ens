"""The experiment pipeline: one command from raw data to figures.

``python -m infl_ens.pipeline --config configs/experiments/<name>.yaml`` runs
the stages of :mod:`infl_ens.pipeline.stages` in order (``manifest`` →
``train`` → ``perround`` → ``routing`` → ``figures``; ``download`` and
``prune`` on request), calling the package APIs directly and recording
progress under the experiment's ``results_dir``.
"""

from __future__ import annotations

from infl_ens.pipeline.stages import STAGES, PipelineContext, run_pipeline, run_smoke

__all__ = ["STAGES", "PipelineContext", "run_pipeline", "run_smoke"]
