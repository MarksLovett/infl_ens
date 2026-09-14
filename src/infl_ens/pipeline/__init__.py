"""The experiment pipeline: one command from raw data to figures.

``python -m infl_ens.pipeline --config configs/experiments/<name>.yaml`` runs
the stages of :mod:`infl_ens.pipeline.stages` in canonical order. Behavioral
generation runs after routing and before figures when requested. Progress is
recorded under the experiment's ``results_dir``.
"""

from __future__ import annotations

from infl_ens.pipeline.stages import STAGES, PipelineContext, run_pipeline, run_smoke

__all__ = ["STAGES", "PipelineContext", "run_pipeline", "run_smoke"]
