"""AI-safety benchmark loaders and learned trait-space construction.

This subpackage groups one module per benchmark used as a trait-space
axis, plus a combined builder. Public surface:

- :class:`BenchmarkSplit`: uniform container of (prompt, score) records.
- :func:`load_beavertails`: BeaverTails loader (harm axis).
- :func:`load_halueval`: HaluEval loader (hallucination axis).
- :func:`load_jbb_behaviors`: JBB-Behaviors loader (jailbreak axis).
- :func:`load_toxicchat`: ToxicChat loader (legacy jailbreak axis).
- :func:`load_ai4privacy`: AI4Privacy PII loader (privacy-density axis).
- :func:`load_orbench`: OR-Bench loader (over-refusal axis).
- :func:`load_prompt_injection`: prompt-injection loader (injection axis).
- :func:`load_do_not_answer`: Do-Not-Answer loader (policy-violation axis).
- :func:`build_safety_trait_space`: combined :math:`N`-axis trait space.
- :class:`LearnedAxis`: a single scoring axis fit from labelled prompts.
- :data:`BEAVERTAILS_CATEGORIES`, :data:`HALUEVAL_TASKS`,
  :data:`TOXICCHAT_SCORE_MODES`, :data:`PII_SCORE_MODES`,
  :data:`ORBENCH_CONFIGS`: dataset taxonomy constants exposed for downstream
  filtering.

All loaders return :class:`BenchmarkSplit` instances so downstream code
(trait-space builders, trainers, evaluation scripts) handles every
benchmark through one interface.
"""

from __future__ import annotations

from infl_ens.data.benchmarks.base import BenchmarkSplit
from infl_ens.data.benchmarks.beavertails import (
    BEAVERTAILS_CATEGORIES,
    load_beavertails,
)
from infl_ens.data.benchmarks.halueval import (
    HALUEVAL_TASKS,
    load_halueval,
)
from infl_ens.data.benchmarks.jbb_behaviors import load_jbb_behaviors
from infl_ens.data.benchmarks.toxicchat import (
    TOXICCHAT_SCORE_MODES,
    load_toxicchat,
)
from infl_ens.data.benchmarks.ai4privacy import (
    PII_SCORE_MODES,
    load_ai4privacy,
)
from infl_ens.data.benchmarks.orbench import (
    ORBENCH_CONFIGS,
    load_orbench,
)
from infl_ens.data.benchmarks.prompt_injection import load_prompt_injection
from infl_ens.data.benchmarks.do_not_answer import load_do_not_answer
from infl_ens.data.benchmarks.safety_trait_space import (
    LearnedAxis,
    build_safety_trait_space,
)

__all__ = [
    "BEAVERTAILS_CATEGORIES",
    "BenchmarkSplit",
    "HALUEVAL_TASKS",
    "LearnedAxis",
    "ORBENCH_CONFIGS",
    "PII_SCORE_MODES",
    "TOXICCHAT_SCORE_MODES",
    "build_safety_trait_space",
    "load_ai4privacy",
    "load_beavertails",
    "load_do_not_answer",
    "load_halueval",
    "load_jbb_behaviors",
    "load_orbench",
    "load_prompt_injection",
    "load_toxicchat",
]
