"""Common interface for AI-safety benchmark adapters.

Each benchmark module under :mod:`infl_ens.data.benchmarks` provides a
loader that returns a :class:`BenchmarkSplit`: a uniform container of
(prompt, score) records plus benchmark-level metadata. The score is the
benchmark's *axis* in the influencer-game trait space:

- :mod:`infl_ens.data.benchmarks.beavertails` → ``harm`` score in ``[0, 1]``,
  where 1 means "the response is harmful". The trait-space axis is
  "harmfulness of prompts the agent serves".
- :mod:`infl_ens.data.benchmarks.halueval` → ``hallucination`` score in
  ``[0, 1]``, where 1 means "the response contains hallucinations".

The combined two-axis trait space is built by
:func:`infl_ens.data.benchmarks.safety_trait_space.build_safety_trait_space`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class BenchmarkSplit:
    """A loaded benchmark split in a single uniform shape.

    :param name: Benchmark identifier (e.g. ``'beavertails'``,
        ``'halueval-qa'``).
    :type name: str
    :param prompts: Raw user-facing prompts, length ``N``. These are what
        the trait-space encoder will embed.
    :type prompts: list[str]
    :param scores: Per-prompt benchmark axis score in ``[0, 1]``, length
        ``N``. Higher means "more of the property the benchmark measures"
        (e.g. more harmful, more hallucinated).
    :type scores: numpy.ndarray
    :param responses: Optional per-prompt response strings (used at
        SFT time). Length ``N`` or empty.
    :type responses: list[str]
    :param axis_name: Human-readable axis name, e.g. ``'harm'`` or
        ``'hallucination'``.
    :type axis_name: str
    :param metadata: Free-form metadata (split name, version, source URL).
    :type metadata: dict
    """

    name: str
    prompts: list[str]
    scores: np.ndarray
    axis_name: str
    responses: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        s = np.asarray(self.scores, dtype=float)
        if s.shape != (len(self.prompts),):
            raise ValueError(
                f"scores must have shape ({len(self.prompts)},), got {s.shape}"
            )
        if ((s < 0.0) | (s > 1.0)).any():
            raise ValueError("scores must lie in [0, 1]")
        # Frozen dataclass: route through object.__setattr__.
        object.__setattr__(self, "scores", s)
        if self.responses and len(self.responses) != len(self.prompts):
            raise ValueError(
                f"responses length {len(self.responses)} != "
                f"prompts length {len(self.prompts)}"
            )

    @property
    def n(self) -> int:
        """Number of records.

        :returns: Length ``N`` of the split.
        :rtype: int
        """
        return len(self.prompts)

    def take(self, indices: Sequence[int]) -> "BenchmarkSplit":
        """Return a sub-split selecting the rows in ``indices``.

        :param indices: Integer row indices.
        :type indices: Sequence[int]
        :returns: New :class:`BenchmarkSplit` with the selected rows.
        :rtype: BenchmarkSplit
        """
        idx = np.asarray(list(indices), dtype=int)
        return BenchmarkSplit(
            name=self.name,
            prompts=[self.prompts[i] for i in idx],
            scores=self.scores[idx],
            axis_name=self.axis_name,
            responses=(
                [self.responses[i] for i in idx] if self.responses else []
            ),
            metadata=dict(self.metadata),
        )
