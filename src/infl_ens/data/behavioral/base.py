"""Typed records for generation-based behavioral evaluation.

Behavioral records deliberately keep model-visible messages separate from
private scoring payloads.  This prevents a router or target model from
accidentally consuming benchmark labels, references, or judge metadata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ChatMessage:
    """One model-visible chat message.

    :param role: Chat role such as ``system``, ``user`` or ``assistant``.
    :type role: str
    :param content: Message text.
    :type content: str
    """

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported chat role {self.role!r}")
        if not self.content:
            raise ValueError("chat message content must not be empty")

    def to_dict(self) -> dict[str, str]:
        """Return a Transformers-compatible message dictionary.

        :returns: Message mapping with ``role`` and ``content``.
        :rtype: dict[str, str]
        """
        return {"role": self.role, "content": self.content}


def stable_case_id(suite: str, messages: Sequence[ChatMessage]) -> str:
    """Derive a deterministic ID from a suite and model-visible messages.

    :param suite: Behavioral-suite identifier.
    :type suite: str
    :param messages: Ordered model-visible messages.
    :type messages: Sequence[ChatMessage]
    :returns: Stable ``<suite>:<sha256-prefix>`` identifier.
    :rtype: str
    """
    payload = json.dumps(
        [message.to_dict() for message in messages],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{suite}:{hashlib.sha256(payload).hexdigest()[:20]}"


@dataclass(frozen=True)
class BehavioralCase:
    """One immutable behavioral-evaluation case.

    :param suite: Suite identifier.
    :type suite: str
    :param case_id: Stable identifier within the suite.
    :type case_id: str
    :param messages: Messages visible to both router and target model.
    :type messages: tuple[ChatMessage, ...]
    :param category: Public benchmark category used only for aggregation.
    :type category: str
    :param metadata: Non-secret provenance metadata.
    :type metadata: Mapping[str, Any]
    :param scoring_payload: References and labels visible only to graders.
    :type scoring_payload: Mapping[str, Any]
    """

    suite: str
    case_id: str
    messages: tuple[ChatMessage, ...]
    category: str = "all"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    scoring_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.suite or not self.case_id:
            raise ValueError("suite and case_id must not be empty")
        if not self.messages:
            raise ValueError("behavioral case needs at least one message")
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("behavioral case needs at least one user message")

    @property
    def routing_text(self) -> str:
        """Serialize only model-visible messages for trait projection.

        :returns: Role-prefixed visible transcript.
        :rtype: str
        """
        return "\n".join(message.content for message in self.messages)

    def visible_dict(self) -> dict[str, Any]:
        """Return the label-free payload allowed into generation code.

        :returns: Case identifier, suite and model-visible messages.
        :rtype: dict[str, Any]
        """
        return {
            "suite": self.suite,
            "case_id": self.case_id,
            "messages": [message.to_dict() for message in self.messages],
        }


@dataclass(frozen=True)
class BehavioralSuite:
    """A pinned behavioral benchmark release.

    :param name: Suite identifier.
    :type name: str
    :param version: Dataset revision, tag or checksum label.
    :type version: str
    :param source: Upstream source identifier.
    :type source: str
    :param cases: Ordered evaluation cases.
    :type cases: tuple[BehavioralCase, ...]
    :param metadata: Suite-level metadata.
    :type metadata: Mapping[str, Any]
    """

    name: str
    version: str
    source: str
    cases: tuple[BehavioralCase, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError(f"behavioral suite {self.name!r} has no cases")
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError(f"behavioral suite {self.name!r} has duplicate case IDs")
        if any(case.suite != self.name for case in self.cases):
            raise ValueError("every case suite must match BehavioralSuite.name")

    def take(self, n: int | None) -> "BehavioralSuite":
        """Return the first ``n`` cases without changing their identifiers.

        :param n: Case cap, or ``None`` for the complete suite.
        :type n: int | None
        :returns: Capped suite.
        :rtype: BehavioralSuite
        """
        if n is None:
            return self
        if n <= 0:
            raise ValueError("case cap must be > 0")
        return BehavioralSuite(
            name=self.name,
            version=self.version,
            source=self.source,
            cases=self.cases[:n],
            metadata=dict(self.metadata),
        )


__all__ = [
    "BehavioralCase",
    "BehavioralSuite",
    "ChatMessage",
    "stable_case_id",
]
