"""Schema-versioned generation and grading artifacts for behavioral evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

BEHAVIORAL_ARTIFACT_SCHEMA_VERSION = 1


def canonical_digest(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-safe mapping using a canonical encoding.

    :param payload: JSON-safe mapping.
    :type payload: Mapping[str, Any]
    :returns: SHA-256 hexadecimal digest.
    :rtype: str
    """
    encoded = json.dumps(
        dict(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GenerationRecord:
    """One generated behavioral response plus routing/resource provenance.

    :param case_id: Stable benchmark case ID.
    :type case_id: str
    :param suite: Behavioral-suite name.
    :type suite: str
    :param target: Evaluated model/arm identifier.
    :type target: str
    :param protocol: Generation protocol.
    :type protocol: str
    :param messages: Model-visible messages.
    :type messages: tuple[dict[str, str], ...]
    :param response: Generated response text.
    :type response: str
    :param token_ids: Newly generated token IDs.
    :type token_ids: tuple[int, ...]
    :param router_weights: Prompt-level routing weights by expert name.
    :type router_weights: Mapping[str, float]
    :param selected_experts: Experts evaluated per decoding step.
    :type selected_experts: tuple[str, ...]
    :param trait_coordinates: Frozen-space prompt coordinates.
    :type trait_coordinates: tuple[float, ...]
    :param finish_reason: ``eos``, ``length`` or ``error``.
    :type finish_reason: str
    :param input_tokens: Number of prompt tokens.
    :type input_tokens: int
    :param output_tokens: Number of generated tokens.
    :type output_tokens: int
    :param model_forwards: Total model forwards used for this response.
    :type model_forwards: int
    :param latency_seconds: Wall-clock generation time.
    :type latency_seconds: float
    :param error: Infrastructure error text, if any.
    :type error: str | None
    """

    case_id: str
    suite: str
    target: str
    protocol: str
    messages: tuple[dict[str, str], ...]
    response: str
    token_ids: tuple[int, ...] = ()
    router_weights: Mapping[str, float] = field(default_factory=dict)
    selected_experts: tuple[str, ...] = ()
    trait_coordinates: tuple[float, ...] = ()
    finish_reason: str = "eos"
    input_tokens: int = 0
    output_tokens: int = 0
    model_forwards: int = 0
    latency_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe record.

        :returns: JSON-safe record mapping.
        :rtype: dict[str, Any]
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GenerationRecord":
        """Construct a record from a JSON mapping.

        :param payload: JSON record.
        :type payload: Mapping[str, Any]
        :returns: Parsed generation record.
        :rtype: GenerationRecord
        """
        return cls(
            case_id=str(payload["case_id"]),
            suite=str(payload["suite"]),
            target=str(payload["target"]),
            protocol=str(payload["protocol"]),
            messages=tuple(dict(value) for value in payload.get("messages", [])),
            response=str(payload.get("response", "")),
            token_ids=tuple(int(value) for value in payload.get("token_ids", [])),
            router_weights={
                str(key): float(value)
                for key, value in dict(payload.get("router_weights") or {}).items()
            },
            selected_experts=tuple(str(value) for value in payload.get("selected_experts", [])),
            trait_coordinates=tuple(float(value) for value in payload.get("trait_coordinates", [])),
            finish_reason=str(payload.get("finish_reason", "eos")),
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            model_forwards=int(payload.get("model_forwards", 0)),
            latency_seconds=float(payload.get("latency_seconds", 0.0)),
            error=(str(payload["error"]) if payload.get("error") is not None else None),
        )


def _records_digest(records: Sequence[GenerationRecord]) -> str:
    hasher = hashlib.sha256()
    for record in records:
        encoded = json.dumps(
            record.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        hasher.update(encoded)
        hasher.update(b"\n")
    return hasher.hexdigest()


@dataclass(frozen=True)
class GenerationArtifact:
    """Loaded behavioral generation artifact.

    :param manifest: Validated artifact manifest.
    :type manifest: Mapping[str, Any]
    :param records: Generated responses.
    :type records: tuple[GenerationRecord, ...]
    """

    manifest: Mapping[str, Any]
    records: tuple[GenerationRecord, ...]

    @property
    def digest(self) -> str:
        """Return the content-addressed generation digest.

        :returns: Generation digest.
        :rtype: str
        """
        return str(self.manifest["generation_digest"])

    @classmethod
    def load(cls, root: str | Path) -> "GenerationArtifact":
        """Load and validate ``generation_manifest.json`` and JSONL records.

        :param root: Artifact directory.
        :type root: str | pathlib.Path
        :returns: Loaded artifact.
        :rtype: GenerationArtifact
        :raises ValueError: If schema, counts or digests disagree.
        """
        directory = Path(root)
        manifest = json.loads(
            (directory / "generation_manifest.json").read_text(encoding="utf-8"),
        )
        if int(manifest.get("schema_version", 0)) != BEHAVIORAL_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported behavioral generation artifact schema")
        records: list[GenerationRecord] = []
        with (directory / "generations.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(GenerationRecord.from_dict(json.loads(line)))
        if len(records) != int(manifest.get("n_records", -1)):
            raise ValueError("generation artifact record count mismatch")
        if _records_digest(records) != manifest.get("records_sha256"):
            raise ValueError("generation artifact record digest mismatch")
        provenance = dict(manifest.get("provenance") or {})
        provenance_digest = canonical_digest(provenance)
        if provenance_digest != manifest.get("provenance_sha256"):
            raise ValueError("generation artifact provenance digest mismatch")
        content_digest = canonical_digest({
            "provenance_sha256": provenance_digest,
            "records_sha256": manifest["records_sha256"],
        })
        if content_digest != manifest.get("generation_digest"):
            raise ValueError("generation artifact content digest mismatch")
        expected_ids = provenance.get("ordered_case_ids")
        observed_ids = [record.case_id for record in records]
        if expected_ids is not None and list(expected_ids) != observed_ids:
            raise ValueError("generation artifact record order mismatch")
        return cls(manifest=manifest, records=tuple(records))


def write_generation_artifact(
    root: str | Path,
    records: Sequence[GenerationRecord],
    provenance: Mapping[str, Any],
) -> GenerationArtifact:
    """Write a generation artifact and immediately validate it.

    :param root: Artifact directory.
    :type root: str | pathlib.Path
    :param records: Generated records in ordered-case order.
    :type records: Sequence[GenerationRecord]
    :param provenance: JSON-safe generation provenance.
    :type provenance: Mapping[str, Any]
    :returns: Reloaded artifact.
    :rtype: GenerationArtifact
    """
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    records_path = directory / "generations.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    provenance_digest = canonical_digest(provenance)
    records_digest = _records_digest(records)
    manifest = {
        "schema_version": BEHAVIORAL_ARTIFACT_SCHEMA_VERSION,
        "generation_digest": canonical_digest({
            "provenance_sha256": provenance_digest,
            "records_sha256": records_digest,
        }),
        "provenance_sha256": provenance_digest,
        "records_sha256": records_digest,
        "n_records": len(records),
        "provenance": dict(provenance),
    }
    (directory / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return GenerationArtifact.load(directory)


def generation_artifact_is_current(
    root: str | Path,
    provenance: Mapping[str, Any],
) -> bool:
    """Return whether a valid cached artifact matches ``provenance``.

    :param root: Artifact directory.
    :type root: str | pathlib.Path
    :param provenance: Expected JSON-safe provenance.
    :type provenance: Mapping[str, Any]
    :returns: ``True`` only for a valid matching artifact.
    :rtype: bool
    """
    try:
        artifact = GenerationArtifact.load(root)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return artifact.manifest.get("provenance_sha256") == canonical_digest(provenance)


def write_score_artifact(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    generation_digest: str,
    grader: Mapping[str, Any],
) -> dict[str, Any]:
    """Write re-runnable grader rows with independent provenance.

    :param path: Destination JSON file.
    :type path: str | pathlib.Path
    :param rows: Per-case score mappings.
    :type rows: Sequence[Mapping[str, Any]]
    :param generation_digest: Source generation digest.
    :type generation_digest: str
    :param grader: Grader identity/configuration.
    :type grader: Mapping[str, Any]
    :returns: Written payload.
    :rtype: dict[str, Any]
    """
    provenance = {"generation_digest": generation_digest, "grader": dict(grader)}
    payload = {
        "schema_version": BEHAVIORAL_ARTIFACT_SCHEMA_VERSION,
        "score_digest": canonical_digest(provenance),
        "provenance": provenance,
        "rows": [dict(row) for row in rows],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


__all__ = [
    "BEHAVIORAL_ARTIFACT_SCHEMA_VERSION",
    "GenerationArtifact",
    "GenerationRecord",
    "canonical_digest",
    "generation_artifact_is_current",
    "write_generation_artifact",
    "write_score_artifact",
]
