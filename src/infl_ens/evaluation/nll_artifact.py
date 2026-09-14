"""Versioned, provenance-checked per-record NLL matrix artifacts.

The expensive part of routed-ensemble evaluation is scoring every saved
adapter on every held-out record.  Routing policies are cheap matrix
operations, so this module stores the model-dependent quantities once and
rejects a cache whenever the data, checkpoints, or scoring configuration no
longer match.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

NLL_ARTIFACT_SCHEMA_VERSION = 1


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def checkpoint_sha256(path: str | Path) -> str:
    """Hash a checkpoint file or every file below a checkpoint directory.

    Directory hashes include relative file names as well as contents so a
    rename or missing adapter component invalidates the artifact.

    :param path: Checkpoint file or directory.
    :type path: str | pathlib.Path
    :returns: Hexadecimal SHA-256 digest.
    :rtype: str
    :raises FileNotFoundError: If ``path`` does not exist.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for file_path in files:
        relative = file_path.name if root.is_file() else file_path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def ordered_records_sha256(
    record_ids: Sequence[str],
    prompts: Sequence[str],
    responses: Sequence[str | None],
) -> str:
    """Hash an ordered evaluation record sequence.

    :param record_ids: Stable record identifiers.
    :type record_ids: Sequence[str]
    :param prompts: Prompt strings.
    :type prompts: Sequence[str]
    :param responses: Target responses.
    :type responses: Sequence[str | None]
    :returns: Hexadecimal SHA-256 digest.
    :rtype: str
    :raises ValueError: If the three sequences are not aligned.
    """
    if not (len(record_ids) == len(prompts) == len(responses)):
        raise ValueError("record_ids, prompts, and responses must have equal length")
    digest = hashlib.sha256()
    for record_id, prompt, response in zip(record_ids, prompts, responses):
        for value in (record_id, prompt, response or ""):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class NllMatrixArtifact:
    """All router-independent arrays needed for ensemble evaluation.

    :param expert_nll: Mean-token NLL matrix, shape ``(M, K)``.
    :type expert_nll: numpy.ndarray
    :param reference_nll: Reference/generalist NLL matrix, shape ``(M, B)``.
    :type reference_nll: numpy.ndarray
    :param token_counts: Supervised token count per record, shape ``(M,)``.
    :type token_counts: numpy.ndarray
    :param expert_names: Expert names aligned with ``expert_nll`` columns.
    :type expert_names: tuple[str, ...]
    :param reference_names: Reference names aligned with ``reference_nll``.
    :type reference_names: tuple[str, ...]
    :param bench_labels: Benchmark label per record.
    :type bench_labels: tuple[str, ...]
    :param record_ids: Stable record identifier per row.
    :type record_ids: tuple[str, ...]
    :param provenance: JSON-safe scoring provenance.
    :type provenance: dict[str, Any]
    """

    expert_nll: np.ndarray
    reference_nll: np.ndarray
    token_counts: np.ndarray
    expert_names: tuple[str, ...]
    reference_names: tuple[str, ...]
    bench_labels: tuple[str, ...]
    record_ids: tuple[str, ...]
    provenance: dict[str, Any]

    def validate(self) -> None:
        """Validate shapes, values, names, and schema provenance.

        :raises ValueError: If any artifact invariant is violated.
        """
        expert = np.asarray(self.expert_nll, dtype=float)
        reference = np.asarray(self.reference_nll, dtype=float)
        tokens = np.asarray(self.token_counts, dtype=float)
        if expert.ndim != 2:
            raise ValueError(f"expert_nll must be 2-D, got {expert.shape}")
        m, k = expert.shape
        if reference.ndim != 2 or reference.shape[0] != m:
            raise ValueError(
                f"reference_nll must have shape ({m}, B), got {reference.shape}"
            )
        if tokens.shape != (m,):
            raise ValueError(f"token_counts must have shape ({m},), got {tokens.shape}")
        if len(self.expert_names) != k:
            raise ValueError("expert_names do not match expert_nll columns")
        if len(self.reference_names) != reference.shape[1]:
            raise ValueError("reference_names do not match reference_nll columns")
        if len(self.bench_labels) != m or len(self.record_ids) != m:
            raise ValueError("bench_labels and record_ids must have one entry per row")
        if len(set(self.expert_names)) != len(self.expert_names):
            raise ValueError("expert_names must be unique")
        if len(set(self.reference_names)) != len(self.reference_names):
            raise ValueError("reference_names must be unique")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("record_ids must be unique")
        if not np.isfinite(expert).all() or not np.isfinite(reference).all():
            raise ValueError("NLL arrays must contain only finite values")
        if not np.isfinite(tokens).all() or np.any(tokens < 1):
            raise ValueError("token_counts must be finite and at least one")
        version = int(self.provenance.get("schema_version", -1))
        if version != NLL_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported NLL artifact schema {version}; "
                f"expected {NLL_ARTIFACT_SCHEMA_VERSION}"
            )

    @property
    def digest(self) -> str:
        """Stable digest used by routing-stage idempotence.

        :returns: Hexadecimal SHA-256 digest.
        :rtype: str
        """
        self.validate()
        digest = hashlib.sha256(_canonical_json(self.provenance).encode("utf-8"))
        for values in (
            self.expert_names,
            self.reference_names,
            self.bench_labels,
            self.record_ids,
        ):
            digest.update("\0".join(values).encode("utf-8"))
        for array in (self.expert_nll, self.reference_nll, self.token_counts):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> Path:
        """Write the artifact as a compressed, pickle-free ``.npz``.

        :param path: Destination path.
        :type path: str | pathlib.Path
        :returns: Written path.
        :rtype: pathlib.Path
        """
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            expert_nll=np.asarray(self.expert_nll, dtype=np.float64),
            reference_nll=np.asarray(self.reference_nll, dtype=np.float64),
            token_counts=np.asarray(self.token_counts, dtype=np.float64),
            expert_names=np.asarray(self.expert_names, dtype=np.str_),
            reference_names=np.asarray(self.reference_names, dtype=np.str_),
            bench_labels=np.asarray(self.bench_labels, dtype=np.str_),
            record_ids=np.asarray(self.record_ids, dtype=np.str_),
            provenance_json=np.asarray(_canonical_json(self.provenance), dtype=np.str_),
        )
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_provenance: Mapping[str, Any] | None = None,
    ) -> "NllMatrixArtifact":
        """Load and validate a pickle-free artifact.

        :param path: Artifact path.
        :type path: str | pathlib.Path
        :param expected_provenance: Exact provenance expected by the caller.
        :type expected_provenance: Mapping[str, Any] | None
        :returns: Validated artifact.
        :rtype: NllMatrixArtifact
        :raises ValueError: If validation or provenance comparison fails.
        """
        with np.load(Path(path), allow_pickle=False) as payload:
            required = {
                "expert_nll",
                "reference_nll",
                "token_counts",
                "expert_names",
                "reference_names",
                "bench_labels",
                "record_ids",
                "provenance_json",
            }
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"NLL artifact missing fields: {missing}")
            provenance = json.loads(str(payload["provenance_json"].item()))
            artifact = cls(
                expert_nll=np.asarray(payload["expert_nll"], dtype=float),
                reference_nll=np.asarray(payload["reference_nll"], dtype=float),
                token_counts=np.asarray(payload["token_counts"], dtype=float),
                expert_names=tuple(str(x) for x in payload["expert_names"].tolist()),
                reference_names=tuple(str(x) for x in payload["reference_names"].tolist()),
                bench_labels=tuple(str(x) for x in payload["bench_labels"].tolist()),
                record_ids=tuple(str(x) for x in payload["record_ids"].tolist()),
                provenance=dict(provenance),
            )
        artifact.validate()
        provenance_mismatch = (
            expected_provenance is not None
            and _canonical_json(artifact.provenance)
            != _canonical_json(expected_provenance)
        )
        if provenance_mismatch:
            raise ValueError("NLL artifact provenance does not match the requested evaluation")
        return artifact


__all__ = [
    "NLL_ARTIFACT_SCHEMA_VERSION",
    "NllMatrixArtifact",
    "checkpoint_sha256",
    "ordered_records_sha256",
]
