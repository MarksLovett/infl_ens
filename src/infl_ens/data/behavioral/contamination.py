"""Outcome-blind overlap checks for behavioral evaluation prompts."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from infl_ens.data.behavioral.base import BehavioralCase

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def normalize_prompt(text: str) -> str:
    """Normalize text for exact cross-corpus comparison.

    :param text: Input prompt or transcript.
    :type text: str
    :returns: NFKC, case-folded alphanumeric token string.
    :rtype: str
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_TOKEN_RE.findall(normalized))


def prompt_sha256(text: str) -> str:
    """Hash normalized prompt text.

    :param text: Input prompt.
    :type text: str
    :returns: SHA-256 hexadecimal digest.
    :rtype: str
    """
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


def token_ngrams(text: str, n: int) -> frozenset[tuple[str, ...]]:
    """Return normalized token n-grams.

    :param text: Input prompt.
    :type text: str
    :param n: N-gram width.
    :type n: int
    :returns: Unique token n-grams.
    :rtype: frozenset[tuple[str, ...]]
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    tokens = normalize_prompt(text).split()
    if len(tokens) < n:
        return frozenset({tuple(tokens)}) if tokens else frozenset()
    return frozenset(tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1))


def jaccard_similarity(left: frozenset[object], right: frozenset[object]) -> float:
    """Compute set Jaccard similarity with empty-set safeguards.

    :param left: First set.
    :type left: frozenset[object]
    :param right: Second set.
    :type right: frozenset[object]
    :returns: Similarity in ``[0, 1]``.
    :rtype: float
    """
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@dataclass(frozen=True)
class ContaminationRecord:
    """Overlap decision for one behavioral case.

    :param case_id: Behavioral case ID.
    :type case_id: str
    :param exact_match: Whether normalized text exactly matched training data.
    :type exact_match: bool
    :param fuzzy_match: Whether n-gram similarity crossed the threshold.
    :type fuzzy_match: bool
    :param maximum_similarity: Largest observed training-prompt similarity.
    :type maximum_similarity: float
    :param matched_training_index: Index of the nearest training prompt.
    :type matched_training_index: int | None
    :param excluded: Whether configured policy removes this case.
    :type excluded: bool
    """

    case_id: str
    exact_match: bool
    fuzzy_match: bool
    maximum_similarity: float
    matched_training_index: int | None
    excluded: bool


def audit_contamination(
    cases: Sequence[BehavioralCase],
    training_prompts: Sequence[str],
    *,
    exact_policy: str = "exclude",
    fuzzy_policy: str = "flag",
    ngram_size: int = 5,
    fuzzy_threshold: float = 0.85,
) -> list[ContaminationRecord]:
    """Audit behavioral prompts against training and validation text.

    :param cases: Behavioral cases.
    :type cases: Sequence[BehavioralCase]
    :param training_prompts: All prompts exposed during model/router fitting.
    :type training_prompts: Sequence[str]
    :param exact_policy: ``exclude`` or ``flag``.
    :type exact_policy: str
    :param fuzzy_policy: ``exclude``, ``flag`` or ``off``.
    :type fuzzy_policy: str
    :param ngram_size: Token n-gram width.
    :type ngram_size: int
    :param fuzzy_threshold: Jaccard threshold.
    :type fuzzy_threshold: float
    :returns: One decision per case in input order.
    :rtype: list[ContaminationRecord]
    """
    if exact_policy not in {"exclude", "flag"}:
        raise ValueError("exact_policy must be exclude or flag")
    if fuzzy_policy not in {"exclude", "flag", "off"}:
        raise ValueError("fuzzy_policy must be exclude, flag or off")
    if not 0.0 <= fuzzy_threshold <= 1.0:
        raise ValueError("fuzzy_threshold must lie in [0, 1]")
    exact_hashes = {prompt_sha256(prompt): index for index, prompt in enumerate(training_prompts)}
    train_ngrams = (
        [token_ngrams(prompt, ngram_size) for prompt in training_prompts]
        if fuzzy_policy != "off" else []
    )
    inverted: dict[tuple[str, ...], list[int]] = {}
    for index, grams in enumerate(train_ngrams):
        for gram in grams:
            inverted.setdefault(gram, []).append(index)
    records: list[ContaminationRecord] = []
    for case in cases:
        exact_index = exact_hashes.get(prompt_sha256(case.routing_text))
        best_similarity = 1.0 if exact_index is not None else 0.0
        best_index = exact_index
        if exact_index is None and train_ngrams:
            query = token_ngrams(case.routing_text, ngram_size)
            candidates = sorted({
                index
                for gram in query
                for index in inverted.get(gram, [])
            })
            for index in candidates:
                candidate = train_ngrams[index]
                score = jaccard_similarity(query, candidate)
                if score > best_similarity:
                    best_similarity = score
                    best_index = index
        fuzzy = exact_index is None and best_similarity >= fuzzy_threshold
        excluded = (
            (exact_index is not None and exact_policy == "exclude")
            or (fuzzy and fuzzy_policy == "exclude")
        )
        records.append(ContaminationRecord(
            case_id=case.case_id,
            exact_match=exact_index is not None,
            fuzzy_match=fuzzy,
            maximum_similarity=float(best_similarity),
            matched_training_index=best_index,
            excluded=excluded,
        ))
    return records


__all__ = [
    "ContaminationRecord",
    "audit_contamination",
    "jaccard_similarity",
    "normalize_prompt",
    "prompt_sha256",
    "token_ngrams",
]
