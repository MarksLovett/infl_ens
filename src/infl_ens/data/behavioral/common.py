"""Private file-reading helpers shared by behavioral benchmark loaders."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from infl_ens.data.behavioral.base import ChatMessage


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON, JSONL or CSV records from a file or directory.

    Directories are traversed recursively in lexical order. JSON mappings
    may contain a conventional ``data``, ``records``, ``examples`` or
    ``items`` list.

    :param path: Input file or directory.
    :type path: str | pathlib.Path
    :returns: Row mappings in deterministic order.
    :rtype: list[dict[str, Any]]
    :raises FileNotFoundError: If ``path`` is absent or has no supported files.
    :raises ValueError: If a file has an unsupported or malformed shape.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    files = (
        [root]
        if root.is_file()
        else sorted(
            file
            for suffix in ("*.jsonl", "*.json", "*.csv")
            for file in root.rglob(suffix)
        )
    )
    if not files:
        raise FileNotFoundError(f"no JSON, JSONL or CSV files under {root}")
    rows: list[dict[str, Any]] = []
    for file in files:
        suffix = file.suffix.lower()
        if suffix == ".jsonl":
            with file.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise ValueError(f"{file}:{line_number}: expected a mapping")
                    row = dict(value)
                    row.setdefault("_source_file", file.as_posix())
                    rows.append(row)
        elif suffix == ".csv":
            with file.open("r", encoding="utf-8-sig", newline="") as handle:
                for value in csv.DictReader(handle):
                    row = dict(value)
                    row.setdefault("_source_file", file.as_posix())
                    rows.append(row)
        else:
            value = json.loads(file.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                nested = next(
                    (
                        value[key]
                        for key in ("data", "records", "examples", "items")
                        if isinstance(value.get(key), list)
                    ),
                    None,
                )
                values = nested if nested is not None else [value]
            elif isinstance(value, list):
                values = value
            else:
                raise ValueError(f"{file}: expected a mapping or list")
            for item in values:
                if not isinstance(item, Mapping):
                    continue
                row = dict(item)
                row.setdefault("_source_file", file.as_posix())
                rows.append(row)
    return rows


def first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    """Return the first non-empty string-like value among ``keys``.

    :param row: Source record.
    :type row: Mapping[str, Any]
    :param keys: Candidate field names.
    :type keys: Sequence[str]
    :returns: Stripped text, or ``None``.
    :rtype: str | None
    """
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def coerce_messages(
    row: Mapping[str, Any],
    *,
    prompt_keys: Sequence[str],
    system_prompt: str | None = None,
) -> tuple[ChatMessage, ...] | None:
    """Build model-visible messages from a row.

    :param row: Source record.
    :type row: Mapping[str, Any]
    :param prompt_keys: Fallback keys for a single user prompt.
    :type prompt_keys: Sequence[str]
    :param system_prompt: Optional fixed system message.
    :type system_prompt: str | None
    :returns: Ordered messages, or ``None`` if no prompt is present.
    :rtype: tuple[ChatMessage, ...] | None
    """
    raw_messages = row.get("messages")
    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage("system", system_prompt))
    if isinstance(raw_messages, list):
        for item in raw_messages:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role", "user")).lower()
            content = item.get("content")
            if content is not None and str(content).strip():
                messages.append(ChatMessage(role, str(content).strip()))
    else:
        prompt = first_text(row, prompt_keys)
        if prompt:
            messages.append(ChatMessage("user", prompt))
    return tuple(messages) if messages else None


def split_values(value: Any) -> list[str]:
    """Coerce common list encodings to a list of strings.

    :param value: List, JSON-encoded list, semicolon-delimited string or scalar.
    :type value: Any
    :returns: String values.
    :rtype: list[str]
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in text.split(";") if part.strip()]


def coerce_bool(value: Any, *, default: bool) -> bool:
    """Coerce common JSON/CSV boolean encodings.

    :param value: Boolean, number, string or ``None``.
    :type value: Any
    :param default: Value used for absent or unrecognized inputs.
    :type default: bool
    :returns: Parsed boolean.
    :rtype: bool
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return default
    normalized = str(value).strip().casefold()
    if normalized in {"true", "yes", "1", "y"}:
        return True
    if normalized in {"false", "no", "0", "n"}:
        return False
    return default


def source_files(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return sorted source-file paths recorded by :func:`read_rows`.

    :param rows: Parsed source records.
    :type rows: Iterable[Mapping[str, Any]]
    :returns: Sorted unique file paths.
    :rtype: list[str]
    """
    return sorted({str(row.get("_source_file")) for row in rows if row.get("_source_file")})
